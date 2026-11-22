"""Nested chronological evaluation; all selection is confined to training rows.

The public period was consulted during historical development and remains a
diagnostic. Outer CV estimates selection performance on the available history;
it is not a claim of a new, prospectively untouched test set.
"""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .allocation import (
    binary_allocation, naive_allocation, ternary_allocation, vol_target_allocation,
)
from .config import DATE_COL, LOOKAHEAD_COLS, SEED, TARGET
from .data import _resolve_data_dir, impute, load_dataset
from .features import blend_signals, build_features
from .metrics import aggregate_folds, evaluate, modified_sharpe, spearman_ic
from .results import save_predictions, save_result
from .validation import assert_no_leakage, get_folds

LAG_COLUMNS = ["M4", "V13", "S5", "S2", "D2", "E19", "P7", "P6",
               "P3", "P13", "P4", "P5", "M2", "V5"]
MODEL_NAMES = {"proposed": "hybrid_domain_lgbm",
               "improved": "ensemble_lgbm_catboost_ridge", "passive": "passive_w1"}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = SEED
    seed_count: int = 20
    model_trials: int = 8
    allocation_trials: int = 16
    inner_splits: int = 3
    max_estimators: int = 300
    n_jobs: int = 2

    def __post_init__(self):
        if min(self.seed_count, self.model_trials, self.allocation_trials,
               self.max_estimators, self.n_jobs) < 1 or self.inner_splits < 2:
            raise ValueError("Experiment budgets must be positive; inner_splits >= 2")


def feature_frames(data_dir=None):
    hull = load_dataset(data_dir)
    featured, features = build_features(hull.full, lag_roll_columns=LAG_COLUMNS)
    if set(features) & set(LOOKAHEAD_COLS):
        raise AssertionError("Look-ahead field in model features")
    domain = [c for c in features if c.startswith(
        ("mom_", "vol_", "revert_", "drawdown", "ret_1d"))]
    cutoff = hull.train[DATE_COL].max()
    train = featured.loc[featured[DATE_COL] <= cutoff].reset_index(drop=True)
    public = featured.loc[featured[DATE_COL] > cutoff].reset_index(drop=True)
    # Statistics are deliberately NOT fitted here. Every fit imputes its own
    # training/validation frames independently of later rows and labels.
    return train, public, features, domain


def fit_predict(family, params, train, validation, features, config, seed=None):
    seed = config.seed if seed is None else seed
    clean = [f[features].replace([np.inf, -np.inf], np.nan)
             for f in (train, validation)]
    x_train, x_val = impute(*clean, columns=features)
    if family == "lgbm":
        model = lgb.LGBMRegressor(
            **params, random_state=seed, n_jobs=config.n_jobs, verbosity=-1,
            deterministic=True, force_col_wise=True, subsample_freq=1,
            importance_type="gain",
        )
    elif family == "catboost":
        model = CatBoostRegressor(
            **params, random_seed=seed, thread_count=config.n_jobs,
            loss_function="RMSE", verbose=False, allow_writing_files=False,
            bootstrap_type="Bernoulli",
        )
    elif family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(**params))
    else:
        raise ValueError(f"Unknown model family: {family}")
    # Fixed iteration count selected on INNER folds. No outer eval_set or
    # early stopping: outer labels cannot affect the fitted estimator.
    model.fit(x_train, train[TARGET])
    importance = (model.booster_.feature_importance(importance_type="gain")
                  if family == "lgbm" else None)
    return model.predict(x_val), importance


def _parameters(trial, family, config):
    if family == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e4, log=True)}
    iterations = trial.suggest_int("iterations", min(60, config.max_estimators),
                                   config.max_estimators)
    params = {"learning_rate": trial.suggest_float("learning_rate", .01, .07, log=True),
              "subsample": trial.suggest_float("subsample", .6, 1.0)}
    if family == "catboost":
        return {**params, "iterations": iterations,
                "depth": trial.suggest_int("depth", 4, 6),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", .01, 10., log=True)}
    return {**params, "n_estimators": iterations,
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", .6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", .001, 10., log=True)}


def _inner_predictions(family, params, train, features, folds, config):
    frames = []
    for fold in folds:
        validation = train.iloc[fold.val_idx]
        history = train.iloc[max(0, fold.val_idx[0] - 60):fold.val_idx[0]]
        prediction_frame = pd.concat([history, validation], ignore_index=True)
        pred, _ = fit_predict(family, params, train.iloc[fold.train_idx],
                              prediction_frame, features, config)
        frame = prediction_frame[[DATE_COL, TARGET, "forward_returns", "risk_free_rate",
                                  "vol_20"]].copy()
        frame["fold"], frame["prediction"] = fold.index, pred
        frame["scored"] = np.r_[np.zeros(len(history), dtype=bool),
                                np.ones(len(validation), dtype=bool)]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _mean_ic(frame, column="prediction"):
    if "scored" in frame:
        frame = frame.loc[frame.scored]
    return float(np.mean([spearman_ic(g[TARGET], g[column])
                          for _, g in frame.groupby("fold", sort=True)]))


def tune_model(family, train, features, folds, config):
    def objective(trial):
        params = _parameters(trial, family, config)
        trial.set_user_attr("model_params", params)
        return _mean_ic(_inner_predictions(family, params, train, features, folds, config))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=config.model_trials)
    params = study.best_trial.user_attrs["model_params"]
    return params, _inner_predictions(family, params, train, features, folds, config)


def blend_predictions(frame):
    out = frame.copy()
    columns = ["lgbm", "catboost", "ridge"]
    out["average"] = out[columns].mean(axis=1)
    out["inverse_vol"] = blend_signals(out[columns]).fillna(0.).to_numpy()
    return out


def allocation(prediction, frame, recipe):
    name, params = recipe["name"], recipe["params"]
    if name == "passive":
        return np.ones(len(prediction))
    if name == "naive":
        return naive_allocation(prediction, **params)
    if name == "binary":
        return binary_allocation(prediction, **params)
    if name == "ternary":
        return ternary_allocation(prediction, **params)
    if name == "vol_target":
        return vol_target_allocation(prediction, frame["vol_20"].to_numpy(),
                                      target_vol=.12, **params)
    raise ValueError(f"Unknown allocation rule: {name}")


def select_allocation(oof, column, config):
    def score(recipe):
        scores = [modified_sharpe(allocation(g[column], g, recipe),
                                  g.forward_returns, g.risk_free_rate)
                  for _, g in oof.groupby("fold", sort=True)]
        return float(np.mean(scores) - .25 * np.std(scores))

    candidates = [{"name": "passive", "params": {}}]
    std = max(float(oof[column].std()), 1e-12)
    for name in ["naive", "binary", "ternary", "vol_target"]:
        def objective(trial):
            if name in ("naive", "vol_target"):
                params = {"k": trial.suggest_float("k", .1, 2000., log=True)}
            else:
                lo = -3 * std if name == "binary" else 0.
                params = {"threshold": trial.suggest_float("threshold", lo, 3 * std)}
            return score({"name": name, "params": params})

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=config.seed))
        study.optimize(objective, n_trials=config.allocation_trials)
        candidates.append({"name": name, "params": study.best_params})
    selected = max(candidates, key=score)
    return selected, [{**c, "inner_objective": score(c)} for c in candidates]


def select_recipe(train, features, config):
    """All input rows precede the outer validation block. No public input."""
    folds = get_folds(train, n_splits=config.inner_splits)
    assert_no_leakage(folds)
    params, frames = {}, {}
    for family in ("lgbm", "catboost", "ridge"):
        params[family], frames[family] = tune_model(family, train, features, folds, config)
    oof = frames["lgbm"].drop(columns="prediction")
    for family, frame in frames.items():
        if not np.array_equal(oof[DATE_COL], frame[DATE_COL]):
            raise AssertionError("Inner OOF alignment mismatch")
        oof[family] = frame.prediction.to_numpy()
    # Reset rolling blend history per fold; never cross a skipped CV gap.
    oof = pd.concat([blend_predictions(g) for _, g in oof.groupby("fold", sort=True)],
                    ignore_index=True)
    # Warm-up rows do not participate in validation objectives or scores.
    oof = oof.loc[oof.scored].copy()
    blend = max(["average", "inverse_vol"], key=lambda col: _mean_ic(oof, col))
    rule, candidates = select_allocation(oof, blend, config)
    recipe = {"models": params, "blend": blend, "allocation": rule,
              "allocation_candidates": candidates,
              "selection_max_date_id": int(train[DATE_COL].max()),
              "inner_tuning_ic": _mean_ic(oof, blend)}
    return recipe


def evaluate_recipe(train, validation, features, recipe, config, history=None):
    history = train.tail(60) if history is None else history
    if len(history) and history[DATE_COL].max() >= validation[DATE_COL].min():
        raise ValueError("Blend history must strictly precede evaluation")
    prediction_frame = pd.concat([history, validation], ignore_index=True)
    preds, gain = {}, None
    for family, params in recipe["models"].items():
        preds[family], importance = fit_predict(family, params, train, prediction_frame,
                                                features, config)
        if importance is not None:
            gain = importance
    blend = blend_predictions(pd.DataFrame(preds))[recipe["blend"]].to_numpy()[-len(validation):]
    preds = {family: values[-len(validation):] for family, values in preds.items()}
    outputs = []
    for name, pred, weights in [
        ("proposed", preds["lgbm"], naive_allocation(preds["lgbm"], k=50)),
        ("improved", blend, allocation(blend, validation, recipe["allocation"])),
        ("passive", np.zeros(len(validation)), np.ones(len(validation))),
    ]:
        frame = validation[[DATE_COL, TARGET, "forward_returns", "risk_free_rate"]].copy()
        frame["model"], frame["prediction"], frame["weight"] = name, pred, weights
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True), gain


def partition_metrics(frame):
    return evaluate(frame[TARGET].to_numpy(), frame.prediction.to_numpy(),
                    weights=frame.weight.to_numpy(),
                    forward_returns=frame.forward_returns.to_numpy(),
                    risk_free_rate=frame.risk_free_rate.to_numpy())


def run_experiment(output_dir, data_dir=None, config=None):
    config = config or ExperimentConfig()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, public, features, domain = feature_frames(data_dir)
    no_domain = [c for c in features if c not in domain]
    folds = get_folds(train)
    assert_no_leakage(folds)
    predictions, ablation, importances, selections = [], [], [], []
    for fold in folds:
        development = train.iloc[fold.train_idx].reset_index(drop=True)
        validation = train.iloc[fold.val_idx].reset_index(drop=True)
        history = train.iloc[max(0, fold.val_idx[0] - 60):fold.val_idx[0]]
        print(f"Outer fold {fold.index}: select on {len(development)} earlier rows, "
              f"evaluate on {len(validation)} later rows", flush=True)
        recipe = select_recipe(development, features, config)
        result, gain = evaluate_recipe(development, validation, features, recipe, config,
                                      history=history)
        result["fold"], result["split"] = fold.index, "cv"
        predictions.append(result)
        importances.append(gain)
        selections.append({"outer_fold": fold.index, **recipe,
                           "evaluation_min_date_id": int(validation[DATE_COL].min())})
        # Paired, fixed-parameter fit-seed sensitivity. Identical params, data,
        # folds and budgets; only inclusion of the domain block differs.
        # This does not rerun the full hyperparameter search for each seed.
        for seed in range(config.seed_count):
            scores = []
            for columns in (features, no_domain):
                prediction_frame = pd.concat([history, validation], ignore_index=True)
                pred, _ = fit_predict("lgbm", recipe["models"]["lgbm"], development,
                                      prediction_frame, columns, config, seed=seed)
                scores.append(spearman_ic(validation[TARGET], pred[-len(validation):]))
            ablation.append({"fold": fold.index, "seed": seed,
                             "with_domain": scores[0], "without_domain": scores[1],
                             "delta": scores[0] - scores[1]})
        print(f"Outer fold {fold.index} complete; paired seeds: {config.seed_count}", flush=True)
    print("Refit selected pipeline for the historically consulted public diagnostic", flush=True)
    public_recipe = select_recipe(train, features, config)
    result, _ = evaluate_recipe(train, public, features, public_recipe, config,
                                history=train.tail(60))
    result["fold"], result["split"] = -1, "public"
    predictions.append(result)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(ablation).to_csv(output_dir / "ablation_folds.csv", index=False)
    gains = pd.DataFrame({"feature": features, "gain": np.mean(importances, axis=0)})
    gains["block"] = ["domain" if c in domain else
                      "lag_roll" if "_lag_" in c or "_roll_" in c else "raw_cross"
                      for c in features]
    gains.to_csv(output_dir / "feature_gain.csv", index=False)
    rows = []
    for (split, model), group in all_predictions.groupby(["split", "model"]):
        stage = {"proposed": "proposed", "improved": "improved", "passive": "benchmark"}[model]
        if split == "cv":
            metrics = aggregate_folds([partition_metrics(g) for _, g in group.groupby("fold")])
        else:
            metrics = partition_metrics(group)
            save_predictions(MODEL_NAMES[model], group[DATE_COL], group[TARGET],
                             group.prediction, group.weight,
                             results_dir=output_dir.parent / "artifacts")
        save_result(MODEL_NAMES[model], stage, metrics, split=split,
                    notes="Nested chronological selection" if split == "cv" else
                          "Historically consulted public diagnostic; not a fresh holdout",
                    params={"config": asdict(config)}, results_dir=output_dir.parent)
        rows.append({"split": split, "model": model, **metrics})
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    raw_path = _resolve_data_dir(data_dir) / "train.csv"
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "scipy", "lightgbm", "catboost", "scikit-learn", "optuna")}
    manifest = {"config": asdict(config), "versions": versions,
                "python": platform.python_version(),
                "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "public_status": "historically consulted; diagnostic only",
                "ablation_protocol": "paired fit seeds; hyperparameters fixed within each outer fold",
                "outer_selections": selections, "public_selection": public_recipe}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return all_predictions
