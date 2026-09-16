"""Leak-safe utilities for the LightGBM + CatBoost + Ridge ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import SEED, TARGET
from .features import top_features_by_gain
from .metrics import spearman_ic
from .preprocessing import (
    ForwardMedianImputer,
    apply_standardization,
    standardization_stats,
)
from .validation import Fold, get_folds


FamilyParams = dict[str, Any]

# Linear models are sensitive to regime shifts in features whose training
# variance is almost zero. Clip train-standardised values using a threshold
# learned without validation data; this prevents a single feature from creating
# absurd Ridge extrapolations while preserving its direction and rank.
RIDGE_Z_CLIP = 10.0


@dataclass(frozen=True)
class ForecastCalibration:
    """Train-only affine mapping from blend signal to target-return units."""

    intercept: float
    slope: float


@dataclass
class PreparedFold:
    fold: Fold
    train: pd.DataFrame
    val: pd.DataFrame
    cat_features: list[str]


@dataclass
class FamilyOOF:
    raw_predictions: np.ndarray
    fold_scores: list[float]
    best_iterations: list[int]
    train_sizes: list[int]


@dataclass
class SignalStats:
    family_stats: dict[str, tuple[float, float]]
    blend_stats: tuple[float, float]


def fit_catboost_selector(
    train: pd.DataFrame,
    features: list[str],
    feature_count: int,
    jobs: int,
    iterations: int = 300,
) -> list[str]:
    """Fit feature selection on the current training rows only."""
    model = CatBoostRegressor(
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        random_seed=SEED,
        verbose=False,
        task_type="CPU",
        thread_count=jobs,
        loss_function="RMSE",
        boosting_type="Plain",
        allow_writing_files=False,
    )

    model.fit(
        train[features],
        train[TARGET],
    )

    return top_features_by_gain(
        model,
        features,
        k=min(
            int(feature_count),
            len(features),
        ),
    )


def prepare_folds(
    frame: pd.DataFrame,
    features: list[str],
    n_splits: int,
    feature_count: int,
    jobs: int,
    selector_iterations: int = 300,
) -> list[PreparedFold]:
    """
    Create inner folds with fold-local imputation and CatBoost selection.
    """
    prepared: list[PreparedFold] = []

    for fold in get_folds(
        frame,
        n_splits=n_splits,
    ):
        raw_train = frame.iloc[
            fold.train_idx
        ].copy()

        raw_val = frame.iloc[
            fold.val_idx
        ].copy()

        imputer = ForwardMedianImputer(
            features
        )

        train, val = imputer.fit_transform_pair(
            raw_train,
            raw_val,
        )

        cat_features = fit_catboost_selector(
            train,
            features,
            feature_count=feature_count,
            jobs=jobs,
            iterations=selector_iterations,
        )

        prepared.append(
            PreparedFold(
                fold=fold,
                train=train,
                val=val,
                cat_features=cat_features,
            )
        )

    return prepared


def fit_lgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    params: FamilyParams,
    jobs: int,
) -> tuple[np.ndarray, int]:
    model = lgb.LGBMRegressor(
        **params,
        random_state=SEED,
        verbosity=-1,
        n_jobs=jobs,
    )

    model.fit(
        train[features],
        train[TARGET],
        eval_set=[
            (
                val[features],
                val[TARGET],
            )
        ],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(
                150,
                verbose=False,
            )
        ],
    )

    best_iter = int(
        model.best_iteration_
        or params.get(
            "n_estimators",
            100,
        )
    )

    predictions = np.asarray(
        model.predict(
            val[features]
        ),
        dtype=float,
    )

    return predictions, best_iter


def fit_catboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    params: FamilyParams,
    jobs: int,
) -> tuple[np.ndarray, int]:
    cfg = dict(params)

    cfg.update(
        {
            "random_seed": SEED,
            "verbose": False,
            "loss_function": "RMSE",
            "task_type": "CPU",
            "thread_count": jobs,
            "boosting_type": "Plain",
            "early_stopping_rounds": 100,
            "allow_writing_files": False,
        }
    )

    model = CatBoostRegressor(
        **cfg
    )

    model.fit(
        train[features],
        train[TARGET],
        eval_set=(
            val[features],
            val[TARGET],
        ),
    )

    best = model.get_best_iteration()

    best_iter = int(
        (best + 1)
        if best is not None
        and best >= 0
        else params.get(
            "iterations",
            100,
        )
    )

    predictions = np.asarray(
        model.predict(
            val[features]
        ),
        dtype=float,
    )

    return predictions, best_iter


def _scaled_ridge_matrix(
    scaler: StandardScaler,
    frame: pd.DataFrame,
    features: list[str],
    z_clip: float = RIDGE_Z_CLIP,
) -> np.ndarray:
    """
    Transform Ridge inputs and cap out-of-regime z-scores.

    The scaler is always fitted on training rows only. Capping is deterministic
    and uses no validation statistics, so it is safe inside chronological CV.
    """
    matrix = np.asarray(
        scaler.transform(
            frame[features]
        ),
        dtype=float,
    )

    return np.clip(
        matrix,
        -float(z_clip),
        float(z_clip),
    )


def fit_ridge(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    params: FamilyParams,
) -> tuple[np.ndarray, None]:
    """
    Fit Ridge using train-only scaling and robust z-score clipping.

    Clipping prevents features with almost-zero training variance from producing
    extreme validation z-scores and exploding Ridge predictions.
    """
    scaler = StandardScaler().fit(
        train[features]
    )

    x_train = _scaled_ridge_matrix(
        scaler,
        train,
        features,
    )

    x_val = _scaled_ridge_matrix(
        scaler,
        val,
        features,
    )

    model = Ridge(
        alpha=float(
            params["alpha"]
        )
    )

    model.fit(
        x_train,
        train[TARGET],
    )

    predictions = np.asarray(
        model.predict(
            x_val
        ),
        dtype=float,
    )

    return predictions, None


def family_oof_predictions(
    prepared: list[PreparedFold],
    family: str,
    params: FamilyParams,
    full_features: list[str],
    jobs: int,
    n_rows: int,
) -> FamilyOOF:
    """
    OOF predictions from folds whose preprocessing is train-local.
    """
    out = np.full(
        n_rows,
        np.nan,
        dtype=float,
    )

    scores: list[float] = []
    iterations: list[int] = []
    train_sizes: list[int] = []

    for item in prepared:
        features = (
            item.cat_features
            if family == "catboost"
            else full_features
        )

        if family == "lgbm":
            pred, n_iter = fit_lgbm(
                item.train,
                item.val,
                features,
                params,
                jobs,
            )

        elif family == "catboost":
            pred, n_iter = fit_catboost(
                item.train,
                item.val,
                features,
                params,
                jobs,
            )

        elif family == "ridge":
            pred, n_iter = fit_ridge(
                item.train,
                item.val,
                features,
                params,
            )

        else:
            raise ValueError(
                f"Unknown family: {family}"
            )

        out[
            item.fold.val_idx
        ] = pred

        scores.append(
            spearman_ic(
                item.val[TARGET].to_numpy(),
                pred,
            )
        )

        train_sizes.append(
            len(item.train)
        )

        if n_iter is not None:
            iterations.append(
                int(n_iter)
            )

    return FamilyOOF(
        raw_predictions=out,
        fold_scores=scores,
        best_iterations=iterations,
        train_sizes=train_sizes,
    )


def lgbm_space(
    trial: optuna.Trial,
) -> FamilyParams:
    """
    Regularised LightGBM space with a valid leaves/depth relationship.
    """
    max_depth = trial.suggest_int(
        "max_depth",
        3,
        9,
    )

    max_leaves = min(
        256,
        2**max_depth,
    )

    return {
        "objective": "regression",
        "metric": "rmse",
        "subsample_freq": 1,
        "n_estimators": trial.suggest_int(
            "n_estimators",
            300,
            2500,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.008,
            0.08,
            log=True,
        ),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int(
            "num_leaves",
            8,
            max_leaves,
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            20,
            250,
        ),
        "min_split_gain": trial.suggest_float(
            "min_split_gain",
            0.0,
            0.05,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-3,
            30.0,
            log=True,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-5,
            5.0,
            log=True,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.55,
            1.0,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.55,
            1.0,
        ),
    }


def catboost_space(
    trial: optuna.Trial,
) -> FamilyParams:
    return {
        "iterations": trial.suggest_int(
            "iterations",
            250,
            1600,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.008,
            0.12,
            log=True,
        ),
        "depth": trial.suggest_int(
            "depth",
            4,
            9,
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            1.0,
            40.0,
            log=True,
        ),
        "random_strength": trial.suggest_float(
            "random_strength",
            1e-3,
            3.0,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.55,
            1.0,
        ),
        "bootstrap_type": "Bernoulli",
    }


def ridge_space(
    trial: optuna.Trial,
) -> FamilyParams:
    return {
        "alpha": trial.suggest_float(
            "alpha",
            1e-3,
            1e5,
            log=True,
        )
    }


def tune_family(
    family: str,
    prepared: list[PreparedFold],
    full_features: list[str],
    jobs: int,
    n_rows: int,
    n_trials: int,
) -> tuple[
    FamilyParams,
    optuna.Study,
]:
    spaces: dict[
        str,
        Callable[
            [optuna.Trial],
            FamilyParams,
        ],
    ] = {
        "lgbm": lgbm_space,
        "catboost": catboost_space,
        "ridge": ridge_space,
    }

    def objective(
        trial: optuna.Trial,
    ) -> float:
        params = spaces[
            family
        ](
            trial
        )

        if family == "catboost":
            params = {
                "boosting_type": "Plain",
                **params,
            }

        result = family_oof_predictions(
            prepared,
            family,
            params,
            full_features,
            jobs,
            n_rows,
        )

        scores = np.asarray(
            result.fold_scores,
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                scores
            )
        ):
            return -1e9

        return float(
            np.mean(
                scores
            )
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED
        ),
    )

    # Start every search from one sensible regularised configuration so even a
    # tiny smoke run contains a meaningful model rather than only random draws.
    if family == "lgbm":
        study.enqueue_trial(
            {
                "n_estimators": 1000,
                "learning_rate": 0.02,
                "max_depth": 5,
                "num_leaves": 31,
                "min_child_samples": 40,
                "min_split_gain": 0.0,
                "reg_lambda": 0.1,
                "reg_alpha": 0.01,
                "colsample_bytree": 0.8,
                "subsample": 0.8,
            }
        )

    elif family == "catboost":
        study.enqueue_trial(
            {
                "iterations": 600,
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 5.0,
                "random_strength": 0.1,
                "subsample": 0.8,
            }
        )

    elif family == "ridge":
        study.enqueue_trial(
            {
                "alpha": 1000.0
            }
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best: FamilyParams = dict(
        study.best_params
    )

    if family == "lgbm":
        best.update(
            {
                "objective": "regression",
                "metric": "rmse",
                "subsample_freq": 1,
            }
        )

    elif family == "catboost":
        best.update(
            {
                "bootstrap_type": "Bernoulli",
                "boosting_type": "Plain",
            }
        )

    return best, study


def standardize_family_oof(
    raw: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        tuple[
            float,
            float,
        ],
    ],
]:
    stats = {
        column: standardization_stats(
            raw[column].to_numpy()
        )
        for column in raw.columns
    }

    scaled = pd.DataFrame(
        {
            column: apply_standardization(
                raw[column].to_numpy(),
                stats[column],
            )
            for column in raw.columns
        },
        index=raw.index,
    )

    return scaled, stats


def blend(
    preds: pd.DataFrame,
    weights: dict[
        str,
        float,
    ],
) -> np.ndarray:
    total = float(
        sum(
            weights.values()
        )
    )

    if total <= 0:
        raise ValueError(
            "Blend weights must sum to a positive value."
        )

    output = np.zeros(
        len(preds),
        dtype=float,
    )

    for column in preds.columns:
        output += (
            preds[column].to_numpy(
                dtype=float
            )
            * float(
                weights[column]
            )
        )

    return output / total


def fit_forecast_calibration(
    signal: np.ndarray,
    target: np.ndarray,
) -> ForecastCalibration:
    """
    Fit a non-negative affine calibration in target-return units.

    Ensemble weights are selected on standardised family predictions because
    Spearman optimisation cares about ranking rather than raw model scale.

    Applying those same weights directly to unscaled family predictions is not
    valid when model families have different variances.

    This calibration maps the already-blended OOF signal back to the target
    scale using inner OOF data only.
    """
    x = np.asarray(
        signal,
        dtype=float,
    )

    y = np.asarray(
        target,
        dtype=float,
    )

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if finite.sum() < 2:
        return ForecastCalibration(
            intercept=float(
                np.nanmean(y)
            ),
            slope=0.0,
        )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    x_mean = float(
        np.mean(x)
    )

    y_mean = float(
        np.mean(y)
    )

    variance = float(
        np.mean(
            (
                x - x_mean
            )
            ** 2
        )
    )

    if (
        not np.isfinite(
            variance
        )
        or variance <= 1e-15
    ):
        return ForecastCalibration(
            intercept=y_mean,
            slope=0.0,
        )

    covariance = float(
        np.mean(
            (
                x - x_mean
            )
            * (
                y - y_mean
            )
        )
    )

    slope = max(
        0.0,
        covariance / variance,
    )

    intercept = (
        y_mean
        - slope * x_mean
    )

    return ForecastCalibration(
        intercept=float(
            intercept
        ),
        slope=float(
            slope
        ),
    )


def apply_forecast_calibration(
    signal: np.ndarray,
    calibration: ForecastCalibration,
) -> np.ndarray:
    """
    Map an ensemble signal to target units with frozen train-only values.
    """
    x = np.asarray(
        signal,
        dtype=float,
    )

    return (
        calibration.intercept
        + calibration.slope
        * x
    )


def tune_blend_weights(
    scaled_oof: pd.DataFrame,
    y_oof: np.ndarray,
    fold_slices: list[
        np.ndarray
    ],
    n_trials: int,
) -> tuple[
    dict[
        str,
        float,
    ],
    optuna.Study,
]:
    """
    Tune blend weights on inner OOF only.

    Outer validation is never seen.
    """

    def objective(
        trial: optuna.Trial,
    ) -> float:
        raw = {
            column: trial.suggest_float(
                f"w_{column}",
                0.0,
                1.0,
            )
            for column in scaled_oof.columns
        }

        total = sum(
            raw.values()
        )

        if total <= 1e-8:
            return -1e9

        pred = blend(
            scaled_oof,
            raw,
        )

        scores = [
            spearman_ic(
                y_oof[
                    fold_slice
                ],
                pred[
                    fold_slice
                ],
            )
            for fold_slice in fold_slices
        ]

        if not np.all(
            np.isfinite(
                scores
            )
        ):
            return -1e9

        return float(
            np.mean(
                scores
            )
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED
        ),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    raw = {
        column: float(
            study.best_params[
                f"w_{column}"
            ]
        )
        for column in scaled_oof.columns
    }

    total = sum(
        raw.values()
    )

    return (
        {
            column: value / total
            for column, value in raw.items()
        },
        study,
    )