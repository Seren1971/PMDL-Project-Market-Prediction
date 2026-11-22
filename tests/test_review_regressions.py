"""Behavioral regressions for checkpoint loading and temporal evaluation."""

import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from report_analysis import paired_block_bootstrap
from src import TARGET, build_features
from src.experiments import ExperimentConfig, _mean_ic, evaluate_recipe, fit_predict, run_experiment

ROOT = Path(__file__).resolve().parents[1]


class ReviewRegressionTests(unittest.TestCase):
    def test_partial_checkpoint_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            model_path = directory / "model.cbm"
            model_path.write_bytes(b"existing model artifact")
            namespace = {"os": os, "warnings": warnings, "MODEL_PATH": model_path,
                         "FEATURES_PATH": directory / "missing-features.joblib"}
            notebook = json.loads((ROOT / "baselines/02_strong_baseline_61st.ipynb").read_text())
            with self.assertRaisesRegex(RuntimeError, "Incomplete checkpoint"):
                exec("".join(notebook["cells"][11]["source"]), namespace)
            self.assertEqual(model_path.read_bytes(), b"existing model artifact")

    def test_checkpoint_branch_restores_plot_inputs_in_fresh_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            model_path, features_path = directory / "model.cbm", directory / "features.joblib"
            x = pd.DataFrame({"M4": np.arange(50.), "V13": np.arange(50.) ** 2})
            trained = lgb.LGBMRegressor(n_estimators=2, min_child_samples=2, n_jobs=1, verbosity=-1)
            trained.fit(x, np.sin(np.arange(50.)))
            joblib.dump(trained, model_path)
            joblib.dump(list(x.columns), features_path)
            notebook = json.loads((ROOT / "baselines/02_strong_baseline_61st.ipynb").read_text())
            namespace = {"os": os, "warnings": warnings, "joblib": joblib,
                         "MODEL_PATH": model_path, "FEATURES_PATH": features_path}
            exec("".join(notebook["cells"][11]["source"]), namespace)
            self.assertEqual(namespace["FEATURES"], ["M4", "V13"])
            self.assertEqual(namespace["final_model"].n_features_in_, 2)
            gain = namespace["final_model"].booster_.feature_importance(importance_type="gain")
            self.assertEqual(gain.shape, (2,))

    def test_outer_labels_do_not_affect_fitted_predictions(self):
        rng = np.random.default_rng(5)
        train = pd.DataFrame({"M4": rng.normal(size=100), "V13": rng.normal(size=100),
                              TARGET: rng.normal(size=100)})
        validation = train.iloc[:20].copy()
        changed = validation.copy()
        changed[TARGET] = 1e6
        config = ExperimentConfig(max_estimators=3, n_jobs=1)
        for family, params in [("lgbm", {"n_estimators": 3}),
                               ("catboost", {"iterations": 3, "depth": 2}),
                               ("ridge", {"alpha": 1.})]:
            with self.subTest(family=family):
                a, _ = fit_predict(family, params, train, validation, ["M4", "V13"], config)
                b, _ = fit_predict(family, params, train, changed, ["M4", "V13"], config)
                np.testing.assert_array_equal(a, b)

    def test_future_features_do_not_change_earlier_imputation(self):
        train = pd.DataFrame({"M4": np.arange(100.), TARGET: np.sin(np.arange(100.))})
        val = pd.DataFrame({"M4": [np.nan, 2., np.nan, 4.], TARGET: [0.] * 4})
        changed = val.copy()
        changed.loc[3, "M4"] = 1e10
        config = ExperimentConfig(n_jobs=1)
        a, _ = fit_predict("ridge", {"alpha": 1.}, train, val, ["M4"], config)
        b, _ = fit_predict("ridge", {"alpha": 1.}, train, changed, ["M4"], config)
        np.testing.assert_array_equal(a[:3], b[:3])

    def test_feature_engineering_is_causal(self):
        rng = np.random.default_rng(8)
        frame = pd.DataFrame({"M4": rng.normal(size=120),
                              "forward_returns": rng.normal(0, .01, 120)})
        changed = frame.copy()
        changed.loc[80:, "forward_returns"] = 100.
        a, columns = build_features(frame, lag_roll_columns=["M4"])
        b, _ = build_features(changed, lag_roll_columns=["M4"])
        np.testing.assert_allclose(a.loc[:80, columns], b.loc[:80, columns], equal_nan=True)

    def test_warmup_labels_are_excluded_from_tuning_score(self):
        frame = pd.DataFrame({"fold": [0] * 6, "scored": [False] * 3 + [True] * 3,
                              TARGET: [9., 8., 7., 1., 2., 3.],
                              "prediction": np.arange(6.)})
        before = _mean_ic(frame)
        frame.loc[~frame.scored, TARGET] = [-1e9, 1e9, -1e9]
        self.assertEqual(before, _mean_ic(frame))
        self.assertEqual(before, 1.)

    def test_inverse_vol_blend_is_warmed_before_first_scored_row(self):
        n = 160
        x = np.linspace(-1., 1., n)
        frame = pd.DataFrame({"date_id": np.arange(n), "M4": x,
                              TARGET: 1. + .1 * x, "forward_returns": .01 + .001 * x,
                              "risk_free_rate": .0001, "vol_20": .2})
        recipe = {"models": {"lgbm": {"n_estimators": 3},
                              "catboost": {"iterations": 3, "depth": 2},
                              "ridge": {"alpha": 1.}},
                  "blend": "inverse_vol", "allocation": {"name": "naive", "params": {"k": .1}}}
        result, _ = evaluate_recipe(frame.iloc[:120], frame.iloc[140:], ["M4"], recipe,
                                     ExperimentConfig(n_jobs=1), history=frame.iloc[80:140])
        ensemble = result.loc[result.model == "improved"]
        self.assertGreater(ensemble.prediction.iloc[0], .5)
        self.assertEqual(ensemble.date_id.iloc[0], 140)
        self.assertEqual(len(ensemble), 20)

    def test_bootstrap_identical_strategies_has_zero_difference(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({"date_id": np.arange(80), "fold": 0,
                              "forward_returns": rng.normal(.001, .01, 80),
                              "risk_free_rate": .0001, "weight": 1.})
        both = pd.concat([frame.assign(model="a"), frame.assign(model="b")])
        result = paired_block_bootstrap(both, "a", "b", draws=30, block_size=10)
        self.assertEqual(result["observed_delta"], 0.)
        self.assertEqual(result["ci_low"], 0.)
        self.assertEqual(result["ci_high"], 0.)
        # Use a uniquely indexed frame to remove just one reference row.
        bad = both.reset_index(drop=True).iloc[:-1]
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_block_bootstrap(bad, "a", "b", draws=10)

    def test_nested_pipeline_records_strict_selection_boundaries(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            data_dir = directory / "data"
            data_dir.mkdir()
            n = 600
            fwd = rng.normal(.0004, .01, n)
            pd.DataFrame({"date_id": np.arange(n), "M4": rng.normal(size=n),
                          "V13": rng.normal(size=n), "forward_returns": fwd,
                          "risk_free_rate": .0001, TARGET: fwd - .0001}).to_csv(
                              data_dir / "train.csv", index=False)
            config = ExperimentConfig(seed_count=1, model_trials=1, allocation_trials=1,
                                      inner_splits=2, max_estimators=3, n_jobs=1)
            output = directory / "results/analysis"
            result = run_experiment(output, data_dir, config)
            manifest = json.loads((output / "manifest.json").read_text())
            for selection in manifest["outer_selections"]:
                self.assertLess(selection["selection_max_date_id"],
                                selection["evaluation_min_date_id"])
            self.assertEqual(set(result.model), {"proposed", "improved", "passive"})
            self.assertEqual(len(result.loc[(result.split == "public") & (result.model == "passive")]), 180)
            self.assertTrue(result.weight.between(0., 2.).all())


if __name__ == "__main__":
    unittest.main()
