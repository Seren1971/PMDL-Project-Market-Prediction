"""Reproduce data quality, distribution and chronological leakage checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import DATE_COL, LOOKAHEAD_COLS, TARGET, assert_no_leakage, describe_folds, get_folds
from src.data import _resolve_data_dir, load_dataset
from src.experiments import feature_frames


def run_checks(data_dir=None, output_dir=Path("results/analysis")):
    path = _resolve_data_dir(data_dir) / "train.csv"
    raw = pd.read_csv(path)
    hull = load_dataset(data_dir)
    assert raw[DATE_COL].is_unique, "Duplicate date_id"
    assert hull.train[DATE_COL].max() < hull.public[DATE_COL].min()
    assert len(hull.public) == 180, "Expected 180 public rows"
    featured, _, features, _ = feature_frames(data_dir)
    assert not set(features) & set(LOOKAHEAD_COLS)
    expected = hull.full.forward_returns.shift(1).iloc[:len(featured)].to_numpy()
    np.testing.assert_allclose(featured.ret_1d, expected, equal_nan=True)
    folds = get_folds(featured)
    assert_no_leakage(folds)
    returns = hull.full.forward_returns.to_numpy()
    jb = stats.jarque_bera(returns)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = hull.full[hull.raw_features].isna().mean().rename("missing_fraction")
    missing.sort_values(ascending=False).to_csv(output_dir / "missingness.csv")
    describe_folds(featured, folds).to_csv(output_dir / "folds.csv", index=False)
    summary = {
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "usable_rows": len(hull.full),
        "train_rows": len(hull.train), "public_rows": len(hull.public),
        "raw_features": len(hull.raw_features), "engineered_features": len(features),
        "train_date_ids": [int(hull.train[DATE_COL].min()), int(hull.train[DATE_COL].max())],
        "public_date_ids": [int(hull.public[DATE_COL].min()), int(hull.public[DATE_COL].max())],
        "complete_features": int((missing == 0).sum()),
        "target_mean": float(hull.full[TARGET].mean()),
        "target_std": float(hull.full[TARGET].std()),
        "return_excess_kurtosis": float(stats.kurtosis(returns)),
        "jarque_bera_statistic": float(jb.statistic), "jarque_bera_pvalue": float(jb.pvalue),
        "return_lag1_autocorrelation": float(pd.Series(returns).autocorr()),
        "squared_return_lag1_autocorrelation": float(pd.Series(returns ** 2).autocorr()),
        "limits": "Distributional checks alone cannot establish data provenance or map anonymous dates. "
                  "No external index reference series is bundled; historical external-return matches "
                  "are not claimed as independently reproduced.",
        "checks_passed": ["unique dates", "chronological split", "180 public rows",
                          "no look-ahead features", "causal return lag", "purged folds"],
    }
    (output_dir / "data_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    print(json.dumps(run_checks(args.data_dir, args.output_dir), indent=2))
