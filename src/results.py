"""
Result persistence and cross-model comparison

Every notebook ends with one :func:`save_result` call. The shared CSV is what
makes the four-way comparison (baselines / proposed / improved) possible without
re-running anything
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, RESULTS_DIR

LEADERBOARD = "leaderboard.csv"

#: Columns shown first by :func:`compare`
HEADLINE = [
    "model",
    "stage",
    "split",
    "spearman_ic_mean",
    "rmse_mean",
    "modified_sharpe_mean",
    "sharpe_mean",
    "vol_ratio_mean",
]


def save_result(
    model: str,
    stage: str,
    metrics: dict[str, Any],
    split: str = "cv",
    notes: str = "",
    params: dict[str, Any] | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Append one row to ``results/leaderboard.csv``

    Parameters
    ----------
    model : short identifier, e.g. ``"lgbm_61st"``. Re-running a notebook with
        the same ``model`` + ``split`` replaces the previous row
    stage : one of ``"baseline"``, ``"proposed"``, ``"improved"``
    metrics : output of :func:`src.metrics.aggregate_folds` or
        :func:`src.metrics.evaluate`
    split : ``"cv"`` for cross-validated results, ``"public"`` for the held-out
        180-day block
    params : hyperparameters, stored as a JSON string for the record
    """
    directory = Path(results_dir or RESULTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LEADERBOARD

    row = {
        "model": model,
        "stage": stage,
        "split": split,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
        "params": json.dumps(params or {}, default=str),
    }
    row.update({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))})

    frame = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        mask = (existing["model"] == model) & (existing["split"] == split)
        existing = existing.loc[~mask]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


def load_results(results_dir: Path | None = None) -> pd.DataFrame:
    path = Path(results_dir or RESULTS_DIR) / LEADERBOARD
    if not path.exists():
        return pd.DataFrame(columns=HEADLINE)
    return pd.read_csv(path)


def compare(
    split: str = "cv",
    results_dir: Path | None = None,
    columns: list[str] | None = None,
    show_mean_cols: bool = False,
) -> pd.DataFrame:
    """Comparison table of all recorded models, best Spearman IC first"""
    frame = load_results(results_dir)
    if frame.empty:
        return frame

    frame = frame.loc[frame["split"] == split]

    # Cross-validated rows carry "<metric>_mean" names, single-split rows
    # the bare metric name; accept either so both appear in one table.
    candidates = [
        c
        for h in HEADLINE
        for c in (h, h.replace("_mean", ""))
    ]

    cols = columns or list(dict.fromkeys(
        c for c in candidates if c in frame.columns
    ))

    cols = list(dict.fromkeys(c for c in cols if c in frame.columns))

    # Drop columns that contain only NaNs
    cols = [c for c in cols if frame[c].notna().any()]

    if not cols:
        return frame.iloc[0:0].copy()

    sort_key = [
        c for c in (
            "modified_sharpe_mean",
            "modified_sharpe",
            "spearman_ic_mean",
            "spearman_ic",
        )
        if c in cols
    ]

    # Fallback so sort_values always has something to sort by
    if not sort_key:
        sort_key = [cols[0]]

    return (
        frame[cols]
        .sort_values(by=sort_key, ascending=False)
        .reset_index(drop=True)
    )

def save_predictions(
    name: str, date_id, y_true, y_pred, weights=None, results_dir: Path | None = None
) -> Path:
    """Persist per-row predictions so later notebooks can blend or re-score
    without refitting"""
    directory = Path(results_dir or ARTIFACT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {"date_id": np.asarray(date_id), "y_true": np.asarray(y_true, float),
         "y_pred": np.asarray(y_pred, float)}
    )
    if weights is not None:
        frame["weight"] = np.asarray(weights, float)
    path = directory / f"{name}_predictions.csv"
    frame.to_csv(path, index=False)
    return path


def load_predictions(name: str, results_dir: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(Path(results_dir or ARTIFACT_DIR) / f"{name}_predictions.csv")
