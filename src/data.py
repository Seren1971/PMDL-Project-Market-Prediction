"""
Universal data loading for the Hull Tactical Market Prediction dataset.

Attribution
-----------
The "hold out the last 180 date_ids" evaluation protocol is taken from the
public leak-safe baseline notebook:
https://www.kaggle.com/code/morodertobias/hull-leak-safe-baseline
Rationale (from the competition data description): the public leaderboard test
set is a *copy* of the last 180 rows of train.csv, so any model fitted on those
rows reports an optimistic public score. We remove them from training and use
them as a held-out public split.

Only the public phase is in scope; the forecasting phase / evaluation API is
deliberately out of scope for this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    DATE_COL,
    FEATURE_PREFIXES,
    KAGGLE_DATA_DIR,
    LOOKAHEAD_COLS,
    PUBLIC_TEST_SIZE,
    TARGET,
)


@dataclass
class HullData:
    """Container returned by :func:`load_dataset`.

    Attributes
    ----------
    train : rows used for fitting and cross-validation.
    public : the held-out last ``PUBLIC_TEST_SIZE`` date_ids.
    full : train + public, chronologically ordered (for feature engineering
        that needs continuous history).
    raw_features : base feature columns present in the source file.
    """

    train: pd.DataFrame
    public: pd.DataFrame
    full: pd.DataFrame
    raw_features: list[str]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HullData(train={self.train.shape}, public={self.public.shape}, "
            f"n_raw_features={len(self.raw_features)})"
        )


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    for candidate in (data_dir, DATA_DIR, KAGGLE_DATA_DIR):
        if candidate is None:
            continue
        path = Path(candidate)
        if (path / "train.csv").exists():
            return path
    raise FileNotFoundError(
        "train.csv not found. Place the Kaggle files in data/raw/ or set the "
        "HULL_DATA_DIR environment variable. See data/README.md."
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the anonymised feature columns of a raw dataframe.

    Excludes ``date_id``, the three look-ahead columns, any ``lagged_*`` /
    ``is_scored`` columns, and anything this project derived later.
    """
    banned = {DATE_COL, "is_scored", *LOOKAHEAD_COLS}
    out = []
    for col in df.columns:
        if col in banned or col.startswith("lagged_"):
            continue
        if col.startswith(FEATURE_PREFIXES) or col.startswith("MOM"):
            out.append(col)
    return out


def load_dataset(
    data_dir: str | Path | None = None,
    public_test_size: int = PUBLIC_TEST_SIZE,
    min_valid_ratio: float = 0.5,
    drop_target_na: bool = True,
) -> HullData:
    """Load train.csv and split off the public leaderboard period.

    Parameters
    ----------
    data_dir : directory holding train.csv. Falls back to ``data/raw`` then to
        the Kaggle mount point.
    public_test_size : number of trailing date_ids reserved as the public split.
    min_valid_ratio : drop leading rows whose share of non-null features is
        below this. Coverage stretches back decades and early rows are mostly
        null; this trims them without touching the recent period.
    drop_target_na : drop rows where the supervised target is missing.
    """
    path = _resolve_data_dir(data_dir)
    df = pd.read_csv(path / "train.csv")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    raw_features = get_feature_columns(df)

    if min_valid_ratio > 0 and raw_features:
        valid_ratio = df[raw_features].notna().mean(axis=1)
        keep_from = valid_ratio.ge(min_valid_ratio).idxmax()
        df = df.loc[keep_from:].reset_index(drop=True)

    if drop_target_na and TARGET in df.columns:
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    cutoff = df[DATE_COL].max() - public_test_size
    train = df.loc[df[DATE_COL] <= cutoff].reset_index(drop=True)
    public = df.loc[df[DATE_COL] > cutoff].reset_index(drop=True)

    return HullData(train=train, public=public, full=df, raw_features=raw_features)


def impute(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Forward-fill then fill remaining gaps with the *train* median.

    Statistics come from ``train`` only and are applied to every frame, so no
    information flows backwards from the held-out period.
    """
    columns = columns or [c for c in train.columns if train[c].dtype != object]
    medians = train[columns].median()

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = [c for c in columns if c in out.columns]
        out[cols] = out[cols].ffill().fillna(medians[cols]).fillna(0.0)
        return out

    return tuple(_apply(df) for df in (train, *others))


def past_returns(df: pd.DataFrame) -> pd.Series:
    """Leak-safe realised daily return series.

    ``forward_returns`` at row *t* spans *t -> t+1*, i.e. it is the future. The
    return already observable at *t* is therefore ``forward_returns.shift(1)``.
    Every price-derived feature in this project is built from this series.
    """
    if "forward_returns" not in df.columns:
        raise KeyError("'forward_returns' is required to derive return history.")
    return df["forward_returns"].shift(1).astype(float)


def excess_returns(df: pd.DataFrame) -> np.ndarray:
    """Market excess return per row: ``forward_returns - risk_free_rate``."""
    return (df["forward_returns"] - df["risk_free_rate"]).to_numpy(dtype=float)
