"""Feature engineering.

Three blocks, each attributed to the public source that inspired it:

``add_temporal_features``
    Lags and rolling mean / std over a small, hand-picked set of columns.
    *Concept adapted from the public 61st-place write-up.*

``add_derived_features``
    Term-structure spread ``U1``, rate-normalised market dynamic ``U2`` and a handful of
    cross products that give linear models the interactions trees would monopolise.
    *Concept adapted from the public 100th-place write-up / notebook.*

``add_domain_features``
    Non-learned technical indicators computed from the realised return series:
    short-horizon mean reversion, momentum, realised-volatility spreads, drawdown, an
    RSI-style breadth measure, and an inverse-volatility blend of the reversal signals.
    *Concept adapted from the public 4th-place write-up ("no machine-learning model",
    a rule-based short-horizon mean-reversion alpha combined with inverse-volatility
    weighting and volatility targeting).*  No code was copied; only the ideas.

Leakage policy
--------------
Every transformation here is **strictly backward-looking**: lags, ``rolling(...)`` and
``shift(...)`` only ever read rows ``<= t``.  Computing them once on the full frame before
splitting is therefore leak-free.  Anything that needs a *fitted statistic* (medians for
imputation, scalers, feature selection) is fitted inside the training fold only - see
:class:`LeakSafeImputer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DATE_COL, LEAKY_COLS, TARGET

__all__ = [
    "TOP_FEATURES_FOR_FE",
    "LAG_PERIODS",
    "ROLLING_WINDOWS",
    "add_temporal_features",
    "add_derived_features",
    "add_domain_features",
    "LeakSafeImputer",
    "model_feature_columns",
]

# Hand-picked columns for temporal expansion (61st-place write-up).  Columns that are
# absent from the loaded dataset are skipped automatically.
TOP_FEATURES_FOR_FE: list[str] = [
    "M4", "V13", "S5", "S2", "D2", "E19", "P7", "P6",
    "P3", "P13", "P4", "P5", "M2", "V5",
]
LAG_PERIODS: list[int] = [1, 3, 5, 7, 14, 20]
ROLLING_WINDOWS: list[int] = [2, 5, 10, 20, 60]


# --------------------------------------------------------------------------------------
# Block 1 - temporal context (61st place)
# --------------------------------------------------------------------------------------
def add_temporal_features(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add lags and rolling mean/std for a small set of promising columns."""
    cols = [c for c in (cols or TOP_FEATURES_FOR_FE) if c in df.columns]
    lags = lags or LAG_PERIODS
    windows = windows or ROLLING_WINDOWS

    out = df.copy()
    new: dict[str, pd.Series] = {}
    for col in cols:
        s = out[col]
        for lag in lags:
            new[f"{col}_lag_{lag}"] = s.shift(lag)
        for w in windows:
            new[f"{col}_roll_mean_{w}"] = s.rolling(w, min_periods=1).mean()
            new[f"{col}_roll_std_{w}"] = s.rolling(w, min_periods=1).std()
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


# --------------------------------------------------------------------------------------
# Block 2 - derived / interaction features (100th place)
# --------------------------------------------------------------------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Term-structure spread, rate-normalised market dynamic and cross products."""
    out = df.copy()
    new: dict[str, pd.Series] = {}

    if {"I1", "I2"}.issubset(out.columns):
        new["U1"] = out["I2"] - out["I1"]
    if {"M11", "I2", "I9", "I7"}.issubset(out.columns):
        denom = (out["I2"] + out["I9"] + out["I7"]) / 3.0
        new["U2"] = out["M11"] / denom.replace(0, np.nan)
    for a, b in (("V1", "S1"), ("M11", "V1"), ("I9", "S1")):
        if {a, b}.issubset(out.columns):
            new[f"{a}_{b}"] = out[a] * out[b]

    if not new:
        return out
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


# --------------------------------------------------------------------------------------
# Block 3 - non-learned domain features (4th place)
# --------------------------------------------------------------------------------------
@dataclass
class DomainFeatureConfig:
    reversal_windows: tuple[int, ...] = (2, 3, 5, 10)
    momentum_windows: tuple[int, ...] = (20, 60, 120)
    vol_windows: tuple[int, ...] = (5, 20, 60)
    breadth_window: int = 14
    vol_weight_window: int = 60
    extra_vol_ratio: tuple[tuple[int, int], ...] = ((5, 60), (20, 60))
    signal_names: list[str] = field(default_factory=list)


def add_domain_features(
    df: pd.DataFrame,
    target_col: str = TARGET,
    cfg: DomainFeatureConfig | None = None,
) -> tuple[pd.DataFrame, DomainFeatureConfig]:
    """Rule-based technical indicators built from the *realised* return series.

    The return realised at row ``t`` is ``target.shift(1)`` (the target of row ``t`` is the
    return of ``t+1``), so nothing here uses information from the future.
    """
    cfg = cfg or DomainFeatureConfig()
    out = df.copy()

    r = out[target_col].shift(1)  # realised excess return known at t
    new: dict[str, pd.Series] = {"ret_realised": r}

    # Realised volatility and its term structure -------------------------------------
    for w in cfg.vol_windows:
        new[f"rv_{w}"] = r.rolling(w, min_periods=max(2, w // 3)).std()
    rv_ref = r.rolling(cfg.vol_windows[-1], min_periods=5).std()
    for short, long in cfg.extra_vol_ratio:
        s = r.rolling(short, min_periods=2).std()
        l = r.rolling(long, min_periods=5).std()
        new[f"vol_ratio_{short}_{long}"] = s / l.replace(0, np.nan)
    new["vol_of_vol_20"] = r.rolling(20, min_periods=5).std().rolling(60, min_periods=10).std()

    # Short-horizon mean reversion (the 4th-place alpha family) -----------------------
    reversal_cols: list[str] = []
    for k in cfg.reversal_windows:
        cum_k = r.rolling(k, min_periods=k).sum()
        z = cum_k / (rv_ref.replace(0, np.nan) * np.sqrt(k))
        name = f"reversal_{k}"
        new[name] = -z.clip(-5, 5)
        reversal_cols.append(name)

    # Momentum (opposite sign family, keeps the model honest across regimes) ----------
    for k in cfg.momentum_windows:
        new[f"momentum_{k}"] = (
            r.rolling(k, min_periods=max(5, k // 4)).mean() / rv_ref.replace(0, np.nan)
        )

    # Path statistics ------------------------------------------------------------------
    equity = r.fillna(0.0).cumsum()
    for k in (20, 60):
        new[f"dist_ma_{k}"] = (equity - equity.rolling(k, min_periods=5).mean()) / (
            rv_ref.replace(0, np.nan) * np.sqrt(k)
        )
    new["drawdown_120"] = equity - equity.rolling(120, min_periods=20).max()
    new["breadth_14"] = (r > 0).rolling(cfg.breadth_window, min_periods=5).mean() - 0.5

    tmp = pd.DataFrame(new, index=out.index)

    # Inverse-volatility blend of the reversal signals --------------------------------
    # w_i = sigma_i^-1 / sum_j sigma_j^-1, with sigma_i the rolling volatility of signal i.
    inv_vols = {}
    for c in reversal_cols:
        sig_vol = tmp[c].rolling(cfg.vol_weight_window, min_periods=20).std()
        inv_vols[c] = 1.0 / sig_vol.replace(0, np.nan)
    inv = pd.DataFrame(inv_vols, index=out.index)
    weights = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0)
    tmp["alpha_invvol_blend"] = (tmp[reversal_cols] * weights).sum(axis=1, min_count=1)

    # Volatility-regime interaction ----------------------------------------------------
    if "V1" in out.columns:
        v1_med = out["V1"].expanding(min_periods=100).median()
        tmp["vol_regime"] = (out["V1"] > v1_med).astype(float)
        tmp["alpha_x_regime"] = tmp["alpha_invvol_blend"] * (1.0 - tmp["vol_regime"])

    cfg.signal_names = list(tmp.columns)
    return pd.concat([out, tmp], axis=1), cfg


# --------------------------------------------------------------------------------------
# Leak-safe imputation
# --------------------------------------------------------------------------------------
class LeakSafeImputer:
    """Forward-fill (past only) + fold-fitted median fill + infinity clean-up.

    The public notebooks impute with the median of the *whole* file, which quietly leaks
    the validation distribution into training.  Here medians are learned on the training
    fold only and reused at validation / inference time.
    """

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None
        self.columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "LeakSafeImputer":
        Xc = X.replace([np.inf, -np.inf], np.nan)
        self.columns_ = list(Xc.columns)
        self.medians_ = Xc.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("LeakSafeImputer must be fitted before transform().")
        Xc = X.replace([np.inf, -np.inf], np.nan).ffill()
        Xc = Xc.fillna(self.medians_).fillna(0.0)
        return Xc.astype("float64")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def model_feature_columns(df: pd.DataFrame, extra_drop: list[str] | None = None) -> list[str]:
    """All numeric columns that are legitimate model inputs."""
    drop = set(LEAKY_COLS) | {DATE_COL} | set(extra_drop or [])
    return [
        c
        for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
