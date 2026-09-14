"""
Feature engineering: non-learning domain features + temporal context.

Attribution
-----------
* Domain block (momentum, short-horizon mean reversion, volatility spreads,
  inverse-volatility signal blending) follows the ideas described in the public
  4th place write-up, "technical model, no learning, short-term mean reversion".
  The original alpha is not disclosed there; what is reproduced here is the
  *class* of indicator and the risk-management framing, rebuilt from scratch.
* Lag / rolling block (lags 1-20, rolling mean and std over 2-60) follows the
  public 61st place notebook:
  https://www.kaggle.com/code/rafanikitas/hull-eda-training-pipeline
* Cross terms ``U1 = I2 - I1`` and ``U2 = M11 / mean(I2, I9, I7)`` appear in
  both the Hull starter notebook and the 100th place write-up.

Leakage rule
------------
``forward_returns`` at row *t* covers *t -> t+1*. The return observable at *t*
is therefore ``forward_returns.shift(1)``. Every price-derived feature below is
built from that shifted series via :func:`src.data.past_returns`; rolling
windows then look strictly backwards. No function here reads the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .data import past_returns

# Defaults, exposed so notebooks can cite them rather than re-declare them.
LAG_PERIODS = (1, 3, 5, 7, 14, 20)
ROLLING_WINDOWS = (2, 5, 10, 20, 60)
MOMENTUM_WINDOWS = (5, 10, 20, 60)
VOL_WINDOWS = (5, 20, 60)
REVERSION_WINDOWS = (3, 5, 10)


def add_cross_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Term-structure spread and rate-normalised market dynamic."""
    out = df.copy()
    if {"I1", "I2"}.issubset(out.columns):
        out["U1"] = out["I2"] - out["I1"]
    if {"M11", "I2", "I9", "I7"}.issubset(out.columns):
        denom = (out["I2"] + out["I9"] + out["I7"]) / 3.0
        out["U2"] = out["M11"] / denom.replace(0.0, np.nan)
    return out


def add_price_features(
    df: pd.DataFrame,
    momentum_windows: tuple[int, ...] = MOMENTUM_WINDOWS,
    vol_windows: tuple[int, ...] = VOL_WINDOWS,
    reversion_windows: tuple[int, ...] = REVERSION_WINDOWS,
) -> pd.DataFrame:
    """Momentum, realised volatility, volatility spreads and mean-reversion
    z-scores, all derived from the lagged realised return series."""
    out = df.copy()
    ret = past_returns(out)
    out["ret_1d"] = ret

    for w in momentum_windows:
        out[f"mom_{w}"] = ret.rolling(w, min_periods=max(2, w // 2)).sum()

    for w in vol_windows:
        out[f"vol_{w}"] = ret.rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(
            TRADING_DAYS
        )

    # Volatility spread / term structure of risk: short vol relative to long vol.
    if {"vol_5", "vol_60"}.issubset(out.columns):
        out["vol_spread_5_60"] = out["vol_5"] - out["vol_60"]
        out["vol_ratio_5_60"] = out["vol_5"] / out["vol_60"].replace(0.0, np.nan)
    if {"vol_20", "vol_60"}.issubset(out.columns):
        out["vol_ratio_20_60"] = out["vol_20"] / out["vol_60"].replace(0.0, np.nan)

    # Short-horizon mean reversion: how stretched is the recent move, in units
    # of its own recent volatility. This is the family the 4th place alpha
    # belongs to.
    for w in reversion_windows:
        roll_mean = ret.rolling(w, min_periods=2).mean()
        roll_std = ret.rolling(20, min_periods=5).std()
        out[f"revert_{w}"] = -(roll_mean / roll_std.replace(0.0, np.nan))

    # Distance from a 20-day equity-curve high: crude drawdown state.
    curve = (1.0 + ret.fillna(0.0)).cumprod()
    out["drawdown_20"] = curve / curve.rolling(20, min_periods=5).max() - 1.0

    return out


def add_lag_roll_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = LAG_PERIODS,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """61st-place temporal context for a shortlist of columns.

    A feature's level is often uninformative on its own; what matters is
    whether it is rising, stable or unusually volatile against its own history.
    """
    out = df.copy()
    new = {}
    for col in columns:
        if col not in out.columns:
            continue
        series = out[col]
        for lag in lags:
            new[f"{col}_lag_{lag}"] = series.shift(lag)
        for w in windows:
            new[f"{col}_roll_mean_{w}"] = series.rolling(w, min_periods=1).mean()
            new[f"{col}_roll_std_{w}"] = series.rolling(w, min_periods=1).std()
    if new:
        out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    return out


def inverse_vol_weights(
    signals: pd.DataFrame, window: int = 60, min_periods: int = 20
) -> pd.DataFrame:
    """Inverse-volatility blending weights, one column per signal.

    From the 4th place write-up: weight each signal by the reciprocal of its own
    rolling volatility, normalised to sum to one. No covariance matrix, no
    optimisation, automatic down-weighting of unstable signals. A sparse but
    strong signal will *not* dominate, which is the intended behaviour.
    """
    vol = signals.rolling(window, min_periods=min_periods).std()
    inv = 1.0 / vol.replace(0.0, np.nan)
    return inv.div(inv.sum(axis=1), axis=0)


def blend_signals(signals: pd.DataFrame, window: int = 60) -> pd.Series:
    """Combine signal columns into one series using inverse-vol weights."""
    weights = inverse_vol_weights(signals, window=window)
    return (signals * weights).sum(axis=1, min_count=1)


def build_features(
    df: pd.DataFrame,
    lag_roll_columns: list[str] | None = None,
    price_features: bool = True,
    cross_terms: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Single entry point. Returns ``(dataframe, feature_names)``.

    Call this on the **full chronological frame** before splitting, so rolling
    windows are continuous across the train/public boundary. This is safe:
    every window looks backwards only.
    """
    from .data import get_feature_columns  # local import avoids a cycle

    before = set(df.columns)
    out = df.copy()

    if cross_terms:
        out = add_cross_terms(out)
    if price_features:
        out = add_price_features(out)
    if lag_roll_columns:
        out = add_lag_roll_features(out, lag_roll_columns)

    derived = [c for c in out.columns if c not in before]
    feature_names = get_feature_columns(df) + derived
    feature_names = [c for c in feature_names if c in out.columns]
    return out, feature_names


def top_features_by_gain(model, feature_names: list[str], k: int = 50) -> list[str]:
    """Top-k features by a fitted tree model's importance."""
    importance = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importance)[::-1][:k]
    return [feature_names[i] for i in order]
