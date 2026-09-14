"""
Position sizing: turning a prediction into an allocation in [0, 2].

Three rules, matching the three strategies the write-ups converged on. Keeping
them here means every notebook scores the *same* mapping from signal to weight,
so differences in the leaderboard come from the model, not from a bespoke sizer.

Attribution
-----------
* :func:`naive_allocation` - the safe, non-overfitting rule specified for
  Stage 3, in the spirit of the 100th place write-up's metric-anchored sizing.
* :func:`binary_allocation` - the 61st place policy: risk-free or fully
  invested, nothing in between, as a form of regularisation.
* :func:`vol_target_allocation` - the 4th place volatility-targeting overlay,
  which that author credits with more leaderboard gain than the alpha itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS

W_MIN, W_MAX = 0.0, 2.0


def naive_allocation(prediction: np.ndarray, k: float = 1.0) -> np.ndarray:
    """``clip(1 + k * prediction, 0, 2)``.

    Centred on the passive ``w = 1`` benchmark, which is already strong under
    this metric, and tilts away from it in proportion to the signal. One
    parameter, no fitting.
    """
    return np.clip(1.0 + k * np.asarray(prediction, float), W_MIN, W_MAX)


def binary_allocation(prediction: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Fully invested when the signal clears ``threshold``, risk-free otherwise."""
    return (np.asarray(prediction, float) > threshold).astype(float)


def vol_target_allocation(
    prediction: np.ndarray,
    realised_vol: np.ndarray,
    target_vol: float = 0.12,
    k: float = 1.0,
) -> np.ndarray:
    """Scale a naive tilt by ``target_vol / realised_vol``.

    ``realised_vol`` should be an *annualised* backward-looking estimate (e.g.
    the ``vol_20`` column from :mod:`src.features`). Deliberately computed over
    a long window so leverage does not chase short-lived noise.
    """
    base = 1.0 + k * np.asarray(prediction, float)
    vol = np.asarray(realised_vol, float)
    leverage = np.divide(
        target_vol, vol, out=np.ones_like(vol), where=(vol > 0) & np.isfinite(vol)
    )
    return np.clip(base * leverage, W_MIN, W_MAX)


def smooth_weights(weights: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Exponential smoothing of the weight path to cut turnover.

    ``alpha`` is the weight on the new value, so 0.25 means 75/25 in favour of
    the previous allocation.
    """
    return (
        pd.Series(np.asarray(weights, float))
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .to_numpy()
    )


def realised_vol(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Annualised backward-looking volatility of a *already lagged* return series."""
    series = pd.Series(np.asarray(returns, float))
    return (
        series.rolling(window, min_periods=max(2, window // 4)).std()
        * np.sqrt(TRADING_DAYS)
    ).to_numpy()
