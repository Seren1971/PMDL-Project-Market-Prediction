"""Position sizing: turning a return forecast into an allocation in ``[0, 2]``.

The competition is as much a portfolio-construction problem as a forecasting problem, so
the sizing rule is treated as a first-class, separately evaluated component.

* :func:`binary_allocation` - the 61st-place policy: cash or full market exposure.
  A hard form of regularisation that refuses to read meaning into small forecast gaps.
* :func:`naive_linear_allocation` - ``clip(1 + k * pred, 0, 2)``: a single-parameter,
  hard-to-overfit rule.  This is the "safe post-processing" used in Stage 3.
* :func:`vol_target_allocation` - the 4th-place overlay: rescale exposure so realised
  strategy volatility sits just below the metric's 120% ceiling.
* :func:`smooth_positions` - EWM smoothing plus a transaction-cost haircut, which cuts
  turnover and acts as a low-pass filter on a noisy signal.
"""

from __future__ import annotations

import numpy as np

from .config import MAX_POSITION, MIN_POSITION, VOL_CEILING_RATIO

__all__ = [
    "binary_allocation",
    "naive_linear_allocation",
    "vol_target_allocation",
    "smooth_positions",
    "tune_naive_k",
]


def binary_allocation(preds, threshold: float = 0.0) -> np.ndarray:
    """0 = risk-free asset, 1 = normal market exposure (61st-place policy)."""
    p = np.asarray(preds, dtype=float).ravel()
    return (p > threshold).astype(float)


def naive_linear_allocation(
    preds,
    k: float = 100.0,
    lo: float = MIN_POSITION,
    hi: float = MAX_POSITION,
) -> np.ndarray:
    """``position = clip(1 + k * prediction, 0, 2)``.

    One parameter, monotone in the forecast, centred on the passive ``w = 1`` baseline,
    which is already a strong benchmark under the competition metric.
    """
    p = np.asarray(preds, dtype=float).ravel()
    return np.clip(1.0 + k * p, lo, hi)


def vol_target_allocation(
    preds,
    vol_estimate,
    market_vol,
    k: float = 100.0,
    ceiling: float = VOL_CEILING_RATIO,
    lo: float = MIN_POSITION,
    hi: float = MAX_POSITION,
) -> np.ndarray:
    """Scale the naive position by ``target_vol / estimated_vol`` and clip.

    ``target_vol`` is anchored to ``ceiling * market_vol`` - the same 1.2 that appears in
    the metric - so the strategy aims just below the penalty cliff instead of at an
    arbitrary volatility level.
    """
    raw = naive_linear_allocation(preds, k=k, lo=-np.inf, hi=np.inf)
    vol = np.asarray(vol_estimate, dtype=float).ravel()
    vol = np.where(np.isfinite(vol) & (vol > 1e-8), vol, np.nan)
    target = ceiling * float(market_vol)
    lev = np.nan_to_num(target / vol, nan=1.0, posinf=1.0)
    return np.clip(raw * lev, lo, hi)


def smooth_positions(
    positions,
    alpha: float = 0.75,
    transaction_cost: float = 3e-5,
    init: float = 1.0,
) -> np.ndarray:
    """EWM smoothing ``w_t = a * w_t + (1 - a) * w_{t-1}`` with a small cost haircut."""
    p = np.asarray(positions, dtype=float).ravel()
    out = np.empty_like(p)
    prev = float(init)
    for i, x in enumerate(p):
        cur = (alpha * x + (1.0 - alpha) * prev) * (1.0 - transaction_cost)
        out[i] = cur
        prev = cur
    return np.clip(out, MIN_POSITION, MAX_POSITION)


def tune_naive_k(
    preds,
    market_returns,
    risk_free=None,
    grid=(10, 25, 50, 100, 150, 200, 300, 500),
) -> tuple[float, float]:
    """Pick ``k`` by penalised Sharpe.

    Must only ever be called on **in-sample / training-fold** data - the notebooks tune it
    on the training part of each fold and apply the frozen value to the validation block.
    """
    from .metrics import penalised_sharpe

    best_k, best_score = float(grid[0]), -np.inf
    for k in grid:
        pos = naive_linear_allocation(preds, k=float(k))
        score = penalised_sharpe(pos, market_returns, risk_free)
        if np.isfinite(score) and score > best_score:
            best_k, best_score = float(k), float(score)
    return best_k, best_score
