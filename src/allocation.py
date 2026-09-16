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
* :func:`ternary_allocation` - a three-state extension of the binary rule
  (risk-free / passive / leveraged) for a signal with a clear neutral band,
  giving the discrete-policy family a middle ground without fitting a
  continuous size.
* :func:`vol_target_allocation` - the 4th place volatility-targeting overlay,
  which that author credits with more leaderboard gain than the alpha itself.
* :func:`vol_budget_allocation` - a 100th-place-style refinement of the naive
  rule: the tilt away from the passive ``w = 1`` benchmark is scaled by
  realised volatility instead of a fixed constant, so leverage adapts to the
  current regime instead of being fitted once and frozen.
* :func:`calibrate_k` - sets :func:`naive_allocation`'s ``k`` by a volatility
  constraint (bisection), not by searching the score.
* :func:`causal_standardise` - expanding, leak-free rescaling of a signal at
  inference time, seeded by frozen out-of-fold statistics.
"""

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


def ternary_allocation(
    prediction: np.ndarray, threshold: float = 0.0, leveraged: float = 2.0
) -> np.ndarray:
    """Risk-free (0), passive (1), or ``leveraged`` (default 2) fully-invested.

    A symmetric extension of :func:`binary_allocation`: a signal that clears
    ``+threshold`` gets leveraged exposure, one below ``-threshold`` sits out
    entirely, and anything in the neutral band between the two just holds the
    passive ``w = 1`` benchmark rather than being forced to pick a side. Still
    two hyperparameters at most (``threshold``, ``leveraged``), so it keeps the
    discrete family's regularising property while adding one state.
    """
    p = np.asarray(prediction, float)
    out = np.ones_like(p)
    out[p > threshold] = leveraged
    out[p < -threshold] = 0.0
    return out


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


def vol_budget_allocation(
    raw_position: np.ndarray,
    realised_vol: np.ndarray,
    target_vol: float = 0.12,
) -> np.ndarray:
    """Scale the *tilt* away from ``w = 1`` to a fixed annualised vol budget.

    ``raw_position`` is an unscaled naive position, typically
    ``1 + prediction`` (i.e. :func:`naive_allocation` with ``k=1``). Unlike
    :func:`vol_target_allocation`, which multiplies the whole position
    (including the passive ``1``) by ``target_vol / realised_vol``, this
    function levers only the deviation from the benchmark - so a day with no
    signal still sits at ``w = 1`` regardless of the current vol regime, and
    only the *size* of a real tilt responds to it.

    This has no fitted parameter: nothing here is chosen against the score, and
    the leverage adapts day to day instead of being frozen at one constant
    ``k`` for every regime the way :func:`naive_allocation` is.
    """
    tilt = np.asarray(raw_position, float) - 1.0
    vol = np.asarray(realised_vol, float)
    leverage = np.divide(
        target_vol, vol, out=np.ones_like(vol, dtype=float),
        where=(vol > 0) & np.isfinite(vol),
    )
    return np.clip(1.0 + tilt * leverage, W_MIN, W_MAX)


def calibrate_k(
    prediction: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    target_vol_ratio: float = 1.05,
    k_bounds: tuple[float, float] = (1e-3, 1e4),
    tol: float = 1e-3,
    max_iter: int = 100,
) -> float:
    """Bisect :func:`naive_allocation`'s ``k`` to hit a target strategy /
    market volatility ratio, instead of searching ``k`` against the score.

    ``vol_ratio`` is monotone increasing in ``k`` (a bigger tilt means more
    strategy volatility), so bisection is well posed. This is a constraint,
    not a fitted parameter: nothing here is selected on the metric, so it
    cannot overfit the objective the way a tuned ``k`` can.
    """
    from .metrics import strategy_returns  # local import avoids a cycle

    fwd = np.asarray(forward_returns, float)
    rf = np.asarray(risk_free_rate, float)
    market_vol = np.std(fwd - rf, ddof=1)
    if market_vol == 0 or not np.isfinite(market_vol):
        return 1.0

    def vol_ratio(k: float) -> float:
        strat = strategy_returns(naive_allocation(prediction, k=k), fwd, rf)
        return float(np.std(strat - rf, ddof=1) / market_vol)

    lo, hi = k_bounds
    if vol_ratio(hi) < target_vol_ratio:
        return hi  # signal too weak to reach the target even at the upper bound
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if vol_ratio(mid) < target_vol_ratio:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def causal_standardise(
    x: np.ndarray,
    warmup_mean: float,
    warmup_sd: float,
    min_periods: int = 20,
) -> np.ndarray:
    """Expanding-window z-score of ``x``, using only *earlier* rows of ``x``.

    The first ``min_periods`` rows fall back to the frozen ``(warmup_mean,
    warmup_sd)`` - typically the training-time statistics of the signal being
    rescaled - since there isn't enough of ``x``'s own history yet to estimate
    anything. After that, row ``i`` is scaled by the mean/sd of rows
    ``0..i-1`` only, so no row ever contributes to its own normalisation and
    nothing here looks ahead.

    Exists because a model refit on the full training period produces a wider
    prediction spread than its out-of-fold counterpart, so freezing the
    out-of-fold scale under-controls volatility at inference; this recovers a
    comparable scale causally, from the held-out block itself.
    """
    x = np.asarray(x, float)
    n = len(x)
    out = np.empty(n, dtype=float)
    cum_sum, cum_sq = 0.0, 0.0
    for i in range(n):
        if i < min_periods:
            mu, sd = warmup_mean, warmup_sd
        else:
            mu = cum_sum / i
            var = max(cum_sq / i - mu * mu, 0.0)
            sd = var ** 0.5
            if sd == 0.0 or not np.isfinite(sd):
                sd = warmup_sd
        sd = sd if sd > 0 else 1.0
        out[i] = (x[i] - mu) / sd
        cum_sum += x[i]
        cum_sq += x[i] * x[i]
    return out


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
