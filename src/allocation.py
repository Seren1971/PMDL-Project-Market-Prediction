"""
Position sizing: turning a prediction into an allocation in [0, 2].

All allocation rules live here so every model uses the same mapping from
prediction signal to portfolio exposure.
"""

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


W_MIN = 0.0
W_MAX = 2.0


def naive_allocation(
    prediction: np.ndarray,
    k: float = 1.0,
) -> np.ndarray:
    """Convert a prediction into an allocation centred around w=1."""
    prediction = np.asarray(prediction, dtype=float)

    return np.clip(
        1.0 + k * prediction,
        W_MIN,
        W_MAX,
    )


def binary_allocation(
    prediction: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Use full market exposure when prediction exceeds the threshold."""
    prediction = np.asarray(prediction, dtype=float)

    return (prediction > threshold).astype(float)


def vol_target_allocation(
    prediction: np.ndarray,
    realised_vol: np.ndarray,
    target_vol: float = 0.12,
    k: float = 1.0,
) -> np.ndarray:
    """
    Scale the complete position according to realised volatility.

    This function is kept for compatibility with the existing project code.
    """
    prediction = np.asarray(prediction, dtype=float)
    vol = np.asarray(realised_vol, dtype=float)

    base = 1.0 + k * prediction

    leverage = np.divide(
        target_vol,
        vol,
        out=np.ones_like(vol),
        where=(vol > 0) & np.isfinite(vol),
    )

    return np.clip(
        base * leverage,
        W_MIN,
        W_MAX,
    )


def vol_budget_allocation(
    raw_position: np.ndarray,
    realised_vol: np.ndarray,
    target_vol: float = 0.12,
) -> np.ndarray:
    """
    Scale only the active tilt away from the passive w=1 benchmark.

    A zero trading signal therefore remains at allocation 1 regardless
    of the current volatility regime.
    """
    raw_position = np.asarray(raw_position, dtype=float)
    vol = np.asarray(realised_vol, dtype=float)

    tilt = raw_position - 1.0

    leverage = np.divide(
        target_vol,
        vol,
        out=np.ones_like(vol),
        where=(vol > 0) & np.isfinite(vol),
    )

    return np.clip(
        1.0 + tilt * leverage,
        W_MIN,
        W_MAX,
    )


def risk_controlled_vol_budget_allocation(
    signal: np.ndarray,
    realised_volatility: np.ndarray,
    tilt_scale: float = 1.0,
    target_vol: float = 0.12,
) -> np.ndarray:
    """
    Apply volatility-aware allocation with an additional risk scale.

    The original vol_budget_allocation adapts exposure to the current
    volatility regime, but it does not explicitly control the final
    strategy/market volatility ratio.

    tilt_scale controls only the active part of the position:

        allocation = 1 + tilt_scale * active_tilt

    Therefore:
        tilt_scale = 0 -> passive allocation 1
        tilt_scale = 1 -> original volatility-budget allocation

    The scale must be calibrated using training or inner-CV data only.
    """
    signal = np.asarray(signal, dtype=float)

    base = vol_budget_allocation(
        raw_position=1.0 + signal,
        realised_vol=realised_volatility,
        target_vol=target_vol,
    )

    active_tilt = base - 1.0

    return np.clip(
        1.0 + float(tilt_scale) * active_tilt,
        W_MIN,
        W_MAX,
    )


def calibrate_vol_budget_scale(
    signal: np.ndarray,
    realised_volatility: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    target_vol_ratio: float = 1.05,
    target_vol: float = 0.12,
    max_scale: float = 4.0,
    n_grid: int = 161,
) -> float:
    """
    Find the largest risk scale that keeps strategy volatility below
    the requested strategy/market volatility ratio.

    This is intentionally NOT a Sharpe optimisation.

    It is a risk constraint. The function should only receive predictions
    generated out-of-fold inside the training period.

    A deterministic grid is used because the relationship between scale
    and realised strategy volatility is not guaranteed to be perfectly
    monotonic due to signal/market covariance.
    """
    from .metrics import modified_sharpe

    if target_vol_ratio <= 0:
        raise ValueError("target_vol_ratio must be positive")

    if max_scale <= 0:
        raise ValueError("max_scale must be positive")

    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")

    candidates = np.linspace(
        0.0,
        float(max_scale),
        int(n_grid),
    )

    feasible_scales = []

    for scale in candidates:
        weights = risk_controlled_vol_budget_allocation(
            signal=signal,
            realised_volatility=realised_volatility,
            tilt_scale=float(scale),
            target_vol=target_vol,
        )

        components = modified_sharpe(
            weights,
            forward_returns,
            risk_free_rate,
            return_components=True,
        )

        vol_ratio = float(components["vol_ratio"])

        if (
            np.isfinite(vol_ratio)
            and vol_ratio <= target_vol_ratio
        ):
            feasible_scales.append(float(scale))

    if not feasible_scales:
        return 0.0

    return max(feasible_scales)


def calibrate_k(
    prediction: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    target_vol_ratio: float = 1.05,
    k_bounds: tuple[float, float] = (1e-3, 1e4),
    tol: float = 1e-3,
    max_iter: int = 100,
) -> float:
    """
    Calibrate naive-allocation strength using a volatility constraint
    instead of maximising the competition score.
    """
    from .metrics import strategy_returns

    prediction = np.asarray(prediction, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    risk_free_rate = np.asarray(risk_free_rate, dtype=float)

    market_vol = np.std(
        forward_returns - risk_free_rate,
        ddof=1,
    )

    if market_vol == 0 or not np.isfinite(market_vol):
        return 1.0

    def calculate_vol_ratio(k: float) -> float:
        allocation = naive_allocation(
            prediction,
            k=k,
        )

        strategy = strategy_returns(
            allocation,
            forward_returns,
            risk_free_rate,
        )

        strategy_vol = np.std(
            strategy - risk_free_rate,
            ddof=1,
        )

        return float(strategy_vol / market_vol)

    lower, upper = k_bounds

    if calculate_vol_ratio(upper) < target_vol_ratio:
        return upper

    for _ in range(max_iter):
        middle = (lower + upper) / 2.0

        if calculate_vol_ratio(middle) < target_vol_ratio:
            lower = middle
        else:
            upper = middle

        if upper - lower < tol:
            break

    return (lower + upper) / 2.0


def causal_standardise(
    x: np.ndarray,
    warmup_mean: float,
    warmup_sd: float,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Standardise each observation using only earlier observations.

    During the initial warm-up period, frozen training statistics are used.

    The current row never contributes to its own normalisation.
    """
    x = np.asarray(x, dtype=float)

    n = len(x)
    output = np.empty(n, dtype=float)

    cumulative_sum = 0.0
    cumulative_squared_sum = 0.0

    for i in range(n):
        if i < min_periods:
            mean = warmup_mean
            std = warmup_sd

        else:
            mean = cumulative_sum / i

            variance = max(
                cumulative_squared_sum / i - mean * mean,
                0.0,
            )

            std = variance ** 0.5

            if std == 0.0 or not np.isfinite(std):
                std = warmup_sd

        if std <= 0 or not np.isfinite(std):
            std = 1.0

        output[i] = (x[i] - mean) / std

        cumulative_sum += x[i]
        cumulative_squared_sum += x[i] * x[i]

    return output


def smooth_weights(
    weights: np.ndarray,
    alpha: float = 0.25,
) -> np.ndarray:
    """Exponentially smooth the allocation path to reduce turnover."""
    weights = np.asarray(weights, dtype=float)

    return (
        pd.Series(weights)
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .to_numpy()
    )


def realised_vol(
    returns: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Calculate annualised backward-looking realised volatility."""
    returns = np.asarray(returns, dtype=float)

    series = pd.Series(returns)

    return (
        series
        .rolling(
            window,
            min_periods=max(2, window // 4),
        )
        .std()
        * np.sqrt(TRADING_DAYS)
    ).to_numpy()