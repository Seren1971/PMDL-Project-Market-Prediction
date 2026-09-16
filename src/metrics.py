"""
Evaluation metrics: prediction quality and strategy quality.

Two metric families are used:

1. Prediction metrics
   - RMSE
   - R²
   - Spearman IC
   - Hit rate

2. Strategy metrics
   - Raw Sharpe
   - Strategy volatility
   - Volatility ratio
   - Modified competition Sharpe
   - Maximum drawdown

The modified Sharpe implementation mirrors the public Hull Tactical
competition metric as closely as possible while returning a large negative
score for degenerate strategies instead of crashing an optimisation run.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict, overload

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


VOL_CEILING = 1.2

# Used when the strategy or market has zero / invalid variance.
_DEGENERATE_SCORE = -1e6


class ModifiedSharpeComponents(TypedDict):
    """Detailed output of the modified Sharpe calculation."""

    modified_sharpe: float
    sharpe: float
    vol_ratio: float
    vol_penalty: float
    return_penalty: float


# --------------------------------------------------------------------------
# Prediction metrics
# --------------------------------------------------------------------------


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Root mean squared error."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (true - pred) ** 2
            )
        )
    )


def r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Standard coefficient of determination."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    ss_res = float(
        np.sum(
            (true - pred) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (true - np.mean(true)) ** 2
        )
    )

    if ss_tot <= 0:
        return float("nan")

    return float(
        1.0 - ss_res / ss_tot
    )


def spearman_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Rank information coefficient (Pearson correlation of average ranks)."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    finite = (
        np.isfinite(true)
        & np.isfinite(pred)
    )

    true = true[finite]
    pred = pred[finite]

    if (
        len(true) < 3
        or np.std(pred) == 0
    ):
        return 0.0

    true_rank = (
        pd.Series(true)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    pred_rank = (
        pd.Series(pred)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    correlation = float(
        np.corrcoef(
            true_rank,
            pred_rank,
        )[0, 1]
    )

    return (
        correlation
        if np.isfinite(correlation)
        else 0.0
    )


def hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Share of observations where predicted and realised signs agree."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    return float(
        np.mean(
            np.sign(true)
            == np.sign(pred)
        )
    )


# --------------------------------------------------------------------------
# Strategy metrics
# --------------------------------------------------------------------------


def strategy_returns(
    weights: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
) -> np.ndarray:
    """
    Calculate daily portfolio return for allocation w in [0, 2].

    w = 0:
        fully risk-free

    w = 1:
        fully invested in the market

    w = 2:
        leveraged 2x market position financed at the risk-free rate
    """
    allocation = np.asarray(
        weights,
        dtype=float,
    )

    market = np.asarray(
        forward_returns,
        dtype=float,
    )

    risk_free = np.asarray(
        risk_free_rate,
        dtype=float,
    )

    return (
        allocation * market
        + (1.0 - allocation) * risk_free
    )


def sharpe(
    excess: np.ndarray,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised arithmetic Sharpe ratio."""
    values = np.asarray(
        excess,
        dtype=float,
    )

    if len(values) < 2:
        return 0.0

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    if (
        std == 0
        or not np.isfinite(std)
    ):
        return 0.0

    mean = float(
        np.mean(values)
    )

    return float(
        mean
        / std
        * np.sqrt(periods)
    )


def annualised_volatility(
    returns: np.ndarray,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised volatility of a daily return series."""
    values = np.asarray(
        returns,
        dtype=float,
    )

    if len(values) < 2:
        return 0.0

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    if not np.isfinite(std):
        return 0.0

    return float(
        std
        * np.sqrt(periods)
    )


def cumulative_return(
    returns: np.ndarray,
) -> float:
    """Compound total return over the supplied period."""
    values = np.asarray(
        returns,
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    return float(
        np.prod(
            1.0 + values
        )
        - 1.0
    )


def max_drawdown(
    returns: np.ndarray,
) -> float:
    """
    Maximum peak-to-trough loss.

    Returned as a positive fraction.

    Example:
        0.10 means a 10% maximum drawdown.
    """
    values = np.asarray(
        returns,
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    equity = np.concatenate(
        (
            np.array(
                [1.0],
                dtype=float,
            ),
            np.cumprod(
                1.0 + values
            ),
        )
    )

    peaks = np.maximum.accumulate(
        equity
    )

    drawdowns = (
        equity / peaks
        - 1.0
    )

    return float(
        -np.min(drawdowns)
    )


# --------------------------------------------------------------------------
# Modified competition Sharpe
# --------------------------------------------------------------------------


@overload
def modified_sharpe(
    weights: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    vol_ceiling: float = VOL_CEILING,
    return_components: Literal[False] = False,
) -> float:
    ...


@overload
def modified_sharpe(
    weights: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    vol_ceiling: float = VOL_CEILING,
    return_components: Literal[True] = True,
) -> ModifiedSharpeComponents:
    ...


def modified_sharpe(
    weights: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    vol_ceiling: float = VOL_CEILING,
    return_components: bool = False,
) -> float | ModifiedSharpeComponents:
    """
    Calculate the Hull Tactical competition-style modified Sharpe score.

    The raw geometric Sharpe is divided by two possible penalties:

    1. Volatility penalty
       Applied when strategy volatility exceeds `vol_ceiling` times
       market volatility.

    2. Return-shortfall penalty
       Applied when the strategy's geometric excess return is below
       the market's geometric excess return.
    """
    allocation = np.asarray(
        weights,
        dtype=float,
    )

    market = np.asarray(
        forward_returns,
        dtype=float,
    )

    risk_free = np.asarray(
        risk_free_rate,
        dtype=float,
    )

    if not (
        len(allocation)
        == len(market)
        == len(risk_free)
    ):
        raise ValueError(
            "weights, forward_returns and risk_free_rate "
            "must have the same length."
        )

    if len(allocation) < 2:
        if return_components:
            return ModifiedSharpeComponents(
                modified_sharpe=float(
                    _DEGENERATE_SCORE
                ),
                sharpe=0.0,
                vol_ratio=float("nan"),
                vol_penalty=float("nan"),
                return_penalty=float("nan"),
            )

        return float(
            _DEGENERATE_SCORE
        )

    strategy = strategy_returns(
        allocation,
        market,
        risk_free,
    )

    n_rows = len(strategy)

    strategy_std = float(
        np.std(
            strategy,
            ddof=1,
        )
    )

    market_std = float(
        np.std(
            market,
            ddof=1,
        )
    )

    if (
        strategy_std == 0
        or not np.isfinite(strategy_std)
        or market_std == 0
        or not np.isfinite(market_std)
    ):
        score = float(
            _DEGENERATE_SCORE
        )

        raw_sharpe = 0.0
        vol_ratio = float("nan")
        vol_penalty = float("nan")
        return_penalty = float("nan")

    else:
        strategy_excess = (
            strategy
            - risk_free
        )

        market_excess = (
            market
            - risk_free
        )

        strategy_growth = float(
            np.prod(
                1.0
                + strategy_excess
            )
        )

        market_growth = float(
            np.prod(
                1.0
                + market_excess
            )
        )

        if (
            strategy_growth <= 0
            or market_growth <= 0
            or not np.isfinite(
                strategy_growth
            )
            or not np.isfinite(
                market_growth
            )
        ):
            score = float(
                _DEGENERATE_SCORE
            )

            raw_sharpe = 0.0
            vol_ratio = float("nan")
            vol_penalty = float("nan")
            return_penalty = float("nan")

        else:
            strategy_mean_excess = float(
                strategy_growth
                ** (
                    1.0
                    / n_rows
                )
                - 1.0
            )

            market_mean_excess = float(
                market_growth
                ** (
                    1.0
                    / n_rows
                )
                - 1.0
            )

            raw_sharpe = float(
                strategy_mean_excess
                / strategy_std
                * np.sqrt(
                    TRADING_DAYS
                )
            )

            vol_ratio = float(
                strategy_std
                / market_std
            )

            vol_penalty = float(
                1.0
                + max(
                    0.0,
                    vol_ratio
                    - vol_ceiling,
                )
            )

            return_gap_pct = float(
                max(
                    0.0,
                    (
                        market_mean_excess
                        - strategy_mean_excess
                    )
                    * 100.0
                    * TRADING_DAYS,
                )
            )

            return_penalty = float(
                1.0
                + return_gap_pct**2
                / 100.0
            )

            score = float(
                min(
                    raw_sharpe
                    / (
                        vol_penalty
                        * return_penalty
                    ),
                    1_000_000.0,
                )
            )

    if not return_components:
        return float(score)

    return ModifiedSharpeComponents(
        modified_sharpe=float(
            score
        ),
        sharpe=float(
            raw_sharpe
        ),
        vol_ratio=float(
            vol_ratio
        ),
        vol_penalty=float(
            vol_penalty
        ),
        return_penalty=float(
            return_penalty
        ),
    )


# --------------------------------------------------------------------------
# Optional official Kaggle scorer
# --------------------------------------------------------------------------


def _official_metric() -> Any | None:
    """Return the official Kaggle score function when available."""
    try:
        from metric import score  # type: ignore[import-not-found]

        return score

    except Exception:
        return None


def hull_score(
    weights: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    prefer_official: bool = True,
) -> float:
    """
    Return the competition score.

    If Kaggle's official `metric.py` is available, use it. Otherwise use
    the local verified implementation.
    """
    if prefer_official:
        official = _official_metric()

        if official is not None:
            solution = pd.DataFrame(
                {
                    "forward_returns": np.asarray(
                        forward_returns,
                        dtype=float,
                    ),
                    "risk_free_rate": np.asarray(
                        risk_free_rate,
                        dtype=float,
                    ),
                }
            )

            submission = pd.DataFrame(
                {
                    "prediction": np.asarray(
                        weights,
                        dtype=float,
                    )
                }
            )

            return float(
                official(
                    solution,
                    submission,
                    "",
                )
            )

    return modified_sharpe(
        weights,
        forward_returns,
        risk_free_rate,
    )


# --------------------------------------------------------------------------
# Complete evaluation
# --------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
    forward_returns: np.ndarray | None = None,
    risk_free_rate: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Calculate all prediction and optional strategy metrics.

    Regression metrics MUST receive the raw regression prediction.

    Do not pass the standardised trading signal as `y_pred`.
    """
    true = np.asarray(
        y_true,
        dtype=float,
    )

    prediction = np.asarray(
        y_pred,
        dtype=float,
    )

    output: dict[str, Any] = {
        "rmse": rmse(
            true,
            prediction,
        ),
        "r2": r2(
            true,
            prediction,
        ),
        "spearman_ic": spearman_ic(
            true,
            prediction,
        ),
        "hit_rate": hit_rate(
            true,
            prediction,
        ),
        "n_rows": int(
            len(true)
        ),
    }

    if (
        weights is None
        or forward_returns is None
        or risk_free_rate is None
    ):
        return output

    allocation = np.asarray(
        weights,
        dtype=float,
    )

    market = np.asarray(
        forward_returns,
        dtype=float,
    )

    risk_free = np.asarray(
        risk_free_rate,
        dtype=float,
    )

    strategy = strategy_returns(
        allocation,
        market,
        risk_free,
    )

    components = modified_sharpe(
        allocation,
        market,
        risk_free,
        return_components=True,
    )

    output.update(
        components
    )

    output.update(
        {
            "ann_return": float(
                np.mean(strategy)
                * TRADING_DAYS
            ),
            "ann_volatility": annualised_volatility(
                strategy
            ),
            "cumulative_return": cumulative_return(
                strategy
            ),
            "max_drawdown": max_drawdown(
                strategy
            ),
            "benchmark_sharpe": sharpe(
                market
                - risk_free
            ),
            "mean_weight": float(
                np.mean(
                    allocation
                )
            ),
            "weight_turnover": (
                float(
                    np.mean(
                        np.abs(
                            np.diff(
                                allocation
                            )
                        )
                    )
                )
                if len(allocation) > 1
                else 0.0
            ),
        }
    )

    return output


# --------------------------------------------------------------------------
# Fold aggregation
# --------------------------------------------------------------------------


def aggregate_folds(
    fold_metrics: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[str, Any]:
    """
    Aggregate fold-level metrics.

    Both mean and standard deviation are reported because financial
    time-series performance can vary strongly across market regimes.
    """
    if not fold_metrics:
        return {
            "n_folds": 0
        }

    frame = pd.DataFrame(
        fold_metrics
    )

    output: dict[str, Any] = {
        "n_folds": int(
            len(
                fold_metrics
            )
        )
    }

    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            continue

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        output[
            f"{column}_mean"
        ] = float(
            values.mean()
        )

        output[
            f"{column}_std"
        ] = float(
            values.std(
                ddof=0
            )
        )

    return output