"""Evaluation metrics: forecasting accuracy + trading / portfolio quality.

Two families are used consistently in every notebook:

**Forecasting metrics** - RMSE, R2 and the Spearman rank IC.  The rank IC is the metric the
61st-place write-up optimised: in a very noisy problem, ranking opportunities correctly
matters more than nailing the magnitude.

**Portfolio metrics** - annualised return / volatility, Sharpe, max drawdown, hit rate and
``penalised_sharpe``, a re-implementation of the *idea* of the competition metric:

    the score is a Sharpe ratio that is punished when the strategy's realised
    volatility exceeds 120% of the market's realised volatility.

The exact official implementation is not public in this repository, so
``penalised_sharpe`` is an explicitly documented **approximation** used only to compare
models against each other on identical folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import ANNUALISATION, VOL_CEILING_RATIO

__all__ = [
    "rmse",
    "r2_score_safe",
    "spearman_ic",
    "strategy_returns",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "hit_rate",
    "penalised_sharpe",
    "evaluate_positions",
    "evaluate_predictions",
]


def _clean(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


# --------------------------------------------------------------------------------------
# Forecasting metrics
# --------------------------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def r2_score_safe(y_true, y_pred) -> float:
    """Out-of-sample R2 against the mean of the *evaluation* window."""
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def spearman_ic(y_true, y_pred) -> float:
    """Spearman rank information coefficient."""
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) < 3 or np.all(y_pred == y_pred[0]):
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


# --------------------------------------------------------------------------------------
# Portfolio metrics
# --------------------------------------------------------------------------------------
def strategy_returns(positions, market_returns, risk_free=None) -> np.ndarray:
    """Return of a portfolio holding ``w`` in the market and ``1 - w`` in cash."""
    w = np.asarray(positions, dtype=float).ravel()
    mkt = np.asarray(market_returns, dtype=float).ravel()
    if risk_free is None:
        rf = np.zeros_like(mkt)
    else:
        rf = np.asarray(risk_free, dtype=float).ravel()
    return w * mkt + (1.0 - w) * rf


def annualised_return(returns, periods: int = ANNUALISATION) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.mean(r) * periods) if len(r) else float("nan")


def annualised_volatility(returns, periods: int = ANNUALISATION) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.std(r, ddof=1) * np.sqrt(periods)) if len(r) > 1 else float("nan")


def sharpe_ratio(returns, risk_free=None, periods: int = ANNUALISATION) -> float:
    r = np.asarray(returns, dtype=float).ravel()
    if risk_free is not None:
        r = r - np.asarray(risk_free, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(periods))


def max_drawdown(returns) -> float:
    r = np.asarray(returns, dtype=float)
    r = np.nan_to_num(r)
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min()) if len(r) else float("nan")


def hit_rate(positions, market_returns, risk_free=None) -> float:
    """Share of periods in which the active bet moved in the right direction."""
    w = np.asarray(positions, dtype=float).ravel()
    mkt = np.asarray(market_returns, dtype=float).ravel()
    rf = np.zeros_like(mkt) if risk_free is None else np.asarray(risk_free, dtype=float).ravel()
    active = (w - 1.0) * (mkt - rf)
    active = active[np.isfinite(active) & (np.abs(w - 1.0) > 1e-9)]
    return float(np.mean(active > 0)) if len(active) else float("nan")


def penalised_sharpe(
    positions,
    market_returns,
    risk_free=None,
    vol_ceiling: float = VOL_CEILING_RATIO,
    periods: int = ANNUALISATION,
) -> float:
    """Sharpe ratio scaled down when strategy volatility breaches the 120% ceiling.

    Approximation of the competition objective::

        score = sharpe(strategy) * min(1, ceiling * vol_market / vol_strategy)

    There is no reward for running *below* the ceiling and no reward for volatility per
    se - only a punishment for exceeding it, which is exactly the "vol is a cliff, not a
    target" observation from the public write-ups.
    """
    strat = strategy_returns(positions, market_returns, risk_free)
    sr = sharpe_ratio(strat, risk_free, periods)
    vol_s = annualised_volatility(strat, periods)
    vol_m = annualised_volatility(market_returns, periods)
    if not np.isfinite(sr) or not np.isfinite(vol_s) or vol_s == 0:
        return float("nan")
    penalty = min(1.0, (vol_ceiling * vol_m) / vol_s)
    return float(sr * penalty)


def evaluate_positions(
    positions,
    market_returns,
    risk_free=None,
    periods: int = ANNUALISATION,
) -> dict[str, float]:
    """Full portfolio report for a position series."""
    strat = strategy_returns(positions, market_returns, risk_free)
    mkt = np.asarray(market_returns, dtype=float).ravel()
    return {
        "ann_return": annualised_return(strat, periods),
        "ann_volatility": annualised_volatility(strat, periods),
        "sharpe": sharpe_ratio(strat, risk_free, periods),
        "penalised_sharpe": penalised_sharpe(positions, mkt, risk_free, periods=periods),
        "vol_ratio_vs_market": annualised_volatility(strat, periods)
        / max(annualised_volatility(mkt, periods), 1e-12),
        "max_drawdown": max_drawdown(strat),
        "hit_rate": hit_rate(positions, mkt, risk_free),
        "mean_position": float(np.nanmean(positions)),
        "turnover": float(np.nanmean(np.abs(np.diff(np.asarray(positions, dtype=float))))),
    }


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Forecasting-only report."""
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score_safe(y_true, y_pred),
        "spearman_ic": spearman_ic(y_true, y_pred),
    }


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Tidy comparison table from a list of ``{'model': ..., **metrics}`` dictionaries."""
    return pd.DataFrame(rows).set_index("model").round(4)
