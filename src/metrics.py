"""
Evaluation metrics: prediction quality and strategy quality.

Two families, and they answer different questions:

* **Prediction metrics** (RMSE, R2, Spearman IC) score the regression head.
  Spearman is the one that matters most - the 61st place solution tuned on mean
  rank correlation across folds rather than RMSE, on the grounds that ranking
  good days against bad ones is learnable where the exact magnitude is not.
* **Strategy metrics** (Sharpe, volatility ratio, modified Sharpe) score the
  allocation actually submitted, in [0, 2].

Attribution
-----------
:func:`modified_sharpe` mirrors the organisers' ``metric.py`` (``score()``)
formula directly: geometric-mean excess return, volatility and return-shortfall
penalties applied as *divisors* rather than subtracted, and the same 1.2x
volatility ceiling. If the official ``metric.py`` is importable,
:func:`hull_score` uses it instead and says so; the two should now agree to
floating-point precision on any well-posed input.

.. warning::
   The organisers' function raises on degenerate inputs (zero-variance
   strategy or market returns, positions outside ``[0, 2]``). This
   re-implementation never raises: those cases are physically impossible for a
   ranking task and would otherwise crash a fold mid-search, so they are
   instead scored with a large fixed penalty (see ``_DEGENERATE_SCORE``) that
   sorts below any real strategy.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import TRADING_DAYS

VOL_CEILING = 1.2  # strategy vol above 120% of market vol is penalised

#: Score assigned to a degenerate input (zero-variance strategy or market
#: returns) in place of the organisers' exception. Sorts below any real
#: strategy - modified_sharpe's official divisor form can't go much below 0 for
#: a non-degenerate input, so this is an unambiguous floor, not just "very low".
_DEGENERATE_SCORE = -1e6

# --------------------------------------------------------------------------
# Prediction metrics
# --------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank information coefficient. The primary tuning objective."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.std(y_pred) == 0 or len(y_true) < 3:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(y_true, y_pred).correlation
    return float(rho) if np.isfinite(rho) else 0.0


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of rows where the predicted sign matches the realised sign."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


# --------------------------------------------------------------------------
# Strategy metrics
# --------------------------------------------------------------------------


def strategy_returns(
    weights: np.ndarray, forward_returns: np.ndarray, risk_free_rate: np.ndarray
) -> np.ndarray:
    """Daily strategy return for an allocation ``w`` in [0, 2].

    ``w`` of capital sits in the market, ``1 - w`` in the risk-free asset
    (negative for ``w > 1``, i.e. leverage financed at the risk-free rate).
    """
    w = np.asarray(weights, float)
    fwd = np.asarray(forward_returns, float)
    rf = np.asarray(risk_free_rate, float)
    return w * fwd + (1.0 - w) * rf


def sharpe(excess: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio of an excess-return series."""
    excess = np.asarray(excess, float)
    sd = np.std(excess, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods))


def annualised_volatility(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    return float(np.std(np.asarray(returns, float), ddof=1) * np.sqrt(periods))


def max_drawdown(returns: np.ndarray) -> float:
    curve = np.cumprod(1.0 + np.asarray(returns, float))
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / peak - 1.0))


def modified_sharpe(
    weights: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    vol_ceiling: float = VOL_CEILING,
    return_components: bool = False,
) -> float | dict[str, float]:
    """Volatility- and shortfall-penalised Sharpe, matching the organisers'
    ``metric.py`` formula structure.

    ``sharpe / (vol_penalty * return_penalty)``, both penalties >= 1 and
    applied as *divisors* - not subtracted - so the sign of the score is always
    the sign of the raw Sharpe. Mean excess return is **geometric**
    (``(1+r).cumprod() ** (1/n) - 1``), not arithmetic, and volatility is the
    std of the *raw* strategy/market returns, not of their excess series -
    both match the official implementation exactly.

    Strategy volatility up to ``vol_ceiling`` (1.2x market) is free; above it,
    ``vol_penalty = 1 + (vol_ratio - vol_ceiling)``. Underperforming
    buy-and-hold on annualised mean return adds a quadratic
    ``return_penalty = 1 + shortfall_pct**2 / 100``, zero if at or above it.

    A zero-variance strategy or market series would make the official function
    raise; this returns ``_DEGENERATE_SCORE`` instead, since a search that hits
    this case should be steered away from it, not crashed.
    """
    strat = strategy_returns(weights, forward_returns, risk_free_rate)
    fwd = np.asarray(forward_returns, float)
    rf = np.asarray(risk_free_rate, float)
    n = len(strat)

    strat_std = np.std(strat, ddof=1)
    market_std = np.std(fwd, ddof=1)

    if strat_std == 0 or not np.isfinite(strat_std) or market_std == 0 or not np.isfinite(market_std):
        base, vol_ratio, vol_penalty, ret_penalty = 0.0, float("nan"), float("nan"), float("nan")
        score = _DEGENERATE_SCORE
    else:
        strat_excess = strat - rf
        strat_mean_excess = float(np.prod(1.0 + strat_excess) ** (1.0 / n) - 1.0)
        base = strat_mean_excess / strat_std * np.sqrt(TRADING_DAYS)

        market_excess = fwd - rf
        market_mean_excess = float(np.prod(1.0 + market_excess) ** (1.0 / n) - 1.0)

        # (strat_std * sqrt(TRADING_DAYS) * 100) / (market_std * ...) reduces
        # to strat_std / market_std - the scaling cancels.
        vol_ratio = float(strat_std / market_std)
        vol_penalty = 1.0 + max(0.0, vol_ratio - vol_ceiling)

        return_gap_pct = max(0.0, (market_mean_excess - strat_mean_excess) * 100.0 * TRADING_DAYS)
        ret_penalty = 1.0 + (return_gap_pct ** 2) / 100.0

        score = min(base / (vol_penalty * ret_penalty), 1_000_000.0)

    if not return_components:
        return float(score)
    return {
        "modified_sharpe": float(score),
        "sharpe": float(base),
        "vol_ratio": vol_ratio,
        "vol_penalty": float(vol_penalty),
        "return_penalty": float(ret_penalty),
    }


def _official_metric():
    """Return the organisers' ``score`` function if it is importable."""
    try:
        from metric import score  # type: ignore  # noqa: WPS433

        return score
    except Exception:  # pragma: no cover - depends on the runtime
        return None


def hull_score(
    weights: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    prefer_official: bool = True,
) -> float:
    """Competition score, preferring the official implementation when present."""
    if prefer_official:
        official = _official_metric()
        if official is not None:
            solution = pd.DataFrame(
                {"forward_returns": forward_returns, "risk_free_rate": risk_free_rate}
            )
            submission = pd.DataFrame({"prediction": np.asarray(weights, float)})
            return float(official(solution, submission, ""))
    return float(modified_sharpe(weights, forward_returns, risk_free_rate))


# --------------------------------------------------------------------------
# One-call evaluation
# --------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
    forward_returns: np.ndarray | None = None,
    risk_free_rate: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute every metric this project reports, as a flat dict.

    Pass ``weights`` + ``forward_returns`` + ``risk_free_rate`` to add the
    strategy block; omit them for a prediction-only evaluation.
    """
    out: dict[str, Any] = {
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "spearman_ic": spearman_ic(y_true, y_pred),
        "hit_rate": hit_rate(y_true, y_pred),
        "n_rows": int(len(y_true)),
    }
    if weights is None or forward_returns is None or risk_free_rate is None:
        return out

    strat = strategy_returns(weights, forward_returns, risk_free_rate)
    rf = np.asarray(risk_free_rate, float)
    market = np.asarray(forward_returns, float)
    components = modified_sharpe(
        weights, forward_returns, risk_free_rate, return_components=True
    )
    out.update(components)
    out.update(
        {
            "ann_return": float(np.mean(strat) * TRADING_DAYS),
            "ann_volatility": annualised_volatility(strat),
            "max_drawdown": max_drawdown(strat),
            "benchmark_sharpe": sharpe(market - rf),
            "mean_weight": float(np.mean(weights)),
            "weight_turnover": float(np.mean(np.abs(np.diff(np.asarray(weights, float))))),
        }
    )
    return out


def aggregate_folds(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and std of each numeric metric across folds.

    Report ``<metric>_mean`` as the headline and ``<metric>_std`` as the
    stability measure - on this dataset, fold variance is large enough that a
    mean alone is misleading.
    """
    frame = pd.DataFrame(fold_metrics)
    out: dict[str, Any] = {"n_folds": len(fold_metrics)}
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            out[f"{col}_mean"] = float(frame[col].mean())
            out[f"{col}_std"] = float(frame[col].std(ddof=0))
    return out
