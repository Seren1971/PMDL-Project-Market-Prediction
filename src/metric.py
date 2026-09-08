"""Официальная метрика соревнования (копия kaggle-ноутбука hull-competition-sharpe)."""
import numpy as np, pandas as pd

MIN_INVESTMENT, MAX_INVESTMENT = 0.0, 2.0
TRADING_DAYS = 252

def hull_sharpe(position, forward_returns, risk_free_rate, strict=True):
    pos = np.asarray(position, float)
    fwd = np.asarray(forward_returns, float)
    rf  = np.asarray(risk_free_rate, float)
    if strict and (pos.max() > MAX_INVESTMENT or pos.min() < MIN_INVESTMENT):
        raise ValueError(f'position out of [0,2]: [{pos.min()}, {pos.max()}]')
    n = len(pos)
    strat = rf * (1 - pos) + pos * fwd
    strat_excess = strat - rf
    strat_mean = (1 + strat_excess).prod() ** (1 / n) - 1
    strat_std = strat.std(ddof=1)
    if strat_std == 0: raise ValueError('strategy std is zero')
    sharpe = strat_mean / strat_std * np.sqrt(TRADING_DAYS)
    strat_vol = strat_std * np.sqrt(TRADING_DAYS) * 100

    mkt_excess = fwd - rf
    mkt_mean = (1 + mkt_excess).prod() ** (1 / n) - 1
    mkt_std = fwd.std(ddof=1)
    mkt_vol = mkt_std * np.sqrt(TRADING_DAYS) * 100
    if mkt_vol == 0: raise ValueError('market std is zero')

    vol_penalty = 1 + max(0.0, strat_vol / mkt_vol - 1.2)
    return_gap = max(0.0, (mkt_mean - strat_mean) * 100 * TRADING_DAYS)
    return_penalty = 1 + return_gap ** 2 / 100
    return dict(score=min(sharpe / (vol_penalty * return_penalty), 1e6),
                sharpe=sharpe, strat_vol=strat_vol, mkt_vol=mkt_vol,
                vol_ratio=strat_vol / mkt_vol, vol_penalty=vol_penalty,
                return_gap=return_gap, return_penalty=return_penalty,
                strat_ann_excess=strat_mean * TRADING_DAYS * 100,
                mkt_ann_excess=mkt_mean * TRADING_DAYS * 100)
