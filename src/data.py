"""Загрузка данных и антиутечковые примитивы."""
import numpy as np, pandas as pd, re
from pathlib import Path

RAW = Path('data/raw')
NON_FEAT = {'date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns'}

def load_train():
    tr = pd.read_csv(RAW / 'train.csv')
    # дефект данных: на 8990 forward_returns дублирует предыдущий день,
    # тогда как excess-таргет показывает другое. Восстанавливаем по excess.
    i = tr.index[tr.date_id == 8990][0]
    drift = tr.forward_returns.iloc[i-1] - tr.market_forward_excess_returns.iloc[i-1]
    tr.loc[i, 'forward_returns'] = tr.market_forward_excess_returns.iloc[i] + drift
    return tr

def feature_cols(df):
    return [c for c in df.columns if re.fullmatch(r'[A-Z]+\d+', c)]

def known_returns(fwd):
    """r_known[t] = доходность, УЖЕ реализованная к закрытию дня t = fwd[t-1].

    В инференсе на день t доступен только lagged_forward_returns = fwd[t-1].
    Любая скользящая статистика по доходностям должна строиться из этого массива.
    """
    r = np.empty_like(fwd)
    r[0] = np.nan
    r[1:] = fwd[:-1]
    return r
