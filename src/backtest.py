"""Бэктестер: скор по окнам + сравнение с бенчмарком."""
import numpy as np, pandas as pd
from metric import hull_sharpe

def window_scores(pos, fwd, rf, width=180, step=30, start=1006):
    """Скор стратегии на скользящих окнах. start=1006 — первый день, где есть фичи."""
    out = []
    for s in range(start, len(fwd) - width + 1, step):
        sl = slice(s, s + width)
        p = np.clip(np.nan_to_num(pos[sl], nan=1.0), 0, 2)
        try:
            out.append((s, hull_sharpe(p, fwd[sl], rf[sl])['score']))
        except ValueError:
            pass
    return pd.DataFrame(out, columns=['start', 'score'])

def summary(scores):
    s = scores.score.values
    return dict(n=len(s), median=np.median(s), mean=s.mean(), std=s.std(ddof=1),
                p10=np.percentile(s, 10), p90=np.percentile(s, 90),
                min=s.min(), max=s.max(), neg=(s < 0).mean())

def compare(pos, base, fwd, rf, width=180, step=30, start=1006, label='strategy'):
    """Парное сравнение с бенчмарком на ОДНИХ И ТЕХ ЖЕ окнах."""
    a = window_scores(pos, fwd, rf, width, step, start)
    b = window_scores(base, fwd, rf, width, step, start)
    m = a.merge(b, on='start', suffixes=('_s', '_b'))
    d = m.score_s - m.score_b
    n = len(d); wins = (d > 0).sum()
    # биномиальный знаковый тест: вероятность получить >= wins побед при p=0.5
    from math import comb
    pval = sum(comb(n, k) for k in range(wins, n + 1)) / 2**n
    return dict(label=label, n=n, win_rate=wins / n, mean_diff=d.mean(),
                median_diff=d.median(), sign_test_p=pval,
                med_s=m.score_s.median(), med_b=m.score_b.median()), m

def fmt(d):
    return (f"{d['label']:28s} окон={d['n']:3d} побед={d['win_rate']*100:5.1f}% "
            f"медиана: {d['med_s']:+.3f} vs {d['med_b']:+.3f} "
            f"(разница {d['median_diff']:+.3f}, p={d['sign_test_p']:.4f})")
