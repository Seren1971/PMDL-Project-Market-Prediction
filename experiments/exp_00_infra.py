"""I1-I3: проверка бэктестера и бенчмарков."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from data import load_train
from metric import hull_sharpe
from backtest import window_scores, summary, compare, fmt

tr = load_train()
fwd, rf = tr.forward_returns.values, tr.risk_free_rate.values
n = len(fwd)

# Окно публичного LB найдено перебором: единственное, где ОБА числа сообщества
# (0.460 при pos=1.0 и 0.4688 при pos=0.806) воспроизводятся, и оно заканчивается
# ровно на 8989 — последнем date_id макетного test.csv.
LB = slice(8810, 8990)
print("=== верификация метрики на найденном окне публичного LB (date_id 8810..8989) ===")
for c, ref in ((0.806, 0.4688), (1.0, 0.460)):
    r = hull_sharpe(np.full(180, c), fwd[LB], rf[LB])
    print(f"  pos={c:.3f}: наш score={r['score']:.4f}  сообщество={ref}  дельта={abs(r['score']-ref):.4f}")

print("\n=== I3: константы на всей истории (date_id>=1006) ===")
for c in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5):
    r = hull_sharpe(np.full(n - 1006, c), fwd[1006:], rf[1006:])
    print(f"  pos={c:.1f}: score={r['score']:.4f}  vol_ratio={r['vol_ratio']:.2f}")

print("\n=== I2: распределение скора констант по окнам 180 дней, шаг 30 ===")
rows = []
for c in (0.6, 0.8, 1.0, 1.2, 1.5):
    s = summary(window_scores(np.full(n, c), fwd, rf))
    s['pos'] = c; rows.append(s)
d = pd.DataFrame(rows).set_index('pos')
print(d[['n','median','mean','std','p10','p90','min','max','neg']].round(3).to_string())

print("\n=== выбор бенчмарка: какая константа лучшая по медиане окон ===")
best = d['median'].idxmax()
print(f"  лучшая по медиане окон: pos={best} (медиана {d['median'].max():.3f})")

print("\n=== санити: парное сравнение 0.8 против 1.0 ===")
r, _ = compare(np.full(n, 0.8), np.full(n, 1.0), fwd, rf, label='const 0.8 vs 1.0')
print(" ", fmt(r))
