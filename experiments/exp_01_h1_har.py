"""H1: HAR предсказывает волатильность не хуже лучшей фичи Hull."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from data import load_train, known_returns, feature_cols
from wf import walk_forward, oos_r2, oos_corr

tr = load_train()
fwd = tr.forward_returns.values
rk = known_returns(fwd)                    # доходность, известная к закрытию дня t
n = len(fwd)

# ---- таргет: реализованная волатильность следующих 21 дня ----
H = 21
fut = pd.Series(fwd**2).rolling(H).mean().shift(-(H-1))
y = np.sqrt(fut.values)                     # RV вперёд
ylog = np.log(y + 1e-8)

# ---- HAR: средние |r| по окнам 1/5/22, только из известных доходностей ----
s = pd.Series(np.abs(rk))
har = np.c_[s.rolling(1).mean(), s.rolling(5).mean(), s.rolling(22).mean()]
harlog = np.log(har + 1e-8)

feats = feature_cols(tr)
V = tr[feats].values

print("=== H1: прогноз волатильности на 21 день вперёд, walk-forward ===")
print("оценка: OOS R^2 против расширяющегося среднего; таргет — log(RV)\n")

def run(X, name):
    p, b = walk_forward(X, ylog)
    r2, k = oos_r2(ylog, p, b)
    c = oos_corr(ylog, p)
    print(f"  {name:38s} OOS R2={r2:+.4f}  corr={c:.3f}  n={k}")
    return r2

res = {}
res['HAR (|r| за 1/5/22 дня)'] = run(harlog, 'HAR (|r| за 1/5/22 дня)')
res['только |r| за 1 день'] = run(harlog[:, :1], 'только |r| за 1 день')
res['только |r| за 22 дня'] = run(harlog[:, 2:3], 'только |r| за 22 дня')

print()
# лучшие одиночные фичи по корреляции с будущей волой
sub = ~np.isnan(y) & (np.arange(n) >= 7000)
cors = {}
for i, c in enumerate(feats):
    m = sub & ~np.isnan(V[:, i])
    if m.sum() > 500: cors[c] = abs(np.corrcoef(V[m, i], y[m])[0, 1])
top = sorted(cors, key=cors.get, reverse=True)[:8]
print(f"  топ фич по |corr| с будущей волой: {', '.join(f'{c}({cors[c]:.2f})' for c in top[:5])}\n")

for c in top[:4]:
    res[c] = run(V[:, [feats.index(c)]], f'одна фича {c}')

res['топ-5 фич вместе'] = run(V[:, [feats.index(c) for c in top[:5]]], 'топ-5 фич вместе')

print("\n=== вердикт H1 ===")
har_r2 = res['HAR (|r| за 1/5/22 дня)']
best_feat = max((res[c] for c in top[:4] if c in res))
print(f"  HAR: {har_r2:+.4f}   лучшая одиночная фича: {best_feat:+.4f}")
if har_r2 >= 0.35 and har_r2 >= best_feat:
    print("  H1 ПОДТВЕРЖДЕНА: HAR >= 0.35 и не хуже лучшей фичи")
elif har_r2 < best_feat:
    print("  H1 ОТВЕРГНУТА: одиночная фича Hull бьёт HAR -> у них уже готовый прогноз волы")
else:
    print(f"  H1 ЧАСТИЧНО: HAR лучший, но R2={har_r2:.3f} < 0.35")
