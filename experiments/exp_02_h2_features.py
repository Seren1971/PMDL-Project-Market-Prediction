"""H2: фичи Hull добавляют информацию поверх HAR (сравнение на ОДНОЙ выборке)."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, collections
from data import load_train, known_returns, feature_cols
from wf import walk_forward, walk_forward_select, oos_r2, oos_corr

tr = load_train(); fwd = tr.forward_returns.values; rk = known_returns(fwd); n = len(fwd)
H = 21
y = np.log(np.sqrt(pd.Series(fwd**2).rolling(H).mean().shift(-(H-1)).values) + 1e-8)
s = pd.Series(np.abs(rk))
har = np.log(np.c_[s.rolling(1).mean(), s.rolling(5).mean(), s.rolling(22).mean()] + 1e-8)
feats = feature_cols(tr); V = tr[feats].values
X = np.c_[har, V]                      # столбцы 0,1,2 = HAR
HARCOLS = (0, 1, 2)

def ev(pred, bench, name, mask=None):
    yy, pp, bb = y.copy(), pred.copy(), bench.copy()
    if mask is not None: pp[~mask] = np.nan
    r2, k = oos_r2(yy, pp, bb); c = oos_corr(yy, pp)
    print(f"  {name:44s} OOS R2={r2:+.4f}  corr={c:.3f}  n={k}")
    return r2, pp

print("=== H2: добавляют ли фичи Hull что-то поверх HAR ===\n")
p_har, b_har = walk_forward(har, y)
# общая маска: сравниваем только там, где ВСЕ модели дали предсказание
runs = {}
runs['HAR'] = (p_har, b_har, [])
for k in (1, 3, 5, 10):
    p, b, log = walk_forward_select(X, y, k=k, keep=HARCOLS)
    runs[f'HAR + топ-{k} фич (отбор в фолде)'] = (p, b, log)
p, b, log = walk_forward_select(X, y, k=5, keep=())
runs['топ-5 фич без HAR'] = (p, b, log)

common = np.ones(n, bool)
for p, _, _ in runs.values(): common &= ~np.isnan(p)
print(f"общая выборка для сравнения: {common.sum()} дней\n")
r2s = {}
for name, (p, b, _) in runs.items():
    pp = p.copy(); pp[~common] = np.nan
    r2, kk = oos_r2(y, pp, b); c = oos_corr(y, pp)
    r2s[name] = r2
    print(f"  {name:44s} OOS R2={r2:+.4f}  corr={c:.3f}  n={kk}")

print("\n=== какие фичи отбирались чаще всего (k=5) ===")
cnt = collections.Counter()
for _, sel in runs['HAR + топ-5 фич (отбор в фолде)'][2]:
    for j in sel: cnt[feats[j-3] if j >= 3 else f'HAR{j}'] += 1
tot = len(runs['HAR + топ-5 фич (отбор в фолде)'][2])
print("  " + ", ".join(f"{c}:{v}/{tot}" for c, v in cnt.most_common(10)))

print("\n=== вердикт H2 ===")
base = r2s['HAR']
best = max(v for k_, v in r2s.items() if k_ != 'HAR' and 'без HAR' not in k_)
gain = best - base
print(f"  HAR: {base:+.4f}   лучший с фичами: {best:+.4f}   прирост: {gain:+.4f}")
print("  H2 ПОДТВЕРЖДЕНА (прирост >= 0.05)" if gain >= 0.05
      else f"  H2 ОТВЕРГНУТА: прирост {gain:+.4f} < 0.05 -> работаем на чистом HAR")
