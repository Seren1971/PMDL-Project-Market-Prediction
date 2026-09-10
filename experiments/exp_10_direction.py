"""H5-H8: предсказывают ли фичи Hull направление; H12-H13: методологические грабли."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns, feature_cols
from metric import hull_sharpe
from wf import walk_forward, oos_r2

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); START = 2300
feats = feature_cols(tr); V = tr[feats].values
S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
z5 = np.nan_to_num(-zc(S.rolling(5).sum().values))
tgt = fwd - rf                                    # сверхдоходность, то что реально надо предсказать

def wins(pos, width=180, step=30):
    pos = np.nan_to_num(pos, nan=1.0)
    return np.array([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                     for s in range(START, n-width+1, step)])
B = wins(np.full(n,1.0))
def stat(pos):
    d = wins(pos) - B; N=len(d); k=int((d>0).sum())
    return np.median(d), k/N, sum(comb(N,i) for i in range(k,N+1))/2**N

print("=== H5: есть ли среди 94 фич предиктор НАПРАВЛЕНИЯ (walk-forward, честно) ===")
rows = []
for i, c in enumerate(feats):
    x = V[:, i]
    m = ~np.isnan(x) & (np.arange(n) >= START)
    if m.sum() < 1500: continue
    xz = zc(x, 504)
    pos = np.clip(1.0 + 0.25*np.nan_to_num(xz), 0, 2)
    md, wr, p = stat(pos)
    # плюс прямая корреляция фичи с завтрашней сверхдоходностью на той же выборке
    mm = m & ~np.isnan(tgt)
    r = np.corrcoef(x[mm], tgt[mm])[0,1]
    t = r*np.sqrt(mm.sum()-2)/np.sqrt(1-r**2)
    rows.append(dict(feat=c, med_diff=md, win=wr, p=p, corr=r, t=t, n=int(mm.sum())))
d = pd.DataFrame(rows).sort_values('med_diff', ascending=False)
print(f"  проверено фич: {len(d)}")
print("  топ-8 по медианному приросту скора:")
print(d.head(8)[['feat','med_diff','win','p','corr','t']].round(4).to_string(index=False))
print("  худшие 3:")
print(d.tail(3)[['feat','med_diff','win','p','corr','t']].round(4).to_string(index=False))
thr = 0.05/len(d)
print(f"\n  порог Бонферрони: p < {thr:.5f}")
print(f"  фич прошли: {(d.p < thr).sum()} -> {list(d[d.p<thr].feat)}")
print(f"  фич с |t| > 3 по корреляции: {(d.t.abs()>3).sum()} -> {list(d[d.t.abs()>3].feat)}")
print(f"  для сравнения, разворот 5 дней: мед.разн={stat(np.clip(1.0+0.25*z5,0,2))[0]:+.4f}")

print("\n=== H6: ансамбль слабых против отбора сильного ===")
top8 = list(d.head(8).feat)
Z = np.column_stack([np.nan_to_num(zc(V[:, feats.index(c)], 504)) for c in top8])
ens = Z.mean(1)
for name, sg in (('среднее 8 лучших фич', ens), ('одна лучшая фича', Z[:,0]),
                 ('среднее 8 фич + разворот', 0.5*ens + 0.5*z5), ('только разворот', z5)):
    md, wr, p = stat(np.clip(1.0+0.25*sg, 0, 2))
    print(f"  {name:28s} мед.разн={md:+.4f} побед={wr*100:5.1f}% p={p:.4f}")

print("\n=== H7: ограничения на прогноз (клип снизу нулём) ===")
pred_log_dummy = None
# линейная модель на топ-8 фичах, walk-forward, прогноз сверхдоходности
X8 = np.column_stack([V[:, feats.index(c)] for c in top8])
p8, b8 = walk_forward(X8, tgt, refit=252, start=2300, embargo=1)
r2, k = oos_r2(tgt, p8, b8)
print(f"  OOS R^2 прогноза сверхдоходности на 8 фичах: {r2:+.5f} (n={k})")
print(f"  для сравнения Hull заявляют ~0.0133 в своей статье")
sc = np.nan_to_num(p8) / (np.nanstd(p8) + 1e-12)
for name, pos in (('без ограничений', np.clip(1.0+0.25*sc, 0, 2)),
                  ('прогноз клипнут снизу 0', np.clip(1.0+0.25*np.maximum(sc,0), 0, 2)),
                  ('только знак', np.where(sc>0, 1.2, 0.8))):
    md, wr, p = stat(pos)
    print(f"  {name:28s} мед.разн={md:+.4f} побед={wr*100:5.1f}% p={p:.4f}")

print("\n=== H12: bfill против ffill (утечка из будущего) ===")
Xf = pd.DataFrame(V).ffill().values
Xb = pd.DataFrame(V).bfill().ffill().values
for name, X in (('ffill (честно)', Xf), ('bfill (утечка)', Xb)):
    pr, bb = walk_forward(X[:, [feats.index(c) for c in top8]], tgt, refit=252, start=2300, embargo=1)
    r2, kk = oos_r2(tgt, pr, bb)
    s2 = np.nan_to_num(pr)/(np.nanstd(pr)+1e-12)
    md, wr, p = stat(np.clip(1.0+0.25*s2, 0, 2))
    print(f"  {name:18s} OOS R2={r2:+.5f}  мед.разн скора={md:+.4f}")

print("\n=== H13: зазор (embargo) между обучением и тестом, прогноз волы ===")
H=21; y = np.log(np.sqrt(pd.Series(fwd**2).rolling(H).mean().shift(-(H-1)).values)+1e-8)
s2_ = pd.Series(np.abs(rk))
har = np.log(np.c_[s2_.rolling(1).mean(), s2_.rolling(5).mean(), s2_.rolling(22).mean()]+1e-8)
for emb in (0, 1, 21, 63):
    pl, bl = walk_forward(har, y, embargo=emb)
    r2, kk = oos_r2(y, pl, bl)
    print(f"  embargo={emb:2d} дней: OOS R2={r2:+.4f}")
print("  (таргет волы охватывает 21 день вперёд -> без зазора часть таргета видна в обучении)")
