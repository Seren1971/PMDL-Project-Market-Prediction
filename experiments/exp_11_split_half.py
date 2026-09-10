"""Честная проверка H5/H6: отбор фич на первой половине, тест на второй."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns, feature_cols
from metric import hull_sharpe

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); feats = feature_cols(tr); V = tr[feats].values
S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
z5 = np.nan_to_num(-zc(S.rolling(5).sum().values))
SPLIT = 6000          # ~2013 год
cal = pd.read_csv('data/interim_calendar_estimate.csv')
print(f"граница разбиения: date_id {SPLIT} = {cal.loc[cal.date_id==SPLIT,'date_est'].values[0]}")

def wins(pos, lo, hi, width=180, step=180):     # НЕПЕРЕСЕКАЮЩИЕСЯ окна
    pos = np.nan_to_num(pos, nan=1.0)
    return np.array([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                     for s in range(lo, hi-width+1, step)])
def stat(pos, lo, hi):
    d = wins(pos, lo, hi) - wins(np.full(n,1.0), lo, hi)
    N=len(d); k=int((d>0).sum())
    return np.median(d), k/N, sum(comb(N,i) for i in range(k,N+1))/2**N, N

print("\n=== шаг 1: отбираем фичи ТОЛЬКО на первой половине (2300..6000) ===")
rows=[]
for i,c in enumerate(feats):
    x = V[:,i]
    if np.isnan(x[2300:SPLIT]).mean() > 0.3: continue
    pos = np.clip(1.0 + 0.25*np.nan_to_num(zc(x,504)), 0, 2)
    md, wr, p, N = stat(pos, 2300, SPLIT)
    rows.append(dict(feat=c, med_in=md, win_in=wr, n_in=N))
d1 = pd.DataFrame(rows).sort_values('med_in', ascending=False)
top8 = list(d1.head(8).feat)
print(f"  кандидатов: {len(d1)}, окон в первой половине: {d1.n_in.iloc[0]}")
print(f"  топ-8 по первой половине: {top8}")

print("\n=== шаг 2: тестируем их на второй половине (6000..конец), не трогая больше ничего ===")
print(f"  {'фича':6s} {'мед.разн IN':>12s} {'мед.разн OUT':>13s} {'побед OUT':>10s}")
out=[]
for c in top8:
    x = V[:, feats.index(c)]
    pos = np.clip(1.0 + 0.25*np.nan_to_num(zc(x,504)), 0, 2)
    md_o, wr_o, p_o, N = stat(pos, SPLIT, n)
    md_i = float(d1.loc[d1.feat==c,'med_in'].iloc[0])
    out.append((c, md_i, md_o, wr_o))
    print(f"  {c:6s} {md_i:+12.3f} {md_o:+13.3f} {wr_o*100:9.1f}%")
o = pd.DataFrame(out, columns=['feat','in','out','win'])
print(f"\n  сохранили положительный знак вне выборки: {(o.out>0).sum()}/8")
print(f"  корреляция in-sample и out-of-sample результата: {o['in'].corr(o['out']):+.3f}")

print("\n=== шаг 3: ансамбль топ-8 (отобранных ТОЛЬКО по первой половине) на второй ===")
Z = np.column_stack([np.nan_to_num(zc(V[:, feats.index(c)],504)) for c in top8])
for name, sg in (('ансамбль 8 фич', Z.mean(1)), ('лучшая фича', Z[:,0]),
                 ('разворот 5 дней', z5), ('ансамбль + разворот', 0.5*Z.mean(1)+0.5*z5)):
    md, wr, p, N = stat(np.clip(1.0+0.25*sg,0,2), SPLIT, n)
    print(f"  {name:22s} мед.разн={md:+.3f} побед={wr*100:5.1f}% p={p:.4f} (окон {N})")

print("\n=== контроль: как разворот вёл себя в ПЕРВОЙ половине (он не отбирался по ней) ===")
md, wr, p, N = stat(np.clip(1.0+0.25*z5,0,2), 2300, SPLIT)
print(f"  разворот 5 дней в первой половине: мед.разн={md:+.3f} побед={wr*100:.1f}% p={p:.4f} (окон {N})")
