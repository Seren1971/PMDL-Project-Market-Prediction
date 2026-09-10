"""Честная проверка порога: подбор ТОЛЬКО на раннем периоде, тест на позднем."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns, feature_cols
from metric import hull_sharpe

rng = np.random.default_rng(21)
tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); S = pd.Series(rk)
EARLY_END = 6500      # подбор только до этой точки (~2015)
feats = feature_cols(tr); V = tr[feats].values
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
Z = np.column_stack([np.nan_to_num(-zc(S.rolling(h).sum().values)) for h in (1,5,22,63)])
W = np.column_stack([1.0/(pd.Series(Z[:,j]).rolling(252).std().values+1e-9) for j in range(4)])
W = W/np.nansum(W,axis=1,keepdims=True); ens = np.nan_to_num(np.nansum(Z*W,axis=1))
zz = ens - np.nan_to_num(pd.Series(ens).rolling(504, min_periods=250).mean().shift(1).values)

def build(thr, mp=1.02, a=0.30, hl=3, lo=0.85, hi=1.5):
    zt = np.where(np.abs(zz) > thr, zz, 0.0)
    return pd.Series(np.clip(mp + a*zt, lo, hi)).ewm(halflife=hl).mean().values

def diffs(pos, lo, hi, width):
    return np.array([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                     - hull_sharpe(np.ones(width), fwd[s:s+width], rf[s:s+width])['score']
                     for s in range(lo, hi-width+1, width)])
def rep(d, tag):
    N=len(d); k=int((d>0).sum()); p=sum(comb(N,i) for i in range(k,N+1))/2**N
    print(f"    {tag:26s} мед={np.median(d):+.3f} средн={d.mean():+.3f} побед={k}/{N} ({k/N*100:.0f}%) p={p:.4f}")
    return p

print("=== шаг 1: подбор порога ТОЛЬКО на раннем периоде 2300..6500 ===")
grid = {}
for thr in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    d = diffs(build(thr), 2300, EARLY_END, 180)
    grid[thr] = (np.median(d), d.mean(), (d>0).mean())
    print(f"  порог {thr:.2f}: мед={grid[thr][0]:+.3f} средн={grid[thr][1]:+.3f} побед={grid[thr][2]*100:.0f}%")
best_med = max(grid, key=lambda t: grid[t][0])
best_win = max(grid, key=lambda t: grid[t][2])
print(f"  -> лучший по медиане на раннем: {best_med};  по доле побед: {best_win}")

print("\n=== шаг 2: применяем выбранный порог к позднему периоду (не подглядывая) ===")
for thr, lab in ((best_med, f'выбран по медиане ({best_med})'), (best_win, f'выбран по побед ({best_win})')):
    print(f"  {lab}:")
    for width in (130, 180):
        rep(diffs(build(thr), EARLY_END, n, width), f'позднее, окно {width}')

print("\n=== устойчивость порога (все значения на позднем периоде) ===")
for thr in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5):
    d = diffs(build(thr), EARLY_END, n, 180)
    print(f"  порог {thr:.2f}: мед={np.median(d):+.3f} средн={d.mean():+.3f} побед={(d>0).mean()*100:.0f}%")

print("\n=== блочный бутстрап на позднем периоде для порога 1.0 ===")
pos = build(1.0); lo = EARLY_END; BL=252; nb=(n-lo)//BL
blocks=[(lo+i*BL, lo+(i+1)*BL) for i in range(nb)]
dd=[]
for _ in range(3000):
    idx = rng.integers(0, nb, nb)
    sel = np.concatenate([np.arange(*blocks[i]) for i in idx])
    try:
        dd.append(hull_sharpe(np.clip(pos[sel],0,2), fwd[sel], rf[sel])['score']
                  - hull_sharpe(np.ones(len(sel)), fwd[sel], rf[sel])['score'])
    except ValueError: pass
dd=np.array(dd)
print(f"  среднее={dd.mean():+.3f} 95%ДИ=[{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}] "
      f"P(>0)={(dd>0).mean()*100:.1f}% (блоков {nb})")

print("\n=== сколько времени стратегия вообще отклоняется от 1.0 ===")
for thr in (0.0, 1.0):
    p = build(thr)
    off = np.mean(np.abs(p[EARLY_END:] - p[EARLY_END:].mean()) > 0.02)
    print(f"  порог {thr}: доля дней с заметным отклонением {off*100:.0f}%, "
          f"std позиции {p[EARLY_END:].std():.3f}, оборот {np.abs(np.diff(p[EARLY_END:])).mean():.4f}")

print("\n=== контрольный вопрос: а просто константа 1.02-1.05 не даёт то же самое? ===")
for c in (1.0, 1.02, 1.05):
    d = diffs(np.full(n, c), EARLY_END, n, 180)
    print(f"  константа {c}: мед={np.median(d):+.3f} средн={d.mean():+.3f} побед={(d>0).mean()*100:.0f}%")
