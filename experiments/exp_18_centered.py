"""Точное центрирование экспозиции + добавка выживших фич."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns, feature_cols
from metric import hull_sharpe

rng = np.random.default_rng(11)
tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); S = pd.Series(rk); LATE = 6500
feats = feature_cols(tr); V = tr[feats].values
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
Z = np.column_stack([np.nan_to_num(-zc(S.rolling(h).sum().values)) for h in (1,5,22,63)])
W = np.column_stack([1.0/(pd.Series(Z[:,j]).rolling(252).std().values+1e-9) for j in range(4)])
W = W/np.nansum(W,axis=1,keepdims=True); ens = np.nan_to_num(np.nansum(Z*W,axis=1))

def stat(pos, lo, hi, width=180):
    d = np.array([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                  - hull_sharpe(np.ones(width), fwd[s:s+width], rf[s:s+width])['score']
                  for s in range(lo, hi-width+1, width)])
    N=len(d); k=int((d>0).sum())
    return np.median(d), d.mean(), k/N, sum(comb(N,i) for i in range(k,N+1))/2**N, N
def show(name, pos):
    a = stat(pos, 2300, n); b = stat(pos, LATE, n)
    mp = np.nanmean(pos[LATE:])
    print(f"  {name:40s} вся: {a[0]:+.3f}/{a[1]:+.3f}/{a[2]*100:3.0f}%  "
          f"посл.10л: {b[0]:+.3f}/{b[1]:+.3f}/{b[2]*100:3.0f}% (ср.поз {mp:.3f})")
    return b

print("=== мед/среднее/побед. Проблема: средняя позиция < 1 в бычьем рынке штрафуется ===")
base = pd.Series(np.clip(0.95+0.25*ens, 0.85, 1.45)).ewm(halflife=3).mean().values
show('кандидат 2-го круга (ср.поз 0.98)', base)

# точное причинное центрирование: вычитаем скользящее среднее САМОЙ позиции
for w in (252, 504, 1008):
    raw = np.clip(0.95+0.25*ens, 0.85, 1.45)
    drift = pd.Series(raw).rolling(w, min_periods=250).mean().shift(1).values
    cen = np.clip(raw - np.nan_to_num(drift, nan=1.0) + 1.0, 0.8, 1.5)
    cen = pd.Series(cen).ewm(halflife=3).mean().values
    show(f'центрировано по окну {w}д', cen)

print()
# центрирование самого сигнала + фиксированная средняя
for mp in (1.0, 1.02, 1.05):
    zz = ens - np.nan_to_num(pd.Series(ens).rolling(504, min_periods=250).mean().shift(1).values)
    pos = pd.Series(np.clip(mp + 0.25*zz, 0.85, 1.5)).ewm(halflife=3).mean().values
    show(f'сигнал центрирован, ср.поз={mp}', pos)

print("\n=== добавка фич, переживших split-half (V7, V13, E19, M11) ===")
surv = ['V7','V13','E19','M11']
Fz = np.column_stack([np.nan_to_num(zc(V[:, feats.index(c)], 504)) for c in surv])
fsig = Fz.mean(1)
zz = ens - np.nan_to_num(pd.Series(ens).rolling(504, min_periods=250).mean().shift(1).values)
for wgt in (0.0, 0.25, 0.5):
    mix = (1-wgt)*zz + wgt*fsig
    pos = pd.Series(np.clip(1.02 + 0.25*mix, 0.85, 1.5)).ewm(halflife=3).mean().values
    show(f'вес фич {wgt:.2f} (0 = только разворот)', pos)

print("\n=== порог: действовать только на крупных движениях (по Нагелю — плата за ликвидность) ===")
for thr in (0.0, 0.5, 1.0, 1.5):
    zt = np.where(np.abs(zz) > thr, zz, 0.0)
    pos = pd.Series(np.clip(1.02 + 0.30*zt, 0.85, 1.5)).ewm(halflife=3).mean().values
    show(f'порог |z| > {thr}', pos)
