"""Правильна ли базовая экспозиция? Метрика даёт бесплатный бюджет волатильности до 1.2."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns
from metric import hull_sharpe

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); LATE = 6500
cal = pd.read_csv('data/interim_calendar_estimate.csv')

print("=== A. Оптимальная КОНСТАНТА по периодам (не подгонка, а диагностика метрики) ===")
print(f"  {'период':22s} " + " ".join(f"{c:>6.1f}" for c in (0.8,0.9,1.0,1.1,1.2,1.3,1.5)))
for lab, lo, hi in (('вся выборка', 2300, n), ('до 2013 (ранний)', 2300, 6000),
                    ('последние 10 лет', LATE, n), ('последние 5 лет', 7800, n)):
    row = []
    for c in (0.8,0.9,1.0,1.1,1.2,1.3,1.5):
        row.append(hull_sharpe(np.full(hi-lo, c), fwd[lo:hi], rf[lo:hi])['score'])
    best = (0.8,0.9,1.0,1.1,1.2,1.3,1.5)[int(np.argmax(row))]
    print(f"  {lab:22s} " + " ".join(f"{v:6.3f}" for v in row) + f"   лучшая: {best}")

print("\n  то же по МЕДИАНЕ непересекающихся окон 180 дней:")
def med_win(pos, lo, hi, width=180):
    return np.median([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                      for s in range(lo, hi-width+1, width)])
print(f"  {'период':22s} " + " ".join(f"{c:>6.1f}" for c in (0.8,0.9,1.0,1.1,1.2,1.3)))
for lab, lo, hi in (('вся выборка', 2300, n), ('до 2013', 2300, 6000), ('последние 10 лет', LATE, n)):
    row = [med_win(np.full(n,c), lo, hi) for c in (0.8,0.9,1.0,1.1,1.2,1.3)]
    best = (0.8,0.9,1.0,1.1,1.2,1.3)[int(np.argmax(row))]
    print(f"  {lab:22s} " + " ".join(f"{v:6.3f}" for v in row) + f"   лучшая: {best}")

print("\n=== B. Бюджет волатильности: метрика разрешает 1.2x бесплатно, мы держим 1.07 ===")
S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
Z = np.column_stack([np.nan_to_num(-zc(S.rolling(h).sum().values)) for h in (1,5,22,63)])
W = np.column_stack([1.0/(pd.Series(Z[:,j]).rolling(252).std().values+1e-9) for j in range(4)])
W = W/np.nansum(W,axis=1,keepdims=True); ens = np.nan_to_num(np.nansum(Z*W,axis=1))

def build(mp, a=0.25, lo_=0.85, hi_=1.45, hl=3):
    return pd.Series(np.clip(mp + a*ens, lo_, hi_)).ewm(halflife=hl).mean().values

def stat(pos, lo, hi, width=180):
    d = np.array([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                  - hull_sharpe(np.ones(width), fwd[s:s+width], rf[s:s+width])['score']
                  for s in range(lo, hi-width+1, width)])
    N=len(d); k=int((d>0).sum())
    return np.median(d), d.mean(), k/N, sum(comb(N,i) for i in range(k,N+1))/2**N, N

print(f"  {'вариант':34s} {'мед':>7s} {'средн':>7s} {'побед':>7s} {'vol_r':>6s}")
for mp, lo_, hi_ in ((0.95,0.85,1.45), (1.05,0.95,1.55), (1.10,1.00,1.60), (1.15,1.05,1.65)):
    pos = build(mp, lo_=lo_, hi_=hi_)
    md, mn, wr, p, N = stat(pos, LATE, n)
    vr = hull_sharpe(np.clip(pos[LATE:],0,2), fwd[LATE:], rf[LATE:])['vol_ratio']
    print(f"  ср.поз≈{mp}, диапазон [{lo_},{hi_}]{'':7s} {md:+7.3f} {mn:+7.3f} {wr*100:6.0f}% {vr:6.2f}")

print("\n  причинная версия: масштаб подбирается так, чтобы прошлая vol_ratio была ~1.15")
base = np.clip(0.95 + 0.25*ens, 0.85, 1.45)
strat_r = rf + base*(fwd-rf)
vr_hist = (pd.Series(strat_r).rolling(500).std() / pd.Series(fwd).rolling(500).std()).shift(1).values
k_adj = np.clip(np.nan_to_num(1.15/vr_hist, nan=1.0), 0.8, 1.4)
pos_adj = pd.Series(np.clip(base*k_adj, 0.8, 1.7)).ewm(halflife=3).mean().values
for lab, lo, hi in (('вся выборка', 2300, n), ('последние 10 лет', LATE, n)):
    md, mn, wr, p, N = stat(pos_adj, lo, hi)
    vr = hull_sharpe(np.clip(pos_adj[lo:],0,2), fwd[lo:], rf[lo:])['vol_ratio']
    print(f"    {lab:20s} мед={md:+.3f} средн={mn:+.3f} побед={wr*100:.0f}% p={p:.3f} "
          f"vol_r={vr:.2f} ср.поз={pos_adj[lo:].mean():.2f}")
