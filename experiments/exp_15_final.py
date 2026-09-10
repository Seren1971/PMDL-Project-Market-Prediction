"""Финальный кандидат второго круга и его честная оценка."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns
from metric import hull_sharpe

rng = np.random.default_rng(7)
tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
Z = np.column_stack([np.nan_to_num(-zc(S.rolling(h).sum().values)) for h in (1,5,22,63)])
W = np.column_stack([1.0/(pd.Series(Z[:,j]).rolling(252).std().values+1e-9) for j in range(4)])
W = W/np.nansum(W,axis=1,keepdims=True)
ens = np.nan_to_num(np.nansum(Z*W, axis=1))

CAND = pd.Series(np.clip(0.95 + 0.25*ens, 0.85, 1.45)).ewm(halflife=3).mean().values
BASE = np.full(n, 1.0); LATE = 6500

def wins(pos, lo, hi, width, step=None):
    step = step or width
    return np.array([hull_sharpe(np.clip(np.nan_to_num(pos[s:s+width],nan=1.0),0,2),
                                 fwd[s:s+width], rf[s:s+width])['score']
                     for s in range(lo, hi-width+1, step)])

print("=== кандидат: ансамбль разворотов 1/5/22/63 (inverse-vol веса),")
print("    линейно, пол 0.85 потолок 1.45, EMA(3) ===\n")
for lab, lo, hi in (('вся выборка', 2300, n), ('первая половина', 2300, 6000),
                    ('вторая половина', 6000, n), ('последние 10 лет', LATE, n)):
    for width in (130, 180):
        a = wins(CAND, lo, hi, width); b = wins(BASE, lo, hi, width)
        d = a-b; N=len(d); k=int((d>0).sum())
        p = sum(comb(N,i) for i in range(k,N+1))/2**N
        print(f"  {lab:18s} окно {width}: побед {k:2d}/{N:2d} ({k/N*100:3.0f}%) "
              f"мед.разн={np.median(d):+.3f} p={p:.4f}")

print("\n=== поправка на СУММАРНОЕ число попыток за оба круга ===")
TRIALS = 110
a = wins(CAND, 2300, n, 180); b = wins(BASE, 2300, n, 180)
d = a-b; N=len(d); k=int((d>0).sum()); p = sum(comb(N,i) for i in range(k,N+1))/2**N
print(f"  вся выборка: сырой p={p:.6f}, p_adj (x{TRIALS}) = {min(1,p*TRIALS):.4f} "
      f"-> {'ПРОЙДЕН' if p*TRIALS<0.05 else 'не пройден'}")
a2 = wins(CAND, LATE, n, 180); b2 = wins(BASE, LATE, n, 180)
d2 = a2-b2; N2=len(d2); k2=int((d2>0).sum()); p2 = sum(comb(N2,i) for i in range(k2,N2+1))/2**N2
print(f"  последние 10 лет: сырой p={p2:.4f}, p_adj = {min(1,p2*TRIALS):.4f} "
      f"-> {'ПРОЙДЕН' if p2*TRIALS<0.05 else 'не пройден'}")

print("\n=== блочный бутстрап ===")
for lo, lab in ((2300,'вся выборка'), (6000,'вторая половина'), (LATE,'последние 10 лет')):
    BL=252; nb=(n-lo)//BL; blocks=[(lo+i*BL, lo+(i+1)*BL) for i in range(nb)]
    dd=[]
    for _ in range(3000):
        idx = rng.integers(0, nb, nb)
        sel = np.concatenate([np.arange(*blocks[i]) for i in idx])
        try:
            dd.append(hull_sharpe(np.clip(CAND[sel],0,2), fwd[sel], rf[sel])['score']
                      - hull_sharpe(np.ones(len(sel)), fwd[sel], rf[sel])['score'])
        except ValueError: pass
    dd=np.array(dd)
    print(f"  {lab:18s} среднее={dd.mean():+.3f} 95%ДИ=[{np.percentile(dd,2.5):+.3f}, "
          f"{np.percentile(dd,97.5):+.3f}] P(>0)={(dd>0).mean()*100:4.1f}% (блоков {nb})")

print("\n=== характеристики позиции и риск нарушить лимит волатильности ===")
for lo, lab in ((2300,'вся выборка'), (LATE,'последние 10 лет')):
    r = hull_sharpe(np.clip(CAND[lo:],0,2), fwd[lo:], rf[lo:])
    c = hull_sharpe(np.ones(n-lo), fwd[lo:], rf[lo:])
    vr = [hull_sharpe(np.clip(CAND[s:s+180],0,2), fwd[s:s+180], rf[s:s+180])['vol_ratio']
          for s in range(lo, n-180+1, 180)]
    print(f"  {lab:18s} score={r['score']:.4f} (константа {c['score']:.4f}) "
          f"vol_ratio={r['vol_ratio']:.2f} | по окнам: доля vol_ratio>1.2: "
          f"{np.mean(np.array(vr)>1.2)*100:.0f}%, макс {max(vr):.2f}")
print(f"  средняя позиция={CAND[2300:].mean():.3f} std={CAND[2300:].std():.3f} "
      f"оборот={np.abs(np.diff(CAND[2300:])).mean():.4f}")
print(f"  сравнение с кандидатом 1-го круга (только разворот 5д): см. reports/exp_12_h14_final.txt")
