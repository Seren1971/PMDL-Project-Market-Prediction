"""Итог третьего круга: бутстрап на уровне окон — то, как реально считается соревнование."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns
from metric import hull_sharpe

rng = np.random.default_rng(33)
tr = load_train(); fwd=tr.forward_returns.values; rf=tr.risk_free_rate.values
rk=known_returns(fwd); n=len(fwd); S=pd.Series(rk); LATE=6500
def zc(x,w=252):
    x=pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
Z=np.column_stack([np.nan_to_num(-zc(S.rolling(h).sum().values)) for h in (1,5,22,63)])
W=np.column_stack([1.0/(pd.Series(Z[:,j]).rolling(252).std().values+1e-9) for j in range(4)])
W=W/np.nansum(W,axis=1,keepdims=True); ens=np.nan_to_num(np.nansum(Z*W,axis=1))
zz=ens-np.nan_to_num(pd.Series(ens).rolling(504,min_periods=250).mean().shift(1).values)

C2 = pd.Series(np.clip(0.95+0.25*ens, 0.85, 1.45)).ewm(halflife=3).mean().values          # 2-й круг
C3 = pd.Series(np.clip(1.02+0.30*np.where(np.abs(zz)>1.0, zz, 0.0), 0.85, 1.5)).ewm(halflife=3).mean().values

print("=== бутстрап НА УРОВНЕ ОКОН: случайно выбираем одно окно, как в соревновании ===")
print("(зачётное окно одно — поэтому нас интересует распределение исхода в одном окне)\n")
for name, pos in (('кандидат 2-го круга', C2), ('кандидат 3-го круга (порог)', C3)):
    for lab, lo in (('вся выборка', 2300), ('последние 10 лет', LATE)):
        for width in (130,):
            d = np.array([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                          - hull_sharpe(np.ones(width), fwd[s:s+width], rf[s:s+width])['score']
                          for s in range(lo, n-width+1, 10)])   # плотная сетка стартов
            bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(3000)]
            bs = np.array(bs)
            print(f"  {name:28s} {lab:18s} P(обойти)={np.mean(d>0)*100:4.0f}%  "
                  f"E[разн]={d.mean():+.3f}  95%ДИ E=[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  "
                  f"худшее={d.min():+.3f}")

print("\n=== поправка на суммарный перебор за три круга (~140 конфигураций) ===")
for name, pos in (('кандидат 2-го круга', C2), ('кандидат 3-го круга', C3)):
    for lab, lo in (('вся выборка', 2300), ('последние 10 лет', LATE)):
        for width in (130, 180):
            d = np.array([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                          - hull_sharpe(np.ones(width), fwd[s:s+width], rf[s:s+width])['score']
                          for s in range(lo, n-width+1, width)])
            N=len(d); k=int((d>0).sum()); p=sum(comb(N,i) for i in range(k,N+1))/2**N
            pa = min(1, p*140)
            print(f"  {name:22s} {lab:18s} окно {width}: {k}/{N} p={p:.4f} p_adj={pa:.4f} "
                  f"{'ПРОЙДЕН' if pa<0.05 else ''}")

print("\n=== сравнение двух кандидатов лоб в лоб на позднем периоде ===")
for width in (130, 180):
    a = np.array([hull_sharpe(np.clip(C3[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                  for s in range(LATE, n-width+1, width)])
    b = np.array([hull_sharpe(np.clip(C2[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                  for s in range(LATE, n-width+1, width)])
    print(f"  окно {width}: 3-й круг лучше 2-го в {(a>b).mean()*100:.0f}% окон, "
          f"медиана разницы {np.median(a-b):+.3f}")

r = hull_sharpe(np.clip(C3[LATE:],0,2), fwd[LATE:], rf[LATE:])
print(f"\n=== характеристики кандидата 3-го круга (поздний период) ===")
print(f"  score={r['score']:.4f} sharpe={r['sharpe']:.3f} vol_ratio={r['vol_ratio']:.2f} "
      f"ret_gap={r['return_gap']:.2f}")
print(f"  ср.поз={C3[LATE:].mean():.3f} std={C3[LATE:].std():.3f} "
      f"оборот={np.abs(np.diff(C3[LATE:])).mean():.4f}")
print(f"  доля дней позиция в [0.99,1.05] (почти пассивно): {np.mean((C3[LATE:]>=0.99)&(C3[LATE:]<=1.05))*100:.0f}%")
