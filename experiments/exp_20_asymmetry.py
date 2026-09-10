"""Частота против величины: что вообще надо максимизировать в этом соревновании."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from data import load_train, known_returns
from metric import hull_sharpe

tr = load_train(); fwd=tr.forward_returns.values; rf=tr.risk_free_rate.values
rk=known_returns(fwd); n=len(fwd); S=pd.Series(rk); LATE=6500
cal=pd.read_csv('data/interim_calendar_estimate.csv')
def zc(x,w=252):
    x=pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
Z=np.column_stack([np.nan_to_num(-zc(S.rolling(h).sum().values)) for h in (1,5,22,63)])
W=np.column_stack([1.0/(pd.Series(Z[:,j]).rolling(252).std().values+1e-9) for j in range(4)])
W=W/np.nansum(W,axis=1,keepdims=True); ens=np.nan_to_num(np.nansum(Z*W,axis=1))
zz=ens-np.nan_to_num(pd.Series(ens).rolling(504,min_periods=250).mean().shift(1).values)
zt=np.where(np.abs(zz)>1.0, zz, 0.0)
CAND=pd.Series(np.clip(1.02+0.30*zt, 0.85, 1.5)).ewm(halflife=3).mean().values

print("=== разложение выигрышей и проигрышей, поздний период, окна 130 и 180 ===")
for width in (130, 180):
    d=[]; st=[]
    for s in range(LATE, n-width+1, width):
        a=hull_sharpe(np.clip(CAND[s:s+width],0,2),fwd[s:s+width],rf[s:s+width])['score']
        b=hull_sharpe(np.ones(width),fwd[s:s+width],rf[s:s+width])['score']
        d.append(a-b); st.append(s)
    d=np.array(d); w=d[d>0]; l=d[d<=0]
    print(f"  окно {width}: побед {len(w)}/{len(d)}, средний выигрыш {w.mean():+.3f}, "
          f"средний проигрыш {l.mean():+.3f}, итог среднее {d.mean():+.3f}")
    for i in np.argsort(d)[:3]:
        print(f"    худшее: {cal.loc[cal.date_id==st[i],'date_est'].values[0]} .. "
              f"{cal.loc[cal.date_id==st[i]+width-1,'date_est'].values[0]}  {d[i]:+.3f}")

print("\n=== вклад каждого года в суммарную разницу (поздний период) ===")
years = cal.date_est.str.slice(0,4).astype(int).values
rows=[]
for y in range(2015, 2026):
    idx=np.where(years==y)[0]; idx=idx[idx>=LATE]
    if len(idx)<150: continue
    a=hull_sharpe(np.clip(CAND[idx],0,2),fwd[idx],rf[idx])['score']
    b=hull_sharpe(np.ones(len(idx)),fwd[idx],rf[idx])['score']
    mkt=((1+fwd[idx]).prod()-1)*100
    rows.append((y,len(idx),a,b,a-b,mkt))
print(f"  {'год':5s} {'дней':>5s} {'стратегия':>10s} {'константа':>10s} {'разница':>9s} {'рынок %':>9s}")
for r in rows:
    print(f"  {r[0]:5d} {r[1]:5d} {r[2]:10.3f} {r[3]:10.3f} {r[4]:+9.3f} {r[5]:+9.1f}")

print("\n=== что важнее для СОРЕВНОВАНИЯ: частота или величина? ===")
print("  зачётное окно одно. Толпа участников кучкуется около пассивного бенчмарка")
print("  (в реальном лидерборде скор 2.866 повторился у 4 команд, 2.983 у 2).")
print("  Значит осмысленная цель — максимизировать ВЕРОЯТНОСТЬ обойти бенчмарк,")
print("  а не среднюю величину превосходства.\n")
d130=[]
for s in range(LATE, n-130+1, 130):
    a=hull_sharpe(np.clip(CAND[s:s+130],0,2),fwd[s:s+130],rf[s:s+130])['score']
    b=hull_sharpe(np.ones(130),fwd[s:s+130],rf[s:s+130])['score']
    d130.append(a-b)
d130=np.array(d130)
print(f"  P(обойти константу) на позднем периоде, окно 130: {(d130>0).mean()*100:.0f}%")
print(f"  E[разница]:                                        {d130.mean():+.3f}")
print(f"  медиана разницы:                                   {np.median(d130):+.3f}")
print(f"  худший случай:                                     {d130.min():+.3f}")
print("\n  Это лотерейный билет наоборот: почти всегда небольшой плюс,")
print("  изредка заметный минус. Для ранга в лидерборде такой профиль скорее хорош,")
print("  для реальных денег — вопрос терпимости к редкому провалу.")
