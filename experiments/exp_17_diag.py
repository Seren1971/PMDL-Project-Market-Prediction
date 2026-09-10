"""Диагностика: от чего зависит успех стратегии в окне."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from data import load_train, known_returns
from metric import hull_sharpe

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); S = pd.Series(rk)
cal = pd.read_csv('data/interim_calendar_estimate.csv')
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
Z = np.column_stack([np.nan_to_num(-zc(S.rolling(h).sum().values)) for h in (1,5,22,63)])
W = np.column_stack([1.0/(pd.Series(Z[:,j]).rolling(252).std().values+1e-9) for j in range(4)])
W = W/np.nansum(W,axis=1,keepdims=True); ens = np.nan_to_num(np.nansum(Z*W,axis=1))
CAND = pd.Series(np.clip(0.95+0.25*ens, 0.85, 1.45)).ewm(halflife=3).mean().values

rows = []
for s in range(2300, n-180+1, 30):
    sl = slice(s, s+180)
    a = hull_sharpe(np.clip(CAND[sl],0,2), fwd[sl], rf[sl])
    b = hull_sharpe(np.ones(180), fwd[sl], rf[sl])
    r = fwd[sl]
    rows.append(dict(
        start=s, year=int(cal.loc[cal.date_id==s,'date_est'].values[0][:4]),
        diff=a['score']-b['score'],
        mkt_ret=(1+r).prod()-1, mkt_vol=r.std()*np.sqrt(252),
        # характеристики режима, известные ЗАРАНЕЕ (по данным до начала окна)
        ac1_prior=pd.Series(rk[s-252:s]).autocorr(1),
        vol_prior=rk[s-252:s].std()*np.sqrt(252),
        trend_prior=(1+rk[s-252:s]).prod()-1,
        # характеристики САМОГО окна (для понимания, не для торговли)
        ac1_in=pd.Series(r).autocorr(1),
        updays=(r>0).mean(),
        maxdd=float((pd.Series((1+r).cumprod()) / pd.Series((1+r).cumprod()).cummax() - 1).min()),
    ))
d = pd.DataFrame(rows)
late = d[d.year >= 2015]

print("=== Что коррелирует с успехом стратегии в окне ===")
print(f"{'признак':26s} {'вся выборка':>13s} {'последние 10 лет':>18s}")
for c in ['mkt_ret','mkt_vol','ac1_prior','vol_prior','trend_prior','ac1_in','updays','maxdd']:
    print(f"  {c:24s} {d['diff'].corr(d[c]):+13.3f} {late['diff'].corr(late[c]):+18.3f}")

print("\n=== Ключевое: автокорреляция доходностей ВНУТРИ окна ===")
print("  (разворот работает, когда автокорреляция отрицательна — рынок 'пилит')")
for lab, sub in (('вся выборка', d), ('последние 10 лет', late)):
    q = pd.qcut(sub.ac1_in, 3, labels=['сильно отриц','около нуля','положит'])
    g = sub.groupby(q, observed=True).agg(окон=('diff','size'), медиана=('diff','median'),
                                          среднее=('diff','mean'), побед=('diff', lambda x:(x>0).mean()))
    print(f"\n  {lab}:"); print(g.round(3).to_string())

print("\n=== Изменилась ли сама автокорреляция рынка со временем? ===")
for y0 in range(1995, 2025, 5):
    idx = np.where((cal.date_est.str.slice(0,4).astype(int).values >= y0) &
                   (cal.date_est.str.slice(0,4).astype(int).values < y0+5))[0]
    idx = idx[idx > 1006]
    if len(idx) < 400: continue
    print(f"  {y0}-{y0+4}: автокорр(1) дневных доходностей = {pd.Series(fwd[idx]).autocorr(1):+.4f}")

print("\n=== Можно ли предсказать режим ЗАРАНЕЕ? ===")
print("  корреляция автокорреляции ДО окна с автокорреляцией ВНУТРИ окна:")
print(f"    вся выборка:      {d.ac1_prior.corr(d.ac1_in):+.3f}")
print(f"    последние 10 лет: {late.ac1_prior.corr(late.ac1_in):+.3f}")
print("  -> если близко к нулю, режим заранее не виден и фильтр построить нельзя")
