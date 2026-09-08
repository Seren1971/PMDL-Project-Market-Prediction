import pandas as pd, numpy as np
tr = pd.read_csv('data/raw/train.csv')
D=[f'D{i}' for i in range(1,10)]
print("=== уникальные значения D ===")
for c in D: print(c, sorted(tr[c].unique()))
print("\nD1 идентична D2:", bool((tr.D1==tr.D2).all()))

r = tr.forward_returns
print("\n=== экстремумы forward_returns (дневная доходность t->t+1) ===")
big = tr.loc[r.abs().sort_values(ascending=False).index[:15], ['date_id','forward_returns']]
print(big.to_string(index=False))
print("\nmin =", r.min(), " max =", r.max())

print("\n=== годовая статистика по блокам в 252 дня (первые/последние) ===")
tr['yr'] = tr.date_id//252
g = tr.groupby('yr').forward_returns.agg(['size','mean','std'])
g['ann_ret_%'] = ((1+tr.groupby('yr').forward_returns.apply(lambda s:(1+s).prod()))**0-0)  # placeholder
g['cum_%'] = tr.groupby('yr').forward_returns.apply(lambda s: ((1+s).prod()-1)*100)
g['vol_%'] = g['std']*np.sqrt(252)*100
print(g[['size','cum_%','vol_%']].round(2).to_string())

print("\n=== реконструкция календаря: сколько торговых дней между датами ===")
import datetime as dt
def easter(y):
    a=y%19;b=y//100;c=y%100;d=b//4;e=b%4;f=(b+8)//25;g_=(b-f+1)//3
    h=(19*a+b-d-g_+15)%30;i=c//4;k=c%4;l=(32+2*e+2*i-h-k)%7;m=(a+11*h+22*l)//451
    mo=(h+l-7*m+114)//31; da=((h+l-7*m+114)%31)+1
    return dt.date(y,mo,da)
def nth_wd(y,m,wd,n):  # n-й weekday месяца
    d=dt.date(y,m,1)
    off=(wd-d.weekday())%7
    return d+dt.timedelta(days=off+7*(n-1))
def last_wd(y,m,wd):
    d=dt.date(y,m+1,1)-dt.timedelta(days=1) if m<12 else dt.date(y,12,31)
    return d-dt.timedelta(days=(d.weekday()-wd)%7)
def observed(d):
    if d.weekday()==5: return d-dt.timedelta(days=1)
    if d.weekday()==6: return d+dt.timedelta(days=1)
    return d
def holidays(y):
    h=[observed(dt.date(y,1,1)), observed(dt.date(y,7,4)), observed(dt.date(y,12,25)),
       nth_wd(y,2,0,3), last_wd(y,5,0), nth_wd(y,9,0,1), nth_wd(y,11,3,4),
       easter(y)-dt.timedelta(days=2)]
    if y>=1998: h.append(nth_wd(y,1,0,3))            # MLK
    if y>=2022: h.append(observed(dt.date(y,6,19)))  # Juneteenth
    return h
special=[dt.date(2001,9,11),dt.date(2001,9,12),dt.date(2001,9,13),dt.date(2001,9,14),
         dt.date(2012,10,29),dt.date(2012,10,30),dt.date(2004,6,11),dt.date(2007,1,2),
         dt.date(2018,12,5),dt.date(1994,4,27),dt.date(2025,1,9)]
def trading_days(a,b):
    hs=set(special)
    for y in range(a.year,b.year+1): hs.update(holidays(y))
    days=[]
    d=a
    while d<=b:
        if d.weekday()<5 and d not in hs: days.append(d)
        d+=dt.timedelta(days=1)
    return days

for start in ['1989-12-01','1990-01-02','1990-02-01','1990-03-01']:
    a=dt.date.fromisoformat(start); b=dt.date(2025,12,8)
    print(f"{start} .. 2025-12-08 -> {len(trading_days(a,b))} торговых дней (нужно 9048 для date_id 0..9047)")
