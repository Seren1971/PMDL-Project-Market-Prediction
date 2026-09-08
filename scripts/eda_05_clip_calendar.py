import pandas as pd, numpy as np, datetime as dt
tr = pd.read_csv('data/raw/train.csv')
r = tr.forward_returns

print("=== обрезаны ли forward_returns? ===")
print(f"min={r.min():.6f}  max={r.max():.6f}  |r| p99.9={r.abs().quantile(0.999):.6f}")
for thr in [0.035,0.038,0.039,0.0395]:
    print(f"  |r| > {thr}: {int((r.abs()>thr).sum()):4d} дней ({100*(r.abs()>thr).mean():.2f}%)")
print("\nреальный S&P500 за 1990-2025 имел дни -12.0% (16.03.2020), -9.0% (15.10.2008), +11.6% (13.10.2008)")
print("здесь таких дней нет -> данные винзоризованы\n")

# профиль порога во времени: максимум |r| в скользящем окне
w = r.abs().rolling(252).max()
print("максимум |forward_returns| по 252-дневным окнам (каждое 1000-е наблюдение):")
print(pd.DataFrame({'date_id':tr.date_id,'roll_max_abs':w}).dropna().iloc[::1000].round(5).to_string(index=False))

print("\n=== кластеризация возле границы: даты 2008 и 2020 кризисов ===")
for lo,hi,name in [(4650,4700,'осень 2008'),(7590,7640,'март 2020')]:
    sub=tr[(tr.date_id>=lo)&(tr.date_id<=hi)]
    print(f"{name}: n={len(sub)}, |r|>0.039: {int((sub.forward_returns.abs()>0.039).sum())}, "
          f"min={sub.forward_returns.min():.4f}, max={sub.forward_returns.max():.4f}")

print("\n=== таргет vs forward_returns ===")
d = tr.forward_returns - tr.market_forward_excess_returns
print("forward_returns - market_forward_excess_returns: описание скользящего среднего 5 лет")
print(pd.DataFrame({'date_id':tr.date_id,'diff':d}).iloc[::1000].round(6).to_string(index=False))
print("годовых, %:", (d.mean()*252*100).round(2), "| std разницы:", d.std().round(6))
print("корреляция таргета с forward_returns:", round(tr.market_forward_excess_returns.corr(tr.forward_returns),6))

print("\n=== привязка календаря по D9 (первые торговые дни месяца) ===")
def easter(y):
    a=y%19;b=y//100;c=y%100;dd=b//4;e=b%4;f=(b+8)//25;g_=(b-f+1)//3
    h=(19*a+b-dd-g_+15)%30;i=c//4;k=c%4;l=(32+2*e+2*i-h-k)%7;m=(a+11*h+22*l)//451
    return dt.date(y,(h+l-7*m+114)//31,((h+l-7*m+114)%31)+1)
def nth_wd(y,m,wd,n):
    d=dt.date(y,m,1); return d+dt.timedelta(days=(wd-d.weekday())%7+7*(n-1))
def last_wd(y,m,wd):
    d=dt.date(y,m+1,1)-dt.timedelta(days=1); return d-dt.timedelta(days=(d.weekday()-wd)%7)
def obs(d):
    return d-dt.timedelta(days=1) if d.weekday()==5 else (d+dt.timedelta(days=1) if d.weekday()==6 else d)
def hol(y):
    h=[obs(dt.date(y,1,1)),obs(dt.date(y,7,4)),obs(dt.date(y,12,25)),nth_wd(y,2,0,3),
       last_wd(y,5,0),nth_wd(y,9,0,1),nth_wd(y,11,3,4),easter(y)-dt.timedelta(days=2)]
    if y>=1998: h.append(nth_wd(y,1,0,3))
    if y>=2022: h.append(obs(dt.date(y,6,19)))
    return h
special={dt.date(2001,9,11),dt.date(2001,9,12),dt.date(2001,9,13),dt.date(2001,9,14),
         dt.date(2012,10,29),dt.date(2012,10,30),dt.date(2004,6,11),dt.date(2007,1,2),
         dt.date(2018,12,5),dt.date(1994,4,27),dt.date(2025,1,9)}
hs=set(special)
for y in range(1989,2027): hs.update(hol(y))
days=[]; d=dt.date(1989,12,1)
while d<=dt.date(2026,1,31):
    if d.weekday()<5 and d not in hs: days.append(d)
    d+=dt.timedelta(days=1)
days=np.array(days)

best=None
for off in range(0,60):
    cal=days[off:off+len(tr)]
    if len(cal)<len(tr): break
    mon=np.array([x.month for x in cal])
    isfirst3=np.zeros(len(cal),bool)
    newm=np.r_[True, mon[1:]!=mon[:-1]]
    idx=np.where(newm)[0]
    for i in idx: isfirst3[i:i+3]=True
    acc=(isfirst3==tr.D9.values.astype(bool)).mean()
    if best is None or acc>best[1]: best=(off,acc,cal[0],cal[-1])
off,acc,d0,dl=best
print(f"лучшая привязка: date_id 0 = {d0}, date_id {len(tr)-1} = {dl}, совпадение с D9 = {acc*100:.2f}%")
cal=days[off:off+len(tr)]
mon=np.array([x.month for x in cal]); newm=np.r_[True, mon[1:]!=mon[:-1]]
isfirst3=np.zeros(len(cal),bool)
for i in np.where(newm)[0]: isfirst3[i:i+3]=True
bad=np.where(isfirst3!=tr.D9.values.astype(bool))[0]
print("первое расхождение на date_id:", bad[0] if len(bad) else "нет", "| всего расхождений:", len(bad))
np.save('/private/tmp/claude-501/-Users-anton-Codes-Marker-Prediction/6b45259a-2061-4e1c-9b3f-5d9db9b3a316/scratchpad/cal.npy', cal)
