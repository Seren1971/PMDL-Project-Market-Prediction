import pandas as pd, numpy as np, datetime as dt
tr = pd.read_csv('data/raw/train.csv')
f = tr.forward_returns

print("=== огибающая: все дни с |r| > 0.038 ===")
b = tr.loc[f.abs()>0.038, ['date_id','forward_returns']].copy()
b['abs']=b.forward_returns.abs()
print(b.to_string(index=False))

print("\n=== месячные длины из D9 (первые 3 торговых дня месяца) ===")
d9 = tr.D9.values.astype(int)
starts = np.where(np.r_[d9[0]==1, (d9[1:]==1)&(d9[:-1]==0)])[0]
lens = np.diff(starts)
print("число месяцев:", len(starts), "| длины: медиана", int(np.median(lens)), "min", lens.min(), "max", lens.max())
print("первые 24 длины:", lens[:24].tolist())

# генерируем календарь NYSE
def easter(y):
    a=y%19;b=y//100;c=y%100;dd=b//4;e=b%4;fq=(b+8)//25;g_=(b-fq+1)//3
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
hs={dt.date(2001,9,11),dt.date(2001,9,12),dt.date(2001,9,13),dt.date(2001,9,14),
    dt.date(2012,10,29),dt.date(2012,10,30),dt.date(2004,6,11),dt.date(2007,1,2),
    dt.date(2018,12,5),dt.date(1994,4,27),dt.date(2025,1,9)}
for y in range(1984,2027): hs.update(hol(y))
days=[]; d=dt.date(1985,1,1)
while d<=dt.date(2026,6,30):
    if d.weekday()<5 and d not in hs: days.append(d)
    d+=dt.timedelta(days=1)
days=np.array(days)
ym = np.array([(x.year, x.month) for x in days])
key = ym[:,0]*12+ym[:,1]
uk, first_idx, counts = np.unique(key, return_index=True, return_counts=True)
order=np.argsort(first_idx); uk=uk[order]; first_idx=first_idx[order]; counts=counts[order]

L=len(lens)
best=None
for off in range(len(counts)-L-1):
    c=counts[off:off+L]
    m=(c==lens).mean()
    if best is None or m>best[1]: best=(off,m)
off,m=best
y0,m0 = divmod(uk[off]-1,12); m0+=1
print(f"\nлучшее совпадение последовательности длин месяцев: {m*100:.1f}%")
print(f"-> месяц date_id 0 = {y0}-{m0:02d}, первый торговый день = {days[first_idx[off]]}")
last_month = uk[off+L]; ly,lm = divmod(last_month-1,12); lm+=1
print(f"-> последний месяц выборки = {ly}-{lm:02d}")
mis=np.where(counts[off:off+L]!=lens)[0]
print("месяцев с расхождением (скорее всего дефекты моего списка праздников):", len(mis), "первые:", mis[:10].tolist())

cal = days[first_idx[off]: first_idx[off]+len(tr)]
print(f"\nитоговая привязка: date_id 0 = {cal[0]}, date_id {len(tr)-1} = {cal[-1]}")
pd.DataFrame({'date_id':tr.date_id,'date_est':cal}).to_csv('data/interim_calendar_estimate.csv', index=False)
print("сохранено в data/interim_calendar_estimate.csv (ОЦЕНКА, не официальные даты)")

# проверка: кризисы на своих местах?
c=pd.DataFrame({'date_id':tr.date_id,'date':cal,'r':f})
for lbl,d1,d2 in [('крах COVID','2020-02-20','2020-03-31'),('кризис 2008','2008-09-15','2008-10-31'),('пузырь дотком','2000-03-01','2000-04-30')]:
    sub=c[(c.date>=dt.date.fromisoformat(d1))&(c.date<=dt.date.fromisoformat(d2))]
    print(f"{lbl} {d1}..{d2}: n={len(sub)}, cum={100*((1+sub.r).prod()-1):.1f}%, вола={sub.r.std()*np.sqrt(252)*100:.0f}%, max|r|={sub.r.abs().max():.4f}")
