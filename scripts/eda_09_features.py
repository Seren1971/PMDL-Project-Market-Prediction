import pandas as pd, numpy as np, re
tr = pd.read_csv('data/raw/train.csv')
feat=[c for c in tr.columns if re.match(r'^[A-Z]+\d+$',c)]
rec = tr[tr.date_id>=7000].reset_index(drop=True)   # период, где все 94 фичи полны

print("=== масштаб фич по группам (на date_id>=7000, все фичи полные) ===")
d=rec[feat].describe().T
d['grp']=[re.match(r'^([A-Z]+)',c).group(1) for c in d.index]
print(d.groupby('grp')[['mean','std','min','max']].agg(['median']).round(3).to_string())

print("\n=== сколько фич выглядят уже стандартизованными / ограниченными ===")
z = d[(d['mean'].abs()<0.3)&(d['std'].between(0.5,2.0))]
b01 = d[(d['min']>=-0.001)&(d['max']<=1.001)]
print(f"похожи на z-score (|mean|<0.3, std 0.5..2): {len(z)}/94")
print(f"лежат в [0,1]: {len(b01)}/94 -> {sorted(b01.index.tolist())}")
print(f"диапазон std по всем фичам: {d['std'].min():.4f} .. {d['std'].max():.4f}")
print("фичи с самым большим std:", d['std'].sort_values(ascending=False).head(5).round(2).to_dict())

print("\n=== дубликаты и почти-дубликаты (|corr|>0.98 на date_id>=7000) ===")
C = rec[feat].corr()
seen=set(); pairs=[]
for i,a in enumerate(feat):
    for b in feat[i+1:]:
        c=C.loc[a,b]
        if abs(c)>0.98: pairs.append((a,b,round(c,4)))
for a,b,c in pairs: print(f"  {a:5s} ~ {b:5s}  corr={c}")
print("всего пар:", len(pairs))

print("\n=== предсказательная сила: корреляция фичи(t) с таргетом(t) ===")
y = rec.market_forward_excess_returns
ic=[]
for c in feat:
    x=rec[c]
    ic.append((c, x.corr(y), x.corr(y,method='spearman')))
ic=pd.DataFrame(ic,columns=['col','pearson','spearman']).dropna()
ic['t_stat']=ic.pearson*np.sqrt(len(rec)-2)/np.sqrt(1-ic.pearson**2)
ic=ic.reindex(ic.pearson.abs().sort_values(ascending=False).index)
print("топ-20 по |pearson|:")
print(ic.head(20).round(4).to_string(index=False))
print(f"\nмедиана |pearson| по всем 94 фичам: {ic.pearson.abs().median():.4f}")
print(f"фич с |t|>2: {int((ic.t_stat.abs()>2).sum())}/94   с |t|>3: {int((ic.t_stat.abs()>3).sum())}/94")
print("для справки: при n=2048 корреляция 0.044 = |t|=2. Чистый шум дал бы ~5 фич с |t|>2")

print("\n=== стабильность знака IC: первая половина vs вторая (date_id>=7000) ===")
h=len(rec)//2
a=rec.iloc[:h]; b=rec.iloc[h:]
st=[]
for c in feat:
    ca=a[c].corr(a.market_forward_excess_returns); cb=b[c].corr(b.market_forward_excess_returns)
    st.append((c,ca,cb,np.sign(ca)==np.sign(cb)))
st=pd.DataFrame(st,columns=['col','ic_1h','ic_2h','same_sign'])
print(f"знак IC совпал в обеих половинах: {int(st.same_sign.sum())}/94")
print("корреляция между IC первой и второй половины:", round(st.ic_1h.corr(st.ic_2h),3))
