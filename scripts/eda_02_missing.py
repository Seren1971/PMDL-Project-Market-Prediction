import pandas as pd, numpy as np, re
pd.set_option('display.width', 200)

tr = pd.read_csv('data/raw/train.csv')
N = len(tr)
feat = [c for c in tr.columns if re.match(r'^[A-Z]+\d+$', c)]

rows=[]
for c in feat:
    s = tr[c]
    fv = s.first_valid_index()
    lv = s.last_valid_index()
    inner = s.loc[fv:lv]
    rows.append(dict(col=c, grp=re.match(r'^([A-Z]+)',c).group(1),
                     first=int(tr.date_id[fv]), last=int(tr.date_id[lv]),
                     n_valid=int(s.notna().sum()),
                     pct_miss_all=round(100*s.isna().mean(),1),
                     pct_miss_inner=round(100*inner.isna().mean(),2),
                     nuniq=int(s.nunique())))
m = pd.DataFrame(rows).sort_values(['first','col'])

print("=== с какого date_id фича начинает существовать (по возрастанию) ===")
print(m.to_string(index=False))

print("\n=== сводка по группам ===")
print(m.groupby('grp').agg(n=('col','size'), first_min=('first','min'), first_max=('first','max'),
                           miss_all_med=('pct_miss_all','median')).to_string())

print("\n=== сколько фич доступно на разных отрезках ===")
for start in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 8500]:
    sub = tr[tr.date_id>=start]
    full = [c for c in feat if sub[c].notna().all()]
    any_ = [c for c in feat if sub[c].notna().any()]
    print(f"date_id >= {start:5d} (n={len(sub):5d}): без единого пропуска {len(full):3d}/94, хоть где-то есть {len(any_):3d}/94")

print("\n=== пропуски ВНУТРИ истории фичи (дыры после первого значения) ===")
holes = m[m.pct_miss_inner>0][['col','first','last','pct_miss_inner']]
print(holes.to_string(index=False) if len(holes) else "дыр нет — все фичи непрерывны после старта")

print("\n=== хвост: есть ли пропуски в последних строках ===")
tail = tr.tail(5)[['date_id']+feat]
print("пропусков в последних 5 строках по фичам:", int(tail[feat].isna().sum().sum()))
print("\nтаргеты — пропуски:", tr[['forward_returns','risk_free_rate','market_forward_excess_returns']].isna().sum().to_dict())
print("хвост таргетов:")
print(tr[['date_id','forward_returns','risk_free_rate','market_forward_excess_returns']].tail(12).to_string(index=False))
