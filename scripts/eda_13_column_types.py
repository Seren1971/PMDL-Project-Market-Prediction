import pandas as pd, numpy as np, re

tr = pd.read_csv('data/raw/train.csv')
feats = [c for c in tr.columns if re.fullmatch(r'[A-Z]+\d+', c)]

def plateau(s):
    """медианная длина участка, на котором значение не меняется (в торговых днях)"""
    s = s.dropna()
    if len(s) < 10: return np.nan
    grp = (s != s.shift()).cumsum()
    return s.groupby(grp).size().median()

def classify(s, nu, lo, hi):
    vals = set(pd.unique(s.dropna()))
    if nu == 2 and vals <= {0, 1}:      return 'бинарный 0/1'
    if nu == 2 and vals <= {-1, 0}:     return 'бинарный -1/0'
    if nu <= 12:                        return f'категориальный ({nu} уровня)'
    if lo >= -0.001 and hi <= 1.001:    return 'непрерывный [0,1]'
    return 'непрерывный, без границ'

rows = []
for c in feats:
    s = tr[c]; d = s.dropna()
    nu = d.nunique()
    lo, hi = d.min(), d.max()
    rows.append(dict(
        col=c, grp=re.match(r'[A-Z]+', c).group(),
        тип=classify(s, nu, lo, hi), уник=nu,
        min=round(lo, 3), max=round(hi, 3), std=round(d.std(), 3),
        плато=plateau(s), старт=int(s.first_valid_index()),
        пропуск_pct=round(s.isna().mean()*100, 1)))

t = pd.DataFrame(rows)
def freq(p):
    if pd.isna(p): return '?'
    if p <= 1.0:  return 'каждый день'
    if p <= 3:    return 'неск. раз в неделю'
    if p <= 8:    return 'раз в неделю'
    if p <= 25:   return 'раз в месяц'
    return 'раз в квартал+'
t['обновление'] = t['плато'].map(freq)

pd.set_option('display.width', 200, 'display.max_rows', 200)
print("=== ВСЕ 94 ПРИЗНАКА ===")
print(t[['col','grp','тип','уник','min','max','плато','обновление','старт','пропуск_pct']].to_string(index=False))

print("\n=== сводка: сколько признаков какого типа ===")
print(t['тип'].value_counts().to_string())
print("\n=== сводка: частота обновления ===")
print(t['обновление'].value_counts().to_string())
print("\n=== типы по группам ===")
print(pd.crosstab(t.grp, t['тип']).to_string())

print("\n=== есть ли one-hot: группы колонок, где сумма всегда ровно 1 ===")
D = [c for c in feats if c.startswith('D')]
import itertools
found = False
for r in range(2, 6):
    for combo in itertools.combinations(D, r):
        s = tr[list(combo)].abs().sum(axis=1)
        if (s == 1).all():
            print("  one-hot:", combo); found = True
if not found:
    print("  one-hot-групп нет — D-флажки независимы и могут гореть одновременно")
    print("  сколько D-флажков активно одновременно:",
          tr[D].abs().sum(axis=1).value_counts().sort_index().to_dict())

print("\n=== целевые колонки ===")
for c in ['date_id','forward_returns','risk_free_rate','market_forward_excess_returns']:
    d = tr[c]
    print(f"  {c:32s} {str(d.dtype):8s} уник={d.nunique():6d} "
          f"min={d.min():.6f} max={d.max():.6f}")
t.to_csv('reports/column_types.csv', index=False)
