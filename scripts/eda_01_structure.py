import pandas as pd, numpy as np, re, collections

tr = pd.read_csv('data/raw/train.csv')
te = pd.read_csv('data/raw/test.csv')

print("train:", tr.shape, "| test:", te.shape)
print("train date_id:", tr.date_id.min(), "->", tr.date_id.max(), "| монотонно+шаг1:",
      bool((tr.date_id.diff().dropna()==1).all()))
print("test  date_id:", te.date_id.min(), "->", te.date_id.max())
print("дубликаты date_id в train:", tr.date_id.duplicated().sum())

def grp(c):
    m = re.match(r'^([A-Z]+)\d+$', c)
    return m.group(1) if m else None

feat = [c for c in tr.columns if grp(c)]
groups = collections.OrderedDict()
for c in feat:
    groups.setdefault(grp(c), []).append(c)

print("\n=== группы фич ===")
tot=0
for g, cols in groups.items():
    nums = sorted(int(re.match(r'^[A-Z]+(\d+)$', c).group(1)) for c in cols)
    gaps = [n for n in range(1, max(nums)+1) if n not in nums]
    print(f"{g:4s} n={len(cols):3d}  {g}1..{g}{max(nums)}  пропущенные номера: {gaps if gaps else 'нет'}")
    tot += len(cols)
print("итого фич:", tot)
print("нефичевые колонки train:", [c for c in tr.columns if c not in feat])
print("нефичевые колонки test :", [c for c in te.columns if c not in feat])
print("есть ли MOM* :", [c for c in tr.columns if c.upper().startswith('MOM')] or "НЕТ")
print("\nколонки train не в test:", [c for c in tr.columns if c not in te.columns])
print("колонки test не в train:", [c for c in te.columns if c not in tr.columns])
print("порядок фич совпадает:", [c for c in tr.columns if c in feat] == [c for c in te.columns if c in feat])
