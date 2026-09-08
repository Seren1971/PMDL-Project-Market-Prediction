import pandas as pd, numpy as np
tr = pd.read_csv('data/raw/train.csv')
D = [f'D{i}' for i in range(1,10)]
print("=== D-фичи: доля единиц и структура ===")
for c in D:
    s = tr[c].astype(int).values
    # длины серий из единиц
    runs=[]; cur=0
    for v in s:
        if v==1: cur+=1
        elif cur: runs.append(cur); cur=0
    if cur: runs.append(cur)
    gaps=[]; cur=0
    for v in s:
        if v==0: cur+=1
        elif cur: gaps.append(cur); cur=0
    print(f"{c}: доля 1 = {s.mean():.4f} ({s.sum():5d} дней), серий единиц={len(runs):5d}, "
          f"медиана длины серии={np.median(runs) if runs else 0:.0f}, медиана паузы={np.median(gaps) if gaps else 0:.0f}")

print("\n=== попарные пересечения (доля совместных единиц / доля 1 у строки) ===")
M = tr[D].astype(int)
print((M.T @ M / M.sum().values[:,None]).round(3).to_string())

print("\n=== первые 60 дней, паттерн D1..D9 ===")
print(tr[['date_id']+D].head(60).astype(int).to_string(index=False))

print("\n=== средний forward_returns при D=1 vs D=0 (годовых, %) ===")
for c in D:
    a = tr.loc[tr[c]==1,'forward_returns'].mean()*252*100
    b = tr.loc[tr[c]==0,'forward_returns'].mean()*252*100
    print(f"{c}: D=1 -> {a:7.2f}%   D=0 -> {b:7.2f}%   разница {a-b:7.2f} п.п.")
