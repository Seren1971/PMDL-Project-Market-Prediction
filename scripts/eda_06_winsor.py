import pandas as pd, numpy as np, datetime as dt
tr = pd.read_csv('data/raw/train.csv')
f = tr.forward_returns.values; y = tr.market_forward_excess_returns.values
diff = f - y
print("=== diff = forward_returns - target (должно быть скользящее 5-летнее среднее) ===")
print(f"min={diff.min():.6f} max={diff.max():.6f} std={diff.std():.6f}")
print("самые большие |diff| (там, где винзоризация могла сработать):")
i = np.argsort(-np.abs(diff))[:10]
print(pd.DataFrame({'date_id':tr.date_id.values[i],'forward':f[i],'target':y[i],'diff':diff[i]}).to_string(index=False))

print("\n=== проверка гипотезы: cap = 4 * MAD скользящего окна ===")
s = pd.Series(f)
for win in [1260, 252]:
    med = s.rolling(win).median()
    mad = (s - med).abs().rolling(win).median()
    cap = 4*mad
    hit = (s.abs() >= cap*0.98) & cap.notna()
    print(f"окно {win}: дней с |r| >= 0.98*4MAD: {int(hit.sum())} ({100*hit.mean():.2f}%), "
          f"медиана cap={cap.median():.4f}, cap на конце={cap.iloc[-1]:.4f}")

print("\nмаксимум |forward_returns| в скользящем окне 252 дня, по годам (block=252):")
tr['blk']=tr.date_id//252
print(tr.groupby('blk').forward_returns.agg(maxabs=lambda x: x.abs().max(), vol=lambda x: x.std()*np.sqrt(252)).round(4).to_string())

print("\n=== эксцесс/хвосты ===")
from statistics import NormalDist
print("kurtosis(forward_returns):", round(pd.Series(f).kurt(),2), " (у реального S&P дневного ~ 10-15)")
print("skew:", round(pd.Series(f).skew(),3))
print("std дневной:", round(f.std(),6), "-> годовая вола:", round(f.std()*np.sqrt(252)*100,2),"%")
print("реальная годовая вола S&P 1990-2025 ~18-19%")
