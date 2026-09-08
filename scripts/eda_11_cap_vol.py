import pandas as pd, numpy as np

tr = pd.read_csv('data/raw/train.csv')
r = tr.forward_returns
print("=== природа обрезки: |r| у экстремальных дней в единицах скользящей std ===")
for w in (60, 252, 1260):
    s = r.rolling(w, min_periods=w//2).std()
    z = (r.abs() / s)
    top = z.nlargest(15)
    print(f"окно {w:4d}: max z={z.max():.2f}  p99.9={z.quantile(0.999):.2f}  "
          f"медиана z топ-15 экстремумов={top.median():.2f}")

env = r.abs()
big = tr[env > 0.038]
print("\nогибающая |r| по времени (дни выше 0.038): растёт ли граница")
print(big.groupby(big.date_id // 1000).forward_returns.apply(lambda x: x.abs().max()).round(4).to_string())

print("\n=== доля дней у самой границы своего периода ===")
roll_max = r.abs().rolling(1260, min_periods=250).max()
at_edge = (r.abs() >= 0.995 * roll_max).sum()
print(f"дней на 99.5% локального максимума за 5 лет: {at_edge} ({at_edge/len(r)*100:.2f}%)")

print("\n=== автокорреляция forward_returns ===")
for lag in (1, 2, 3, 5, 10, 21):
    print(f"  lag {lag:2d}: {r.autocorr(lag):+.4f}")
print("автокорр |r| (кластеризация волатильности):")
for lag in (1, 5, 21):
    print(f"  lag {lag:2d}: {r.abs().autocorr(lag):+.4f}")

print("\n=== предсказуемость волатильности vs доходности (date_id>=7000) ===")
sub = tr[tr.date_id >= 7000].copy()
sub['fut_vol'] = sub.forward_returns.abs().rolling(21).mean().shift(-21)
feats = [c for c in tr.columns if c[0] in 'DEIMPSV' and c not in
         ('date_id','forward_returns','risk_free_rate','market_forward_excess_returns')]
rows = []
for c in feats:
    x = sub[c]
    m = x.notna() & sub.fut_vol.notna()
    if m.sum() < 500: continue
    rows.append((c, x[m].corr(sub.fut_vol[m]), x[m].corr(sub.market_forward_excess_returns[m])))
d = pd.DataFrame(rows, columns=['col','corr_fut_vol','corr_target']).assign(
    a=lambda d: d.corr_fut_vol.abs()).sort_values('a', ascending=False)
print("топ-15 фич по корреляции с будущей реализованной волой (21д):")
print(d.head(15)[['col','corr_fut_vol','corr_target']].round(3).to_string(index=False))
print(f"\nмедиана |corr| с будущей волой: {d.a.median():.3f}  |  с таргетом: "
      f"{d.corr_target.abs().median():.3f}")
print(f"фич с |corr(vol)|>0.3: {(d.a>0.3).sum()}/{len(d)}")

print("\n=== рынок по декадам (блоки по 2520 дней ~ 10 лет) ===")
tr['dec'] = tr.date_id // 2520
g = tr.groupby('dec').agg(n=('forward_returns','size'),
                          ann_ret=('forward_returns', lambda x: ((1+x).prod()**(252/len(x))-1)*100),
                          vol=('forward_returns', lambda x: x.std()*np.sqrt(252)*100),
                          rf=('risk_free_rate', lambda x: x.mean()*252*100))
g['sharpe'] = (g.ann_ret - g.rf) / g.vol
print(g.round(2).to_string())
