import pandas as pd, numpy as np
tr=pd.read_csv('data/raw/train.csv'); te=pd.read_csv('data/raw/test.csv')
tr['diff']=tr.forward_returns-tr.market_forward_excess_returns
print("=== окрестность date_id 8990 (аномалия в diff) ===")
print(tr.loc[(tr.date_id>=8984)&(tr.date_id<=8996),['date_id','forward_returns','risk_free_rate','market_forward_excess_returns','diff']].round(6).to_string(index=False))
print("\nскачки diff > 5*std(изменений):")
dd=tr['diff'].diff()
j=tr.loc[dd.abs()>5*dd.std(),['date_id','diff']]
print(j.round(6).to_string(index=False) if len(j) else "нет")

print("\n=== test.csv целиком (кроме фич) ===")
print(te[['date_id','is_scored','lagged_forward_returns','lagged_risk_free_rate','lagged_market_forward_excess_returns']].round(6).to_string(index=False))
print("\nсверка лага: train.forward_returns[8979..8988] =")
print(tr.loc[(tr.date_id>=8979)&(tr.date_id<=8988),['date_id','forward_returns','market_forward_excess_returns']].round(6).to_string(index=False))
print("\nсовпадает ли test.lagged_forward_returns[t] с train.forward_returns[t-1]:",
      bool(np.allclose(te.lagged_forward_returns.values, tr.set_index('date_id').loc[te.date_id-1,'forward_returns'].values, atol=1e-9)))
print("совпадают ли фичи test с фичами train на тех же date_id:",
      bool(np.allclose(te[[c for c in te.columns if c in tr.columns and c!='date_id']].values,
                       tr.set_index('date_id').loc[te.date_id, [c for c in te.columns if c in tr.columns and c!='date_id']].values,
                       equal_nan=True)))
print("\n=== хвост train: последние 12 строк, поведение diff ===")
print(tr.tail(12)[['date_id','forward_returns','market_forward_excess_returns','diff']].round(6).to_string(index=False))
