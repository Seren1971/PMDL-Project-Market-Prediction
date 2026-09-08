import pandas as pd, numpy as np, sys
sys.path.insert(0,'.')
from src.metric import hull_sharpe
tr = pd.read_csv('data/raw/train.csv')

print("=== метрика для константных позиций (вся история, date_id 0..9047) ===")
print(f"{'pos':>5} {'score':>8} {'sharpe':>8} {'vol_ratio':>10} {'ret_pen':>8} {'strat_exc%':>11} {'mkt_exc%':>9}")
for p in [0.0,0.5,0.8,1.0,1.1,1.2,1.3,1.5,2.0]:
    r=hull_sharpe(np.full(len(tr),p), tr.forward_returns, tr.risk_free_rate)
    print(f"{p:5.1f} {r['score']:8.4f} {r['sharpe']:8.4f} {r['vol_ratio']:10.3f} {r['return_penalty']:8.3f} {r['strat_ann_excess']:11.2f} {r['mkt_ann_excess']:9.2f}")

print("\n=== то же на последних 180 днях (размер зачётного окна) ===")
w = tr.tail(180)
print(f"{'pos':>5} {'score':>8} {'sharpe':>8} {'vol_ratio':>10} {'ret_pen':>8}")
for p in [0.0,0.5,1.0,1.2,1.5,2.0]:
    r=hull_sharpe(np.full(len(w),p), w.forward_returns, w.risk_free_rate)
    print(f"{p:5.1f} {r['score']:8.4f} {r['sharpe']:8.4f} {r['vol_ratio']:10.3f} {r['return_penalty']:8.3f}")

print("\n=== разброс скора константы 1.0 по непересекающимся 180-дневным окнам ===")
sc=[]
for i in range(0, len(tr)-180, 180):
    w=tr.iloc[i:i+180]
    sc.append(hull_sharpe(np.ones(len(w)), w.forward_returns, w.risk_free_rate)['score'])
sc=np.array(sc)
print(f"окон={len(sc)}  медиана={np.median(sc):.3f}  среднее={sc.mean():.3f}  std={sc.std():.3f}  "
      f"min={sc.min():.3f}  max={sc.max():.3f}  доля отрицательных={100*(sc<0).mean():.0f}%")
print("это шум, с которым придётся соревноваться: скор одной и той же тривиальной стратегии гуляет в этих пределах")

print("\n=== оракул: идеальный тайминг (pos=2 если завтра рынок выше rf, иначе 0) на последних 180 днях ===")
w=tr.tail(180)
oracle=np.where(w.forward_returns>w.risk_free_rate,2.0,0.0)
r=hull_sharpe(oracle,w.forward_returns,w.risk_free_rate)
print({k:round(v,3) for k,v in r.items()})
