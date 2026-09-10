"""Vol-targeting — это страховка? Смотрим хвосты, а не медиану."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from data import load_train, known_returns
from wf import walk_forward
from metric import hull_sharpe

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); H = 21
y = np.log(np.sqrt(pd.Series(fwd**2).rolling(H).mean().shift(-(H-1)).values) + 1e-8)
s = pd.Series(np.abs(rk))
har = np.log(np.c_[s.rolling(1).mean(), s.rolling(5).mean(), s.rolling(22).mean()] + 1e-8)
pred_log, _ = walk_forward(har, y); sig = np.exp(pred_log)

def vt_pos(mean_pos=1.0, lo=0.0, hi=2.0, invert=False, warm=2000):
    st = pd.Series(sig).expanding(250).median().shift(1).values
    raw = (sig / st) if invert else (st / sig)
    scale = pd.Series(raw).expanding(250).mean().shift(1).values
    pos = np.clip(mean_pos * raw / scale, lo, hi)
    pos[:warm] = mean_pos; pos[np.isnan(pos)] = mean_pos
    return pos

START = 2300
def wins(pos, width=180, step=30):
    out = []
    for st in range(START, n - width + 1, step):
        sl = slice(st, st + width)
        out.append(hull_sharpe(np.clip(pos[sl],0,2), fwd[sl], rf[sl])['score'])
    return np.array(out)

base = wins(np.full(n, 1.0))
vt   = wins(vt_pos(1.0))
inv  = wins(vt_pos(1.0, invert=True))
d = vt - base

print("=== распределение РАЗНИЦЫ (volTarget − константа) по 219 окнам ===")
for q in (5, 10, 25, 50, 75, 90, 95):
    print(f"  {q:2d}-й перцентиль: {np.percentile(d, q):+.3f}")
print(f"  среднее {d.mean():+.3f}   медиана {np.median(d):+.3f}   доля побед {(d>0).mean()*100:.1f}%")
print("  -> если среднее > 0, а медиана < 0, это страховка: часто чуть хуже, редко сильно лучше")

print("\n=== где именно volTarget выигрывает крупно ===")
idx = np.argsort(d)[::-1][:6]
starts = list(range(START, n - 180 + 1, 30))
cal = pd.read_csv('data/interim_calendar_estimate.csv')
for i in idx:
    st = starts[i]
    d0 = cal.loc[cal.date_id == st, 'date_est'].values[0]
    d1 = cal.loc[cal.date_id == st+179, 'date_est'].values[0]
    print(f"  {d0} .. {d1}: константа {base[i]:+.2f} -> volTgt {vt[i]:+.2f}  (разница {d[i]:+.2f})")
print("  и где проигрывает крупнее всего:")
for i in np.argsort(d)[:4]:
    st = starts[i]
    d0 = cal.loc[cal.date_id == st, 'date_est'].values[0]
    d1 = cal.loc[cal.date_id == st+179, 'date_est'].values[0]
    print(f"  {d0} .. {d1}: константа {base[i]:+.2f} -> volTgt {vt[i]:+.2f}  (разница {d[i]:+.2f})")

print("\n=== проверка наоборот: а если наращивать позицию В высокую волатильность? ===")
di = inv - base
print(f"  инвертированный volTarget: медиана {np.median(di):+.3f}, среднее {di.mean():+.3f}, "
      f"побед {(di>0).mean()*100:.1f}%")

print("\n=== по десятилетиям: медиана разницы ===")
dec = [(cal.loc[cal.date_id == st, 'date_est'].values[0][:4]) for st in starts]
df = pd.DataFrame({'year': [int(x) for x in dec], 'diff': d})
df['dec'] = (df.year // 10) * 10
print(df.groupby('dec').agg(окон=('diff','size'), медиана=('diff','median'),
                            среднее=('diff','mean'), побед=('diff', lambda x:(x>0).mean())).round(3).to_string())
