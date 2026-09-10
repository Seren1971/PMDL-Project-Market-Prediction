"""Почему vol-targeting выигрывает на длинной истории, но проигрывает по окнам."""
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

def vol_target(sig, mean_pos=1.0, lo=0.0, hi=2.0, warm=2000):
    st = pd.Series(sig).expanding(250).median().shift(1).values
    raw = st / sig
    scale = pd.Series(raw).expanding(250).mean().shift(1).values
    pos = np.clip(mean_pos * raw / scale, lo, hi)
    pos[:warm] = mean_pos; pos[np.isnan(pos)] = mean_pos
    return pos

START = 2300
def per_window(pos, width=180, step=30):
    rows = []
    for st in range(START, n - width + 1, step):
        sl = slice(st, st + width)
        r = hull_sharpe(np.clip(pos[sl], 0, 2), fwd[sl], rf[sl])
        rows.append(dict(start=st, score=r['score'], sharpe=r['sharpe'],
                         vol_ratio=r['vol_ratio'], ret_pen=r['return_penalty'],
                         vol_pen=r['vol_penalty'], gap=r['return_gap'],
                         mkt=r['mkt_ann_excess']))
    return pd.DataFrame(rows)

base = per_window(np.full(n, 1.0))
vt   = per_window(vol_target(sig, 1.0))

print("=== разложение: где теряется скор (окна 180 дней, n=%d) ===" % len(base))
print(f"{'':22s} {'константа 1.0':>14s} {'volTarget':>12s}")
for col in ['sharpe', 'score', 'vol_ratio', 'ret_pen', 'vol_pen']:
    print(f"  медиана {col:14s} {base[col].median():14.3f} {vt[col].median():12.3f}")
print(f"  доля окон ret_pen>1  {(base.ret_pen>1).mean():14.3f} {(vt.ret_pen>1).mean():12.3f}")

print("\n--- ключевой вопрос: СЫРОЙ Sharpe (без штрафов) улучшается? ---")
d = vt.sharpe - base.sharpe
print(f"  разница сырых Sharpe: медиана {d.median():+.3f}, побед {(d>0).mean()*100:.1f}%")
d2 = vt.score - base.score
print(f"  разница СКОРА:        медиана {d2.median():+.3f}, побед {(d2>0).mean()*100:.1f}%")

print("\n--- в каких окнах volTarget проигрывает ---")
m = base.copy(); m['diff'] = vt.score - base.score; m['vt_gap'] = vt.gap
print("  по квартилям доходности рынка в окне:")
m['q'] = pd.qcut(m.mkt, 4, labels=['рынок падал','слабо','хорошо','сильный рост'])
print(m.groupby('q', observed=True).agg(n=('diff','size'), медиана_разницы=('diff','median'),
      побед=('diff', lambda x: (x>0).mean()), штраф_gap=('vt_gap','median')).round(3).to_string())

print("\n=== лечение: асимметричные варианты (не опускаться низко, только срезать верх) ===")
variants = {
    'volTgt свободный [0,2] @1.0':   vol_target(sig, 1.0, 0.0, 2.0),
    'volTgt пол 0.7 @1.0':           vol_target(sig, 1.0, 0.7, 2.0),
    'volTgt пол 0.9 @1.0':           vol_target(sig, 1.0, 0.9, 2.0),
    'volTgt пол 0.9 потолок 1.3':    vol_target(sig, 1.05, 0.9, 1.3),
    'volTgt пол 1.0 потолок 1.5':    vol_target(sig, 1.1, 1.0, 1.5),
    'volTgt @1.2 пол 0.9':           vol_target(sig, 1.2, 0.9, 2.0),
    'volTgt @1.3 пол 1.0':           vol_target(sig, 1.3, 1.0, 2.0),
}
print(f"  {'вариант':32s} {'мед.скор':>9s} {'побед%':>7s} {'ср.поз':>7s} {'сырой Sh':>9s}")
print(f"  {'константа 1.0 (бенчмарк)':32s} {base.score.median():9.3f} {'—':>7s} {1.0:7.2f} {base.sharpe.median():9.3f}")
for name, pos in variants.items():
    w = per_window(pos); d = w.score - base.score
    print(f"  {name:32s} {w.score.median():9.3f} {(d>0).mean()*100:7.1f} "
          f"{pos[START:].mean():7.2f} {w.sharpe.median():9.3f}")
