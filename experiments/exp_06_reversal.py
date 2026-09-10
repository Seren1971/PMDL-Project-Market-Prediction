"""Инвертированный vol-targeting и краткосрочный разворот. Плюс проверка на артефакты."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from data import load_train, known_returns
from wf import walk_forward
from metric import hull_sharpe
from backtest import compare, fmt

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); H = 21
y = np.log(np.sqrt(pd.Series(fwd**2).rolling(H).mean().shift(-(H-1)).values) + 1e-8)
s = pd.Series(np.abs(rk))
har = np.log(np.c_[s.rolling(1).mean(), s.rolling(5).mean(), s.rolling(22).mean()] + 1e-8)
pred_log, _ = walk_forward(har, y); sig = np.exp(pred_log)
START = 2300; cal = pd.read_csv('data/interim_calendar_estimate.csv')
years = cal.date_est.str.slice(0,4).astype(int).values

def zscore_causal(x, w=252):
    m = pd.Series(x).rolling(w).mean().shift(1)
    sd = pd.Series(x).rolling(w).std().shift(1)
    return ((pd.Series(x).shift(1) - m) / sd).values     # всё сдвинуто: только прошлое

def to_pos(z, mean_pos=1.0, alpha=0.25, lo=0.0, hi=2.0):
    p = np.clip(mean_pos + alpha * np.nan_to_num(z), lo, hi)
    p[np.isnan(z)] = mean_pos
    return p

def wins(pos, width=180, step=30):
    out = []
    for st in range(START, n - width + 1, step):
        sl = slice(st, st + width)
        out.append(hull_sharpe(np.clip(pos[sl],0,2), fwd[sl], rf[sl])['score'])
    return np.array(out)

base = wins(np.full(n, 1.0))
def rep(name, pos):
    w = wins(pos); d = w - base
    r = hull_sharpe(np.clip(pos[START:],0,2), fwd[START:], rf[START:])
    print(f"  {name:34s} мед={np.median(w):+.3f} побед={(d>0).mean()*100:5.1f}% "
          f"ср.разн={d.mean():+.3f} | вся история: score={r['score']:.3f} "
          f"vol_ratio={r['vol_ratio']:.2f} ср.поз={pos[START:].mean():.2f}")
    return w, d

print("=== A. Сигналы: чем выше — тем больше ставим ===")
print(f"  {'константа 1.0 (бенчмарк)':34s} мед={np.median(base):+.3f}")
sig_hi_vol = zscore_causal(np.log(sig))                       # высокая ожидаемая вола
rev1  = -zscore_causal(rk)                                    # вчера упало -> покупаем
rev5  = -zscore_causal(pd.Series(rk).rolling(5).sum().values)
rev22 = -zscore_causal(pd.Series(rk).rolling(22).sum().values)
mom252= zscore_causal(pd.Series(rk).rolling(252).sum().values)

res = {}
res['высокая вола (инверсия volTgt)'] = rep('высокая вола (инверсия volTgt)', to_pos(sig_hi_vol))
res['разворот 1 день'] = rep('разворот 1 день', to_pos(rev1))
res['разворот 5 дней'] = rep('разворот 5 дней', to_pos(rev5))
res['разворот 22 дня'] = rep('разворот 22 дня', to_pos(rev22))
res['моментум 252 дня'] = rep('моментум 252 дня', to_pos(mom252))

print("\n=== B. Устойчивость по десятилетиям (медиана разницы с константой) ===")
starts = np.array(list(range(START, n - 180 + 1, 30)))
wy = years[starts]
tab = {}
for name, (w, d) in res.items():
    tab[name] = pd.Series(d).groupby((wy//10)*10).median()
print(pd.DataFrame(tab).round(3).to_string())

print("\n=== C. Проверка на артефакт винзоризации ===")
print("данные обрезаны на ±4%: реальные обвалы срезаны, поэтому плечо в кризис")
print("выглядит безопаснее, чем оно есть. Смотрим, что будет, если вернуть хвосты грубо:")
# грубая реконструкция: дни на границе |r|>0.037 растягиваем в 2 раза
fat = fwd.copy(); edge = np.abs(fwd) > 0.037
fat[edge] = np.sign(fwd[edge]) * (0.037 + (np.abs(fwd[edge]) - 0.037) * 8)
print(f"  дней растянуто: {edge.sum()}, новый min={fat.min():.3f} max={fat.max():.3f}")
def wins_fat(pos, width=180, step=30):
    out = []
    for st in range(START, n - width + 1, step):
        sl = slice(st, st+width)
        out.append(hull_sharpe(np.clip(pos[sl],0,2), fat[sl], rf[sl])['score'])
    return np.array(out)
bf = wins_fat(np.full(n,1.0))
for name, pos in (('высокая вола', to_pos(sig_hi_vol)), ('разворот 5 дней', to_pos(rev5)),
                  ('разворот 1 день', to_pos(rev1))):
    w = wins_fat(pos); d = w - bf
    print(f"  на 'жирных хвостах' {name:18s} мед={np.median(w):+.3f} побед={(d>0).mean()*100:5.1f}% ср.разн={d.mean():+.3f}")
