"""Разворот: строгая причинность + парная статистика + устойчивость к параметрам."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns
from metric import hull_sharpe

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd)
cal = pd.read_csv('data/interim_calendar_estimate.csv')
years = cal.date_est.str.slice(0,4).astype(int).values
START = 2300

# rk[t] = fwd[t-1] — доходность, УЖЕ известная к закрытию дня t.
# Значит скользящие статистики по rk можно брать ВКЛЮЧИТЕЛЬНО по t. Лишнего сдвига не нужно.
S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x)
    return ((x - x.rolling(w).mean()) / x.rolling(w).std()).values

def to_pos(z, mean_pos=1.0, alpha=0.25, lo=0.0, hi=2.0):
    p = np.clip(mean_pos + alpha * np.nan_to_num(z), lo, hi)
    p[np.isnan(z)] = mean_pos
    return p

def wins(pos, width, step=30):
    out, st_ = [], []
    for s0 in range(START, n - width + 1, step):
        sl = slice(s0, s0 + width)
        out.append(hull_sharpe(np.clip(pos[sl],0,2), fwd[sl], rf[sl])['score']); st_.append(s0)
    return np.array(out), np.array(st_)

def paired(pos, width=180):
    w, st_ = wins(pos, width); b, _ = wins(np.full(n,1.0), width)
    d = w - b; k = int((d>0).sum()); N = len(d)
    p = sum(comb(N,i) for i in range(k, N+1)) / 2**N
    return dict(med_d=np.median(d), mean_d=d.mean(), win=k/N, p=p, N=N, d=d, st=st_)

signals = {
    'разворот 1 день':   -zc(rk),
    'разворот 3 дня':    -zc(S.rolling(3).sum().values),
    'разворот 5 дней':   -zc(S.rolling(5).sum().values),
    'разворот 10 дней':  -zc(S.rolling(10).sum().values),
    'разворот 22 дня':   -zc(S.rolling(22).sum().values),
    'разворот 63 дня':   -zc(S.rolling(63).sum().values),
    'моментум 252 дня':   zc(S.rolling(252).sum().values),
}
print("=== Парное сравнение с константой 1.0, окна 180 дней, шаг 30 ===")
print(f"  {'сигнал':22s} {'мед.разн':>9s} {'ср.разн':>8s} {'побед':>7s} {'p (знаковый)':>13s}")
res = {}
for name, z in signals.items():
    r = paired(to_pos(z)); res[name] = r
    print(f"  {name:22s} {r['med_d']:+9.3f} {r['mean_d']:+8.3f} {r['win']*100:6.1f}% {r['p']:13.5f}")

print("\n=== Устойчивость к силе сигнала alpha (разворот 5 дней) ===")
z5 = signals['разворот 5 дней']
for a in (0.10, 0.15, 0.25, 0.4, 0.6):
    r = paired(to_pos(z5, alpha=a))
    print(f"  alpha={a:.2f}: мед.разн={r['med_d']:+.3f} побед={r['win']*100:5.1f}% p={r['p']:.5f}")

print("\n=== Устойчивость к ширине окна оценки ===")
for width in (130, 180, 252):
    r = paired(to_pos(z5), width=width)
    print(f"  окно {width:3d} дней: мед.разн={r['med_d']:+.3f} побед={r['win']*100:5.1f}% "
          f"p={r['p']:.5f} (окон {r['N']})")

print("\n=== Разбивка по десятилетиям ===")
tab = {}
for name in ('разворот 1 день','разворот 5 дней','разворот 22 дня','моментум 252 дня'):
    r = res[name]
    tab[name] = pd.Series(r['d']).groupby((years[r['st']]//10)*10).agg(['median','mean',lambda x:(x>0).mean()])
for name, t in tab.items():
    t.columns = ['медиана','среднее','побед']
    print(f"\n  {name}:")
    print(t.round(3).to_string())

print("\n=== Поправка на множественное тестирование ===")
print(f"  проверено {len(signals)} сигналов. Порог Бонферрони p < {0.05/len(signals):.4f}")
for name, r in res.items():
    ok = "ПРОХОДИТ" if r['p'] < 0.05/len(signals) else "не проходит"
    print(f"  {name:22s} p={r['p']:.5f}  {ok}")
