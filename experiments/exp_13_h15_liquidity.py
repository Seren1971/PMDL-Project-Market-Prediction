"""H15: премия за разворот пропорциональна волатильности (механизм Нагеля)."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns
from metric import hull_sharpe

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd)
S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
z5 = np.nan_to_num(-zc(S.rolling(5).sum().values))
vol22 = pd.Series(rk**2).rolling(22).mean().pow(0.5).values          # прокси VIX, причинный
volq = pd.Series(vol22).expanding(500).rank(pct=True).values          # где вола относительно истории

print("=== A. Прямая проверка механизма: окупается ли разворот сильнее при высокой воле ===")
sig = z5.copy(); tgtx = fwd - rf
m = ~np.isnan(volq) & ~np.isnan(sig) & (np.arange(n) > 1000)
q = pd.qcut(volq[m], 4, labels=['вола низкая','ниже средн','выше средн','вола высокая'])
df = pd.DataFrame({'sig': sig[m], 'ret': tgtx[m], 'q': q})
print("  корреляция сигнала разворота с завтрашней сверхдоходностью, по квартилям волатильности:")
for name, g in df.groupby('q', observed=True):
    r = np.corrcoef(g.sig, g.ret)[0,1]
    t = r*np.sqrt(len(g)-2)/np.sqrt(1-r**2)
    print(f"    {name:14s} corr={r:+.4f}  t={t:+.2f}  n={len(g)}")
print("  -> по Нагелю ожидаем рост слева направо")

print("\n=== B. Стратегия: сила разворота масштабируется волатильностью ===")
def wins(pos, lo, hi, width, step):
    return np.array([hull_sharpe(np.clip(np.nan_to_num(pos[s:s+width],nan=1.0),0,2),
                                 fwd[s:s+width], rf[s:s+width])['score']
                     for s in range(lo, hi-width+1, step)])
def stat(pos, lo=2300, hi=n, width=180, step=180):
    d = wins(pos, lo, hi, width, step) - wins(np.full(n,1.0), lo, hi, width, step)
    N=len(d); k=int((d>0).sum())
    return np.median(d), k/N, sum(comb(N,i) for i in range(k,N+1))/2**N, N

scale = np.nan_to_num(volq, nan=0.5)
variants = {
 'разворот, сила постоянна':          np.clip(0.95 + 0.25*z5, 0, 2),
 'разворот x квантиль волы (Нагель)': np.clip(0.95 + 0.50*z5*scale, 0, 2),
 'разворот x вола, резко (^2)':       np.clip(0.95 + 0.60*z5*scale**2, 0, 2),
 'разворот только при высокой воле':  np.clip(0.95 + 0.50*z5*(scale>0.6), 0, 2),
 'разворот ОБРАТНО воле (контроль)':  np.clip(0.95 + 0.50*z5*(1-scale), 0, 2),
}
for lab, lo, hi in (('вся выборка', 2300, n), ('вторая половина', 6000, n)):
    print(f"\n  --- {lab} (непересекающиеся окна 180) ---")
    for name, pos in variants.items():
        md, wr, p, N = stat(pos, lo, hi)
        print(f"    {name:36s} мед.разн={md:+.3f} побед={wr*100:5.1f}% p={p:.4f} (окон {N})")

print("\n=== C. Затухание во времени: разворот по пятилеткам ===")
cal = pd.read_csv('data/interim_calendar_estimate.csv')
years = cal.date_est.str.slice(0,4).astype(int).values
pos = np.clip(0.95 + 0.25*z5, 0, 2)
rows=[]
for y0 in range(1995, 2025, 5):
    idx = np.where((years >= y0) & (years < y0+5))[0]
    idx = idx[idx > 2300]
    if len(idx) < 400: continue
    a = hull_sharpe(np.clip(pos[idx],0,2), fwd[idx], rf[idx])['score']
    b = hull_sharpe(np.ones(len(idx)), fwd[idx], rf[idx])['score']
    # IC сигнала в этом периоде
    ic = np.corrcoef(z5[idx], (fwd-rf)[idx])[0,1]
    rows.append((f"{y0}-{y0+4}", len(idx), a, b, a-b, ic))
print(f"  {'период':10s} {'дней':>6s} {'стратегия':>10s} {'константа':>10s} {'разница':>9s} {'IC':>8s}")
for r in rows:
    print(f"  {r[0]:10s} {r[1]:6d} {r[2]:10.3f} {r[3]:10.3f} {r[4]:+9.3f} {r[5]:+8.4f}")
