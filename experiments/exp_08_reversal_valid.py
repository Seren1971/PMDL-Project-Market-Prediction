"""Честная статистика: непересекающиеся окна + блочный бутстрап."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns
from metric import hull_sharpe

rng = np.random.default_rng(0)
tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); START = 2300
S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x); return ((x - x.rolling(w).mean()) / x.rolling(w).std()).values
def to_pos(z, mp=1.0, a=0.25, lo=0.0, hi=2.0):
    p = np.clip(mp + a*np.nan_to_num(z), lo, hi); p[np.isnan(z)] = mp; return p

sigs = {'разворот 5 дней': -zc(S.rolling(5).sum().values),
        'разворот 22 дня': -zc(S.rolling(22).sum().values),
        'разворот 63 дня': -zc(S.rolling(63).sum().values)}

def wins(pos, width, step):
    return np.array([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                     for s in range(START, n-width+1, step)])

print("=== НЕПЕРЕСЕКАЮЩИЕСЯ окна (step = width) — статистика честная ===")
for width in (130, 180):
    b = wins(np.full(n,1.0), width, width)
    print(f"\n  ширина {width}, окон {len(b)}:")
    for name, z in sigs.items():
        w = wins(to_pos(z), width, width); d = w - b
        k = int((d>0).sum()); N = len(d)
        p = sum(comb(N,i) for i in range(k,N+1)) / 2**N
        print(f"    {name:18s} мед.разн={np.median(d):+.3f} побед={k}/{N} ({k/N*100:.0f}%) p={p:.4f}")

print("\n=== Блочный бутстрап (блоки по 252 дня, 2000 повторов) ===")
print("проверяем: устойчива ли разница, если пересобрать историю из годовых блоков")
BL = 252
nb = (n - START) // BL
for name, z in sigs.items():
    pos = to_pos(z)
    # доходности стратегии и бенчмарка по дням
    st_r = rf + pos*(fwd - rf); bs_r = rf + 1.0*(fwd - rf)
    blocks = [(START+i*BL, START+(i+1)*BL) for i in range(nb)]
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, nb, nb)
        sel = np.concatenate([np.arange(*blocks[i]) for i in idx])
        try:
            a = hull_sharpe(np.clip(pos[sel],0,2), fwd[sel], rf[sel])['score']
            c = hull_sharpe(np.ones(len(sel)), fwd[sel], rf[sel])['score']
            diffs.append(a - c)
        except ValueError: pass
    d = np.array(diffs)
    print(f"  {name:18s} среднее={d.mean():+.3f} "
          f"95%ДИ=[{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}] "
          f"P(>0)={(d>0).mean()*100:.1f}%")

print("\n=== Комбинация: разворот + vol-targeting ===")
from wf import walk_forward
H=21; y = np.log(np.sqrt(pd.Series(fwd**2).rolling(H).mean().shift(-(H-1)).values)+1e-8)
s2 = pd.Series(np.abs(rk))
har = np.log(np.c_[s2.rolling(1).mean(), s2.rolling(5).mean(), s2.rolling(22).mean()]+1e-8)
pl,_ = walk_forward(har, y); sig = np.exp(pl)
volz = zc(np.log(sig))
z5 = sigs['разворот 5 дней']
combos = {
 'только разворот 5д':                to_pos(z5),
 'разворот 5д + высокая вола':        to_pos(0.5*np.nan_to_num(z5)+0.5*np.nan_to_num(volz)),
 'разворот 5д, срез волы сверху':     np.minimum(to_pos(z5), to_pos(-volz, mp=1.3, a=0.3)),
 'разворот 5д x обратная вола':       to_pos(z5) * np.clip(np.nanmedian(sig)/sig, 0.6, 1.4),
}
b180 = wins(np.full(n,1.0), 180, 30)
for name, pos in combos.items():
    pos = np.nan_to_num(pos, nan=1.0)
    w = wins(pos, 180, 30); d = w - b180
    full = hull_sharpe(np.clip(pos[START:],0,2), fwd[START:], rf[START:])
    print(f"  {name:32s} мед.разн={np.median(d):+.3f} побед={(d>0).mean()*100:5.1f}% "
          f"| история: score={full['score']:.3f} vol_r={full['vol_ratio']:.2f}")
