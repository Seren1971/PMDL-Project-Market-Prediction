"""Второй круг: H16 асимметрия, H17 ансамбль горизонтов, H19 календарь, H20 ограничитель."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns
from metric import hull_sharpe
from wf import walk_forward

rng = np.random.default_rng(2)
tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd)
S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
Z = {h: np.nan_to_num(-zc(S.rolling(h).sum().values)) for h in (1,5,22,63)}
z5 = Z[5]
LATE = 6500     # последние ~10 лет

def wins(pos, lo, hi, width=180, step=180):
    return np.array([hull_sharpe(np.clip(np.nan_to_num(pos[s:s+width],nan=1.0),0,2),
                                 fwd[s:s+width], rf[s:s+width])['score']
                     for s in range(lo, hi-width+1, step)])
def stat(pos, lo, hi, width=180):
    d = wins(pos, lo, hi, width) - wins(np.full(n,1.0), lo, hi, width)
    N=len(d); k=int((d>0).sum())
    return np.median(d), k/N, sum(comb(N,i) for i in range(k,N+1))/2**N, N
def show(name, pos):
    a = stat(pos, 2300, n); b = stat(pos, LATE, n)
    print(f"  {name:38s} вся: {a[0]:+.3f}/{a[1]*100:4.0f}%  посл.10л: {b[0]:+.3f}/{b[1]*100:4.0f}% (p={b[2]:.3f}, окон {b[3]})")
    return b

print("=== H16: асимметрия — только добавлять, не снижать ===")
show('симметрично 0.95 + 0.25z', np.clip(0.95 + 0.25*z5, 0, 2))
show('односторонне 1.0 + 0.25*max(z,0)', np.clip(1.0 + 0.25*np.maximum(z5,0), 0, 2))
show('односторонне 1.0 + 0.4*max(z,0)', np.clip(1.0 + 0.40*np.maximum(z5,0), 0, 2))
show('пол 1.0, потолок 1.5', np.clip(1.0 + 0.25*z5, 1.0, 1.5))
show('пол 0.9, потолок 1.4', np.clip(1.0 + 0.25*z5, 0.9, 1.4))
show('пол 1.0 + 0.25z, ср.поз выровнена', np.clip(0.93 + 0.25*z5, 0.85, 2))

print("\n=== H17: ансамбль горизонтов через inverse-vol weighting (приём 4-го места) ===")
# вес каждого сигнала обратно пропорционален его скользящей волатильности
W = np.column_stack([1.0/(pd.Series(Z[h]).rolling(252).std().values + 1e-9) for h in (1,5,22,63)])
W = W / np.nansum(W, axis=1, keepdims=True)
ens_iv = np.nansum(np.column_stack([Z[h] for h in (1,5,22,63)]) * W, axis=1)
ens_eq = np.column_stack([Z[h] for h in (1,5,22,63)]).mean(1)
show('только разворот 5д', np.clip(0.95 + 0.25*z5, 0, 2))
show('ансамбль, равные веса', np.clip(0.95 + 0.25*np.nan_to_num(ens_eq), 0, 2))
show('ансамбль, inverse-vol веса', np.clip(0.95 + 0.25*np.nan_to_num(ens_iv), 0, 2))

print("\n=== H19: календарные флажки поверх разворота ===")
D = {c: tr[c].values.astype(float) for c in ['D1','D4','D5','D8','D9','D6']}
cal_sig = D['D8'] + D['D9'] + D['D5']            # окно начала месяца
show('разворот 5д (база)', np.clip(0.95 + 0.25*z5, 0, 2))
show('разворот + начало месяца', np.clip(0.95 + 0.25*z5 + 0.15*cal_sig, 0, 2))
show('разворот + ноябрь-апрель D4', np.clip(0.95 + 0.25*z5 + 0.15*D['D4'], 0, 2))
show('только календарь', np.clip(0.95 + 0.15*cal_sig + 0.15*D['D4'], 0, 2))

print("\n=== H20: прогноз волы как ограничитель, а не сайзер ===")
H=21; y = np.log(np.sqrt(pd.Series(fwd**2).rolling(H).mean().shift(-(H-1)).values)+1e-8)
s2 = pd.Series(np.abs(rk))
har = np.log(np.c_[s2.rolling(1).mean(), s2.rolling(5).mean(), s2.rolling(22).mean()]+1e-8)
pl,_ = walk_forward(har, y); sig = np.exp(pl)
mkt_sig = pd.Series(sig).expanding(250).median().shift(1).values
base_pos = np.clip(0.95 + 0.25*z5, 0, 2)
capped = np.minimum(base_pos, np.nan_to_num(1.2*mkt_sig/sig, nan=2.0))
show('без ограничителя', base_pos)
show('с ограничителем 1.2*sigma_mkt/sigma', capped)
for lo, hi in ((2300, n), (LATE, n)):
    for nm, p in (('без огр.', base_pos), ('с огр.', capped)):
        vr = [hull_sharpe(np.clip(p[s:s+180],0,2), fwd[s:s+180], rf[s:s+180])['vol_ratio']
              for s in range(lo, n-180+1, 180)]
        print(f"    [{lo}..] {nm:10s} доля окон vol_ratio>1.2: {np.mean(np.array(vr)>1.2)*100:.1f}%, макс={max(vr):.2f}")

print("\n=== H21: оптимизация по худшим 25% окон вместо медианы ===")
grid = []
for mp in (0.85, 0.9, 0.95, 1.0, 1.05):
    for a in (0.10, 0.15, 0.25, 0.40):
        for hl in (1, 5, 10):
            pos = pd.Series(np.clip(mp + a*z5, 0, 2)).ewm(halflife=hl).mean().values
            d = wins(pos, 2300, LATE) - wins(np.full(n,1.0), 2300, LATE)   # подбор ТОЛЬКО на раннем периоде
            grid.append((mp, a, hl, np.median(d), np.percentile(d, 25)))
g = pd.DataFrame(grid, columns=['mp','alpha','hl','med','p25'])
b_med = g.loc[g.med.idxmax()]; b_p25 = g.loc[g.p25.idxmax()]
print(f"  лучшая по медиане на раннем периоде: mp={b_med.mp} a={b_med.alpha} hl={b_med.hl:.0f}")
print(f"  лучшая по 25-му перцентилю:          mp={b_p25.mp} a={b_p25.alpha} hl={b_p25.hl:.0f}")
for lab, b in (('по медиане', b_med), ('по p25', b_p25)):
    pos = pd.Series(np.clip(b.mp + b.alpha*z5, 0, 2)).ewm(halflife=int(b.hl)).mean().values
    md, wr, p, N = stat(pos, LATE, n)
    print(f"    {lab:12s} на отложенных последних 10 годах: мед.разн={md:+.3f} побед={wr*100:.0f}% p={p:.3f}")
