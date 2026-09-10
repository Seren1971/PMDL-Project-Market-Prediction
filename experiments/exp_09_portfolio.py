"""H4, H9, H10, H11: портфельный слой поверх сигнала разворота."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns
from metric import hull_sharpe
from wf import walk_forward

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); START = 2300
S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x); return ((x - x.rolling(w).mean())/x.rolling(w).std()).values
z5 = -zc(S.rolling(5).sum().values)
z5 = np.nan_to_num(z5)

def wins(pos, width=180, step=30):
    pos = np.nan_to_num(pos, nan=1.0)
    return np.array([hull_sharpe(np.clip(pos[s:s+width],0,2), fwd[s:s+width], rf[s:s+width])['score']
                     for s in range(START, n-width+1, step)])
B = wins(np.full(n,1.0))
def rep(name, pos, extra=''):
    w = wins(pos); d = w - B
    N=len(d); k=int((d>0).sum()); p = sum(comb(N,i) for i in range(k,N+1))/2**N
    f = hull_sharpe(np.clip(np.nan_to_num(pos[START:],nan=1.0),0,2), fwd[START:], rf[START:])
    print(f"  {name:38s} мед.разн={np.median(d):+.3f} побед={k/N*100:5.1f}% p={p:.4f} "
          f"| ср.поз={np.nanmean(pos[START:]):.2f} vol_r={f['vol_ratio']:.2f} {extra}")
    return np.median(d)

print("=== H9: форма отображения сигнал -> позиция (сигнал ОДИН И ТОТ ЖЕ) ===")
b = 0.006
lin   = np.clip(1.0 + 0.25*z5, 0, 2)
tanh_ = np.tanh(np.clip(z5, -3, 3)) + 1.0
disc2 = np.where(z5 > 0, 1.0, 0.0)
disc3 = np.where(z5 > 0.5, 2.0, np.where(z5 < -0.5, 0.0, 1.0))
disc01_hi = np.where(z5 > 0, 1.3, 0.7)
rank  = 2.0 * pd.Series(z5).rolling(504, min_periods=250).rank(pct=True).fillna(0.5).values
for name, pos in (('линейное 1+0.25z, клип [0,2]', lin), ('tanh(z)+1', tanh_),
                  ('дискретно 0/1', disc2), ('дискретно 0/1/2', disc3),
                  ('дискретно 0.7/1.3', disc01_hi), ('ранг в [0,2] (скольз. 2 года)', rank)):
    rep(name, pos)

print("\n=== H10: средняя экспозиция ===")
for mp in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
    rep(f'линейное @ср.поз={mp}', np.clip(mp + 0.25*z5, 0, 2))

print("\n=== H11: сглаживание позиции ===")
lin_base = np.clip(1.0 + 0.25*z5, 0, 2)
for half in (1, 2, 3, 5, 10, 21):
    sm = pd.Series(lin_base).ewm(halflife=half).mean().values
    turn = np.abs(np.diff(sm[START:])).mean()
    rep(f'EMA сглаживание, полураспад {half}д', sm, f'оборот={turn:.4f}')
print(f"  {'без сглаживания':38s} оборот={np.abs(np.diff(lin_base[START:])).mean():.4f}")
for cap in (0.05, 0.1, 0.2, 0.5):
    p = lin_base.copy()
    for i in range(1, n):
        p[i] = np.clip(p[i], p[i-1]-cap, p[i-1]+cap)
    rep(f'лимит изменения {cap}/день', p, f'оборот={np.abs(np.diff(p[START:])).mean():.4f}')

print("\n=== H4: длина окна оценки волатильности (для vol-слоя) ===")
H = 21
y = np.log(np.sqrt(pd.Series(fwd**2).rolling(H).mean().shift(-(H-1)).values)+1e-8)
s2 = pd.Series(np.abs(rk))
print("  вариант A: прогноз волы из HAR с разной длиной таргета/переобучения")
for refit in (63, 126, 252, 504):
    har = np.log(np.c_[s2.rolling(1).mean(), s2.rolling(5).mean(), s2.rolling(22).mean()]+1e-8)
    pl,_ = walk_forward(har, y, refit=refit)
    sig = np.exp(pl)
    pos = np.clip(1.0 + 0.25*z5, 0, 2) * np.clip(np.nanmedian(sig)/sig, 0.6, 1.5)
    rep(f'разворот x обр.вола, переобуч {refit}д', pos)
print("  вариант B: простая реализованная вола разной длины окна")
for w in (10, 22, 60, 120, 250):
    sv = pd.Series(rk**2).rolling(w).mean().pow(0.5).values
    pos = np.clip(1.0 + 0.25*z5, 0, 2) * np.clip(np.nanmedian(sv)/sv, 0.6, 1.5)
    rep(f'разворот x обр.вола, окно {w}д', pos)
