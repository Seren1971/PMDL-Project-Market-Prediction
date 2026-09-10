"""H3: vol-targeting даёт прирост скора против константы."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from data import load_train, known_returns, feature_cols
from wf import walk_forward
from backtest import compare, fmt, window_scores, summary

tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd)
H = 21
y = np.log(np.sqrt(pd.Series(fwd**2).rolling(H).mean().shift(-(H-1)).values) + 1e-8)
s = pd.Series(np.abs(rk))
har = np.log(np.c_[s.rolling(1).mean(), s.rolling(5).mean(), s.rolling(22).mean()] + 1e-8)

# прогноз волы (walk-forward, всё причинно)
pred_log, _ = walk_forward(har, y)
sig_model = np.exp(pred_log)

# наивная альтернатива: просто скользящая реализованная вола по известным доходностям
sig_naive22 = pd.Series(rk**2).rolling(22).mean().pow(0.5).values
sig_naive60 = pd.Series(rk**2).rolling(60).mean().pow(0.5).values

def vol_target(sig, mean_pos=1.0, cap=2.0, warm=2000):
    """pos = c * sigma_target / sigma_hat. sigma_target и c оцениваются ТОЛЬКО по прошлому:
    на каждый день берём расширяющуюся медиану sigma до вчерашнего дня."""
    st = pd.Series(sig).expanding(250).median().shift(1).values
    raw = st / sig
    # нормировка так, чтобы СРЕДНЯЯ позиция по прошлому равнялась mean_pos
    scale = pd.Series(raw).expanding(250).mean().shift(1).values
    pos = mean_pos * raw / scale
    pos = np.clip(pos, 0, cap)
    pos[:warm] = mean_pos
    pos[np.isnan(pos)] = mean_pos
    return pos

START = 2300   # первые предсказания появляются с 2000, даём запас на разогрев нормировки
print("=== H3: vol-targeting против константы, парное сравнение по окнам 180 дней ===\n")

results = []
for mp in (0.8, 1.0, 1.2):
    base = np.full(n, mp)
    for name, sig in (('модель HAR', sig_model), ('наив 22д', sig_naive22), ('наив 60д', sig_naive60)):
        pos = vol_target(sig, mean_pos=mp)
        r, m = compare(pos, base, fwd, rf, width=180, step=30, start=START,
                       label=f'volTgt {name} @{mp}')
        results.append(r); print("  " + fmt(r))
    print()

print("=== средняя позиция и её оборот ===")
for name, sig in (('модель HAR', sig_model), ('наив 22д', sig_naive22), ('наив 60д', sig_naive60)):
    p = vol_target(sig, 1.0)[START:]
    print(f"  {name:12s} средняя={p.mean():.3f} std={p.std():.3f} "
          f"оборот|dpos|={np.abs(np.diff(p)).mean():.4f} доля на клипе 2.0={np.mean(p>=1.999):.3f}")

print("\n=== скор на всей истории целиком (не по окнам) ===")
from metric import hull_sharpe
sl = slice(START, n)
for mp in (0.8, 1.0, 1.2):
    c = hull_sharpe(np.full(n-START, mp), fwd[sl], rf[sl])
    print(f"  константа {mp}: score={c['score']:.4f}")
    for name, sig in (('модель HAR', sig_model), ('наив 22д', sig_naive22), ('наив 60д', sig_naive60)):
        p = np.clip(vol_target(sig, mp)[sl], 0, 2)
        r = hull_sharpe(p, fwd[sl], rf[sl])
        print(f"    volTgt {name:12s} score={r['score']:.4f} sharpe={r['sharpe']:.3f} "
              f"vol_ratio={r['vol_ratio']:.2f} ret_gap={r['return_gap']:.2f}")

print("\n=== вердикт H3 ===")
best = max(results, key=lambda r: r['median_diff'])
print(f"  лучший вариант: {best['label']}")
print(f"  побед {best['win_rate']*100:.1f}% окон, разница медиан {best['median_diff']:+.3f}, p={best['sign_test_p']:.4f}")
ok = best['median_diff'] > 0 and (1 - best['win_rate']) < 0.40
print("  H3 ПОДТВЕРЖДЕНА" if ok else "  H3 НЕ ПОДТВЕРЖДЕНА по заявленному критерию")
