"""H14: отличима ли итоговая стратегия от константы. Честно, на непересекающихся окнах."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from math import comb
from data import load_train, known_returns
from metric import hull_sharpe

rng = np.random.default_rng(1)
tr = load_train(); fwd = tr.forward_returns.values; rf = tr.risk_free_rate.values
rk = known_returns(fwd); n = len(fwd); cal = pd.read_csv('data/interim_calendar_estimate.csv')
S = pd.Series(rk)
def zc(x, w=252):
    x = pd.Series(x); return ((x-x.rolling(w).mean())/x.rolling(w).std()).values
z5 = np.nan_to_num(-zc(S.rolling(5).sum().values))

# кандидат: разворот 5д -> линейно -> средняя экспозиция 0.95 -> EMA сглаживание 5 дней
raw = np.clip(0.95 + 0.25*z5, 0, 2)
CAND = pd.Series(raw).ewm(halflife=5).mean().values
BASE = np.full(n, 1.0)

def nonoverlap(pos, lo, hi, width):
    return np.array([hull_sharpe(np.clip(np.nan_to_num(pos[s:s+width],nan=1.0),0,2),
                                 fwd[s:s+width], rf[s:s+width])['score']
                     for s in range(lo, hi-width+1, width)])

print("=== кандидат: разворот 5д, линейное отображение, ср.поз 0.95, EMA(5) ===")
for label, lo, hi in (('вся выборка 2300..конец', 2300, n),
                      ('первая половина 2300..6000', 2300, 6000),
                      ('вторая половина 6000..конец', 6000, n),
                      ('последние 10 лет 6500..конец', 6500, n)):
    for width in (130, 180):
        a = nonoverlap(CAND, lo, hi, width); b = nonoverlap(BASE, lo, hi, width)
        d = a-b; N=len(d); k=int((d>0).sum())
        p = sum(comb(N,i) for i in range(k,N+1))/2**N
        print(f"  {label:30s} окно {width}: побед {k}/{N} ({k/N*100:.0f}%) "
              f"мед.разн={np.median(d):+.3f} p={p:.4f}")

print("\n=== поправка на число перебранных нами конфигураций ===")
TRIALS = 60      # сигналы x отображения x параметры, честно посчитано по журналу экспериментов
a = nonoverlap(CAND, 2300, n, 180); b = nonoverlap(BASE, 2300, n, 180)
d = a-b; N=len(d); k=int((d>0).sum())
p = sum(comb(N,i) for i in range(k,N+1))/2**N
print(f"  сырой p={p:.5f}; после поправки на {TRIALS} попыток: p_adj={min(1, p*TRIALS):.4f}")
print(f"  порог 0.05 {'ПРОЙДЕН' if p*TRIALS < 0.05 else 'НЕ пройден'}")

print("\n=== блочный бутстрап на второй половине (где сигнал слабее) ===")
BL = 252; lo = 6000; nb = (n-lo)//BL
blocks = [(lo+i*BL, lo+(i+1)*BL) for i in range(nb)]
diffs=[]
for _ in range(3000):
    idx = rng.integers(0, nb, nb)
    sel = np.concatenate([np.arange(*blocks[i]) for i in idx])
    try:
        diffs.append(hull_sharpe(np.clip(CAND[sel],0,2), fwd[sel], rf[sel])['score']
                     - hull_sharpe(np.ones(len(sel)), fwd[sel], rf[sel])['score'])
    except ValueError: pass
dd = np.array(diffs)
print(f"  среднее={dd.mean():+.3f} 95%ДИ=[{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}] "
      f"P(>0)={(dd>0).mean()*100:.1f}% (блоков {nb})")

print("\n=== итоговые цифры кандидата ===")
for lo, hi, lab in ((2300, n, 'вся выборка'), (6000, n, 'вторая половина')):
    r = hull_sharpe(np.clip(CAND[lo:hi],0,2), fwd[lo:hi], rf[lo:hi])
    c = hull_sharpe(np.ones(hi-lo), fwd[lo:hi], rf[lo:hi])
    print(f"  {lab}: кандидат score={r['score']:.4f} sharpe={r['sharpe']:.3f} "
          f"vol_ratio={r['vol_ratio']:.2f} | константа score={c['score']:.4f}")
print(f"  средняя позиция={CAND[2300:].mean():.3f}, std={CAND[2300:].std():.3f}, "
      f"оборот={np.abs(np.diff(CAND[2300:])).mean():.4f}")
print(f"  доля дней на границах: pos<=0.01: {np.mean(CAND[2300:]<=0.01)*100:.2f}%, "
      f"pos>=1.99: {np.mean(CAND[2300:]>=1.99)*100:.2f}%")
