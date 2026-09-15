## TL;DR
A streaming ensemble (ElasticNet 0.30 / XGBoost 0.35 / LightGBM 0.35) predicting the winsorised excess return target,
with a volatility-regime-aware position sizer that explicitly anchors to the metric's 120% vol ceiling. Online
retraining every row, narrow-band hyperopt every 50 rows. Sliding 800-row training window. \
**The key insight: the metric is the model**
The official metric is a penalised Sharpe, accounting for volatility and return.
Three things shape everything else:

- Vol is a cliff, not a target. No reward for 119% vs 50% market vol — only punishment above 120%.
- Underperformance is quadratic. Being below buy-and-hold hurts super-linearly; being above is free.
- Long-only, $[0, 2]$. A passive $w=1$ is a strong baseline.
So the entire pipeline is engineered around not triggering the penalties while extracting a small edge.

## What worked
1. Predict market_forward_excess_returns, not raw forward_returns. The target is already de-meaned (5y rolling) and
MAD-winsorised. Predicting it directly removes drift GBDTs handle poorly and tames the tails that dominate MSE. Single
biggest stability win.
2. 800-row sliding window. Decades of history dilute recent signal and pre-2000 rows are mostly null. ~3 years of
trading days adapts to regime shifts without drowning in stale regimes.
3. Position sizer anchored to 1.2. The position is $w = \text{clip} (s / (\hat{\sigma} \cdot 1.2), 0, 2)$. The 1.2 is
the same 1.2 as in the metric — we target strategy vol just below the penalty cliff. This is the most important design
choice in the pipeline.
4. Vol-regime multiplier. A binary split: multiplier 600 in low-vol ($V1 < V1_{\text{median}}$), 400 in high-vol.
Low-vol: underinvestment penalty dominates, lean in. High-vol: vol ceiling binds, dampen. Crude but effective.
5. GARCH-blended vol estimate. $\hat{\sigma} = \sqrt{0.3 \cdot \text{Var}(\text{returns}_{20d}) + 0.7 \cdot V1^2}$. V1
is forward-looking but noisy; realised vol is robust but lagging. Blend beats either alone.
6. Two-tier hyperopt. Heavy Optuna search at startup (30 trials × 5-fold TSCV, cached to pickle). Every 50 rows:
10-trial, 15s-timeout narrow-band search around current best (e.g. lr ∈ [0.8×, 1.2×], depth ± 1). A controlled random
walk through HP space that tracks regime shifts without blowing the time budget.
7. Fixed ensemble weights. Stacking with a Ridge meta-learner overfit the ~135-row CV folds. Fixed 0.30/0.35/0.35 was
strictly better on this sample size.
8. 75/25 EWM smoothing + 0.003% transaction cost haircut. Cuts turnover, acts as a low-pass filter on the signal.
Almost always a net Sharpe win.
Feature engineering (the bits that survived)
- U1 = I2 - I1 (term-structure spread) and U2 = M11 / mean(I2, I9, I7) (rate-normalised market dynamic). U2 ended up
the #1 feature by XGB gain.
- Cross products V1×S1, M11×V1, I9×S1 — give the linear model interaction terms the trees would otherwise monopolise.
- Top-50 by XGB gain selection from the ~85 base + engineered set. Diverse across E/P/S/V/M/I families — no single
family carries the signal.
Online loop
Each predict() call appends the previous test row (using lagged_market_forward_excess_returns as the target) to the
training set, trims to 800 rows, refits all three models, and predicts. Every 50th row triggers Tier-2 hyperopt. The
derived features on appended rows are computed via the test feature pipeline (is_train=False) to avoid train/test
skew.
## What didn't work
- Training-only rolling/lag features (S1_lag1, V1_roll_*, V1_S1_corr_10) — useful offline but the API serves one row
at a time, so they're dropped at inference. A stateful ring buffer inside predict() would fix this; left for v2.
- Per-row full hyperopt — blew the time budget. 50-row cadence was the compromise.
- CatBoost / extra XGB seeds — added runtime, not enough diversity to justify.
- Targeting raw forward_returns — far noisier positions; the de-meaned target is materially easier to learn.
## One-line takeaway
The edge is real but fragile — it lives in a narrow band of volatility scaling, ensemble weight, and retrain cadence,
and dissolves quickly if any dial is turned too far. The two-tier hyperopt and the sliding window are, in a sense, the
actual model; the trees just produce the next number.