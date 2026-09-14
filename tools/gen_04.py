from nb_common import FOLDS, LOAD, SETUP, build

cells = [
    ("md", """\
# Stage 2 - Minimal Requirement 2: Proposed Model
## Hybrid single model: 4th-place domain features + 61st-place gradient-boosting pipeline

**Notebook 4 of 5.**

### The proposal in one sentence
Keep the *learning* machinery deliberately simple - **one** LightGBM regressor, tuned by
rank correlation, exactly as in the 61st-place pipeline - and spend the modelling budget on
**non-learned domain features** derived from the 4th-place solution, whose central claim is
that portfolio-relevant structure (short-horizon mean reversion, volatility term structure,
inverse-volatility blending) beats marginal gains in raw predictive accuracy.

### Why this combination should help
| Public solution | What it got right | What it left on the table |
|---|---|---|
| 61st place | strong ML hygiene: temporal context, rank-based tuning, one model, robust binary sizing | features are generic lag/rolling transforms of anonymised columns - no financial structure |
| 4th place | financially motivated, hand-built signals; **no ML at all** | a rule-based alpha cannot exploit interactions or non-linearities between signals and regimes |

The hybrid feeds the second column's signals into the first column's pipeline: the domain
features supply economically meaningful inputs, and the gradient-boosted trees learn the
non-linear regime dependence that a fixed rule cannot express.

---
### Attribution
* Domain-feature concepts: public 4th-place write-up,
  <https://kaggle.com/competitions/hull-tactical-market-prediction/writeups/4th-place-technical-model-no-learning-short-te>
  (short-horizon mean-reversion alpha, inverse-volatility weighting of several signals,
  volatility-targeting overlay, "a feature does not need to predict returns to improve a
  strategy"). The public solution's exact alpha construction was never disclosed; the
  indicators here are an independent implementation of the described *family* of signals.
* Pipeline concepts: public 61st-place notebook / write-up
  <https://www.kaggle.com/code/rafanikitas/hull-eda-training-pipeline>
  (temporal expansion of selected columns, single LightGBM, Optuna on mean Spearman rank
  correlation across chronological folds, binary allocation).

No code was copied verbatim from either source; both are re-implemented on top of `src/`.
Evaluation uses the identical purged folds as every other notebook in this project.
"""),
    ("code", SETUP),
    ("code", LOAD),
    ("code", FOLDS),
    ("md", """\
## 1. Domain features (non-learned, 4th-place inspired)

All indicators are built from the **realised** return series, `target.shift(1)`, which is
the last return actually observable at decision time:

* `reversal_k` - negative of the k-day cumulative return scaled by its volatility
  (k = 2, 3, 5, 10): the short-horizon mean-reversion family;
* `momentum_k` - volatility-scaled mean return over 20 / 60 / 120 days;
* `rv_k`, `vol_ratio_5_60`, `vol_ratio_20_60`, `vol_of_vol_20` - realised-volatility level,
  term structure and instability;
* `dist_ma_k`, `drawdown_120`, `breadth_14` - path statistics;
* `alpha_invvol_blend` - the reversal signals combined with **inverse-volatility weights**
  `w_i = sigma_i^-1 / sum_j sigma_j^-1`. No optimisation, no covariance matrix, automatic
  down-weighting of unstable signals - the 4th-place argument for robustness over
  optimality;
* `vol_regime`, `alpha_x_regime` - the blended alpha interacted with the volatility regime,
  which is the part a fixed rule cannot express and the trees can."""),
    ("code", '''\
dom_df, dom_cfg = F.add_domain_features(df, target_col=TARGET)
DOMAIN_COLS = [c for c in dom_cfg.signal_names]
print(f"{len(DOMAIN_COLS)} domain features:")
print(", ".join(DOMAIN_COLS))

ic_table = (
    pd.Series({c: M.spearman_ic(dom_df[TARGET], dom_df[c]) for c in DOMAIN_COLS})
    .sort_values(key=np.abs, ascending=False).round(4)
)
print("\\nfull-sample rank IC of each domain feature (diagnostic only, not a selection step):")
display(ic_table.to_frame("spearman_ic").head(12))
'''),
    ("md", """\
## 2. The hybrid design matrix

Three blocks stacked into one matrix for one model:

1. raw anonymised features (breadth),
2. 61st-place temporal expansion of the 14 selected columns (history),
3. 4th-place domain indicators (financial structure)."""),
    ("code", '''\
COLS_TO_DROP = ["E7", "V10", "S3", "M1", "M14"]
hybrid = F.add_temporal_features(
    dom_df.drop(columns=[c for c in COLS_TO_DROP if c in dom_df.columns]),
    cols=F.TOP_FEATURES_FOR_FE, lags=F.LAG_PERIODS, windows=F.ROLLING_WINDOWS,
)

FEATURES = F.model_feature_columns(hybrid)
X = hybrid[FEATURES].copy()
y = hybrid[TARGET].copy()

raw_cols = [c for c in feature_columns(df) if c in FEATURES]
temporal_cols = [c for c in FEATURES if "_lag_" in c or "_roll_" in c]
domain_cols = [c for c in DOMAIN_COLS if c in FEATURES]
print(f"raw: {len(raw_cols)} | temporal: {len(temporal_cols)} | domain: {len(domain_cols)} | total: {X.shape[1]}")
'''),
    ("md", """\
## 3. Ablation: does each block earn its place?

Fixed, modest LightGBM settings are used here so the comparison isolates the *features*.
All three variants run on the identical purged folds."""),
    ("code", '''\
import lightgbm as lgb

ABLATION_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, max_depth=6, num_leaves=31,
    min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=1.0, random_state=SEED, verbosity=-1, n_jobs=-1,
)

blocks = {
    "A: raw only": raw_cols,
    "B: raw + temporal (61st)": raw_cols + temporal_cols,
    "C: raw + temporal + domain (hybrid)": raw_cols + temporal_cols + domain_cols,
    "D: domain only (4th-place signals)": domain_cols,
}

ablation = []
for label, cols in blocks.items():
    r = run_cv(
        X[cols], y, market, risk_free,
        model_factory=lambda: lgb.LGBMRegressor(**ABLATION_PARAMS),
        allocator=A.binary_allocation, name=label, stage="2-ablation", cv=cv, verbose=False,
    )
    ablation.append({"variant": label, "n_features": len(cols), **{k: r.metrics[k] for k in
                     ("spearman_ic", "rmse", "penalised_sharpe", "sharpe", "vol_ratio_vs_market")}})

display(pd.DataFrame(ablation).set_index("variant").round(4))
'''),
    ("md", """\
## 4. Tuning the single hybrid model

Same protocol as the 61st-place baseline, so the comparison is fair: Optuna, Spearman rank
IC, inner purged CV, **development region only** (first 60% of rows), then frozen.

Two deliberate departures, both aimed at generalisation rather than at the search score:

* the objective is **stability-aware**, `mean(IC) - 0.5 * std(IC)` across inner folds, so a
  configuration that wins one fold and loses another is not preferred over a consistent one;
* the search space is narrower and biased towards regularisation (shallower trees, larger
  leaves, lower learning rates), because the ablation above already shows that the signal
  is small and easy to overfit."""),
    ("code", '''\
import optuna
from src.features import LeakSafeImputer
from src.validation import default_cv as _cv

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEV_END = int(0.6 * len(X))
X_dev, y_dev = X.iloc[:DEV_END], y.iloc[:DEV_END]
inner_cv = _cv(n_splits=3, purge=1, embargo=10)
N_TRIALS = 12 if QUICK else 60


def objective(trial):
    params = dict(
        objective="regression", metric="rmse",
        n_estimators=trial.suggest_int("n_estimators", 200, 900),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        num_leaves=trial.suggest_int("num_leaves", 8, 64, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 40, 300),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 0.9),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1, random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    scores = []
    for tr, va in inner_cv.split(X_dev):
        imp = LeakSafeImputer().fit(X_dev.iloc[tr])
        model = lgb.LGBMRegressor(**params).fit(imp.transform(X_dev.iloc[tr]), y_dev.iloc[tr])
        scores.append(M.spearman_ic(y_dev.iloc[va], model.predict(imp.transform(X_dev.iloc[va]))))
    scores = np.asarray(scores, dtype=float)
    return float(np.nanmean(scores) - 0.5 * np.nanstd(scores))   # stability-aware objective


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
BEST_PARAMS = dict(study.best_params, random_state=SEED, verbosity=-1, n_jobs=-1, subsample_freq=1)
print(f"best stability-adjusted inner-CV rank IC: {study.best_value:+.4f}")
print(json.dumps(study.best_params, indent=2, default=str))
'''),
    ("md", """\
## 5. Evaluation of the proposed model

The headline configuration keeps the binary allocation of the 61st-place pipeline, so any
difference against that baseline is attributable to the features, not to sizing. A second
configuration adds the 4th-place **volatility-targeting overlay**, in which exposure is
rescaled so that expected strategy volatility sits just under the metric's 120% ceiling."""),
    ("code", '''\
res_hybrid = run_cv(
    X, y, market, risk_free,
    model_factory=lambda: lgb.LGBMRegressor(**BEST_PARAMS),
    allocator=A.binary_allocation,
    name="PROPOSED hybrid (domain + temporal, single LGBM)", stage="2-proposed", cv=cv,
)
save_result(res_hybrid)
'''),
    ("code", '''\
# --- variant with the 4th-place volatility-targeting overlay --------------------------
from src.experiment import aggregate_folds

fold_rows = []
for k, (tr, va) in enumerate(cv.split(X), start=1):
    imp = LeakSafeImputer().fit(X.iloc[tr])
    model = lgb.LGBMRegressor(**BEST_PARAMS).fit(imp.transform(X.iloc[tr]), y.iloc[tr])
    pred = model.predict(imp.transform(X.iloc[va]))

    # volatility estimate: long, slowly updated window (deliberately not reactive)
    vol_hat = X["rv_60"].iloc[va].to_numpy() if "rv_60" in X.columns else None
    vol_hat = np.where(np.isfinite(vol_hat), vol_hat, np.nanmedian(vol_hat))
    market_vol = float(np.nanstd(y.iloc[tr].to_numpy(), ddof=1))   # training-fold estimate only

    pos = A.vol_target_allocation(pred, vol_hat, market_vol, k=100.0)
    pos = A.smooth_positions(pos, alpha=0.75, transaction_cost=3e-5)

    mkt = market.iloc[va].to_numpy()
    rf = None if risk_free is None else risk_free.iloc[va].to_numpy()
    fold_rows.append({"fold": k, "n_train": len(tr), "n_valid": len(va),
                      **M.evaluate_predictions(y.iloc[va].to_numpy(), pred),
                      **M.evaluate_positions(pos, mkt, rf)})
    print(f"  fold {k}: penalised Sharpe={fold_rows[-1]['penalised_sharpe']:+.3f} "
          f"mean w={fold_rows[-1]['mean_position']:.2f} vol/market={fold_rows[-1]['vol_ratio_vs_market']:.2f}")

res_hybrid_vt = aggregate_folds(
    "PROPOSED hybrid + vol targeting", fold_rows, stage="2-proposed",
    notes="4th-place volatility-targeting overlay anchored to the 120% ceiling",
)
save_result(res_hybrid_vt)
print(f"  >>> penalised Sharpe={res_hybrid_vt.metrics['penalised_sharpe']:+.3f}")
'''),
    ("md", "## 6. Which features does the hybrid actually rely on?"),
    ("code", '''\
imp = LeakSafeImputer().fit(X.iloc[:DEV_END])
final_model = lgb.LGBMRegressor(**BEST_PARAMS).fit(imp.transform(X.iloc[:DEV_END]), y.iloc[:DEV_END])
gain = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)

block_of = lambda c: ("domain" if c in domain_cols else ("temporal" if c in temporal_cols else "raw"))
share = gain.groupby(gain.index.map(block_of)).sum() / max(gain.sum(), 1e-9)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
top = gain.head(20)[::-1]
colors = {"domain": "crimson", "temporal": "seagreen", "raw": "steelblue"}
ax[0].barh(top.index, top.to_numpy(), color=[colors[block_of(c)] for c in top.index])
ax[0].set_title("Top 20 features (red = domain, green = temporal, blue = raw)")
ax[1].bar(share.index, share.to_numpy(), color=[colors[b] for b in share.index])
ax[1].set_title("Share of total importance by feature block")
for a in ax:
    a.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.show()
print((share * 100).round(1).to_string())
'''),
    ("md", "## 7. Comparison against Stage 1"),
    ("code", "display(comparison_table())"),
    ("md", """\
### Reading of the results

* The ablation isolates the contribution of the domain block on identical folds: if variant
  C beats variant B, the 4th-place signals add information the generic lag/rolling
  transforms do not carry, at a cost of only a couple of dozen extra columns.
* Variant D (domain features alone, ~25 columns) is the most interesting diagnostic: a very
  small, economically motivated matrix that can rival matrices an order of magnitude larger
  is a direct empirical echo of the 4th-place thesis.
* The importance chart shows how much of the model's attention the domain block attracts
  relative to its size - the honest way to judge whether the hybrid is more than an
  concatenation of two public solutions.
* Remaining weakness: the sizing rule is still fixed and the model is a single learner.
  Stage 3 addresses both."""),
]

build("../proposed_model/proposed_hybrid_model.ipynb", cells)
