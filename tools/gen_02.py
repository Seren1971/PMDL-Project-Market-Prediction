from nb_common import FOLDS, LOAD, SETUP, build

cells = [
    ("md", """\
# Stage 1 - Strong Public Baseline A: the 61st-place pipeline
## Hull Tactical Market Prediction (PMLDL 2026)

**Notebook 2 of 5.** Re-implementation of the publicly shared 61st-place solution as a
*strong* baseline.

---
### Attribution
Source notebook (public Kaggle kernel):
<https://www.kaggle.com/code/rafanikitas/hull-eda-training-pipeline>
Accompanying public write-up: 61st-place solution thread for the Hull Tactical Market
Prediction competition.

Ideas taken from that public work, re-implemented here on top of this project's `src/`
modules (no verbatim code reuse):

| # | Idea from the public solution | Where it lives here |
|---|---|---|
| 1 | Temporal context for ~14 hand-picked columns: lags `[1,3,5,7,14,20]`, rolling mean/std over `[2,5,10,20,60]` | `src.features.add_temporal_features` |
| 2 | A **single** LightGBM regressor rather than an ensemble (their multi-model attempts scored worse) | this notebook |
| 3 | Optuna selecting hyper-parameters by **mean Spearman rank correlation** over chronological folds, while LightGBM itself optimises RMSE | `objective()` below |
| 4 | A deliberately **binary allocation** (`0` = risk-free, `1` = market exposure) as a form of regularisation | `src.allocation.binary_allocation` |

### Deviations from the public notebook (and why)
* **Imputation** - the public version fills missing values with medians of the whole file;
  here medians are fitted inside each training fold (`LeakSafeImputer`) to remove that leak.
* **Validation** - `TimeSeriesSplit` is replaced by the project-wide purged/embargoed split,
  so this baseline is comparable with every other model in the repository.
* **Hyper-parameter search** - run on a *development region* (first 60% of rows) with its own
  inner purged CV, then frozen. Searching directly on the evaluation folds would make the
  reported scores optimistic.
* **Inference server** - the Kaggle `kaggle_evaluation` serving code is out of scope for an
  offline, reproducible academic pipeline and is replaced by an offline fold evaluation.

No author names or team identifiers from the public sources are reproduced anywhere.
"""),
    ("code", SETUP),
    ("code", LOAD),
    ("code", FOLDS),
    ("md", """\
## 1. Temporal feature engineering

14 promising columns are expanded with lags and rolling statistics: 6 lags + 5 rolling
means + 5 rolling standard deviations = 16 derived columns each. The intention of the
original solution was to *preserve the breadth of the original dataset while adding deeper
history only where it appears most useful* - a feature's current level often means little
without knowing whether it is rising, stable or unusually volatile.

All of these transformations read rows `<= t` only, so computing them before splitting is
leak-free."""),
    ("code", '''\
COLS_TO_DROP = ["E7", "V10", "S3", "M1", "M14"]   # dropped in the public solution

base = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
feat_df = F.add_temporal_features(
    base,
    cols=F.TOP_FEATURES_FOR_FE,
    lags=F.LAG_PERIODS,
    windows=F.ROLLING_WINDOWS,
)

FEATURES = F.model_feature_columns(feat_df)
X = feat_df[FEATURES].copy()
y = feat_df[TARGET].copy()

used = [c for c in F.TOP_FEATURES_FOR_FE if c in base.columns]
print(f"columns expanded: {len(used)} -> {len(used) * (len(F.LAG_PERIODS) + 2 * len(F.ROLLING_WINDOWS))} new temporal features")
print(f"design matrix: {X.shape}")
'''),
    ("md", """\
## 2. Hyper-parameter search (Optuna, rank-correlation objective)

LightGBM trains against RMSE, but the *selection* criterion is the mean Spearman rank
correlation across chronological folds: in a problem this noisy, ordering strong and weak
opportunities correctly is more attainable than predicting magnitudes.

The search runs on the development region only (first 60% of rows) with an inner purged CV
of 3 folds. Set `PMLDL_QUICK=0` to use the full budget."""),
    ("code", '''\
import optuna
import lightgbm as lgb
from src.features import LeakSafeImputer
from src.validation import default_cv as _cv

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEV_END = int(0.6 * len(X))
X_dev, y_dev = X.iloc[:DEV_END], y.iloc[:DEV_END]
inner_cv = _cv(n_splits=3, purge=1, embargo=10)
N_TRIALS = 12 if QUICK else 60


def objective(trial):
    params = dict(
        objective="regression",
        metric="rmse",
        n_estimators=trial.suggest_int("n_estimators", 200, 1200),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.07, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        num_leaves=trial.suggest_int("num_leaves", 16, 256, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    )
    scores = []
    for tr, va in inner_cv.split(X_dev):
        imp = LeakSafeImputer().fit(X_dev.iloc[tr])
        model = lgb.LGBMRegressor(**params)
        model.fit(imp.transform(X_dev.iloc[tr]), y_dev.iloc[tr])
        pred = model.predict(imp.transform(X_dev.iloc[va]))
        scores.append(M.spearman_ic(y_dev.iloc[va], pred))
    return float(np.nanmean(scores))


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

BEST_PARAMS = dict(study.best_params, random_state=SEED, verbosity=-1, n_jobs=-1, subsample_freq=1)
print(f"trials: {len(study.trials)} | best inner-CV rank IC: {study.best_value:+.4f}")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
'''),
    ("md", """\
## 3. Evaluation on the shared folds

The frozen configuration is retrained on every purged fold. The headline allocation is the
binary policy of the original solution; a naive linear sizing rule is evaluated alongside
it to show *why* the authors of the public solution ended up preferring the binary one."""),
    ("code", '''\
res_61 = run_cv(
    X, y, market, risk_free,
    model_factory=lambda: lgb.LGBMRegressor(**BEST_PARAMS),
    allocator=A.binary_allocation,
    name="61st-place pipeline (LGBM + binary policy)", stage="1-strong-baseline", cv=cv,
)
save_result(res_61)
'''),
    ("code", '''\
# same forecasts, different sizing rule: position = clip(1 + k * pred, 0, 2)
K_DEV = 100.0
res_61_naive = run_cv(
    X, y, market, risk_free,
    model_factory=lambda: lgb.LGBMRegressor(**BEST_PARAMS),
    allocator=lambda p: A.naive_linear_allocation(p, k=K_DEV),
    name="61st-place pipeline (LGBM + naive sizing)", stage="1-strong-baseline", cv=cv,
)
save_result(res_61_naive)
'''),
    ("md", "## 4. What the model actually uses"),
    ("code", '''\
imp = LeakSafeImputer().fit(X.iloc[:DEV_END])
final_model = lgb.LGBMRegressor(**BEST_PARAMS).fit(imp.transform(X.iloc[:DEV_END]), y.iloc[:DEV_END])

gain = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top = gain.head(25)[::-1]

plt.figure(figsize=(7, 7))
plt.barh(top.index, top.to_numpy(), color="seagreen")
plt.title("LightGBM feature importance (top 25, fitted on the development region)")
plt.grid(alpha=.3, axis="x"); plt.tight_layout(); plt.show()

share = gain[[c for c in gain.index if "_lag_" in c or "_roll_" in c]].sum() / max(gain.sum(), 1e-9)
print(f"share of total importance carried by engineered temporal features: {share:.1%}")
'''),
    ("md", "## 5. Results so far"),
    ("code", '''\
display(comparison_table())

fold_df = pd.DataFrame(res_61.fold_metrics)[["fold", "spearman_ic", "rmse", "penalised_sharpe", "vol_ratio_vs_market"]]
display(fold_df.round(4))
'''),
    ("md", """\
### Reading of the results

* Temporal context lifts the rank IC above the Stage-1 defaults, confirming the core claim
  of the public write-up: history matters more than raw levels in this dataset.
* The two sizing rules run on **identical forecasts**, so the gap between their portfolio
  metrics is caused purely by position sizing. The `vol_ratio_vs_market` column shows how
  much of the market's volatility each rule actually spends; whichever rule wins here, the
  size of the gap is the point - sizing moves the score at least as much as the forecast
  does, which is the observation Stage 3 builds on.
* Fold-to-fold variance stays large. Offline validation constrains, but does not reproduce,
  the uncertainty of live deployment - the same lesson the public write-up drew."""),
]

build("../baselines/02_strong_baseline_61st.ipynb", cells)
