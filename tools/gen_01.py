from nb_common import FOLDS, LOAD, SETUP, build

cells = [
    ("md", """\
# Stage 1 - Minimal Requirement 1: Simple Baselines
## Hull Tactical Market Prediction (PMLDL 2026)

**Notebook 1 of 5.** Establishes the three reference points every later model must beat:

1. **Buy & Hold** - constant full market exposure (`w = 1`). Under the competition's
   penalised-Sharpe metric this is deliberately hard to beat.
2. **ElasticNet** - regularised linear model on the raw anonymised features.
3. **LightGBM (default parameters)** - untuned gradient boosting on the same inputs.

---
### Attribution
The leak-safe baseline structure (constant benchmark + linear model + untuned GBDT,
evaluated with strictly chronological validation) follows the publicly shared Kaggle
notebook *"Hull leak-safe baseline"*:
<https://www.kaggle.com/code/morodertobias/hull-leak-safe-baseline>

No code was copied verbatim. The public notebook was used as a **conceptual reference**
for what a leak-free starter pipeline must contain; all code here is an independent
re-implementation on top of this project's `src/` modules.

### Compliance notes
* Random seed fixed at `42` everywhere (`src.config.set_seed`).
* No author names, team identifiers or personal data appear anywhere in this repository.
* Validation is a **purged, embargoed** time-series split shared by all five notebooks,
  so every model in the project is compared on mathematically identical folds.
"""),
    ("code", SETUP),
    ("md", """\
## 1. Data

`src.data.load_dataset` loads `data/train.csv` when the Kaggle file is available. If it is
not (no Kaggle credentials in the grading environment), a **deterministic synthetic
surrogate** with the same schema, the same target definition and realistic market dynamics
is generated instead, so the whole project stays executable end-to-end. The printout below
states which mode is active - all reported numbers must be read in that light."""),
    ("code", LOAD),
    ("code", '''\
# --- compact EDA ---------------------------------------------------------------------
feat_cols = feature_columns(df)
fam = pd.Series([c[0] for c in feat_cols]).value_counts().sort_index()
miss = df[feat_cols].isna().mean().sort_values(ascending=False)

print(f"rows: {len(df):,} | anonymised features: {len(feat_cols)}")
print("feature families:", dict(fam))
print(f"columns with missing values: {(miss > 0).sum()} (worst: {miss.iloc[0]:.1%})")
print(df[TARGET].describe()[["mean", "std", "min", "max"]].to_string())

fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
ax[0].plot(np.cumsum(df[TARGET].to_numpy()), lw=1)
ax[0].set_title("Cumulative market excess return")
ax[1].plot(df[TARGET].rolling(60).std().to_numpy() * np.sqrt(252), lw=1, color="darkred")
ax[1].set_title("Rolling 60d annualised volatility")
ax[2].hist(df[TARGET].dropna(), bins=80, color="steelblue")
ax[2].set_title("Target distribution")
for a in ax:
    a.grid(alpha=.3)
plt.tight_layout(); plt.show()
'''),
    ("md", """\
## 2. Leak-safe validation protocol

Row *t* carries the return realised at *t+1*, and the engineered features of later
notebooks are rolling windows ending at *t*. Two leakage channels therefore have to be
closed:

| channel | fix |
|---|---|
| a training label overlaps the validation block | **purge** the last row(s) of the training window |
| rolling features of early validation rows overlap the training window | **embargo** a further 10 rows |

`PurgedTimeSeriesSplit` implements both. The `gap` column below shows the number of rows
physically removed between each training window and its validation block."""),
    ("code", FOLDS),
    ("code", '''\
# --- model inputs: raw anonymised features only (no engineering in Stage 1) ----------
X = df[feat_cols].copy()
y = df[TARGET].copy()
print("design matrix:", X.shape)
results = []
'''),
    ("md", """\
## 3. Baseline A - Buy & Hold

A trivial "model" that always predicts zero and is always fully invested. It is scored on
exactly the same folds as everything else, so its Sharpe, volatility and drawdown are
directly comparable."""),
    ("code", '''\
class ConstantPredictor:
    """Predicts a constant; used so the benchmark passes through the identical CV harness."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


res_bh = run_cv(
    X, y, market, risk_free,
    model_factory=lambda: ConstantPredictor(0.0),
    allocator=lambda p: np.ones_like(np.asarray(p, dtype=float)),
    name="Buy & Hold", stage="1-baseline", cv=cv,
)
results.append(res_bh)
'''),
    ("md", """\
## 4. Baseline B - ElasticNet

Standardisation and imputation are fitted **inside each training fold** (`LeakSafeImputer`
+ `StandardScaler`), never on the full series. Positions use the binary policy
(`0` = cash, `1` = market) so that all learned baselines share one sizing rule and
differences in the portfolio metrics come from the forecast, not from leverage.

(The rank IC of Buy & Hold is undefined - a constant forecast has no ranking - so it is
reported as `NaN` in the tables.)"""),
    ("code", '''\
from sklearn.linear_model import ElasticNet

res_enet = run_cv(
    X, y, market, risk_free,
    model_factory=lambda: ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=50_000, random_state=SEED),
    allocator=A.binary_allocation,
    name="ElasticNet (raw features)", stage="1-baseline", cv=cv, scale=True,
)
results.append(res_enet)
'''),
    ("md", "## 5. Baseline C - LightGBM with default parameters"),
    ("code", '''\
import lightgbm as lgb

res_lgb = run_cv(
    X, y, market, risk_free,
    model_factory=lambda: lgb.LGBMRegressor(random_state=SEED, verbosity=-1, n_jobs=-1),
    allocator=A.binary_allocation,
    name="LightGBM (default)", stage="1-baseline", cv=cv,
)
results.append(res_lgb)
'''),
    ("md", """\
## 6. Results

Every result is persisted to `results/*.json`; the final notebook of the project
(`model_improvements/improved_ensemble_optuna.ipynb`) rebuilds one comparison table from
these files."""),
    ("code", '''\
for r in results:
    save_result(r)

table = comparison_table(results)
display(table)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
names = [r.name for r in results]
ax[0].barh(names, [r.metrics["penalised_sharpe"] for r in results], color="steelblue")
ax[0].set_title("Penalised Sharpe (mean over folds)")
ax[1].barh(names, [r.metrics["spearman_ic"] for r in results], color="darkorange")
ax[1].set_title("Spearman rank IC (mean over folds)")
for a in ax:
    a.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.show()
'''),
    ("md", """\
### Reading of the Stage-1 results

* Raw anonymised features alone give a very small rank IC - the expected outcome in a
  problem with a signal-to-noise ratio this low.
* The untuned GBDT does **not** automatically beat Buy & Hold on the penalised metric:
  turning a weak forecast into a position is where most of the score is won or lost.
* This motivates the two strong public baselines (notebooks 02 and 03), the domain-feature
  hybrid (Stage 2) and the ensemble with explicit position sizing (Stage 3)."""),
]

build("../baselines/01_simple_baselines.ipynb", cells)
