# Hull Tactical Market Prediction — PMLDL 2026, Stage 2

An end-to-end, reproducible machine-learning project for the Kaggle competition
[*Hull Tactical - Market Prediction*](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

The task: each trading day, predict the next day's **excess market return** and convert that
forecast into a portfolio allocation in `[0, 2]` (`0` = fully in the risk-free asset,
`1` = passive market exposure, `2` = double leverage). The competition score is a Sharpe
ratio that is **penalised when strategy volatility exceeds 120% of market volatility**, so
the problem is as much portfolio construction as forecasting.

---

## 1. Repository layout

```
project_root/
├── data/                                  # dataset placeholder + download instructions
│   └── README.md
├── src/                                   # reusable core modules
│   ├── __init__.py
│   ├── config.py                          # SEED=42, paths, column conventions
│   ├── data.py                            # loader + deterministic synthetic surrogate
│   ├── metrics.py                         # RMSE, R2, Spearman IC, Sharpe, penalised Sharpe
│   ├── validation.py                      # PurgedTimeSeriesSplit (purge + embargo)
│   ├── features.py                        # temporal / derived / domain feature blocks
│   ├── allocation.py                      # binary, naive linear, vol-targeting sizing
│   └── experiment.py                      # shared CV runner + results registry
├── baselines/                             # Stage 1 - Minimal Requirement 1
│   ├── 01_simple_baselines.ipynb          # Buy & Hold, ElasticNet, default LightGBM
│   ├── 02_strong_baseline_61st.ipynb      # 61st-place public solution, re-implemented
│   └── 03_strong_baseline_100th.ipynb     # 100th-place public solution, re-implemented
├── proposed_model/                        # Stage 2 - Minimal Requirement 2
│   └── proposed_hybrid_model.ipynb        # single hybrid model: 4th-place features + 61st pipeline
├── model_improvements/                    # Stage 3 - Minimal Requirement 3
│   └── improved_ensemble_optuna.ipynb     # ensemble + Optuna + naive allocation
├── results/                               # auto-generated JSON metric records
├── tools/                                 # notebook generation / batch execution helpers
├── requirements.txt
└── README.md
```

`tools/` contains the scripts that generated the notebooks and the batch runner
(`tools/run_all.sh`). It is development scaffolding, not part of the deliverable pipeline —
the notebooks are self-contained and can be edited directly.

---

## 2. Quick start

```bash
# 1. environment (Python 3.11+ recommended)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. data (optional but preferred - see data/README.md)
kaggle competitions download -c hull-tactical-market-prediction -p data/
unzip -o data/hull-tactical-market-prediction.zip -d data/
#    without this step the notebooks run on a deterministic synthetic surrogate

# 3. run the notebooks IN ORDER
jupyter lab
```

Execution order matters, because each notebook appends its metrics to `results/` and the
last one builds the cross-stage comparison table from those files:

1. `baselines/01_simple_baselines.ipynb`
2. `baselines/02_strong_baseline_61st.ipynb`
3. `baselines/03_strong_baseline_100th.ipynb`
4. `proposed_model/proposed_hybrid_model.ipynb`
5. `model_improvements/improved_ensemble_optuna.ipynb`

### Headless reproduction

```bash
bash tools/run_all.sh          # clears results/ and executes all five notebooks in order
```

or one at a time:

```bash
cd baselines
PMLDL_QUICK=1 jupyter nbconvert --to notebook --execute --inplace 01_simple_baselines.ipynb
```

### Search budget

Every notebook reads the `PMLDL_QUICK` environment variable:

| value | meaning | approximate total runtime (8 cores, CPU only) |
|---|---|---|
| `1` (default) | reduced Optuna budgets — enough to reproduce the pipeline and the ranking | ~25–35 min for all five notebooks |
| `0` | full search budgets used for the reported tuning | ~3–5 hours |

```bash
export PMLDL_QUICK=0     # full budgets
```

---

## 3. Methodology

### 3.1 No data leakage

Row *t* carries the return realised at *t+1*, and engineered features at *t* are rolling
windows ending at *t*. Three leakage channels are closed explicitly:

| channel | mitigation |
|---|---|
| a training label overlapping the validation block | **purge** the last row(s) of each training window (`purge=1`, the label horizon) |
| rolling features of the first validation rows overlapping training rows | **embargo** a further 10 rows |
| statistics fitted on the whole series (medians, scalers, feature selection) | `LeakSafeImputer`, `StandardScaler` and top-k selection are **fitted inside the training fold only** |
| hyper-parameters chosen on the evaluation folds | all tuning happens on a **development region** (first 60% of rows) with its own inner purged CV, then the configuration is frozen |

`src.validation.default_cv()` is called with identical arguments in every notebook, so all
models — from Buy & Hold to the final ensemble — are scored on **mathematically identical
folds** (5 chronological blocks, 999 validation rows each, an 11-row gap before each block).
`describe_folds()` prints the layout in every notebook so this is verifiable at a glance.

Two public-notebook leaks were deliberately *not* reproduced: whole-file median imputation
and whole-file feature selection. Both are called out in the relevant notebooks.

### 3.2 Metrics

*Forecasting*: RMSE, out-of-sample R², and the **Spearman rank IC** — the criterion the
61st-place solution tuned on, since ordering opportunities is more attainable than
predicting magnitudes in a series this noisy.

*Portfolio*: annualised return and volatility, Sharpe, max drawdown, hit rate, turnover,
`vol_ratio_vs_market`, and **`penalised_sharpe`**:

```
score = sharpe(strategy) * min(1, 1.2 * vol_market / vol_strategy)
```

This is a documented **approximation** of the competition objective — the official
implementation is not reproduced here. It captures the property that matters for design:
volatility is a cliff, not a target. There is no reward for running at 119% of market
volatility rather than 50%, only a penalty above 120%.

### 3.3 Feature blocks

| block | source of the idea | contents |
|---|---|---|
| **raw** | dataset | ~95 anonymised columns across the D/E/I/M/P/S/V families |
| **temporal** | 61st-place solution | lags `[1,3,5,7,14,20]` and rolling mean/std over `[2,5,10,20,60]` for 14 hand-picked columns → 224 columns |
| **derived** | 100th-place solution | `U1 = I2 - I1`, `U2 = M11 / mean(I2, I9, I7)`, cross products `V1×S1`, `M11×V1`, `I9×S1` |
| **domain** | 4th-place solution | ~21 non-learned indicators: `reversal_{2,3,5,10}`, `momentum_{20,60,120}`, `rv_{5,20,60}`, volatility ratios, `vol_of_vol_20`, `dist_ma_{20,60}`, `drawdown_120`, `breadth_14`, the inverse-volatility blend `alpha_invvol_blend`, and the regime interaction `alpha_x_regime` |

All are strictly backward-looking (`shift`, `rolling`), so computing them before splitting
is leak-free.

### 3.4 Position sizing

Sizing is treated as a first-class, separately evaluated component (`src/allocation.py`):

* **binary** — `0` or `1`; a hard regulariser that refuses to read meaning into small
  forecast differences (61st place);
* **naive linear** — `clip(1 + k · prediction, 0, 2)`; one parameter, monotone, centred on
  the strong passive baseline (Stage 3's "safe post-processing");
* **volatility targeting** — rescale exposure so expected strategy volatility sits just
  below the 120% ceiling (4th place);
* **regime sizer** — vol-regime multiplier plus GARCH-blended volatility estimate and EWM
  smoothing (100th place).

---

## 4. The five notebooks

### Stage 1 — Baselines (`baselines/`)

**`01_simple_baselines.ipynb`** — Buy & Hold, ElasticNet and an untuned LightGBM on the raw
features, plus a compact EDA and the fold layout.
Reference: <https://www.kaggle.com/code/morodertobias/hull-leak-safe-baseline>

**`02_strong_baseline_61st.ipynb`** — the 61st-place pipeline: temporal expansion of 14
columns, a **single** LightGBM tuned by Optuna on mean Spearman IC, binary allocation. Both
the binary and the naive sizing rules are reported on identical forecasts to isolate the
effect of sizing.
Reference: <https://www.kaggle.com/code/rafanikitas/hull-eda-training-pipeline>

**`03_strong_baseline_100th.ipynb`** — the 100th-place streaming ensemble, offline: fixed-weight
ElasticNet 0.30 / XGBoost 0.35 / LightGBM 0.35, top-50 selection by XGBoost gain (refitted
per fold), GARCH-blended volatility, vol-regime multiplier, EWM smoothing. Evaluated with
both an expanding window and the 800-row sliding window of the original.
Reference: <https://www.kaggle.com/code/tingkaigong/hull-market-prediction-just-improved?scriptVersionId=285734689>

### Stage 2 — Proposed model (`proposed_model/proposed_hybrid_model.ipynb`)

A **single** hybrid model: the 4th-place *non-learned domain features* fed into the
61st-place *gradient-boosting pipeline*. The rationale is that the 61st-place solution has
good ML hygiene but financially generic features, while the 4th-place solution has
economically meaningful signals but no learner to model their regime dependence.

The notebook includes a four-way ablation on identical folds (raw / +temporal / +domain /
domain-only), a stability-aware Optuna objective (`mean(IC) − 0.5·std(IC)`) that prefers
consistent configurations over fold-lucky ones, feature-importance attribution by block,
and a variant with the 4th-place volatility-targeting overlay.
Reference: <https://kaggle.com/competitions/hull-tactical-market-prediction/writeups/4th-place-technical-model-no-learning-short-te>

### Stage 3 — Improvements (`model_improvements/improved_ensemble_optuna.ipynb`)

1. **Ensemble** of three inductive biases — LightGBM (leaf-wise), CatBoost (oblivious trees,
   ordered boosting), Ridge (linear).
2. **Optuna** for each family's hyper-parameters *and* for the blend weights and the
   allocation coefficient `k`, the latter two optimised directly against penalised Sharpe on
   development out-of-fold predictions.
3. **Naive allocation** `clip(1 + k · prediction, 0, 2)`, with a sensitivity curve over `k`
   to show the result is not a razor-thin artefact.

Closes with the full cross-stage comparison table, a penalised-Sharpe bar chart, and
out-of-fold equity curves against Buy & Hold.

---

## 5. Results

Produced with `PMLDL_QUICK=1` on the **synthetic surrogate** (no Kaggle credentials in the
authoring environment). Metrics are means over the five identical purged folds. Re-running
with `data/train.csv` present regenerates every number for the real dataset.

See the final table rendered at the end of
`model_improvements/improved_ensemble_optuna.ipynb`, and the machine-readable records in
`results/*.json`. Indicative ordering from the committed run:

| stage | model | Spearman IC | penalised Sharpe |
|---|---|---|---|
| 1 – baseline | Buy & Hold | n/a (constant) | 0.46 |
| 1 – baseline | ElasticNet (raw) | 0.005 | 0.48 |
| 1 – baseline | LightGBM (default) | −0.006 | 0.11 |
| 1 – strong | 61st-place pipeline (binary) | 0.012 | 0.30 |
| 1 – strong | 100th-place ensemble (expanding) | 0.021 | 0.59 |
| 2 – proposed | **hybrid, single LightGBM** | **0.024** | **0.59** |
| 3 – improved | ensemble + naive allocation | see notebook | see notebook |

The ablation inside the Stage-2 notebook is the cleanest evidence that the hybrid is more
than a concatenation: on identical folds and identical model settings, adding the 21 domain
features lifts rank IC from ≈0.013 (raw + temporal) to ≈0.027, and the 21 domain features
*alone* are competitive with 90 raw ones.

**How to read these numbers.** Fold dispersion is large relative to the differences between
models; small gaps should not be over-interpreted, and none of these figures is a
leaderboard estimate.

---

## 6. Reproducibility

* `SEED = 42` is set in `src/config.py` and applied via `set_seed()` at the top of every
  notebook; it is also passed to every estimator, every Optuna sampler and the synthetic
  data generator.
* Dependencies are pinned in `requirements.txt` to the exact versions used for the committed
  outputs.
* Folds are deterministic and shared (`src.validation.default_cv`).
* Metric records are written to `results/*.json`, each including the seed, per-fold metrics
  and out-of-fold predictions, so any table in the project can be rebuilt without re-running
  the models.
* Residual non-determinism: LightGBM/XGBoost/CatBoost multithreading can produce
  floating-point differences in the last digits across machines. Set `n_jobs=1` /
  `thread_count=1` for bitwise reproducibility at the cost of runtime.

---

## 7. Attribution and academic integrity

This project re-implements ideas from publicly shared Kaggle solutions. **No code was copied
verbatim.** Every notebook opens with an attribution block naming the source, a table
mapping each borrowed idea to where it lives in this repository, and an explicit list of
deviations and the reasons for them.

| public source | what was borrowed |
|---|---|
| [Leak-safe baseline notebook](https://www.kaggle.com/code/morodertobias/hull-leak-safe-baseline) | structure of a leak-free starter pipeline |
| [61st-place notebook](https://www.kaggle.com/code/rafanikitas/hull-eda-training-pipeline) + write-up | temporal feature expansion, single-model discipline, rank-IC Optuna objective, binary allocation |
| [100th-place notebook](https://www.kaggle.com/code/tingkaigong/hull-market-prediction-just-improved?scriptVersionId=285734689) + write-up | fixed-weight three-family ensemble, `U1`/`U2`/cross-product features, top-k selection by gain, sliding window, vol-regime sizer anchored to 1.2, GARCH-blended volatility, EWM smoothing |
| [4th-place write-up](https://kaggle.com/competitions/hull-tactical-market-prediction/writeups/4th-place-technical-model-no-learning-short-te) | short-horizon mean-reversion signal family, inverse-volatility weighting, volatility-targeting overlay, "portfolio = alpha + risk management" framing |

Everything else — the `src/` package, the purged/embargoed validation, the leak-safe
imputation, the metric implementations, the hybrid design, the ablation, the stability-aware
tuning objective and the Stage-3 ensemble — is original work for this project.

**Anonymisation.** No author names, team names, Kaggle handles, emails or other personal
identifiers appear anywhere in this repository. Public sources are cited by URL only.

---

## 8. Known limitations

* The competition's official metric implementation and its live monthly evaluation are not
  reproduced; `penalised_sharpe` is a documented approximation and offline folds cannot
  capture deployment uncertainty.
* Committed outputs come from the synthetic surrogate, so absolute values describe the
  pipeline rather than the competition.
* Volatility statistics used for sizing are estimated in-fold from a single historical path.
* Model differences are point estimates over five folds with a single seed; a multi-seed
  variance study would be needed to attach confidence intervals.
* The Kaggle inference-server / streaming loop is deliberately out of scope; the 100th-place
  online design is approximated by fold-wise refitting on a sliding window.
