# Hull Tactical Market Prediction — PMLDL 2026

Daily S&P 500 allocation in **[0, 2]** from anonymised market features.
The project contains three reproduced baselines, a hybrid LightGBM and a
LightGBM + CatBoost + Ridge ensemble. Baselines are evaluated on the same
180-day public period using their original preprocessing and internal splits.
The proposed and improved pipelines additionally have nested temporal CV.

The public period was consulted during historical development: an early score
of 0.011 motivated a change. It is a **diagnostic**, not a fresh independent test.
The historical 0.453 → 0.770 CV comparison used the tuning folds and must not be
interpreted as an independent estimate of improvement.

Competition: <https://www.kaggle.com/competitions/hull-tactical-market-prediction>

## Attribution

| Component | Source |
| --- | --- |
| `baselines/01_simple_baselines.ipynb` | <https://www.kaggle.com/code/morodertobias/hull-leak-safe-baseline> |
| `baselines/02_strong_baseline_61st.ipynb` | <https://www.kaggle.com/code/rafanikitas/hull-eda-training-pipeline> |
| `baselines/03_strong_baseline_100th.ipynb` | <https://www.kaggle.com/code/tingkaigong/hull-market-prediction-just-> |
| Domain indicators and volatility overlay | `baselines/04_4th_writeup.md` |
| Cross terms `U1`, `U2` | Hull starter notebook and 100th-place write-up |

Baseline adaptations include local data loading, exclusion of the public rows,
offline inference replay, results saving, checkpoint restoration and explicit
random seeds. Their own internal splitters are retained. The inherited B3
online branch never runs in the replay: its saved score describes a **frozen**
ensemble, not an 800-row online model retuned every 50 observations.

## Layout

```text
src/                      shared features, metrics, allocation, nested experiments
baselines/                three source-based baseline notebooks
proposed_model/            hybrid model notebook
model_improvements/       model_improvement.ipynb
reports/                  revised report and validation record
data/raw/                 local competition files (ignored)
results/                  generated results, checkpoints and figures (ignored)
data_checks.py            dataset and chronological checks
report_analysis.py        nested evaluation, seed ablation, bootstrap, figures
update_report.py          regenerate revised Word report from the supplied source
INTERFACE.md              shared API and evaluation contract
tests/                    behavioral regression tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `train.csv` in `data/raw/`, or set `HULL_DATA_DIR` before starting Python.
See [data/README.md](data/README.md) for download instructions. Do not commit
the raw data, virtual environment, checkpoints or caches.

## Reproduce the revised analysis

From the project root:

```bash
python data_checks.py
python report_analysis.py
python -m unittest discover -s tests -v
```

The analysis performs four purged outer temporal folds. Each outer training
period has three inner folds for model hyperparameters, blend and allocation
selection. Imputation is fitted inside each fit. Outer labels are not passed to
models for fitting or early stopping. The final model is refitted with a fixed
inner-selected tree count. The passive `w=1` strategy appears in all comparisons.

Rolling ensemble weights are warmed with predictions on the preceding 60 feature
rows using the same fitted models. Warm-up rows are excluded from validation
objectives and scoring; their labels are used only where they already belong
to the permitted training set. Each temporal fold has its own history.

Default budgets: 8 seeded Optuna trials **per model family per selection**,
16 trials per allocation rule, at most 300 trees and two CPU threads. Candidate
rules are naive, binary, ternary, vol_target, plus the fixed passive reference.
These are new experiments with explicit budgets, not a reconstruction of the
old 120/40/25/30-trial tuning scores. There is no wall-clock stopping condition.

A paired domain-feature ablation uses fit seeds **0–19** on identical outer
folds. Within each fold the model parameters are fixed. Its mean, sample SD,
positive-seed count and paired test describe sensitivity to fit randomness;
they do not measure the uncertainty of a new market period or 20 repeated
hyperparameter searches. Importance is explicitly **gain**, not split count.
The old single-seed +0.013 IC is not evidence of a stable benefit.

Outputs in `results/analysis/` include:

- `manifest.json`: budgets, seeds, package versions, data hash and selection ranges;
- `predictions.csv`, `metrics.csv`, `fold_metrics.csv`: aligned model and index results;
- `ablation_folds.csv`, `ablation_seeds.csv`, `ablation_summary.json`;
- `feature_gain.csv`, `feature_gain_blocks.csv`;
- `bootstrap.csv`: paired moving-block intervals (20-day blocks, 2,000 draws,
  seed 42, sampled within each fold);
- `data_checks.json`, `missingness.csv`, `folds.csv`;
- `figures/figure_2.png`, `figure_4a.png`, `figure_4b.png`, `figure_5.png`.

Intervals are conditional on fitted strategies, not repeated model selection.
The public period is historically consulted; retrospective nested CV also
cannot replace a new prospectively untouched market period. Distributional
checks do not prove real-world provenance or reconstruct calendar dates.
External historical index-return references are not bundled or silently fetched.

The revised Word report and a plain-text review copy are in `reports/`.
Rebuild it from the supplied original Word document, or from the checked-in
revised Word document, with a distinct output path:

```bash
python update_report.py --source reports/project_report.docx --output results/rebuilt_report.docx
```

Run all three baselines first to populate the public comparison. Report generation
requires the full 20-seed analysis and refuses smoke results.

For a short execution check, with separate outputs:

```bash
python report_analysis.py --quick
python report_analysis.py --reuse             # rebuild statistics/figures only
```

`--quick` uses two seeds, one trial per family and at most 60 trees, and writes
to `results/smoke/analysis`. Its results are not report results.
`--output-dir` and `--data-dir` provide explicit path overrides.

## Baseline notebooks and checkpoints

Run these using `jupyter lab` or `jupyter nbconvert --to notebook --execute`:

| Notebook | Result identifier | Evaluation |
| --- | --- | --- |
| `baselines/01_simple_baselines.ipynb` | `elasticnet_leak_safe` | public only |
| `baselines/02_strong_baseline_61st.ipynb` | `lgbm_61st` | public only |
| `baselines/03_strong_baseline_100th.ipynb` | `online_ensemble_100th` (historical identifier; frozen model) | public only |
| `proposed_model/proposed_hybrid_model.ipynb` | `hybrid_domain_lgbm` | nested CV + public diagnostic |
| `model_improvements/model_improvement.ipynb` | `ensemble_lgbm_catboost_ridge` | nested CV + public diagnostic |

The last two notebooks load the shared analysis, generating it if absent.
Run `report_analysis.py` after code changes to regenerate it. Baseline results
are added to public comparison tables when available; no baseline CV rows are
invented. Notebook outputs from previous versions have been cleared.

B2 reuses a checkpoint only when **both** model and feature files exist in
`results/b_02_checkpoint/`; it restores `final_model` and `FEATURES` before the
importance plot. A fresh run trains and saves both files. New training uses a
seeded Optuna sampler with 200 trials, potentially taking substantial time.
An incomplete checkpoint raises a clear error and preserves the remaining file;
restore its companion or select another `HULL_RESULTS_DIR` to retrain.
The original unseeded search is not claimed to be reconstructable from seed 42.
B3 similarly caches its tuned parameters in `results/b_03_checkpoint/`.

To retrain without overwriting existing artifacts, use another results directory:

```bash
HULL_RESULTS_DIR=results/retrained jupyter lab
```

For a baseline smoke run, set `PMLDL_QUICK=1` and a separate `HULL_RESULTS_DIR`.
B2 then fits 80 trees; B3 uses one search trial per family. Those runs do not
reproduce the historical baseline scores. Never load checkpoints from an
untrusted source: these baseline artifacts use Python pickle/joblib.

`save_result` replaces the row with the same model and split. Compare with:

```python
from src import compare
compare(split="cv")       # proposed, improved and passive; nested outer results
compare(split="public")   # public diagnostic; baselines when executed
```

`modified_sharpe` is the implementation in `src/metrics.py`; the organisers'
forecasting service was not used. The `benchmark_sharpe` arithmetic diagnostic
is not the passive modified-Sharpe score: use the explicit `passive_w1` row.
See [INTERFACE.md](INTERFACE.md) for the shared implementation contract.
