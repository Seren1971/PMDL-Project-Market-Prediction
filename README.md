# Hull Tactical Market Prediction — PMLDL 2026 Stage 2

Predicting daily S&P 500 allocation from anonymised market, macro, rate,
valuation, volatility and sentiment features. The deliverable is three
reproduced baselines, one proposed hybrid model, and an improved ensemble, all
evaluated on **identical purged time-series folds** so the comparison is valid.

Competition:
<https://www.kaggle.com/competitions/hull-tactical-market-prediction>

## Attribution

This project reproduces and builds on publicly shared Kaggle work. Every
notebook carries an attribution header naming its source. No author names,
usernames or team identifiers appear anywhere in this repository — sources are
cited by URL and by placement number only.

| Component | Source |
| --- | --- |
| `baselines/01_simple_baselines.ipynb` | <https://www.kaggle.com/code/morodertobias/hull-leak-safe-baseline> |
| `baselines/02_strong_baseline_61st.ipynb` | <https://www.kaggle.com/code/rafanikitas/hull-eda-training-pipeline> |
| `baselines/03_strong_baseline_100th.ipynb` | <https://www.kaggle.com/code/tingkaigong/hull-market-prediction-just-> |
| Domain feature ideas in `src/features.py` and Stage 2 | 4th place write-up (competition write-ups page) |
| Cross terms `U1`, `U2` | Hull starter notebook; also used in the 100th place write-up |

The baseline notebooks keep their original modelling logic. Changes are limited
to data loading and results saving, and each one is marked in-place with an
`ADDED (PMLDL Stage 1)` comment.

## Layout

```
.
├── data/                 # dataset goes here (not committed) — see data/README.md
├── src/                  # shared library: loading, features, CV, metrics, results
├── baselines/            # Stage 1 — three reproduced baselines
├── proposed_model/       # Stage 2 — single hybrid model
├── model_improvements/   # Stage 3 — ensemble + Optuna + naive allocation
├── results/              # generated: leaderboard.csv, artifacts/ (not committed)
├── INTERFACE.md          # usage contract for src/ — read before editing notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then download the data into `data/raw/`:

```bash
kaggle competitions download -c hull-tactical-market-prediction -p data/raw
unzip -o data/raw/hull-tactical-market-prediction.zip -d data/raw
```

Only `train.csv` is required. Full instructions, the schema, and the two
leakage traps in this dataset are in [`data/README.md`](data/README.md). To read
the data from elsewhere, set `HULL_DATA_DIR`.

## Running

Run in order — later notebooks read `results/leaderboard.csv` to build the
comparison table.

```bash
jupyter lab
```

| # | Notebook | Produces |
| --- | --- | --- |
| 1 | `baselines/01_simple_baselines.ipynb` | `elasticnet_leak_safe` |
| 2 | `baselines/02_strong_baseline_61st.ipynb` | `lgbm_61st` |
| 3 | `baselines/03_strong_baseline_100th.ipynb` | `online_ensemble_100th` |
| 4 | `proposed_model/proposed_hybrid_model.ipynb` | `hybrid_domain_lgbm` |
| 5 | `model_improvements/improved_ensemble_optuna.ipynb` | `improved_ensemble` |

Or headless:

```bash
jupyter nbconvert --to notebook --execute --inplace baselines/01_simple_baselines.ipynb
```

Runtimes: baselines 01 and 03 are minutes; 02 and 04 run Optuna and take up to
an hour each. Both expose a `QUICK`/trial-count switch near the top for a fast
pass.

Results accumulate in `results/leaderboard.csv`; re-running a notebook replaces
its own row rather than appending a duplicate. Compare at any point with:

```python
from src import compare
compare(split="cv")        # cross-validated
compare(split="public")    # held-out 180-day block
```

## Reproducibility

* `SEED = 42` in `src/config.py`, applied via `set_seed()` in the first cell of
  every notebook, passed as `random_state` to every model, and used to seed the
  Optuna `TPESampler` so the hyperparameter search itself is deterministic.
* Dependencies pinned exactly in `requirements.txt`.
* Folds are a deterministic function of row count, so all five notebooks
  validate on byte-identical splits.

## Methodology

**Validation.** `get_folds()` gives 4 expanding-window splits with `purge=1`
(the target horizon) and `embargo=20` trading days. Purging stops the last
training row's forward-looking target from overlapping the validation block;
the embargo stops serially correlated rows sitting flush against it.
`assert_no_leakage()` is called in every notebook. Plain `TimeSeriesSplit`,
`KFold` and manual slicing are prohibited by the interface contract.

**Two leakage traps specific to this dataset.**

1. The public leaderboard set is a *copy of the last 180 rows of `train.csv`*.
   `load_dataset()` holds them out; nothing is fitted on them.
2. `forward_returns` at row *t* spans *t → t+1* — it is the future of its own
   row, as are `risk_free_rate` and `market_forward_excess_returns`. The return
   observable at *t* is `forward_returns.shift(1)`, which is what
   `src.data.past_returns()` returns and what every price-derived feature is
   built from.

**Metrics.** Tune on Spearman rank correlation, report modified Sharpe. Ranking
days from weak to strong is learnable in this noise regime; exact return
magnitudes are not, and the ranking is all the allocation rule consumes.

> **`modified_sharpe` is our re-implementation, not the organisers' metric.** It
> applies a volatility cliff at 120% of market volatility and a quadratic
> penalty for underperforming buy-and-hold, following the public description of
> the scoring function. `hull_score()` automatically prefers the competition's
> own `metric.py` if it is importable. Treat our number as a consistent internal
> yardstick for ranking these models against each other and against
> `benchmark_sharpe` — not as a leaderboard prediction.

**Scope.** The public phase only. The forecasting phase and the
`kaggle_evaluation` inference server are out of scope; the baselines replay the
held-out block offline instead of serving predictions.

## Extending

Read [`INTERFACE.md`](INTERFACE.md) first. The rule is that loading, folds,
metrics and allocation live in `src/` and are never re-implemented inside a
notebook — if something is missing, add it to `src/` and re-run the others, so
that every row in the leaderboard stays comparable.
