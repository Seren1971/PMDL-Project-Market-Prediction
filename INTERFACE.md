# `src/` Interface Contract

**Read this before writing any notebook.** Every notebook in `baselines/`,
`proposed_model/` and `model_improvements/` must use this interface. Rewriting
any of it inline breaks the comparison the project is graded on.

## Rules

1. **Import from `src`, never from submodules.** `from src import load_dataset`,
   not `from src.data import load_dataset`.
2. **Never re-implement loading, folds, metrics or allocation in a notebook.**
   Missing something? Add it to `src/` and re-run the other notebooks.
3. **Call `set_seed()` in the first code cell.** Pass `random_state=SEED` to
   every model.
4. **Use `get_folds()` with default arguments.** Identical folds across all
   notebooks is a hard requirement. Never use `TimeSeriesSplit`, `KFold`, or a
   manual slice.
5. **Never feed `forward_returns`, `risk_free_rate` or
   `market_forward_excess_returns` to a model.** They describe the future of
   their own row. `get_feature_columns()` and `build_features()` already
   exclude them; do not add them back.
6. **Never fit on the public block.** `load_dataset()` splits it off; it is for
   final reporting only.
7. **Fit preprocessing on training rows only.** `impute()` enforces this — pass
   the training frame first.
8. **Every notebook ends with `save_result(...)`.**
9. **Attribution header in the first markdown cell of every notebook**, naming
   the source it borrows from. **No author names, usernames, team names or
   personal details anywhere** — cite notebooks by URL and placements by number.

## Setup cell (copy verbatim)

```python
import sys; sys.path.append("..")   # notebooks live one level below the root

from src import *

set_seed()                          # SEED = 42
```

## The five calls

### 1. Load

```python
data = load_dataset()               # -> HullData(train, public, full, raw_features)
```

Reads `data/raw/train.csv` (falls back to the Kaggle mount, or `HULL_DATA_DIR`),
sorts by `date_id`, trims the mostly-null early rows, and holds out the last 180
`date_id`s as `data.public`. Fit on `data.train`; report on both.

### 2. Engineer features

```python
feat_df, features = build_features(
    data.full,                      # full frame: rolling windows stay continuous
    lag_roll_columns=["M4", "V13", "S5", "S2", "P13"],   # 61st-style, optional
)
```

Returns the frame plus the feature name list. Call it on `data.full`, then
re-split on `date_id` — safe because every window looks strictly backwards.

Available separately: `add_cross_terms` (U1, U2), `add_price_features`
(momentum, realised vol, vol spreads, mean-reversion z-scores, drawdown),
`add_lag_roll_features`, `inverse_vol_weights`, `blend_signals`,
`top_features_by_gain`.

Then impute, training statistics only:

```python
train_df, public_df = impute(train_df, public_df, columns=features)
```

### 3. Fold

```python
folds = get_folds(train_df)         # defaults: 4 splits, purge=1, embargo=20
assert_no_leakage(folds)
display(describe_folds(train_df, folds))   # print this in every notebook

for fold in folds:
    X_tr = train_df.loc[fold.train_idx, features]
    y_tr = train_df.loc[fold.train_idx, TARGET]
    X_va = train_df.loc[fold.val_idx, features]
```

Expanding window. Positional indices, deterministic in row count, so two
notebooks on the same frame get identical folds.

### 4. Score

```python
weights = naive_allocation(pred, k=50)     # clip(1 + k*pred, 0, 2)

m = evaluate(
    y_true, pred,
    weights=weights,
    forward_returns=val_df["forward_returns"].to_numpy(),
    risk_free_rate=val_df["risk_free_rate"].to_numpy(),
)
agg = aggregate_folds(fold_metrics)        # -> {"<metric>_mean", "<metric>_std"}
```

`evaluate` returns prediction metrics (`rmse`, `r2`, `spearman_ic`, `hit_rate`)
and, when the three strategy arrays are supplied, strategy metrics
(`modified_sharpe`, `sharpe`, `vol_ratio`, `vol_penalty`, `return_penalty`,
`ann_return`, `ann_volatility`, `max_drawdown`, `benchmark_sharpe`,
`mean_weight`, `weight_turnover`).

Drop the strategy arrays for a prediction-only evaluation.

**Tune on `spearman_ic`, report `modified_sharpe`.** Ranking is what is
learnable here; exact magnitudes are not.

Allocation rules: `naive_allocation` (Stage 3 default), `binary_allocation`
(61st), `vol_target_allocation` (4th), plus `smooth_weights` and `realised_vol`.

### 5. Save

```python
save_result(
    model="lgbm_61st",              # short id; re-running replaces the row
    stage="baseline",               # "baseline" | "proposed" | "improved"
    metrics=agg,
    split="cv",                     # "cv" | "public"
    params=best_params,
    notes="4-fold purged CV, Optuna on mean Spearman",
)
display(compare())                  # comparison table across all notebooks
```

Writes `results/leaderboard.csv`. Also available: `save_predictions` /
`load_predictions` for per-row outputs, so a later notebook can blend without
refitting.

## Two facts that trip people up

**`modified_sharpe` is our re-implementation, not the organisers' code.** It
applies a volatility cliff at 120% of market volatility (below it is free, above
it scales the Sharpe down) and a quadratic penalty for annualised
underperformance against buy-and-hold. If the competition's `metric.py` is
importable, `hull_score()` uses that instead. Treat our number as a consistent
internal yardstick, not a leaderboard prediction.

**`past_returns(df)` is `forward_returns.shift(1)`.** `forward_returns` at row
*t* is the return from *t* to *t+1*, so the return already observable at *t* is
the previous row's. Build every price-derived feature from `past_returns()`;
never touch `forward_returns` directly outside of scoring.

## Reference

| Call | Module | Purpose |
| --- | --- | --- |
| `set_seed()`, `SEED`, `TARGET`, `DATE_COL` | `config` | Reproducibility, constants |
| `load_dataset()`, `impute()`, `past_returns()`, `get_feature_columns()` | `data` | Loading and splitting |
| `build_features()` + block-level helpers | `features` | Feature engineering |
| `get_folds()`, `describe_folds()`, `assert_no_leakage()` | `validation` | Purged CV |
| `evaluate()`, `aggregate_folds()`, `modified_sharpe()`, `hull_score()` | `metrics` | Scoring |
| `naive_allocation()`, `binary_allocation()`, `vol_target_allocation()` | `allocation` | Position sizing |
| `save_result()`, `compare()`, `save_predictions()` | `results` | Persistence |
