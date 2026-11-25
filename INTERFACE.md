# Shared project interface

Run commands from the repository root. Notebooks locate the root through
`src/experiments.py` and import the same implementation used by the CLI.

## Data and features

`load_dataset(data_dir=None)` reads `train.csv`, trims low-coverage leading
history, and reserves the last 180 date IDs as the public diagnostic. Set
`HULL_DATA_DIR` to use another directory. The public period influenced historical
development and must not be described as a prospectively untouched test set.

`build_features(full_frame, lag_roll_columns=...)` builds causal return features,
cross terms and temporal features. `past_returns` shifts forward returns by one
row. Building features on the full chronological frame preserves past history;
it does not fit any statistics. Model inputs exclude all three look-ahead columns.

`impute(training, validation, columns=features)` forward-fills each frame, then
uses training medians and finally zero. Call it separately inside every model
fit. Do not impute the whole development period before splitting into folds.

## Evaluation

`get_folds(frame)` returns four deterministic expanding temporal folds with
purge 1 and a 20-row gap at the head of each validation block. These are the
outer evaluation folds in `src.experiments`. `assert_no_leakage` checks them.

`ExperimentConfig` declares the trial, seed and tree budgets.
`select_recipe(outer_training, features, config)` selects all model parameters,
blend and sizing on inner temporal folds. `evaluate_recipe` refits using fixed
iteration counts and scores the next outer block. Neither its imputer nor its
models receive outer labels for fitting or early stopping.

The inverse-volatility blend is warmed with predictions on the preceding 60
feature rows, including gap rows, using the same already fitted models. Warm-up
rows are excluded from validation objectives and scores; each fold has its own
history. Their labels are used only where they already belong to the permitted
training set.

`run_experiment(output_dir, data_dir=None, config=None)` evaluates both model
stages and the passive `w=1` reference, records paired domain-feature ablations
for seeds 0–19, then scores the historically consulted public diagnostic. The
ablation holds hyperparameters fixed within each outer fold; it measures fit
randomness, not repeated-search or temporal uncertainty. LightGBM importance
is explicitly requested with `importance_type="gain"`.

Baseline notebooks retain their own internal splitters and preprocessing. Their
saved comparisons are **public only**; they do not have shared outer CV results.
B3's replay uses a frozen ensemble: its inherited online branch is inactive.

## Results

`save_result` replaces the row with the same `(model, split)` in
`results/leaderboard.csv`. Set `HULL_RESULTS_DIR` before importing `src` for
isolated notebook runs. `save_predictions` stores aligned dates, targets,
predictions and positions. Position bounds are `[0, 2]`.

`python data_checks.py` regenerates dataset diagnostics and fold boundaries.
`python report_analysis.py` regenerates nested evaluation, seed ablation,
paired moving-block intervals and figures. `--quick` writes a smoke experiment
to `results/smoke/analysis`; `--reuse` rebuilds statistics from existing results
and checks that the dataset hash matches. Defaults and package versions are
stored in `manifest.json`.

The modified Sharpe is the implementation in `src/metrics.py`. The arithmetic
`benchmark_sharpe` diagnostic is not the passive modified-Sharpe score. Compare
strategies against the explicit `passive_w1` row using the same metric.
