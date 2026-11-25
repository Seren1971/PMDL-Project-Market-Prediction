"""Regenerate nested evaluation, paired seed ablation, bootstrap and figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from data_checks import run_checks
from src import modified_sharpe, strategy_returns
from src.experiments import ExperimentConfig, partition_metrics, run_experiment

LABELS = {"proposed": "Hybrid LightGBM", "improved": "Ensemble", "passive": "Passive w=1",
          "elasticnet_leak_safe": "B1 ElasticNet", "lgbm_61st": "B2 LightGBM",
          "online_ensemble_100th": "B3 Frozen ensemble"}


def paired_block_bootstrap(frame, model, reference, draws=2000, block_size=20, seed=42):
    """Paired moving-block CI conditional on fitted strategies, within each fold.

    Resampling never crosses fold gaps. This estimates time-sampling uncertainty,
    not the uncertainty from rerunning selection or fitting models.
    """
    pairs = []
    for _, group in frame.groupby("fold", sort=True):
        left = group.loc[group.model == model].sort_values("date_id")
        right = group.loc[group.model == reference].sort_values("date_id")
        if not np.array_equal(left.date_id, right.date_id) or left.empty:
            raise ValueError("Paired bootstrap requires identical, nonempty date rows")
        pairs.append((left, right))
    if not pairs or draws < 1 or block_size < 1:
        raise ValueError("Nonempty folds and positive bootstrap budgets required")

    def difference(left, right, index):
        fwd = left.forward_returns.to_numpy()[index]
        rf = left.risk_free_rate.to_numpy()[index]
        return (modified_sharpe(left.weight.to_numpy()[index], fwd, rf) -
                modified_sharpe(right.weight.to_numpy()[index], fwd, rf))

    observed = float(np.mean([difference(l, r, np.arange(len(l))) for l, r in pairs]))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        scores = []
        for left, right in pairs:
            n = len(left)
            width = min(block_size, n)
            starts = rng.integers(0, n - width + 1, size=int(np.ceil(n / width)))
            indices = np.concatenate([np.arange(s, s + width) for s in starts])[:n]
            scores.append(difference(left, right, indices))
        differences.append(float(np.mean(scores)))
    low, high = np.quantile(differences, [.025, .975])
    return {"model": model, "reference": reference, "observed_delta": observed,
            "ci_low": float(low), "ci_high": float(high),
            "fraction_positive": float(np.mean(np.asarray(differences) > 0)),
            "draws": draws, "block_size": block_size, "seed": seed}


def build_report(directory, draws=2000):
    directory = Path(directory)
    predictions = pd.read_csv(directory / "predictions.csv")
    manifest = json.loads((directory / "manifest.json").read_text())
    ablation = pd.read_csv(directory / "ablation_folds.csv")
    seeds = ablation.groupby("seed")[["with_domain", "without_domain", "delta"]].mean()
    seeds.to_csv(directory / "ablation_seeds.csv")
    delta = seeds.delta.to_numpy()
    test = stats.ttest_1samp(delta, 0.) if len(delta) > 1 else None
    seed_summary = {"n_seeds": len(delta), "mean_gain": float(delta.mean()),
                    "std_gain": float(delta.std(ddof=1)) if len(delta) > 1 else None,
                    "positive_seeds": int((delta > 0).sum()),
                    "paired_t": float(test.statistic) if test is not None else None,
                    "paired_pvalue": float(test.pvalue) if test is not None else None,
                    "scope": "Fit-seed sensitivity with parameters fixed within each outer fold; "
                             "seeds share data and do not measure temporal uncertainty."}
    (directory / "ablation_summary.json").write_text(json.dumps(seed_summary, indent=2) + "\n")
    bootstraps = []
    for split, group in predictions.groupby("split"):
        for model, reference in [("proposed", "passive"), ("improved", "passive"),
                                 ("improved", "proposed")]:
            bootstraps.append({"split": split, **paired_block_bootstrap(
                group, model, reference, draws=draws)})
    pd.DataFrame(bootstraps).to_csv(directory / "bootstrap.csv", index=False)
    gains = pd.read_csv(directory / "feature_gain.csv")
    shares = gains.groupby("block").agg(total_gain=("gain", "sum"), n_features=("gain", "size"))
    shares["gain_share"] = shares.total_gain / shares.total_gain.sum()
    shares["gain_per_feature"] = shares.total_gain / shares.n_features
    shares.to_csv(directory / "feature_gain_blocks.csv")
    metric_rows = []
    for (split, model, fold), group in predictions.groupby(["split", "model", "fold"]):
        metric_rows.append({"split": split, "model": model, "fold": fold,
                            **partition_metrics(group)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(directory / "fold_metrics.csv", index=False)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    missing = pd.read_csv(directory / "missingness.csv", index_col=0)
    ax = missing.head(15).iloc[::-1].plot.barh(legend=False, figsize=(8, 5))
    ax.set(xlabel="Missing fraction after coverage trim", title="Feature coverage")
    plt.tight_layout(); plt.savefig(figures / "figure_2.png", dpi=150); plt.close()
    cv = metrics.loc[metrics.split == "cv"]
    ax = cv.pivot(index="fold", columns="model", values="modified_sharpe").rename(columns=LABELS).plot.bar(figsize=(8, 4))
    ax.set(ylabel="Modified Sharpe", title="Nested outer evaluation (passive w=1 included)")
    plt.tight_layout(); plt.savefig(figures / "figure_4a.png", dpi=150); plt.close()
    public = metrics.loc[metrics.split == "public"].set_index("model")
    # Baselines have public-only results; do not invent CV rows for them.
    leaderboard = directory.parent / "leaderboard.csv"
    if leaderboard.exists():
        recorded = pd.read_csv(leaderboard)
        for _, row in recorded.loc[(recorded.stage == "baseline") & (recorded.split == "public")].iterrows():
            public.loc[row.model, "modified_sharpe"] = row.modified_sharpe
    ax = public.modified_sharpe.rename(index=LABELS).sort_values().plot.barh(figsize=(8, 4))
    ax.set(xlabel="Modified Sharpe", ylabel="", title="Public diagnostic (historically consulted)")
    ax.bar_label(ax.containers[0], fmt="%.3f", padding=3)
    ax.set_xlim(right=float(public.modified_sharpe.max()) * 1.15)
    plt.tight_layout(); plt.savefig(figures / "figure_4b.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, group in predictions.loc[predictions.split == "public"].groupby("model"):
        group = group.sort_values("date_id")
        curve = np.cumprod(1. + strategy_returns(group.weight, group.forward_returns, group.risk_free_rate))
        ax.plot(group.date_id, curve, label=LABELS[model])
    ax.set(xlabel="date_id", ylabel="Wealth from initial capital 1", title="Public diagnostic, including passive w=1")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "figure_5.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ("elasticnet_leak_safe", "lgbm_61st", "online_ensemble_100th"):
        artifact = directory.parent / "artifacts" / f"{name}_predictions.csv"
        if artifact.exists():
            allocations = pd.read_csv(artifact)
            ax.plot(allocations.date_id, allocations.weight, label=LABELS[name], alpha=.8)
    ax.axhline(1., color="black", linestyle="--", label=LABELS["passive"])
    ax.set(xlabel="date_id", ylabel="Allocation", title="Baseline public allocations (frozen B3 replay)")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "figure_1.png", dpi=150); plt.close(fig)
    print("Ablation:", json.dumps(seed_summary), flush=True)
    print(metrics[["split", "model", "fold", "modified_sharpe", "spearman_ic"]].to_string(index=False))
    return {"ablation": seed_summary, "bootstrap": bootstraps, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quick", action="store_true", help="Smoke run; separate results/smoke directory")
    parser.add_argument("--reuse", action="store_true", help="Rebuild figures/statistics from existing predictions")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    output = args.output_dir or Path("results/smoke/analysis" if args.quick else "results/analysis")
    checks = run_checks(args.data_dir, output)
    if not args.reuse:
        config = ExperimentConfig(seed_count=2, model_trials=1, allocation_trials=2,
                                  inner_splits=2, max_estimators=60) if args.quick else ExperimentConfig()
        run_experiment(output, args.data_dir, config)
    else:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["data_sha256"] != checks["data_sha256"]:
            raise ValueError("Stored predictions belong to a different dataset")
    build_report(output, args.bootstrap_draws)


if __name__ == "__main__":
    main()
