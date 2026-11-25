"""Revise the supplied Stage 2 Word report using computed analysis artifacts.

Usage: python update_report.py --source original.docx
The original document is never overwritten. No private filesystem paths are
embedded in the revised document or its plain-text review copy.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import struct
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = {"w": W[1:-1]}


def text_of(element):
    return "".join(n.text or "" for n in element.iter(W + "t"))


def set_text(element, text):
    """Preserve paragraph/cell formatting, replacing its text content."""
    if element.tag == W + "tc":
        para = element.find(W + "p")
        for child in list(element):
            if child.tag != W + "tcPr":
                element.remove(child)
        element.append(deepcopy(para) if para is not None else ET.Element(W + "p"))
        set_text(element.find(W + "p"), text)
        return
    for child in list(element):
        if child.tag != W + "pPr":
            element.remove(child)
    run = ET.SubElement(element, W + "r")
    ET.SubElement(run, W + "t").text = str(text)


def set_table(table, rows):
    template = table.findall(W + "tr")
    width = len(template[0].findall(W + "tc"))
    if any(len(row) != width for row in rows):
        raise ValueError("Report table width mismatch")
    for row in template:
        table.remove(row)
    for i, values in enumerate(rows):
        row = deepcopy(template[0 if i == 0 else min(1, len(template) - 1)])
        for cell, value in zip(row.findall(W + "tc"), values):
            set_text(cell, value)
        table.append(row)


def revise(source, directory, destination):
    directory, destination = Path(directory), Path(destination)
    if Path(source).resolve() == destination.resolve():
        raise ValueError("Source and output must differ; the input report is never overwritten")
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "ablation_summary.json").read_text())
    metrics = pd.read_csv(directory / "fold_metrics.csv")
    intervals = pd.read_csv(directory / "bootstrap.csv")
    bootstrap_config = intervals.iloc[0]
    gains = pd.read_csv(directory / "feature_gain_blocks.csv").set_index("block")
    checks = json.loads((directory / "data_checks.json").read_text())
    versions = manifest["versions"]
    config = manifest["config"]
    leaderboard = pd.read_csv(directory.parent / "leaderboard.csv")
    baselines = leaderboard.loc[(leaderboard.stage == "baseline") &
                                (leaderboard.split == "public")].set_index("model")
    baseline_labels = {"elasticnet_leak_safe": "B1 ElasticNet",
                       "lgbm_61st": "B2 LightGBM (61st)",
                       "online_ensemble_100th": "B3 Frozen ensemble (100th)"}
    if not set(baseline_labels).issubset(baselines.index):
        raise ValueError("Run all three baseline notebooks before generating the complete report")
    if config["seed_count"] < 20:
        raise ValueError("Smoke experiments cannot be used for the revised report")

    def metric(model, split="cv", key="modified_sharpe"):
        return metrics.loc[(metrics.model == model) & (metrics.split == split), key]

    def display(model, split="cv", key="modified_sharpe"):
        values = metric(model, split, key)
        return f"{values.mean():.4f} ± {values.std(ddof=0):.4f}" if split == "cv" else f"{values.iloc[0]:.4f}"

    with zipfile.ZipFile(source) as original:
        document = original.read("word/document.xml")
        namespaces = {prefix: uri for _, (prefix, uri) in
                      ET.iterparse(BytesIO(document), events=["start-ns"])}
        for prefix, uri in namespaces.items():
            ET.register_namespace(prefix, uri)
        root = ET.fromstring(document)
        paragraphs = root.findall("w:body/w:p", NS)
        tables = root.findall(".//w:tbl", NS)
        if len(tables) != 18 or not text_of(paragraphs[68]).startswith("Ablation."):
            raise ValueError("This is not the supplied Stage 2 Word report revision")
        diff = metric("improved").mean() - metric("passive").mean()
        seed_statement = (
            f"Across {summary['n_seeds']} paired fit seeds (0–{summary['n_seeds'] - 1}), "
            f"the mean domain-block IC gain is {summary['mean_gain']:+.5f} "
            f"(sample SD {summary['std_gain']:.5f}), positive in "
            f"{summary['positive_seeds']} runs; paired t={summary['paired_t']:.3f}, "
            f"p={summary['paired_pvalue']:.4f}. Parameters are fixed within each outer fold. "
            "This measures fit-seed sensitivity, not 20 independent datasets or repeated searches. "
            "The original single-seed +0.013 does not establish a stable contribution."
        )
        ci_lines = []
        for _, row in intervals.iterrows():
            ci_lines.append(f"{row['split']}: {row['model']} − {row['reference']} = "
                            f"{row.observed_delta:+.4f}, 95% CI [{row.ci_low:+.4f}, {row.ci_high:+.4f}]")
        replacements = {
            5: "We study Hull Tactical Market Prediction: each trading day a model outputs an S&P 500 "
               "allocation in [0, 2], scored by volatility- and shortfall-penalised modified Sharpe. "
               "We reproduce three public baselines, evaluate a hybrid LightGBM with domain indicators, "
               "and extend it with a LightGBM/CatBoost/Ridge ensemble and allocation selection. "
               "The revised study nests selection inside chronological training periods and compares "
               "every evaluated strategy with passive w=1. The label improved names a pipeline stage; "
               "it does not establish a statistically reliable performance gain.",
            7: f"Revised findings. Nested outer modified Sharpe: proposed {display('proposed')}, "
               f"improved {display('improved')}, passive {display('passive')}. "
               f"The improved-minus-passive mean difference is {diff:+.4f}; no superiority is "
               "assumed from the model name. Public scores are retrospective diagnostics. "
               "Historical tuning-fold scores 0.453 and 0.770 are retained only as development history.",
            13: "The last 180 date_ids are excluded from model fitting and form the public diagnostic. "
                "They are not an untouched test: the historical score of 0.011 influenced development (§4.5). "
                "No new prospective test data are available; the forecasting phase is out of scope.",
            17: f"Recomputed data checks (data_checks.py): {checks['raw_rows']} raw rows, "
                f"{checks['usable_rows']} usable rows, {checks['train_rows']} development and "
                f"{checks['public_rows']} public rows; {checks['complete_features']} fully observed "
                f"raw features after trimming. Target mean={checks['target_mean']:.6g}, "
                f"SD={checks['target_std']:.6g}. Forward-return excess kurtosis="
                f"{checks['return_excess_kurtosis']:.3f}, Jarque–Bera p={checks['jarque_bera_pvalue']:.3g}, "
                f"lag-1 autocorrelation={checks['return_lag1_autocorrelation']:.4f}, squared-return "
                f"lag-1 autocorrelation={checks['squared_return_lag1_autocorrelation']:.4f}. "
                "Unique dates, chronological splitting, the causal return lag and purged boundaries "
                "are checked programmatically. These internal checks alone do not prove real-market "
                "provenance or reconstruct anonymous calendar dates; no external reference-return match is claimed.",
            27: "The revised evaluation separates selection from scoring chronologically. Baselines have "
                "public-only comparisons with their original internal splitters. Proposed, improved and "
                "passive strategies additionally use the same four outer evaluation folds.",
            28: "Public diagnostic (180 days). The pipeline is selected on preceding training rows and "
                "refitted before scoring this period. Because the block influenced historical development, "
                "its results cannot be treated as independent test evidence.",
            29: "Nested purged temporal CV. For each outer fold, three inner temporal folds select model "
                "parameters, ensemble and allocation using only the outer training rows. Imputation is fitted "
                "inside each fit. Outer labels never control early stopping; iteration counts are selected "
                "internally and fixed before outer scoring. Purge=1 and the validation-head gap=20 are "
                "applied at both levels. This retrospective protocol removes within-run selection reuse "
                "but cannot erase earlier human exposure to the dataset.",
            31: f"Reference strategy. Buy-and-hold (w=1) is scored using the same modified Sharpe: "
                f"{display('passive')} on outer folds and {display('passive', 'public')} publicly. "
                "The separate benchmark_sharpe diagnostic is arithmetic Sharpe and is not this reference score.",
            37: "The three source baselines represent a linear allocation model, a single LightGBM with "
                "temporal features, and an ElasticNet/XGBoost/LightGBM ensemble. B3 executes as a frozen "
                "ensemble in our replay; the online procedure advertised by the source does not run.",
            44: "Baselines retain their source preprocessing and internal validation. The B2 model and B3 "
                "hyperparameters can be restored from the submitted archive; replaying those artifacts "
                "reproduces their historical scores. New seeded searches are distinct experiments and are "
                "not guaranteed to recover parameters from the original unseeded searches. No baseline "
                "outer CV score is reported.",
            53: "Figure 1. Replayed baseline allocations on the public diagnostic, with the passive w=1 reference. "
                "B3 is a frozen ensemble; no online refitting occurs.",
            60: f"Nested model search: seeded Optuna TPE, {config['model_trials']} trials per family per "
                f"selection on {config['inner_splits']} inner purged temporal folds, at most "
                f"{config['max_estimators']} trees. The objective is mean inner Spearman IC. Budgets are "
                "fixed trial counts, with no wall-clock cutoff. These revised experiments differ from "
                "the original 120-trial search.",
            61: "Each outer evaluation refits on its own earlier training rows using inner-selected parameters. "
                "The public diagnostic uses a fresh selection on the full development period followed by refitting.",
            63: f"Determinism: seed={config['seed']}, LightGBM deterministic=True and force_col_wise=True; "
                f"CPU thread budget={config['n_jobs']}. Versions and all selected parameters are in manifest.json.",
            64: "Table 7. Revised search space and inner-selected LightGBM parameters for the public diagnostic; "
                "outer-fold selections are recorded separately in manifest.json.",
            68: "Ablation. " + seed_statement + f" Actual LightGBM gain gives the domain block "
                f"{gains.loc['domain', 'gain_share']:.1%} of outer-model gain. Split counts are not labelled "
                "gain. Importance describes model usage, not evidence that a feature improves unseen returns.",
            69: f"Analysis. Revised public diagnostic modified Sharpe: proposed "
                f"{display('proposed', 'public')} versus passive {display('passive', 'public')}. "
                "This direct comparison replaces the claim that beating the weaker baselines demonstrates "
                "outperformance of the index. Public selection history prevents an independent-test interpretation.",
            76: "Table 11. Fixed revised search budgets and inner-selected public-diagnostic parameters. "
                "Inner tuning objectives are not outer evaluation scores; no outer early stopping is used.",
            78: "Table 12. Nested outer Spearman IC for the two evaluated pipelines; the constant passive "
                "signal has no forecasting interpretation. Historical per-family selection scores are superseded.",
            79: "Table 13. HISTORICAL tuning-fold allocation comparison. These original numbers were used "
                "to select the rule and are optimistic. They are development diagnostics, not independent "
                "evidence. Revised outer results are in Tables 14 and 16.",
            81: "Figure 2. Recomputed feature missingness after coverage trimming (data_checks.py). "
                "The earlier allocation-search figure has been replaced; old rule scores remain historical in Table 13.",
            82: "Table 14. Revised nested outer and historically consulted public results, including the passive index.",
            83: "Table 15. HISTORICAL exploratory public allocation comparison, with the passive reference added. "
                "These rows do not select the revised headline rule.",
            85: "Ensembling and sizing are assessed jointly as a selection procedure on later outer blocks. "
                "Model families, blend and sizing are selected on earlier inner folds; this avoids treating "
                "the best inner configuration as an unbiased evaluation.",
            86: f"Revised outer modified Sharpe is {display('improved')} for improved, "
                f"{display('proposed')} for proposed and {display('passive')} for passive. "
                "The historical 0.453 → 0.770 increase is not retained as confirmation of improvement.",
            87: f"Public diagnostic: improved {display('improved', 'public')}, proposed "
                f"{display('proposed', 'public')}, passive {display('passive', 'public')}. "
                "The public recipe was fixed using the preceding training data; this does not undo earlier "
                "development exposure. Volatility and turnover are reported rather than assumed acceptable.",
            88: "4.4 Blend warm-up and historical development bug",
            89: "The inverse-volatility blend requires history. The revised pipeline predicts the 60 "
                "chronologically preceding feature rows using the same fitted models, prepends these "
                "predictions for rolling weights, then scores only the evaluation rows. The models and "
                "imputation medians are fitted solely on the permitted earlier training rows. Warm-up "
                "rows do not enter validation objectives or scores; their labels are used only if already "
                "part of the permitted training set. Each inner/outer fold has its own history, "
                "including the feature rows in its purge/validation-head gap. Public inference follows "
                "the same warm-up procedure.",
            90: "Historical context: an earlier version filled the undefined first 19 blend values with "
                "zero and reported 2.523 publicly; a later fix reported 1.239 with its historical models. "
                "Those numbers describe the earlier pipeline, not the revised nested experiment. The "
                "current code tests warm-up explicitly and does not assume that disjoint OOF folds are contiguous.",
            92: "Figure 3. HISTORICAL illustration of the blend warm-up bug from the original report. "
                "It is retained as development history, not as a plot of the revised nested experiment.",
            103: "Table 16. Revised results. Baseline public rows use their recorded procedures; notes in "
                 "leaderboard.csv distinguish artifact replay from new training. Their outer CV cells are "
                 "intentionally empty. Proposed, improved and passive CV rows use nested evaluation.",
            105: "Figure 4a. Revised nested outer modified Sharpe, including passive w=1.",
            107: "Figure 4b. Historically consulted public diagnostic, including the passive index and available baselines.",
            109: "Figure 5. Recomputed public wealth curves for proposed, improved and passive w=1, "
                 "using the aligned predictions and allocations in results/analysis/predictions.csv.",
            111: "Baselines versus proposed. The original baseline scores reproduce when their archived "
                 "artifacts are replayed. They are public-only reference points with different preprocessing, "
                 "not a controlled comparison on the nested outer folds. " + seed_statement,
            113: "Improved versus proposed. The selection procedure is now evaluated on later outer blocks. "
                 "A gain is only descriptive unless the paired interval and temporal limitations support a "
                 "stronger conclusion. The historical 70% tuning-fold increase is not independent evidence.",
            114: "Failures and regime dependence are visible in the per-fold tables and wealth curves. "
                 "A selected policy can lose to the passive reference on later blocks; the analysis retains "
                 "those outcomes rather than switching the rule after looking at them.",
            118: f"Paired moving-block bootstrap: {int(bootstrap_config.block_size)}-day non-circular blocks, "
                 f"{int(bootstrap_config.draws)} resamples, seed {int(bootstrap_config.seed)}, "
                 "sampled within each fold; all strategies, market returns and risk-free returns use the same "
                 "resampled rows. Results: " + "; ".join(ci_lines) + ". Intervals are conditional on fitted "
                 "strategies and do not include retraining/selection uncertainty. Fit seeds share the same "
                 "time series, so their paired t-test is a sensitivity diagnostic, not proof of market generalisation.",
            121: f"The revised verification used Python {manifest.get('python', platform.python_version())}. "
                 "Direct dependencies are pinned in requirements.txt; installed model-library versions "
                 "are recorded in manifest.json. The earlier report was developed on Python 3.12/3.13.",
            123: f"Model-search seed={config['seed']}; ablation fit seeds 0–{config['seed_count']-1}; "
                 f"CPU threads={config['n_jobs']}. Package versions: " + json.dumps(versions, sort_keys=True) + ".",
            136: "python data_checks.py && python report_analysis.py  # nested evaluation, ablation, paired bootstrap and figures",
            138: "Table 17. Current execution paths and outputs. Full retraining runtimes depend on hardware; "
                 "old laptop timing estimates are not claimed for the revised experiments.",
            139: "Run report_analysis.py before the two model notebooks; they load its shared artifacts. "
                 "Use --quick for isolated results/smoke output and --reuse to regenerate statistics from "
                 "stored predictions. B2 loads both model and feature list before plotting; absent "
                 "checkpoints trigger training. Incomplete checkpoints are preserved with a clear error. "
                 "To force a fresh baseline run, choose a new HULL_RESULTS_DIR "
                 "instead of deleting existing artifacts. Python unittest discovers the regression suite.",
            140: "Reproducibility limits: original B2/B3 searches were not fully seeded, so archived-artifact "
                 "replay and new seeded retraining are distinct. B3's online branch is inactive. New searches "
                 "have fixed trial counts and explicit seeds; cross-platform bitwise identity is not claimed. "
                 "data_checks.py reproduces internal distributional/chronological checks; it does not claim "
                 "to recreate external index-return matches without a bundled reference series.",
            144: "Domain features are used by the model, but their original single-seed +0.013 IC is "
                 "insufficient evidence of a robust benefit. " + seed_statement,
            145: f"The revised improved pipeline scores {display('improved')} on outer CV versus "
                 f"{display('passive')} for passive and {display('proposed')} for proposed. "
                 "This replaces selection-fold improvement claims; the public results remain retrospective.",
            148: "Small diagnostic period. The public block has only 180 contiguous observations from one "
                 "market regime and was consulted during development. Bootstrap precision cannot restore test independence.",
            149: "Historical versus revised CV. The old model/blend/policy selection reused reported folds. "
                 "The revised code nests all selection within outer training periods, with train-only "
                 "imputation and no outer early stopping. Retrospective validation on a previously studied "
                 "dataset still cannot establish prospective performance.",
            150: "Selection and seed sensitivity. There are only four outer periods and a deliberately finite "
                 "search budget. The 20 paired seeds hold hyperparameters fixed per fold and do not cover "
                 "the variability of rerunning the entire search. Allocation choice may vary across periods.",
            152: "Volatility and costs. Check the measured volatility ratios against 1.2 in the revised tables. "
                 "The metric penalises excess volatility but does not enforce a hard constraint; transaction "
                 "costs are not modelled and high turnover can reduce realised performance.",
            156: "Prospective evaluation on newly arriving market data, plus more nested walk-forward windows "
                 "and repeated full-search seeds. Nested temporal selection is already implemented in this revision.",
        }
        for index, replacement in replacements.items():
            set_text(paragraphs[index], replacement)
        table0 = [["Model", "Nested outer modified Sharpe (mean ± SD)", "Public diagnostic modified Sharpe"]]
        for name, label in [("passive", "Buy-and-hold (w=1)"), ("proposed", "Proposed hybrid LightGBM"),
                            ("improved", "Improved selection procedure")]:
            table0.append([label, display(name), display(name, "public")])
        table0.append(["B2 public baseline", "Not evaluated",
                       f"{baselines.loc['lgbm_61st', 'modified_sharpe']:.4f}"])
        set_table(tables[0], table0)
        # Preserve source data and baseline tables, correcting the executed B3 procedure.
        cells = tables[1].findall(W + "tr")[-1].findall(W + "tc")
        set_text(cells[1], "Forward-fill and training-fold median, recomputed inside every inner/outer model fit")
        cells = tables[4].findall(W + "tr")[-1].findall(W + "tc")
        set_text(cells[2], "Frozen ElasticNet + XGBoost + LightGBM, weights 0.30/0.35/0.35; volatility-scaled smoothed allocation")
        set_text(cells[3], "Top-50 features; offline fit and optional cached parameters. The intended 800-row "
                 "updates / 50-row retuning do not execute: create_features removes the lagged target "
                 "before previous_lagged is stored. New searches use seeded samplers.")
        for index in (5, 16):
            for cell in tables[index].iter(W + "tc"):
                if "Online ensemble" in text_of(cell):
                    set_text(cell, text_of(cell).replace("Online ensemble", "Frozen ensemble"))
        rows = [["Model", "Spearman IC", "Hit rate", "Mod. Sharpe", "Sharpe", "Vol ratio",
                 "Return penalty", "Ann. return", "Mean weight"]]
        for name, label in baseline_labels.items():
            row = baselines.loc[name]
            rows.append([label] + [f"{row[key]:.4f}" for key in ["spearman_ic", "hit_rate", "modified_sharpe",
                         "sharpe", "vol_ratio", "return_penalty", "ann_return", "mean_weight"]])
        public_passive = metrics.loc[(metrics.split == "public") & (metrics.model == "passive")].iloc[0]
        rows.append(["Buy-and-hold (w=1)", "—", "—"] +
                    [f"{public_passive[key]:.4f}" for key in ["modified_sharpe", "sharpe", "vol_ratio",
                     "return_penalty", "ann_return", "mean_weight"]])
        set_table(tables[5], rows)
        params = manifest["public_selection"]["models"]["lgbm"]
        ranges = {"n_estimators": f"{min(60, config['max_estimators'])}–{config['max_estimators']}",
                  "learning_rate": "0.01–0.07 log", "max_depth": "4–8",
                  "num_leaves": "16–64", "reg_lambda": "0.001–10 log",
                  "colsample_bytree": "0.6–1.0", "subsample": "0.6–1.0"}
        set_table(tables[7], [["Parameter", "Revised search range", "Selected for public diagnostic"]] +
                  [[key, ranges[key], f"{value:.5g}"] for key, value in params.items()])
        outer_proposed = metrics.loc[(metrics.split == "cv") & (metrics.model == "proposed")]
        passive = metric("passive").to_numpy()
        rows = [["Fold", "Spearman IC", "RMSE", "Hit rate", "Mod. Sharpe", "Passive mod. Sharpe", "Vol ratio", "Mean weight"]]
        for i, (_, row) in enumerate(outer_proposed.iterrows()):
            rows.append([str(row.fold), f"{row.spearman_ic:.4f}", f"{row.rmse:.4f}", f"{row.hit_rate:.4f}",
                         f"{row.modified_sharpe:.4f}", f"{passive[i]:.4f}", f"{row.vol_ratio:.4f}", f"{row.mean_weight:.4f}"])
        set_table(tables[8], rows)
        rows = [["Model", "Spearman IC", "RMSE", "Hit rate", "Mod. Sharpe", "Sharpe", "Vol ratio", "Ann. return", "Max DD", "Turnover"]]
        for model in ("passive", "proposed", "improved"):
            row = metrics.loc[(metrics.split == "public") & (metrics.model == model)].iloc[0]
            rows.append([model] + [f"{row[key]:.4f}" for key in ["spearman_ic", "rmse", "hit_rate", "modified_sharpe",
                         "sharpe", "vol_ratio", "ann_return", "max_drawdown", "weight_turnover"]])
        set_table(tables[9], rows)
        set_table(tables[10], [["#", "Change", "Type", "Motivation"],
            ["1", "LightGBM, CatBoost and standardised Ridge, tuned on inner folds", "Model families", "Different inductive biases"],
            ["2", "Average or inverse-volatility blend, selected by inner mean IC", "Ensemble", "Choice made before outer evaluation"],
            ["3", f"Naive, binary, ternary and vol_target: {config['allocation_trials']} trials per rule; fixed passive alternative", "Sizing", "Select mean minus 0.25 SD of inner modified Sharpe"],
            ["4", "Fixed inner-selected tree counts; imputation inside every fit; no outer early stopping", "Evaluation", "Outer labels excluded from selection"],
            ["5", f"Paired {config['seed_count']}-seed domain ablation, explicit gain importance and paired block intervals", "Reliability", "Quantify fit randomness and temporal sampling limits separately"]])
        rows = [["Component", "Trials per selection", "Search space", "Public-diagnostic selection", "Evaluation"]]
        for family, chosen in manifest["public_selection"]["models"].items():
            rows.append([family, str(config["model_trials"]), "See src/experiments.py", json.dumps(chosen), "Inner tuning only"])
        rows.append(["Allocation", str(config["allocation_trials"]) + " per rule", "Four rules plus passive", json.dumps(manifest["public_selection"]["allocation"]), "Inner tuning only"])
        set_table(tables[11], rows)
        rows = [["Pipeline", "Fold 0 IC", "Fold 1 IC", "Fold 2 IC", "Fold 3 IC", "Mean IC"]]
        for model in ("passive", "proposed", "improved"):
            values = metric(model, key="spearman_ic")
            rows.append([model] + [f"{v:.4f}" for v in values] + [f"{values.mean():.4f}"])
        set_table(tables[12], rows)
        rows = [["Split", "Model", "Spearman IC", "Mod. Sharpe", "Sharpe", "Vol ratio", "Ann. return", "Max DD", "Mean weight", "Turnover"]]
        for split in ("cv", "public"):
            for model in ("passive", "proposed", "improved"):
                rows.append(["Nested outer" if split == "cv" else "Public diagnostic", model] +
                            [display(model, split, key) for key in ["spearman_ic", "modified_sharpe", "sharpe", "vol_ratio",
                             "ann_return", "max_drawdown", "mean_weight", "weight_turnover"]])
        set_table(tables[14], rows)
        old_rows = [[text_of(c) for c in row.findall(W + "tc")] for row in tables[15].findall(W + "tr")
                    if "Buy-and-hold" not in text_of(row)]
        set_table(tables[15], old_rows + [["Buy-and-hold (w=1)", display("passive", "public"), display("passive", "public")]])
        rows = [["Model", "Stage", "Nested CV mod. Sharpe", "Nested CV IC", "Public mod. Sharpe", "Public IC", "Public Sharpe", "Public vol ratio"]]
        for name, label in baseline_labels.items():
            row = baselines.loc[name]
            rows.append([label, "baseline", "—", "—"] +
                        [f"{row[key]:.4f}" for key in ["modified_sharpe", "spearman_ic", "sharpe", "vol_ratio"]])
        for model in ("passive", "proposed", "improved"):
            rows.append([model, "reference" if model == "passive" else model,
                         display(model), display(model, key="spearman_ic"), display(model, "public"),
                         display(model, "public", "spearman_ic"), display(model, "public", "sharpe"), display(model, "public", "vol_ratio")])
        set_table(tables[16], rows)
        set_table(tables[17], [["Step", "Output", "Runtime scope"],
            ["Baseline artifact replay", "leaderboard and aligned allocations", "Verified with supplied B2/B3 artifacts"],
            ["Baseline fresh training", "model/features or cached parameters", "Separate seeded search; full retraining time not measured"],
            ["python report_analysis.py", "Nested evaluation, 20-seed ablation, manifest, bootstrap and figures", "Explicit budgets in manifest"],
            ["Both model notebooks", "Views of the same shared analysis artifacts", "Reuse existing analysis; generate if absent"],
            ["python -m unittest discover -s tests -v", "Behavioral regression checks", "CPU"]])
        # Replace plot assets in-place, preserving the Word layout and captions.
        relations = ET.fromstring(original.read("word/_rels/document.xml.rels"))
        targets = {r.get("Id"): "word/" + r.get("Target") for r in relations}
        replacements_media = {targets[rid]: (directory / "figures" / filename).read_bytes()
                              for rid, filename in [("rId8", "figure_1.png"),
                                                    ("rId9", "figure_2.png"),
                                                    ("rId11", "figure_4a.png"),
                                                    ("rId12", "figure_4b.png"),
                                                    ("rId13", "figure_5.png")]}
        # Match the Word drawing height to the new PNG aspect ratio.
        drawing_ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
        wp_ns = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
        rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
        for drawing in root.iter(W + "drawing"):
            blip = drawing.find(".//" + drawing_ns + "blip")
            if blip is None:
                continue
            target = targets.get(blip.get(rel_ns + "embed"))
            if target in replacements_media:
                width, height = struct.unpack(">II", replacements_media[target][16:24])
                for extent in [*drawing.iter(wp_ns + "extent"),
                               *drawing.iter(drawing_ns + "ext")]:
                    if "cx" in extent.attrib:
                        extent.set("cy", str(round(int(extent.get("cx")) * height / width)))
        # ElementTree discards unused namespace declarations, but Word's
        # mc:Ignorable may refer to them. Preserve those original declarations.
        preview = ET.tostring(root, encoding="unicode")
        for prefix, uri in namespaces.items():
            if prefix and f"xmlns:{prefix}=" not in preview:
                root.set(f"xmlns:{prefix}", uri)
        revised = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for item in original.infolist():
                data = revised if item.filename == "word/document.xml" else replacements_media.get(item.filename, original.read(item.filename))
                if item.filename == "docProps/core.xml":
                    for _, (prefix, uri) in ET.iterparse(BytesIO(data), events=["start-ns"]):
                        ET.register_namespace(prefix, uri)
                    core = ET.fromstring(data)
                    modified = core.find("{http://purl.org/dc/terms/}modified")
                    if modified is not None:
                        modified.text = datetime.now(timezone.utc).isoformat()
                    data = ET.tostring(core, encoding="utf-8", xml_declaration=True)
                output.writestr(item, data)
        readable = []
        for element in root.find(W + "body"):
            if element.tag == W + "p":
                readable.append(text_of(element))
            elif element.tag == W + "tbl":
                readable.extend(" | ".join(text_of(c) for c in row.findall(W + "tc"))
                                for row in element.findall(W + "tr"))
            readable.append("")
        destination.with_suffix(".txt").write_text("\n".join(readable).rstrip() + "\n")
        record = {"source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  "data_sha256": manifest["data_sha256"], "config": config,
                  "data_checks": checks,
                  "ablation": summary, "bootstrap": intervals.to_dict(orient="records"),
                  "versions": versions, "python": manifest.get("python", platform.python_version()),
                  "report_source": Path(source).name,
                  "baseline_public_metrics": baselines[["modified_sharpe", "spearman_ic", "vol_ratio"]].to_dict(orient="index"),
                  "outer_fold_metrics": metrics.loc[metrics.split == "cv", ["model", "fold", "modified_sharpe", "spearman_ic", "vol_ratio"]].to_dict(orient="records"),
                  "historical_pdf": "Latest PDF metadata: project.zip/project.pdf, 2026-09-16 22:04:22 Europe/Moscow"}
        (destination.parent / "validation_summary.json").write_text(json.dumps(record, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--output", type=Path, default=Path("reports/project_report.docx"))
    args = parser.parse_args()
    print(revise(args.source, args.analysis_dir, args.output))
