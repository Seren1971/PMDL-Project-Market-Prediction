"""Shared experiment runner + results registry.

All five notebooks evaluate their models through :func:`run_cv`, which guarantees:

* identical folds (``src.validation.default_cv``),
* identical imputation / scaling protocol (fitted inside the training fold only),
* identical metric computation (``src.metrics``),
* results persisted to ``results/<name>.json`` so the final notebook can build one
  comparison table across every stage of the project.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .allocation import binary_allocation
from .config import RESULTS_DIR, RISK_FREE_COL, RETURN_COL, SEED
from .features import LeakSafeImputer
from .metrics import evaluate_positions, evaluate_predictions
from .validation import PurgedTimeSeriesSplit, default_cv

__all__ = [
    "FoldResult",
    "CVResult",
    "run_cv",
    "aggregate_folds",
    "result_from_positions",
    "save_result",
    "load_results",
    "comparison_table",
    "market_series",
]


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_valid: int
    metrics: dict[str, float]


@dataclass
class CVResult:
    name: str
    stage: str
    metrics: dict[str, float]
    fold_metrics: list[dict] = field(default_factory=list)
    oof_index: list[int] = field(default_factory=list)
    oof_pred: list[float] = field(default_factory=list)
    notes: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"model": self.name, **self.metrics}]).set_index("model")


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    market_returns: pd.Series,
    risk_free: pd.Series | None,
    model_factory: Callable[[], object],
    name: str,
    stage: str = "",
    cv: PurgedTimeSeriesSplit | None = None,
    allocator: Callable[[np.ndarray], np.ndarray] = binary_allocation,
    scale: bool = False,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> CVResult:
    """Fit ``model_factory()`` on every purged fold and report averaged metrics.

    Parameters
    ----------
    allocator
        Maps validation-fold predictions to positions in ``[0, 2]``.
    scale
        Standardise features with statistics fitted on the training fold (linear models).
    """
    cv = cv or default_cv()
    fit_kwargs = fit_kwargs or {}
    fold_rows: list[dict] = []
    oof_idx: list[int] = []
    oof_pred: list[float] = []

    for k, (tr, va) in enumerate(cv.split(X), start=1):
        imp = LeakSafeImputer().fit(X.iloc[tr])
        X_tr, X_va = imp.transform(X.iloc[tr]), imp.transform(X.iloc[va])

        if scale:
            from sklearn.preprocessing import StandardScaler

            sc = StandardScaler().fit(X_tr)
            X_tr = pd.DataFrame(sc.transform(X_tr), index=X_tr.index, columns=X_tr.columns)
            X_va = pd.DataFrame(sc.transform(X_va), index=X_va.index, columns=X_va.columns)

        model = model_factory()
        model.fit(X_tr, y.iloc[tr], **fit_kwargs)
        pred = np.asarray(model.predict(X_va), dtype=float).ravel()

        pos = np.asarray(allocator(pred), dtype=float).ravel()
        mkt = market_returns.iloc[va].to_numpy()
        rf = None if risk_free is None else risk_free.iloc[va].to_numpy()

        metrics = {
            **evaluate_predictions(y.iloc[va].to_numpy(), pred),
            **evaluate_positions(pos, mkt, rf),
        }
        fold_rows.append({"fold": k, "n_train": len(tr), "n_valid": len(va), **metrics})
        oof_idx.extend(np.asarray(X.index[va]).tolist())
        oof_pred.extend(pred.tolist())

        if verbose:
            print(
                f"  fold {k}: IC={metrics['spearman_ic']:+.4f} "
                f"RMSE={metrics['rmse']:.5f} "
                f"penalised Sharpe={metrics['penalised_sharpe']:+.3f}"
            )

    folds = pd.DataFrame(fold_rows)
    mean_metrics = folds.drop(columns=["fold", "n_train", "n_valid"]).mean(numeric_only=True)
    result = CVResult(
        name=name,
        stage=stage,
        metrics={k: float(v) for k, v in mean_metrics.items()},
        fold_metrics=fold_rows,
        oof_index=[int(i) for i in oof_idx],
        oof_pred=[float(p) for p in oof_pred],
    )
    if verbose:
        print(
            f"  >>> {name}: mean IC={result.metrics['spearman_ic']:+.4f} | "
            f"mean penalised Sharpe={result.metrics['penalised_sharpe']:+.3f}"
        )
    return result


def result_from_positions(
    name: str,
    positions: np.ndarray,
    market_returns: np.ndarray,
    risk_free: np.ndarray | None,
    stage: str = "",
    notes: str = "",
    predictions: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> CVResult:
    """Build a :class:`CVResult` directly from a position series (e.g. walk-forward loops)."""
    metrics = dict(evaluate_positions(positions, market_returns, risk_free))
    if predictions is not None and y_true is not None:
        metrics.update(evaluate_predictions(y_true, predictions))
    return CVResult(name=name, stage=stage, metrics=metrics, notes=notes)


def aggregate_folds(
    name: str,
    fold_rows: list[dict],
    stage: str = "",
    notes: str = "",
    oof_index: list[int] | None = None,
    oof_pred: list[float] | None = None,
) -> CVResult:
    """Average per-fold metric dictionaries into a :class:`CVResult` (custom CV loops)."""
    folds = pd.DataFrame(fold_rows)
    drop = [c for c in ("fold", "n_train", "n_valid") if c in folds.columns]
    mean_metrics = folds.drop(columns=drop).mean(numeric_only=True)
    return CVResult(
        name=name,
        stage=stage,
        metrics={k: float(v) for k, v in mean_metrics.items()},
        fold_metrics=fold_rows,
        oof_index=[int(i) for i in (oof_index or [])],
        oof_pred=[float(p) for p in (oof_pred or [])],
        notes=notes,
    )


def save_result(result: CVResult, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{_slug(result.name)}.json"
    payload = asdict(result)
    payload["seed"] = SEED
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_results(results_dir: Path = RESULTS_DIR) -> list[CVResult]:
    results_dir = Path(results_dir)
    out: list[CVResult] = []
    for p in sorted(results_dir.glob("*.json")):
        d = json.loads(p.read_text())
        d.pop("seed", None)
        out.append(CVResult(**d))
    return out


def comparison_table(
    results: list[CVResult] | None = None,
    columns: list[str] | None = None,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """One tidy table with every model saved so far, sorted by penalised Sharpe."""
    results = results if results is not None else load_results(results_dir)
    columns = columns or [
        "spearman_ic",
        "rmse",
        "r2",
        "sharpe",
        "penalised_sharpe",
        "ann_return",
        "ann_volatility",
        "vol_ratio_vs_market",
        "max_drawdown",
        "mean_position",
    ]
    rows = []
    for r in results:
        row = {"stage": r.stage, "model": r.name}
        row.update({c: r.metrics.get(c, float("nan")) for c in columns})
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.set_index(["stage", "model"])
        .sort_values("penalised_sharpe", ascending=False)
        .round(4)
    )


def market_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series | None]:
    """Forward market return and risk-free series aligned with the model rows."""
    mkt = df[RETURN_COL] if RETURN_COL in df.columns else df["market_forward_excess_returns"]
    rf = df[RISK_FREE_COL] if RISK_FREE_COL in df.columns else None
    return mkt, rf


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.lower())
