"""
Purged / embargoed time-series cross-validation.

Every notebook in this project must call :func:`get_folds` with the default
settings so that baseline, proposed and improved models are compared on
mathematically identical folds.

Concept
-------
Standard ``TimeSeriesSplit`` leaks in two ways on financial data:

1. The last training row's target overlaps the first validation row (the target
   is a *forward* return). Fixed by **purging** ``purge`` rows before each
   validation block.
2. Serial correlation makes rows immediately after a validation block
   informative about it. Fixed by an **embargo** of ``embargo`` rows after each
   block, relevant once training windows are allowed to extend past a block.

The folds are a deterministic function of the number of rows only, so two
notebooks operating on the same dataframe get byte-identical splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CV_EMBARGO, CV_N_SPLITS, CV_PURGE, DATE_COL


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold, stored as positional indices."""

    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Fold({self.index}: train={len(self.train_idx)} rows, "
            f"val={len(self.val_idx)} rows)"
        )


class PurgedTimeSeriesSplit:
    """Expanding-window splitter with purge and embargo gaps.

    Sklearn-compatible: exposes ``split`` and ``get_n_splits``.
    """

    def __init__(
        self,
        n_splits: int = CV_N_SPLITS,
        purge: int = CV_PURGE,
        embargo: int = CV_EMBARGO,
        max_train_size: int | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.max_train_size = max_train_size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803
        return self.n_splits

    def split(self, X, y=None, groups=None):  # noqa: N803
        n_samples = len(X)
        # Equal-sized validation blocks covering the tail of the series.
        fold_size = n_samples // (self.n_splits + 1)
        if fold_size <= self.purge + self.embargo:
            raise ValueError(
                f"{n_samples} rows is too few for {self.n_splits} splits with "
                f"purge={self.purge}, embargo={self.embargo}."
            )

        indices = np.arange(n_samples)
        for i in range(self.n_splits):
            val_start = fold_size * (i + 1)
            val_end = n_samples if i == self.n_splits - 1 else fold_size * (i + 2)

            train_end = val_start - self.purge
            train_idx = indices[:train_end]
            if self.max_train_size is not None:
                train_idx = train_idx[-self.max_train_size :]

            val_idx = indices[val_start:val_end]
            # Embargo trims the head of the validation block so it never sits
            # flush against training rows it is correlated with.
            if self.embargo:
                val_idx = val_idx[self.embargo :]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield train_idx, val_idx


def get_folds(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    purge: int = CV_PURGE,
    embargo: int = CV_EMBARGO,
    max_train_size: int | None = None,
) -> list[Fold]:
    """Return the canonical fold list. **Use the defaults in every notebook.**"""
    splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits, purge=purge, embargo=embargo, max_train_size=max_train_size
    )
    return [
        Fold(index=i, train_idx=tr, val_idx=va)
        for i, (tr, va) in enumerate(splitter.split(df))
    ]


def describe_folds(df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Human-readable fold summary; print this in every notebook as evidence
    that the splits match across models."""
    rows = []
    has_date = DATE_COL in df.columns
    for fold in folds:
        row = {
            "fold": fold.index,
            "n_train": len(fold.train_idx),
            "n_val": len(fold.val_idx),
        }
        if has_date:
            dates = df[DATE_COL].to_numpy()
            row["train_end_date_id"] = dates[fold.train_idx[-1]]
            row["val_start_date_id"] = dates[fold.val_idx[0]]
            row["val_end_date_id"] = dates[fold.val_idx[-1]]
            row["gap_days"] = row["val_start_date_id"] - row["train_end_date_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def assert_no_leakage(folds: list[Fold], purge: int = CV_PURGE) -> None:
    """Fail loudly if any fold's training indices reach into its validation
    block or violate the purge gap."""
    for fold in folds:
        overlap = np.intersect1d(fold.train_idx, fold.val_idx)
        if overlap.size:
            raise AssertionError(f"Fold {fold.index}: {overlap.size} overlapping rows.")
        gap = fold.val_idx[0] - fold.train_idx[-1]
        if gap <= purge:
            raise AssertionError(
                f"Fold {fold.index}: gap of {gap} rows violates purge={purge}."
            )
