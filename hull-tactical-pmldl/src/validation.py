"""Leak-safe validation for a forward-looking financial target.

Why not ``KFold``?
------------------
The target at row *t* (``market_forward_excess_returns``) is realised at *t+1* and the
engineered features at *t* are built from a rolling window that ends at *t*.  A random
split would therefore let the model see the future, and a plain ``TimeSeriesSplit`` would
still let the last training rows overlap the first validation rows through both the target
horizon and the rolling windows.

``PurgedTimeSeriesSplit`` fixes both problems:

* **purge**  - drop the last ``purge`` training rows before every validation block so that
  no training label depends on a validation observation (label-horizon overlap);
* **embargo** - additionally drop ``embargo`` rows so that rolling-window features of the
  first validation rows do not overlap the training window (serial-correlation leakage);
* **expanding or sliding** training windows (``max_train_size``) for regime adaptivity.

Every notebook in this project instantiates the folds through :func:`default_cv`, so all
models are compared on **mathematically identical folds**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

__all__ = ["PurgedTimeSeriesSplit", "default_cv", "describe_folds"]


@dataclass
class PurgedTimeSeriesSplit:
    """Chronological CV with purging and an embargo period.

    Parameters
    ----------
    n_splits
        Number of validation blocks.
    test_size
        Rows per validation block.  ``None`` -> ``n_samples // (n_splits + 1)``.
    purge
        Rows removed at the end of the training window because their label overlaps the
        validation block (>= the label horizon, which is 1 day here).
    embargo
        Extra rows removed after the purge, covering feature-side leakage from rolling
        windows.
    max_train_size
        If set, the training window slides instead of expanding (keeps at most this many
        of the most recent rows).
    min_train_size
        Folds whose training window would be smaller than this are skipped.
    """

    n_splits: int = 5
    test_size: int | None = None
    purge: int = 1
    embargo: int = 10
    max_train_size: int | None = None
    min_train_size: int = 250

    # ---------------------------------------------------------------- sklearn API
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _n_samples(X)
        test_size = self.test_size or max(1, n // (self.n_splits + 1))
        gap = int(self.purge) + int(self.embargo)

        # Validation blocks are laid out contiguously at the end of the series.
        starts = [n - (self.n_splits - i) * test_size for i in range(self.n_splits)]
        for start in starts:
            stop = start + test_size
            train_end = start - gap
            if train_end <= 0:
                continue
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            if train_end - train_start < self.min_train_size:
                continue
            yield (
                np.arange(train_start, train_end, dtype=int),
                np.arange(start, min(stop, n), dtype=int),
            )


def _n_samples(X) -> int:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return len(X)
    if isinstance(X, (list, tuple)):
        return len(X)
    return int(np.asarray(X).shape[0])


def default_cv(
    n_splits: int = 5,
    embargo: int = 10,
    purge: int = 1,
    max_train_size: int | None = None,
) -> PurgedTimeSeriesSplit:
    """The single CV configuration shared by *all* notebooks of this project."""
    return PurgedTimeSeriesSplit(
        n_splits=n_splits,
        purge=purge,
        embargo=embargo,
        max_train_size=max_train_size,
    )


def describe_folds(cv: PurgedTimeSeriesSplit, X, index: Sequence | None = None) -> pd.DataFrame:
    """Human-readable fold layout (used in the notebooks to prove folds are identical)."""
    rows = []
    for k, (tr, va) in enumerate(cv.split(X), start=1):
        rows.append(
            {
                "fold": k,
                "train_start": int(tr[0]),
                "train_end": int(tr[-1]),
                "n_train": len(tr),
                "gap": int(va[0] - tr[-1] - 1),
                "valid_start": int(va[0]),
                "valid_end": int(va[-1]),
                "n_valid": len(va),
            }
        )
    return pd.DataFrame(rows)
