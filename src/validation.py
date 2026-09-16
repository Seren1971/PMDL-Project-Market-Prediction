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

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None,
    ) -> int:  # noqa: N803
        return self.n_splits

    def split(
        self,
        X,
        y=None,
        groups=None,
    ):  # noqa: N803
        n_samples = len(X)

        # Equal-sized validation blocks covering the tail of the series.
        fold_size = n_samples // (
            self.n_splits + 1
        )

        if fold_size <= (
            self.purge
            + self.embargo
        ):
            raise ValueError(
                f"{n_samples} rows is too few for {self.n_splits} splits with "
                f"purge={self.purge}, embargo={self.embargo}."
            )

        indices = np.arange(
            n_samples
        )

        for i in range(
            self.n_splits
        ):
            val_start = (
                fold_size
                * (i + 1)
            )

            val_end = (
                n_samples
                if i
                == self.n_splits - 1
                else fold_size
                * (i + 2)
            )

            train_end = (
                val_start
                - self.purge
            )

            train_idx = indices[
                :train_end
            ]

            if (
                self.max_train_size
                is not None
            ):
                train_idx = train_idx[
                    -self.max_train_size :
                ]

            val_idx = indices[
                val_start:val_end
            ]

            # Embargo trims the head of the validation block so it never sits
            # flush against training rows it is correlated with.
            if self.embargo:
                val_idx = val_idx[
                    self.embargo :
                ]

            if (
                len(train_idx) == 0
                or len(val_idx) == 0
            ):
                continue

            yield (
                train_idx,
                val_idx,
            )


def get_folds(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    purge: int = CV_PURGE,
    embargo: int = CV_EMBARGO,
    max_train_size: int | None = None,
) -> list[Fold]:
    """Return the canonical fold list. Use the defaults in every notebook."""
    splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        purge=purge,
        embargo=embargo,
        max_train_size=max_train_size,
    )

    return [
        Fold(
            index=i,
            train_idx=tr,
            val_idx=va,
        )
        for i, (tr, va)
        in enumerate(
            splitter.split(df)
        )
    ]


def describe_folds(
    df: pd.DataFrame,
    folds: list[Fold],
) -> pd.DataFrame:
    """Human-readable fold summary with JSON-friendly Python scalars."""
    rows: list[
        dict[
            str,
            int,
        ]
    ] = []

    dates = (
        df[
            DATE_COL
        ].to_numpy()
        if DATE_COL in df.columns
        else None
    )

    for fold in folds:
        row: dict[
            str,
            int,
        ] = {
            "fold": int(
                fold.index
            ),
            "n_train": int(
                len(
                    fold.train_idx
                )
            ),
            "n_val": int(
                len(
                    fold.val_idx
                )
            ),
        }

        if dates is not None:
            train_end_position = int(
                fold.train_idx[
                    -1
                ]
            )

            val_start_position = int(
                fold.val_idx[
                    0
                ]
            )

            val_end_position = int(
                fold.val_idx[
                    -1
                ]
            )

            train_end = int(
                np.asarray(
                    dates[
                        train_end_position
                    ]
                ).item()
            )

            val_start = int(
                np.asarray(
                    dates[
                        val_start_position
                    ]
                ).item()
            )

            val_end = int(
                np.asarray(
                    dates[
                        val_end_position
                    ]
                ).item()
            )

            row.update(
                {
                    "train_end_date_id": train_end,
                    "val_start_date_id": val_start,
                    "val_end_date_id": val_end,
                    "gap_days": (
                        val_start
                        - train_end
                    ),
                }
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def assert_no_leakage(
    folds: list[Fold],
    purge: int = CV_PURGE,
) -> None:
    """
    Fail loudly if any fold's training indices reach into its validation
    block or violate the purge gap.
    """
    for fold in folds:
        overlap = np.intersect1d(
            fold.train_idx,
            fold.val_idx,
        )

        if overlap.size:
            raise AssertionError(
                f"Fold {fold.index}: "
                f"{overlap.size} overlapping rows."
            )

        gap = int(
            fold.val_idx[
                0
            ]
            - fold.train_idx[
                -1
            ]
        )

        if gap <= purge:
            raise AssertionError(
                f"Fold {fold.index}: gap of {gap} rows "
                f"violates purge={purge}."
            )