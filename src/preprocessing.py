"""Leak-safe preprocessing helpers for chronological model validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForwardMedianImputer:
    """Forward-fill with train history, then use train-only medians.

    ``fit`` learns both the median and the last observed value from the training
    frame. ``transform_future`` seeds forward-fill with the last training value,
    so the first validation row may use information available at the end of the
    training period but can never use a future validation observation.
    """

    columns: list[str]
    medians_: pd.Series | None = None
    last_values_: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "ForwardMedianImputer":
        frame = train[self.columns]
        self.medians_ = frame.median().fillna(0.0)
        self.last_values_ = frame.ffill().iloc[-1].fillna(self.medians_)
        return self

    def _check_fitted(self) -> None:
        if self.medians_ is None or self.last_values_ is None:
            raise RuntimeError("ForwardMedianImputer must be fitted first.")

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        out = train.copy()
        out[self.columns] = (
            out[self.columns]
            .ffill()
            .fillna(self.medians_)
            .fillna(0.0)
        )
        return out

    def transform_future(self, future: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        out = future.copy()
        values = out[self.columns].copy()
        seed = pd.DataFrame([self.last_values_], columns=self.columns)
        seeded = pd.concat([seed, values], ignore_index=True).ffill().iloc[1:]
        seeded.index = values.index
        out[self.columns] = seeded.fillna(self.medians_).fillna(0.0)
        return out

    def fit_transform_pair(
        self, train: pd.DataFrame, future: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.fit(train)
        return self.transform_train(train), self.transform_future(future)


def standardization_stats(values: np.ndarray) -> tuple[float, float]:
    """Return stable mean/std values for a one-dimensional signal."""
    arr = np.asarray(values, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return mu, sd


def apply_standardization(
    values: np.ndarray, stats: tuple[float, float]
) -> np.ndarray:
    """Apply frozen train-derived standardization statistics."""
    mu, sd = stats
    return (np.asarray(values, dtype=float) - mu) / sd