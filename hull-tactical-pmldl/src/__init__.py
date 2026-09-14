"""
Reusable core modules for the Hull Tactical Market Prediction project (PMLDL 2026, Stage 2).

Modules
-------
config      : global constants, random seed, paths.
data        : dataset loading (real Kaggle CSV or reproducible synthetic surrogate).
features    : domain / temporal feature engineering + leak-safe imputation pipeline.
validation  : purged & embargoed time-series cross-validation.
metrics     : RMSE, R2, Spearman IC, Sharpe and the competition-style penalised Sharpe.
allocation  : position-sizing rules (binary, naive linear, volatility targeting).
"""

from . import allocation, config, data, experiment, features, metrics, validation  # noqa: F401

__all__ = ["config", "data", "features", "validation", "metrics", "allocation", "experiment"]
