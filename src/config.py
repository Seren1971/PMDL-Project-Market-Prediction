"""
Central configuration: seeds, paths, column conventions.

PMLDL 2026 Stage 2 project - Hull Tactical Market Prediction.
Every notebook imports from here; nothing is hard-coded downstream.
"""

import importlib
import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
TRADING_DAYS = 252


def set_seed(seed: int = SEED) -> int:
    """Seed every RNG this project can touch. Call once per notebook."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# src/ lives directly under the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(
    os.environ.get(
        "HULL_DATA_DIR",
        PROJECT_ROOT / "data" / "raw",
    )
)

RESULTS_DIR = Path(
    os.environ.get(
        "HULL_RESULTS_DIR",
        PROJECT_ROOT / "results",
    )
)

ARTIFACT_DIR = RESULTS_DIR / "artifacts"

# Kaggle kernels mount the competition here; used as a fallback by the loader.
KAGGLE_DATA_DIR = Path(
    "/kaggle/input/hull-tactical-market-prediction"
)


# --------------------------------------------------------------------------
# Column conventions
# --------------------------------------------------------------------------
DATE_COL = "date_id"

#: Default supervised target. De-meaned (5y rolling) and MAD-winsorised by
#: the organisers, so it is materially easier to learn than raw
#: ``forward_returns``.
TARGET = "market_forward_excess_returns"

#: Columns present in train.csv only. They describe the future relative to a
#: row and must never be used as model inputs.
LOOKAHEAD_COLS = (
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
)

#: Anonymised feature family prefixes, per the competition data description.
FEATURE_PREFIXES = (
    "M",
    "E",
    "I",
    "P",
    "V",
    "S",
    "D",
)

# Only the public phase is in scope for this project: the public leaderboard
# set is a copy of the last 180 date_ids of train.csv.
PUBLIC_TEST_SIZE = 180


# --------------------------------------------------------------------------
# Shared cross-validation settings - IDENTICAL IN EVERY NOTEBOOK
# --------------------------------------------------------------------------
CV_N_SPLITS = 4

# Trading days dropped after each validation block.
CV_EMBARGO = 20

# Target horizon of 1 day -> purge 1 row before each block.
CV_PURGE = 1