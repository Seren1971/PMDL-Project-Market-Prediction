"""Global configuration: seeds, paths and column conventions.

Every notebook imports SEED from here so that all experiments are reproducible
and comparable on identical folds.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
SEED: int = 42


def set_seed(seed: int = SEED) -> int:
    """Set every random seed we can reach and return the seed (for logging)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional dependency
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass
    return seed


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = Path(os.environ.get("HULL_DATA_DIR", PROJECT_ROOT / "data"))
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Column conventions of the competition dataset
# --------------------------------------------------------------------------------------
TARGET: str = "market_forward_excess_returns"
RETURN_COL: str = "forward_returns"
RISK_FREE_COL: str = "risk_free_rate"
DATE_COL: str = "date_id"

# Columns that must never be used as model inputs (they contain the future).
LEAKY_COLS: tuple[str, ...] = (
    "forward_returns",
    "risk_free_rate",
    "excess_return",
    "market_forward_excess_returns",
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
    "is_scored",
)

# Anonymised feature family prefixes used by the competition.
FEATURE_PREFIXES: tuple[str, ...] = ("D", "E", "I", "M", "P", "S", "V")

# Trading days per year, used to annualise Sharpe / volatility.
ANNUALISATION: int = 252

# Competition constraints.
MIN_POSITION: float = 0.0
MAX_POSITION: float = 2.0
VOL_CEILING_RATIO: float = 1.2  # strategy vol may not exceed 120% of market vol
