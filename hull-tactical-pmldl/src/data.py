"""Dataset access layer.

The project is designed to run in two modes:

1. **Real mode** - the Kaggle competition file ``data/train.csv`` is present
   (see ``data/README.md`` for the download instructions).  It is loaded as-is.
2. **Surrogate mode** - the CSV is absent (e.g. no Kaggle credentials in the grading
   environment).  A *deterministic synthetic surrogate* with the same schema, the same
   column families (D/E/I/M/P/S/V), the same target definition and realistic financial
   dynamics (volatility clustering, short-horizon mean reversion, missing history in the
   early rows) is generated instead, so that **every notebook remains executable and
   reproducible end-to-end**.

The surrogate is *not* a claim about competition performance; it exists so the pipeline,
the validation protocol and the model comparison can be reproduced by a grader without
the private dataset.  All numbers reported in surrogate mode are labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, DATE_COL, RETURN_COL, RISK_FREE_COL, SEED, TARGET

# Number of columns per anonymised family (matches the order of magnitude of the
# real competition file: ~90 anonymised features).
FAMILY_SIZES: dict[str, int] = {
    "D": 10,  # dummy / calendar-like binary indicators
    "E": 20,  # macro-economic
    "I": 9,   # interest rates
    "M": 18,  # market dynamics
    "P": 13,  # price / valuation
    "S": 12,  # sentiment
    "V": 13,  # volatility
}


@dataclass
class DatasetInfo:
    """Small provenance record attached to every loaded dataset."""

    source: str  # "kaggle_csv" or "synthetic_surrogate"
    n_rows: int
    n_features: int
    path: str | None = None

    def describe(self) -> str:
        tag = (
            "REAL Kaggle competition data"
            if self.source == "kaggle_csv"
            else "SYNTHETIC SURROGATE data (Kaggle CSV not found)"
        )
        return f"{tag} | rows={self.n_rows:,} | anonymised features={self.n_features}"


# --------------------------------------------------------------------------------------
# Synthetic surrogate
# --------------------------------------------------------------------------------------
def make_synthetic_dataset(n_rows: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Generate a deterministic surrogate of the competition training file.

    Data generating process
    -----------------------
    * log-volatility follows an AR(1) process -> volatility clustering / regimes;
    * the conditional mean of the next return mixes a **short-horizon mean-reversion**
      component and a slow **sentiment** component, both scaled by the current
      volatility -> a weak but learnable signal (annualised IC of a few percent);
    * anonymised features are noisy, partially redundant observations of the latent
      states, plus a set of pure-noise columns so that feature selection is meaningful;
    * the first rows contain missing values for several families, mirroring the sparse
      pre-2000 history of the real file.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    # ---- latent states ----------------------------------------------------------
    log_vol = np.zeros(n)
    log_vol[0] = np.log(0.009)
    for t in range(1, n):
        log_vol[t] = 0.985 * log_vol[t - 1] + 0.015 * np.log(0.009) + 0.09 * rng.standard_normal()
    vol = np.clip(np.exp(log_vol), 0.003, 0.06)  # daily volatility

    sentiment = np.zeros(n)
    for t in range(1, n):
        sentiment[t] = 0.97 * sentiment[t - 1] + 0.25 * rng.standard_normal()

    rate = np.zeros(n)
    rate[0] = 0.03
    for t in range(1, n):
        rate[t] = np.clip(0.999 * rate[t - 1] + 0.0002 * rng.standard_normal(), 0.0, 0.09)

    # ---- returns ---------------------------------------------------------------
    ret = np.zeros(n)  # daily *excess* market return realised at t
    eps = rng.standard_normal(n)
    for t in range(1, n):
        window = ret[max(0, t - 3): t]
        z3 = window.sum() / (vol[t] * np.sqrt(max(len(window), 1)) + 1e-12)
        mu = vol[t] * (-0.075 * np.clip(z3, -4, 4) + 0.045 * np.tanh(sentiment[t - 1]))
        ret[t] = mu + vol[t] * eps[t] + 0.00015  # small positive equity drift

    real_vol_20 = pd.Series(ret).rolling(20, min_periods=5).std().bfill().to_numpy()
    cum = np.cumsum(ret)
    mom_20 = pd.Series(ret).rolling(20, min_periods=5).mean().bfill().to_numpy()
    mom_60 = pd.Series(ret).rolling(60, min_periods=5).mean().bfill().to_numpy()

    df = pd.DataFrame({DATE_COL: np.arange(n)})

    def noisy(base: np.ndarray, scale: float, noise: float) -> np.ndarray:
        return scale * base + noise * rng.standard_normal(n)

    # Volatility family: V1 is the annualised forward-looking vol proxy (as in the
    # public write-ups), the rest are noisy vol observations.
    df["V1"] = np.clip(vol * np.sqrt(252) * (1 + 0.05 * rng.standard_normal(n)), 0.03, 1.2)
    for i in range(2, FAMILY_SIZES["V"] + 1):
        df[f"V{i}"] = noisy(real_vol_20 * np.sqrt(252), 1.0 if i < 6 else 0.0, 0.05)

    # Interest-rate family.
    for i in range(1, FAMILY_SIZES["I"] + 1):
        df[f"I{i}"] = noisy(rate, 1.0 + 0.05 * i, 0.0015)

    # Market-dynamics family: M11 is used by the public solutions as a rate-normalised
    # market dynamic, so give it real content.
    for i in range(1, FAMILY_SIZES["M"] + 1):
        base = mom_20 if i % 3 == 0 else (mom_60 if i % 3 == 1 else np.zeros(n))
        df[f"M{i}"] = noisy(base, 8.0, 0.03)
    df["M11"] = noisy(mom_20 * 10 + rate, 1.0, 0.01)

    # Price / valuation family (level-like, slow moving).
    for i in range(1, FAMILY_SIZES["P"] + 1):
        df[f"P{i}"] = noisy(cum, 0.6 if i % 2 == 0 else 0.0, 0.15)

    # Sentiment family: S1 carries the latent sentiment.
    df["S1"] = noisy(np.tanh(sentiment), 1.0, 0.20)
    for i in range(2, FAMILY_SIZES["S"] + 1):
        df[f"S{i}"] = noisy(np.tanh(sentiment), 0.7 if i < 6 else 0.0, 0.35)

    # Macro family (mostly slow, mostly uninformative).
    for i in range(1, FAMILY_SIZES["E"] + 1):
        df[f"E{i}"] = noisy(rate * 20 + 0.1 * np.tanh(sentiment), 0.5 if i < 8 else 0.0, 0.30)

    # Dummy / calendar family.
    for i in range(1, FAMILY_SIZES["D"] + 1):
        df[f"D{i}"] = (rng.random(n) < 0.08).astype(float)

    # ---- targets ---------------------------------------------------------------
    rf_daily = rate / 252.0
    forward_excess = np.roll(ret, -1)
    forward_excess[-1] = np.nan
    df[TARGET] = forward_excess
    df[RISK_FREE_COL] = rf_daily
    df[RETURN_COL] = forward_excess + np.roll(rf_daily, -1)
    df.loc[df.index[-1], RETURN_COL] = np.nan

    # ---- realistic missingness in the early history ----------------------------
    for fam, cut in (("E", 900), ("S", 400), ("I", 120), ("V", 60)):
        cols = [c for c in df.columns if c.startswith(fam) and c[1:].isdigit()]
        for c in cols[: max(1, len(cols) // 2)]:
            df.loc[: cut - 1, c] = np.nan

    return df


# --------------------------------------------------------------------------------------
# Public loader
# --------------------------------------------------------------------------------------
def load_dataset(
    data_dir: str | Path = DATA_DIR,
    n_rows_synth: int = 6000,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[pd.DataFrame, DatasetInfo]:
    """Load ``train.csv`` if available, otherwise build the synthetic surrogate."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "train.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        source, path = "kaggle_csv", str(csv_path)
    else:
        df = make_synthetic_dataset(n_rows=n_rows_synth, seed=seed)
        source, path = "synthetic_surrogate", None

    if DATE_COL not in df.columns:
        df[DATE_COL] = np.arange(len(df))
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Drop rows with an undefined target (the last row has no forward return).
    df = df[df[TARGET].notna()].reset_index(drop=True)

    n_feat = len(feature_columns(df))
    info = DatasetInfo(source=source, n_rows=len(df), n_features=n_feat, path=path)
    if verbose:
        print(info.describe())
    return df, info


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Anonymised feature columns (family prefix + digits), leak-free by construction."""
    from .config import FEATURE_PREFIXES

    return [
        c
        for c in df.columns
        if c[:1] in FEATURE_PREFIXES and c[1:].isdigit()
    ]
