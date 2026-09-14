# Data

The competition data is **not** committed to this repository. Download it once
into `data/raw/`.

## Expected layout

```
data/
├── README.md
└── raw/
    ├── train.csv          # required
    ├── test.csv           # optional (mock test set, not used in this project)
    └── kaggle_evaluation/  # optional, unused: forecasting phase is out of scope
```

## Download

Kaggle CLI (needs `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle competitions download -c hull-tactical-market-prediction -p data/raw
unzip -o data/raw/hull-tactical-market-prediction.zip -d data/raw
```

Or download manually from
<https://www.kaggle.com/competitions/hull-tactical-market-prediction/data>
and unzip into `data/raw/`.

Running on a Kaggle kernel instead? Nothing to do — the loader falls back to
`/kaggle/input/hull-tactical-market-prediction`. To point at any other
location, set `HULL_DATA_DIR`.

## Schema

`train.csv`, one row per trading day, history stretching back decades with
extensive missing values in the early rows.

| Column | Meaning |
| --- | --- |
| `date_id` | Identifier for a single trading day. Monotonic, not a calendar date. |
| `M*` | Market dynamics / technical features |
| `E*` | Macro-economic features |
| `I*` | Interest rate features |
| `P*` | Price / valuation features |
| `V*` | Volatility features |
| `S*` | Sentiment features |
| `MOM*` | Momentum features |
| `D*` | Dummy / binary features |
| `forward_returns` | Return from buying the S&P 500 and selling it one day later. **Train only.** |
| `risk_free_rate` | Federal funds rate. **Train only.** |
| `market_forward_excess_returns` | `forward_returns` minus its rolling 5-year mean, MAD-winsorised at criterion 4. **Train only.** This is the project's supervised target. |

All feature names are anonymised; the organisers publish no mapping to real
instruments.

## Two things that decide how this data may be used

**1. The last 180 rows are the public leaderboard set.** The public test set is
a copy of the final 180 `date_id`s of `train.csv`, which is why public
leaderboard scores in the competition were not meaningful. `load_dataset()`
splits them off as a held-out block and they are never used for fitting.

**2. The three columns above describe the future.** `forward_returns` at row
*t* spans *t → t+1*. Using any of them as a model input is look-ahead leakage.
The return actually observable at *t* is `forward_returns.shift(1)`, which is
what `src.data.past_returns()` returns and what every price-derived feature is
built from.

## Scope

Only the public phase is in scope. The forecasting phase and the
`kaggle_evaluation` inference server are deliberately excluded — the project
evaluates offline on purged time-series folds plus the held-out 180-day block.

## Generated outputs

`results/leaderboard.csv` and `results/artifacts/` are written by the notebooks
and are also not committed.
