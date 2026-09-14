# `data/` - dataset placeholder

This folder is intentionally empty in version control. The project runs in either of two
modes, decided automatically by `src.data.load_dataset`.

## Mode 1 - real Kaggle data (preferred)

Place the competition training file here:

```
data/train.csv
```

Download it with the Kaggle CLI:

```bash
pip install kaggle
# put your kaggle.json API token in ~/.kaggle/kaggle.json  (chmod 600)
kaggle competitions download -c hull-tactical-market-prediction -p data/
unzip -o data/hull-tactical-market-prediction.zip -d data/
```

Or manually from
<https://www.kaggle.com/competitions/hull-tactical-market-prediction/data>.

Expected columns:

| column | meaning |
|---|---|
| `date_id` | chronological index of the trading day |
| `D*`, `E*`, `I*`, `M*`, `P*`, `S*`, `V*` | anonymised feature families (dummy, macro-economic, interest-rate, market-dynamics, price/valuation, sentiment, volatility) |
| `forward_returns` | total market return realised on the next day |
| `risk_free_rate` | daily risk-free rate |
| `market_forward_excess_returns` | **the modelling target**: de-meaned, winsorised forward excess return |

Nothing else has to be configured - every notebook picks the file up automatically. You can
also point the loader elsewhere with an environment variable:

```bash
export HULL_DATA_DIR=/absolute/path/to/folder/containing/train.csv
```

## Mode 2 - synthetic surrogate (automatic fallback)

If `data/train.csv` is absent (for example, a grading environment without Kaggle
credentials), `src.data.make_synthetic_dataset` generates a **deterministic surrogate**
seeded with `SEED = 42`. It reproduces the schema and the statistical character of the real
problem, not its content:

* the same column families and roughly the same feature count (~95);
* AR(1) log-volatility, so volatility clusters into regimes;
* a weak, learnable conditional mean mixing short-horizon mean reversion with a slow
  sentiment component - a realistic signal-to-noise ratio for this kind of problem;
* pure-noise columns alongside informative ones, so feature selection is meaningful;
* missing values across the early history, mirroring the sparse pre-2000 rows of the real
  file.

Every notebook prints which mode is active in its first data cell, and the committed outputs
were produced in surrogate mode. **Numbers obtained in surrogate mode describe the pipeline,
not the competition**: they show that the comparison between models is valid and
reproducible, but they are not leaderboard estimates.
