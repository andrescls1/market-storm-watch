# Market Storm Watch

A causal early-warning model for severe stock-market declines, following the
structure of Fabozzi & Focardi, *Causal Modeling for Finance and Business*.
It runs every weekday on GitHub Actions and publishes a dashboard on GitHub Pages.

**Status: version 1, unvalidated.** Version 1 did not beat the historical base
rate out of sample (1925–2023). Readings describe conditions and are not forecasts.
Research tool, not investment advice.

## What it shows

- **Watch**: monthly probability of a 20%+ S&P 500 fall within 24 months
  (causal watch model, walk-forward, no look-ahead).
- **Warning**: daily stress gauge, the average percentile of the high-yield spread,
  the Baa spread, VIX, yield-curve inversion, the Chicago Fed financial conditions
  index and the S&P drawdown. Each is ranked only against its own past.
- **Run log**: one row per daily run, showing how risk progresses over time.

## Files

| File | Role |
|---|---|
| `fetch_data.py` | Downloads FRED series and the Shiller/CPI/10-yr mirrors into `data/` |
| `features.py` | Builds the causal-graph variables with publication lags |
| `causal_graph.py` | Theory graph plus a PCMCI test across eras |
| `watch_model.py` | Walk-forward watch model, scored against baselines |
| `robustness.py` | Alternative model variants, all reported |
| `daily_run.py` | Daily update: model, stress gauge, run log, `docs/data/dashboard.json` |
| `docs/index.html` | The dashboard (GitHub Pages) |
| `.github/workflows/daily.yml` | Weekday schedule, 11:17 UTC |

## Setup (one time)

1. **Settings → Pages**: under Source, choose *Deploy from a branch*, set the branch to `main` and the folder to `/docs`, then click Save.
2. **Actions tab**: enable workflows if prompted, open *Daily model run*, and click *Run workflow*.
3. Optional: add a free FRED API key under **Settings → Secrets and variables → Actions** as `FRED_API_KEY`.
   Without it, the public CSV download is used.

## Run locally

```bash
pip install -r requirements.txt
python fetch_data.py && python daily_run.py
cd docs && python -m http.server 8000   # open http://localhost:8000
```
