"""
Daily data fetch (runs on GitHub Actions, which can reach FRED).

- FRED series -> data/fred/<ID>.csv   (columns: date, value)
- Shiller monthly S&P file, CPI and 10-yr yield mirrors -> data/
- CPI and 10-yr yield from FRED also overwrite the mirror files, so the
  model always has the latest month.

FRED access: uses the official API when a FRED_API_KEY secret is set,
otherwise the public CSV download. A failed series is logged and skipped;
the last saved copy stays in place, so one bad day never breaks the run.
"""
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
FRED_DIR = DATA / "fred"

# series id -> (layer, description)
FRED_SERIES = {
    # Fuel: credit and leverage
    "QUSPAM770A": ("fuel", "Private non-financial credit, % of GDP (BIS), quarterly"),
    # Stress transmission: spreads and volatility
    "BAA":          ("stress", "Moody's Baa corporate yield, monthly, 1919+"),
    "AAA":          ("stress", "Moody's Aaa corporate yield, monthly, 1919+"),
    "BAA10Y":       ("stress", "Baa minus 10-yr Treasury spread, daily, 1986+"),
    "BAMLH0A0HYM2": ("stress", "High-yield option-adjusted spread, daily, 1996+"),
    "VIXCLS":       ("stress", "VIX, daily, 1990+"),
    "NFCI":         ("stress", "Chicago Fed financial conditions, weekly, 1971+"),
    # Policy: rates and the yield curve
    "TB3MS":    ("policy", "3-month T-bill, monthly, 1934+"),
    "GS10":     ("policy", "10-yr Treasury, monthly, 1953+"),
    "T10Y3M":   ("policy", "10-yr minus 3-month curve, daily, 1982+"),
    "FEDFUNDS": ("policy", "Fed funds rate, monthly, 1954+"),
    # Inflation and prices
    "CPIAUCNS": ("inflation", "CPI-U, not seasonally adjusted, monthly"),
    "SP500":    ("price", "S&P 500 daily close (last 10 years on FRED)"),
}

MIRRORS = {
    "shiller_sp500.csv": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    "cpi_us.csv": "https://raw.githubusercontent.com/datasets/cpi-us/main/data/cpiai.csv",
    "us10y_monthly.csv": "https://raw.githubusercontent.com/datasets/bond-yields-us-10y/main/data/monthly.csv",
}

HEADERS = {"User-Agent": "market-storm-watch/1.0 (research project)"}


def parse_fred_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]            # observation_date/DATE, <ID>
    df["value"] = pd.to_numeric(df["value"], errors="coerce")   # "." = missing
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna()


def fetch_fred(series_id: str) -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY")
    if key:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params=dict(series_id=series_id, api_key=key, file_type="json"),
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
        obs = pd.DataFrame(r.json()["observations"])[["date", "value"]]
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs["date"] = pd.to_datetime(obs["date"])
        return obs.dropna()
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                     params={"id": series_id}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return parse_fred_csv(r.text)


def main() -> int:
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for name, url in MIRRORS.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            (DATA / name).write_text(r.text)
            print(f"ok   {name}")
        except Exception as e:  # keep the last good copy
            failures.append(name)
            print(f"FAIL {name}: {e}")

    for sid in FRED_SERIES:
        for attempt in range(3):
            try:
                df = fetch_fred(sid)
                df.to_csv(FRED_DIR / f"{sid}.csv", index=False)
                print(f"ok   {sid:14s} {df.date.min().date()} -> {df.date.max().date()}  n={len(df)}")
                break
            except Exception as e:
                if attempt == 2:
                    failures.append(sid)
                    print(f"FAIL {sid}: {e}")
                time.sleep(3 * (attempt + 1))

    # FRED CPI / 10-yr are fresher than the mirrors: overwrite in the mirror format
    cpi, gs10 = FRED_DIR / "CPIAUCNS.csv", FRED_DIR / "GS10.csv"
    if cpi.exists():
        c = pd.read_csv(cpi, parse_dates=["date"])
        c.rename(columns={"date": "Date", "value": "Index"}).to_csv(DATA / "cpi_us.csv", index=False)
    if gs10.exists():
        g = pd.read_csv(gs10, parse_dates=["date"])
        g.rename(columns={"date": "Date", "value": "Rate"}).to_csv(DATA / "us10y_monthly.csv", index=False)

    print(f"\n{len(failures)} failures: {failures}")
    # Fail the job only if the core price file is missing entirely
    return 1 if not (DATA / "shiller_sp500.csv").exists() else 0


if __name__ == "__main__":
    sys.exit(main())
