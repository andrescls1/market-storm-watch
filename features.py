"""
Step 1 — Data layer and feature engineering.

Source: Robert Shiller's monthly S&P 500 dataset (1871–present), mirrored at
github.com/datasets/s-and-p-500.

Every feature is built ONLY from information that would have been available at
the end of month t (publication lags applied), so the backtest has no look-ahead.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"

# Publication lags (months). Earnings are reported quarterly with a delay and
# Shiller interpolates them, so we lag 6 months to be safe. CPI comes out ~2-3
# weeks after month end, so 1 month.
EARNINGS_LAG = 6
CPI_LAG = 1


def load_shiller(path=DATA / "shiller_sp500.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    df = df.rename(columns={
        "SP500": "price", "Dividend": "div", "Earnings": "eps",
        "Consumer Price Index": "cpi", "Long Interest Rate": "long_rate",
    })[["price", "div", "eps", "cpi", "long_rate"]]
    # Recent months not yet filled in by the source are coded as 0 -> missing
    df = df.replace(0.0, np.nan)

    # Extend CPI and the 10-yr rate past where the Shiller mirror stops
    # (Sep 2023) using the same underlying series (CPI-U NSA, GS10),
    # ratio-linked at the splice point so there is no jump.
    cpi_path, rate_path = DATA / "cpi_us.csv", DATA / "us10y_monthly.csv"
    if cpi_path.exists():
        c = pd.read_csv(cpi_path, parse_dates=["Date"]).set_index("Date")["Index"]
        last = df["cpi"].last_valid_index()
        c = c * (df.loc[last, "cpi"] / c.loc[last])
        df["cpi"] = df["cpi"].fillna(c.reindex(df.index))
    if rate_path.exists():
        r = pd.read_csv(rate_path, parse_dates=["Date"]).set_index("Date")["Rate"]
        df["long_rate"] = df["long_rate"].fillna(r.reindex(df.index))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    price = df["price"]
    cpi = df["cpi"].ffill().shift(CPI_LAG)           # known with 1m lag
    eps = df["eps"].shift(EARNINGS_LAG)              # known with 6m lag
    real_price = price / cpi
    real_eps = eps / cpi
    logp = np.log(price)
    ret = logp.diff()

    # --- VALUATION layer ------------------------------------------------
    # CAPE = real price / 10-yr average real earnings (lag-safe version)
    cape = real_price / real_eps.rolling(120, min_periods=120).mean()
    f["log_cape"] = np.log(cape)

    # --- FUEL / EUPHORIA layer (asset-price boom) -----------------------
    # 3-year real price growth: the "rapid asset price growth" half of the
    # Greenwood-Shleifer et al. crisis predictor. Credit growth (other half)
    # arrives in Step 2 with FRED/BIS data.
    f["boom_3y"] = np.log(real_price).diff(36)

    # --- BEHAVIOR / MOMENTUM layer --------------------------------------
    # Gap between price and its 12-month average: herding / extrapolation
    f["trend_gap"] = logp - np.log(price.rolling(12).mean())

    # --- POLICY / RATE layer ("jet stream") -----------------------------
    f["rate_level"] = df["long_rate"]
    f["rate_chg_12"] = df["long_rate"].diff(12)      # tightening proxy
    f["inflation"] = np.log(cpi).diff(12)

    # --- EARNINGS layer ("the sun") -------------------------------------
    f["eps_growth"] = np.log(real_eps).diff(12)

    # --- STRESS TRANSMISSION layer --------------------------------------
    f["vol_12"] = ret.rolling(12).std() * np.sqrt(12)

    # --- MINSKY term: how long the market has been calm -----------------
    # Real-time market state: drawdown from the current cycle's peak, and
    # whether a bear market is under way (both knowable at month t).
    state = realtime_state(price)
    f["drawdown_now"] = state["dd"]
    f["in_bear"] = state["in_bear"]
    months_since, c = [], 0
    for b in state["in_bear"]:
        c = 0 if b else c + 1
        months_since.append(c)
    f["calm_years"] = np.log1p(np.array(months_since) / 12)

    f["price"] = price
    return f


def realtime_state(price: pd.Series, down=-0.20, up=0.20) -> pd.DataFrame:
    """Walk forward through prices using only past data: track the cycle peak,
    flag a bear market once the drop from it reaches 20%, and end it once
    prices rebound 20% off the trough."""
    out = []
    peak = trough = None
    in_bear = False
    for v in price.values:
        if np.isnan(v):
            out.append((np.nan, in_bear)); continue
        if peak is None:
            peak = trough = v
        if not in_bear:
            peak = max(peak, v)
            if v / peak - 1 <= down:
                in_bear, trough = True, v
        else:
            trough = min(trough, v)
            if v / trough - 1 >= up:
                in_bear, peak = False, v
        out.append((v / peak - 1, in_bear))
    return pd.DataFrame(out, index=price.index, columns=["dd", "in_bear"])


def add_targets(f: pd.DataFrame, horizons=(12, 24), threshold=-0.20) -> pd.DataFrame:
    """y_h = 1 if, within the next h months, the index falls >= 20% below
    today's level at some point (nominal, monthly-average prices)."""
    p = f["price"].values
    n = len(p)
    for h in horizons:
        y = np.full(n, np.nan)
        for t in range(n - h):
            fut = p[t + 1: t + 1 + h]
            if np.isnan(fut).any() or np.isnan(p[t]):
                continue
            y[t] = float(fut.min() / p[t] - 1 <= threshold)
        f[f"y_{h}m"] = y
    return f


def bear_markets(price: pd.Series, down=-0.20, up=0.20):
    """Classic bull/bear dating on monthly-average prices: a bear market is a
    fall of >= 20% from a peak; it ends once prices rise >= 20% off the trough."""
    price = price.dropna()
    eps = []
    peak_i, peak_v = price.index[0], price.iloc[0]
    trough_i, trough_v = peak_i, peak_v
    in_bear = False
    for d, v in price.items():
        if not in_bear:
            if v > peak_v:
                peak_i, peak_v = d, v
            elif v / peak_v - 1 <= down:
                in_bear, trough_i, trough_v = True, d, v
        else:
            if v < trough_v:
                trough_i, trough_v = d, v
            elif v / trough_v - 1 >= up:   # new bull market begins
                eps.append((peak_i, trough_i, trough_v / peak_v - 1))
                in_bear = False
                peak_i, peak_v = d, v
    if in_bear:
        eps.append((peak_i, trough_i, trough_v / peak_v - 1))
    return pd.DataFrame(eps, columns=["peak", "trough", "decline"])


FEATURES = ["log_cape", "boom_3y", "trend_gap", "rate_level", "rate_chg_12",
            "inflation", "eps_growth", "vol_12", "calm_years", "drawdown_now"]


if __name__ == "__main__":
    raw = load_shiller()
    f = add_targets(build_features(raw))
    print(f[FEATURES].describe().T.round(3))
    print("\nLast valid feature row:", f[FEATURES].dropna().index[-1].date())
    print("\nBear markets (>=20%, monthly avg prices):")
    print(bear_markets(raw["price"]).to_string())
    for h in (12, 24):
        y = f[f"y_{h}m"].dropna()
        print(f"\nBase rate y_{h}m: {y.mean():.3f} over {len(y)} months")
