"""
Daily run: update the model and write everything the dashboard reads.

1. Re-run the watch model (24-month risk of a 20%+ fall) on the latest data.
2. Compute the daily stress gauge from market series (spreads, VIX, curve,
   drawdown, financial conditions). Each component is ranked against its OWN
   PAST ONLY (expanding percentile), so the history has no look-ahead.
3. Append today's snapshot to docs/data/history.csv (the risk progression log).
4. Write docs/data/dashboard.json for the web page.
"""
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features import load_shiller, build_features, bear_markets

ROOT = Path(__file__).parent
FRED = ROOT / "data" / "fred"
OUT = ROOT / "docs" / "data"

# Shown on the dashboard until a version passes out-of-sample validation
MODEL_STATUS = {
    "version": "1",
    "validated": False,
    "note": "Version 1 did not beat the historical base rate out of sample. "
            "Treat readings as conditions, not forecasts, until version 2 is validated.",
}

# Stress gauge components: series id, label, transform (higher = more stress)
STRESS = [
    # High-yield spread left out: FRED now only publishes its last 3 years,
    # too short to rank against history. The Baa spread covers credit stress.
    ("BAA10Y", "Baa corporate spread", lambda s: s),
    ("VIXCLS", "VIX (volatility)", lambda s: s),
    ("T10Y3M", "Yield-curve inversion", lambda s: -s),
    ("NFCI", "Financial conditions", lambda s: s),
    ("SP500", "S&P drawdown from 1-yr high", lambda s: -(s / s.rolling(252, min_periods=60).max() - 1)),
]
MIN_HISTORY_DAYS = 756   # ~3 years before a component's percentile counts


def read_fred(sid):
    p = FRED / f"{sid}.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["value"].dropna()
    return s if len(s) else None


def expanding_pct(s: pd.Series) -> pd.Series:
    """Percentile of each value within all values up to that date."""
    v = s.values
    out = np.full(len(v), np.nan)
    sorted_hist = []
    import bisect
    for i, x in enumerate(v):
        bisect.insort(sorted_hist, x)
        if i + 1 >= MIN_HISTORY_DAYS:
            out[i] = bisect.bisect_left(sorted_hist, x) / (len(sorted_hist) - 1)
    return pd.Series(out, index=s.index)


def stress_gauge():
    idx = pd.bdate_range("1990-01-01", pd.Timestamp.today().normalize())
    comps, meta = {}, []
    for sid, label, fn in STRESS:
        s = read_fred(sid)
        if s is None:
            continue
        raw_s = s
        s = fn(s.reindex(idx.union(s.index)).ffill(limit=10)).reindex(idx)
        s = s.loc[:raw_s.index[-1]].dropna()      # never extend past the last real print
        if len(s) < MIN_HISTORY_DAYS:
            continue
        comps[sid] = expanding_pct(s)
        meta.append(dict(id=sid, label=label, last_date=str(raw_s.index[-1].date()),
                         value=round(float(raw_s.iloc[-1] if sid != "SP500" else
                                           -(raw_s.iloc[-1] / raw_s.iloc[-252:].max() - 1) * 100), 2)))
    if not comps:
        return None, None, meta
    c = pd.DataFrame(comps).reindex(idx)
    last_obs = max(pd.Timestamp(m["last_date"]) for m in meta)
    c = c.loc[:last_obs].ffill(limit=5)   # carry weekly/lagging prints a few days
    gauge = c.mean(axis=1, skipna=True).where(c.notna().sum(axis=1) >= 2)
    for m in meta:
        m["percentile"] = round(float(c[m["id"]].dropna().iloc[-1]), 3)
    return gauge.dropna(), c, meta


def clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    subprocess.run([sys.executable, "watch_model.py"], check=True, cwd=ROOT)

    wp = pd.read_csv(ROOT / "outputs/watch_predictions.csv", index_col=0, parse_dates=True)
    wr = json.load(open(ROOT / "outputs/watch_results.json"), parse_constant=lambda c: float("nan"))

    raw = load_shiller()
    f = build_features(raw)
    peaks = bear_markets(raw["price"])

    # Driver readings: latest value and percentile vs full history
    drivers = []
    for col, label in [("log_cape", "CAPE (valuation)"), ("boom_3y", "3-yr real price boom"),
                       ("trend_gap", "Trend gap (momentum)"), ("rate_chg_12", "Rate change 12m"),
                       ("inflation", "Inflation"), ("calm_years", "Calm years (Minsky)"),
                       ("vol_12", "Volatility 12m"), ("eps_growth", "Earnings growth")]:
        s = f[col].dropna()
        drivers.append(dict(id=col, label=label, asof=s.index[-1].strftime("%Y-%m"),
                            value=float(s.iloc[-1]), percentile=float((s < s.iloc[-1]).mean())))

    gauge, comps, stress_meta = stress_gauge()

    # Latest readings
    ne = wp["No-earnings variant"].dropna()
    cm = wp["Causal watch model"].dropna()
    base = wp["Base rate"].dropna()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot = dict(
        run_date=today,
        watch_prob=float(ne.iloc[-1]) if len(ne) else None,
        watch_asof=ne.index[-1].strftime("%Y-%m") if len(ne) else None,
        full_model_prob=float(cm.iloc[-1]) if len(cm) else None,
        full_model_asof=cm.index[-1].strftime("%Y-%m") if len(cm) else None,
        base_rate=float(base.iloc[-1]) if len(base) else None,
        stress=float(gauge.iloc[-1]) if gauge is not None else None,
        stress_asof=str(gauge.index[-1].date()) if gauge is not None else None,
    )
    for m in stress_meta:
        snapshot[f"pct_{m['id']}"] = m.get("percentile")

    # Append to the progression log (one row per run date; re-runs replace)
    hist_path = OUT / "history.csv"
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    if not hist.empty:
        hist = hist[hist.run_date != today]
    hist = pd.concat([hist, pd.DataFrame([snapshot])], ignore_index=True)
    hist.to_csv(hist_path, index=False)

    # Monthly watch series since 1925 (backtest + live)
    w = wp.loc["1925":]
    watch_monthly = [[t.strftime("%Y-%m"), r["Causal watch model"], r["No-earnings variant"],
                      r["Base rate"], r["price"]] for t, r in w.iterrows()]

    # Stress: weekly points for the long history, daily for the last 2 years
    stress_series = []
    if gauge is not None:
        cut = gauge.index[-1] - pd.DateOffset(years=2)
        long = gauge.loc[:cut].resample("W-FRI").last().dropna()
        recent = gauge.loc[cut:]
        g = pd.concat([long, recent])
        stress_series = [[str(d.date()), round(float(v), 4)] for d, v in g.items()]

    dash = dict(
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        status=MODEL_STATUS,
        latest=snapshot,
        drivers=drivers,
        stress_components=stress_meta,
        watch_monthly=watch_monthly,
        stress_series=stress_series,
        history=hist.to_dict(orient="records"),
        peaks=[dict(peak=r.peak.strftime("%Y-%m"), trough=r.trough.strftime("%Y-%m"),
                    decline=round(float(r.decline), 3)) for r in peaks.itertuples()],
        backtest=wr.get("scores"),
    )
    (OUT / "dashboard.json").write_text(json.dumps(clean(dash), allow_nan=False))
    print("snapshot:", json.dumps(clean(snapshot)))


if __name__ == "__main__":
    main()
