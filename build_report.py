"""Step 4 — Assemble the results page from the pipeline outputs."""
import json, math
def clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o
import pandas as pd
nan = lambda c: float("nan")
p = pd.read_csv("outputs/watch_predictions.csv", index_col=0, parse_dates=True).loc["1920":]
f = lambda v: None if pd.isna(v) else round(float(v), 3)
series = [[t.strftime("%Y-%m"), f(r.price), f(r["Causal watch model"]), f(r["CAPE only"]),
           f(r["Base rate"]), f(r["No-earnings variant"])] for t, r in p.iterrows()]
d = dict(series=series,
         watch=json.load(open("outputs/watch_results.json"), parse_constant=nan),
         graph=json.load(open("outputs/causal_graph_results.json"), parse_constant=nan)["comparison"],
         robust=json.load(open("outputs/robustness.json"), parse_constant=nan))
html = open("report_template.html").read().replace("__DATA__", json.dumps(clean(d), allow_nan=False))
open("outputs/market_storm_watch.html", "w").write(html)
print("ok", len(html))
