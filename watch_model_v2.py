"""
Version 2 — walk-forward backtest of the watch model with the credit,
spread and yield-curve nodes, against version 1 and the baselines.

The model specs below were fixed BEFORE looking at any version 2 test
result. All of them are reported, whatever they score.

Two test windows, because the new data starts at different dates:
  A: 1950-2023  (spreads from 1919, yield curve from 1934)
  B: 1970-2023  (private credit/GDP from 1951, plus a 3-year change and lag)
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

import watch_model as wm
from features import load_shiller, build_features, add_macro, add_targets, bear_markets

V1 = wm.MODELS["Causal watch model"]
V1_NO_LEVEL = [c for c in V1 if c != "rate_level"]
SPREADS = ["term_spread", "default_spread", "spread_chg_6"]

SPECS = {
    # name: (columns, C, windows)
    "v1 causal model": (V1, 0.5, "AB"),
    "v2 + spreads & curve": (V1_NO_LEVEL + SPREADS, 0.5, "AB"),
    "v2 + spreads, curve & credit": (V1_NO_LEVEL + SPREADS + ["credit_3y"], 0.5, "B"),
    "v2 core: credit, curve, valuation, boom": (["credit_3y", "term_spread", "log_cape", "boom_3y"], 0.5, "B"),
    "v2 core, heavy regularization": (["credit_3y", "term_spread", "log_cape", "boom_3y"], 0.05, "B"),
    "Curve only": (["term_spread"], 0.5, "AB"),
    "Credit only": (["credit_3y"], 0.5, "B"),
    "CAPE only": (["log_cape"], 0.5, "AB"),
}
WINDOWS = {"A": "1950-01-01", "B": "1970-01-01"}


def evaluate(preds, base, y, common):
    out = {}
    bb = brier_score_loss(y.loc[common], base.loc[common])
    for name, p in preds.items():
        yy, pp = y.loc[common], p.loc[common]
        bs = brier_score_loss(yy, pp)
        on = pp >= wm.WATCH
        out[name] = dict(n=int(len(common)), auc=roc_auc_score(yy, pp), brier_skill=1 - bs / bb,
                         watch_share=float(on.mean()),
                         hit_when_watch=float(yy[on].mean()) if on.any() else None,
                         hit_otherwise=float(yy[~on].mean()))
    return out


def main():
    raw = load_shiller()
    f = add_macro(add_targets(build_features(raw), horizons=(wm.H,)), raw)
    y = f[wm.TARGET]
    peaks = bear_markets(raw["price"])
    results, all_preds = {}, {}

    for w, start in WINDOWS.items():
        wm.OOS_START = start
        base = wm.base_rate_forecast(f)
        preds = {"Base rate": base}
        for name, (cols, C, wins) in SPECS.items():
            if w in wins:
                preds[name], _ = wm.walk_forward(f, cols, C)
        common = pd.concat(preds, axis=1).dropna().index.intersection(y.dropna().index)
        results[w] = dict(start=start, end=str(common[-1].date()), scores=evaluate(preds, base, y, common))
        all_preds[w] = preds
        print(f"\nWINDOW {w}: {common[0].date()} to {common[-1].date()}, n={len(common)} months")
        print(pd.DataFrame(results[w]["scores"]).T.round(3).to_string())

    # Episode table for the full v2 model in window B
    wm.OOS_START = WINDOWS["B"]
    ep = wm.episode_table(all_preds["B"]["v2 + spreads, curve & credit"], peaks)
    ep_core = wm.episode_table(all_preds["B"]["v2 core: credit, curve, valuation, boom"], peaks)
    print("\nEpisodes, v2 full (window B):\n", ep.to_string(index=False))
    print("\nEpisodes, v2 core (window B):\n", ep_core.to_string(index=False))

    out = pd.concat({f"{w}|{k}": v for w, d in all_preds.items() for k, v in d.items()}, axis=1).reindex(f.index)
    out["y"], out["price"], out["in_bear"] = y, f["price"], f["in_bear"]
    out.to_csv("outputs/watch_v2_predictions.csv")
    json.dump(dict(results=results, episodes_full=ep.to_dict(orient="records"),
                   episodes_core=ep_core.to_dict(orient="records"),
                   specs={k: dict(cols=v[0], C=v[1], windows=v[2]) for k, v in SPECS.items()}),
              open("outputs/watch_v2_results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
