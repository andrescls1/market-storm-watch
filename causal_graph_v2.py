"""
Version 2 — extend the theory graph with the credit, spread and yield-curve
nodes, then re-test it with PCMCI (1952-2023, plus two eras).
Reuses the v1 machinery in causal_graph.py.
"""
import json
import pandas as pd

import causal_graph as cg
from features import load_shiller, build_features, add_macro

THEORY_V2 = cg.THEORY + [
    ("credit_3y",      "drawdown_now",   -1, "causal",     "Credit boom is the fuel for severe busts"),
    ("credit_3y",      "boom_3y",        +1, "causal",     "Credit finances asset-price booms"),
    ("calm_years",     "credit_3y",      +1, "causal",     "Minsky: calm periods invite leverage"),
    ("rate_chg_12",    "term_spread",    -1, "causal",     "Tightening flattens / inverts the curve"),
    ("term_spread",    "drawdown_now",   +1, "causal",     "Inverted curve precedes recessions and bear markets"),
    ("default_spread", "drawdown_now",   -1, "causal",     "Wide credit spreads signal stress"),
    ("spread_chg_6",   "drawdown_now",   -1, "causal",     "Widening spreads: stress spreading"),
    ("vol_12",         "default_spread", +1, "causal",     "Volatility raises default risk pricing"),
    ("drawdown_now",   "default_spread", -1, "causal",     "Feedback: falling prices widen spreads"),
]

VARS_V2 = ["inflation", "rate_chg_12", "term_spread", "credit_3y", "default_spread",
           "spread_chg_6", "log_cape", "boom_3y", "trend_gap", "vol_12",
           "calm_years", "eps_growth", "drawdown_now"]
ERAS_V2 = {"1952-1989": ("1952", "1989"), "1990-2023": ("1990", "2023")}


if __name__ == "__main__":
    raw = load_shiller()
    f = add_macro(build_features(raw), raw)
    q = f[VARS_V2].dropna()
    q = q[q.index.month.isin([3, 6, 9, 12])]
    print(f"Quarterly panel: {q.index[0].date()} to {q.index[-1].date()}, n={len(q)}")

    results = {"Full sample": cg.run_pcmci(q)}
    for era, (a, b) in ERAS_V2.items():
        results[era] = cg.run_pcmci(q.loc[a:b])

    table = cg.compare(THEORY_V2, results)
    pd.set_option("display.width", 250)
    print(table.drop(columns="mechanism").to_string(index=False))

    print("\nALL LINKS INTO drawdown_now BY SAMPLE:")
    for name, links in results.items():
        m = links[links.effect == "drawdown_now"] if not links.empty else links
        if not m.empty:
            m = m.loc[m.groupby("cause").strength.apply(lambda s: s.abs().idxmax())]
        print(f"  {name}: " + ", ".join(f"{r.cause} {r.strength:+.2f}@{r.lag_q}q" for r in m.itertuples()))

    with open("outputs/causal_graph_v2.json", "w") as fh:
        json.dump(dict(comparison=table.to_dict(orient="records"),
                       by_sample={k: v.to_dict(orient="records") for k, v in results.items()},
                       panel=[str(q.index[0].date()), str(q.index[-1].date()), len(q)]),
                  fh, indent=1, default=str)
