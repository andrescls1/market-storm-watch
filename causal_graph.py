"""
Step 2 — Theory causal graph, then test it against the data with PCMCI.

Fabozzi & Focardi's workflow: start from a graph built on economic mechanisms
(the expert prior), then use a structure-learning algorithm to check which
links the data supports, and whether they are stable across regimes.
Links that hold in every era are candidates for real causal mechanisms; links
that appear in one era only are more likely descriptive (functional) patterns.
"""
import json
import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from features import load_shiller, build_features

# ---------------------------------------------------------------------------
# 1. THEORY GRAPH  (cause, effect, expected sign, type, mechanism)
#    type = "causal"     -> a mechanism that should survive an intervention
#           "functional" -> anticipation / descriptive pattern, not a lever
# ---------------------------------------------------------------------------
THEORY = [
    ("inflation",    "rate_chg_12",  +1, "causal",     "Inflation pushes up yields / policy tightening"),
    ("inflation",    "log_cape",     -1, "causal",     "Inflation erodes real earnings value, lowers multiples"),
    ("rate_chg_12",  "log_cape",     -1, "causal",     "Higher discount rate lowers valuations"),
    ("rate_chg_12",  "drawdown_now", -1, "causal",     "Tightening is the classic trigger (jet stream shift)"),
    ("rate_level",   "log_cape",     -1, "causal",     "Level of rates sets the discount rate"),
    ("eps_growth",   "log_cape",     +1, "causal",     "Earnings (the 'sun') support prices"),
    ("eps_growth",   "drawdown_now", +1, "causal",     "Earnings growth cushions the market"),
    ("log_cape",     "drawdown_now", -1, "causal",     "Valuation fuel: expensive markets fall further"),
    ("boom_3y",      "drawdown_now", -1, "causal",     "Asset-price booms precede busts (fuel)"),
    ("trend_gap",    "drawdown_now", +1, "functional", "Short-run momentum / herding (descriptive)"),
    ("calm_years",   "boom_3y",      +1, "causal",     "Minsky: stability breeds risk-taking"),
    ("calm_years",   "vol_12",       -1, "causal",     "Long calm suppresses volatility"),
    ("vol_12",       "drawdown_now", -1, "causal",     "Stress transmission: volatility forces de-risking"),
    ("drawdown_now", "vol_12",       -1, "causal",     "Feedback loop: falling prices raise volatility"),
    ("drawdown_now", "eps_growth",   +1, "functional", "Market anticipates earnings (not a lever)"),
]

VARS = ["inflation", "rate_level", "rate_chg_12", "eps_growth", "log_cape",
        "boom_3y", "trend_gap", "vol_12", "calm_years", "drawdown_now"]

ERAS = {"1881-1945": ("1881", "1945"), "1946-1989": ("1946", "1989"),
        "1990-2023": ("1990", "2023")}

TAU_MAX = 8          # quarters -> up to 2 years of lag
PC_ALPHA = 0.05
ALPHA = 0.01         # significance for reporting a link


def quarterly_panel(f: pd.DataFrame) -> pd.DataFrame:
    # Sample every 3rd month (quarter end) to reduce overlap in rolling features
    q = f[VARS].dropna()
    return q[q.index.month.isin([3, 6, 9, 12])]


def run_pcmci(q: pd.DataFrame):
    z = (q - q.mean()) / q.std()
    df = pp.DataFrame(z.values, var_names=list(z.columns))
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmci(tau_min=1, tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
    names = list(z.columns)
    links = []
    p, v = res["p_matrix"], res["val_matrix"]
    for i, cause in enumerate(names):
        for j, effect in enumerate(names):
            if i == j:
                continue  # own-lag persistence is expected, not reported
            for tau in range(1, TAU_MAX + 1):
                if p[i, j, tau] < ALPHA:
                    links.append(dict(cause=cause, effect=effect, lag_q=tau,
                                      strength=round(float(v[i, j, tau]), 3),
                                      p=float(p[i, j, tau])))
    return pd.DataFrame(links)


def strongest(links: pd.DataFrame, cause, effect):
    if links.empty:
        return None
    m = links[(links.cause == cause) & (links.effect == effect)]
    if m.empty:
        return None
    r = m.loc[m.strength.abs().idxmax()]
    return dict(lag_q=int(r.lag_q), strength=float(r.strength))


def compare(theory, results_by_sample):
    rows = []
    for cause, effect, sign, typ, why in theory:
        row = dict(cause=cause, effect=effect, expected=sign, type=typ, mechanism=why)
        for name, links in results_by_sample.items():
            s = strongest(links, cause, effect)
            if s is None:
                row[name] = "—"
            else:
                ok = np.sign(s["strength"]) == sign
                row[name] = f"{'✓' if ok else '✗ sign'} {s['strength']:+.2f} @{s['lag_q']}q"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    f = build_features(load_shiller())
    q = quarterly_panel(f)
    print(f"Quarterly panel: {q.index[0].date()} to {q.index[-1].date()}, n={len(q)}")

    results = {"Full sample": run_pcmci(q)}
    for era, (a, b) in ERAS.items():
        results[era] = run_pcmci(q.loc[a:b])

    table = compare(THEORY, results)
    pd.set_option("display.width", 250)
    print("\nTHEORY LINKS vs DATA (strongest lagged link, partial corr @ lag in quarters)")
    print(table.drop(columns="mechanism").to_string(index=False))

    theory_pairs = {(c, e) for c, e, *_ in THEORY}
    full = results["Full sample"]
    new = full[~full.apply(lambda r: (r.cause, r.effect) in theory_pairs, axis=1)]
    new = new.loc[new.groupby(["cause", "effect"]).strength.apply(lambda s: s.abs().idxmax())]
    print("\nLINKS THE DATA FOUND THAT THEORY DID NOT INCLUDE (full sample):")
    print(new.sort_values("strength", key=abs, ascending=False).to_string(index=False))

    # Era stability of all links into the market outcome
    print("\nALL LINKS INTO drawdown_now BY ERA:")
    for name, links in results.items():
        m = links[links.effect == "drawdown_now"] if not links.empty else links
        m = m.loc[m.groupby("cause").strength.apply(lambda s: s.abs().idxmax())] if not m.empty else m
        print(f"  {name}: " + ", ".join(f"{r.cause} {r.strength:+.2f}@{r.lag_q}q" for r in m.itertuples()))

    out = dict(
        comparison=table.to_dict(orient="records"),
        new_links=new.to_dict(orient="records"),
        by_sample={k: v.to_dict(orient="records") for k, v in results.items()},
    )
    with open("outputs/causal_graph_results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
