"""
Step 3 — "Watch" model: probability of a >=20% fall within the next 24 months,
estimated walk-forward (the model never sees the future).

Predictors = the parents of the market outcome in the causal graph (theory
links plus links PCMCI confirmed), not every variable that happens to
correlate. Scored against two baselines: the historical base rate and a
CAPE-only model.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

from features import load_shiller, build_features, add_targets, bear_markets

H = 24                      # horizon in months
TARGET = f"y_{H}m"
OOS_START = "1925-01-01"    # first out-of-sample prediction
REFIT_EVERY = 12            # months
WATCH = 0.35                # "watch" threshold (~1.75x the base rate)

MODELS = {
    # Causal parents of the outcome (theory + PCMCI-confirmed)
    "Causal watch model": ["log_cape", "boom_3y", "rate_chg_12", "rate_level",
                           "inflation", "eps_growth", "trend_gap", "vol_12",
                           "calm_years"],
    # Same graph minus the earnings-based nodes, so it can run up to 2026
    # (the public earnings series currently stops in 2023)
    "No-earnings variant": ["boom_3y", "rate_chg_12", "rate_level", "inflation",
                            "trend_gap", "vol_12", "calm_years"],
    "CAPE only": ["log_cape"],
}


def walk_forward(f, cols, C=0.5):
    data = f[cols + [TARGET, "in_bear"]].copy()
    # Predict the ONSET of trouble: only months not already in a bear market
    data = data[(data.in_bear == False)].drop(columns="in_bear")
    X_all = data[cols].dropna()
    preds = pd.Series(np.nan, index=X_all.index)
    dates = X_all.loc[OOS_START:].index
    model = None
    for k, t in enumerate(dates):
        if k % REFIT_EVERY == 0:
            # Embargo: a label at month s needs prices up to s+H, so only
            # rows with s <= t - H were fully known at time t.
            cutoff = t - pd.DateOffset(months=H)
            train = data.loc[:cutoff].dropna()
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=C, max_iter=1000))
            model.fit(train[cols], train[TARGET])
        preds[t] = model.predict_proba(X_all.loc[[t], cols])[0, 1]
    return preds.loc[OOS_START:], model


def base_rate_forecast(f):
    d = f[[TARGET, "in_bear"]]
    d = d[d.in_bear == False]
    out = {}
    for t in d.loc[OOS_START:].index:
        hist = d.loc[:t - pd.DateOffset(months=H), TARGET].dropna()
        out[t] = hist.mean()
    return pd.Series(out)


def score(pred, y):
    m = pd.concat([pred.rename("p"), y.rename("y")], axis=1).dropna()
    bs = brier_score_loss(m.y, m.p)
    return dict(n=len(m), auc=roc_auc_score(m.y, m.p), brier=bs,
                watch_share=float((m.p >= WATCH).mean()),
                hit_rate_when_watch=float(m.y[m.p >= WATCH].mean()) if (m.p >= WATCH).any() else np.nan,
                hit_rate_otherwise=float(m.y[m.p < WATCH].mean()))


def calibration(pred, y, bins=(0, .1, .2, .3, .4, .5, 1)):
    m = pd.concat([pred.rename("p"), y.rename("y")], axis=1).dropna()
    m["bin"] = pd.cut(m.p, bins)
    return m.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"),
                                               actual=("y", "mean")).round(3)


def episode_table(pred, peaks):
    rows = []
    for r in peaks.itertuples():
        if r.peak < pd.Timestamp(OOS_START) + pd.DateOffset(months=12):
            continue
        w = pred.loc[r.peak - pd.DateOffset(months=24): r.peak].dropna()
        if w.empty:
            continue
        first = w[w >= WATCH]
        rows.append(dict(peak=r.peak.strftime("%Y-%m"), decline=f"{r.decline:.0%}",
                         avg_p_24m_before=round(w.mean(), 2), max_p=round(w.max(), 2),
                         p_at_peak=round(w.iloc[-1], 2),
                         watch_lead_months=(r.peak - first.index[0]).days // 30 if len(first) else None))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw = load_shiller()
    f = add_targets(build_features(raw), horizons=(H,))
    peaks = bear_markets(raw["price"])
    y = f[TARGET]

    results, preds = {}, {}
    base = base_rate_forecast(f)
    preds["Base rate"] = base
    for name, cols in MODELS.items():
        preds[name], last_model = walk_forward(f, cols)
        if name == "Causal watch model":
            final_model, final_cols = last_model, cols

    # Score everyone on the SAME months (where all models have forecasts and labels)
    common = pd.concat(preds, axis=1).dropna().index.intersection(y.dropna().index)
    for name, p in preds.items():
        s = score(p.loc[common], y.loc[common])
        s["brier_skill_vs_base"] = 1 - s["brier"] / score(base.loc[common], y.loc[common])["brier"]
        results[name] = s
    table = pd.DataFrame(results).T.round(3)
    print(f"Out-of-sample {common[0].date()} to {common[-1].date()} (months not already in a bear market)")
    print(table.to_string())

    print("\nCalibration — causal watch model:")
    print(calibration(preds["Causal watch model"].loc[common], y.loc[common]).to_string())

    eps = episode_table(preds["Causal watch model"], peaks)
    print("\nBear markets — what the causal watch model said in the 24 months before each peak:")
    print(eps.to_string(index=False))
    eps_ne = episode_table(preds["No-earnings variant"], peaks)

    # Coefficients of the most recent fit (standardized -> comparable)
    coefs = pd.Series(final_model[-1].coef_[0], index=final_cols).sort_values()
    print("\nLatest-fit standardized coefficients (log-odds per 1 std):")
    print(coefs.round(3).to_string())

    latest = {n: (p.dropna().index[-1].strftime("%Y-%m"), round(float(p.dropna().iloc[-1]), 3))
              for n, p in preds.items()}
    print("\nLatest readings:", latest)

    out = pd.concat(preds, axis=1).reindex(f.index)
    out["y"] = y
    out["price"] = f["price"]
    out["in_bear"] = f["in_bear"]
    out.to_csv("outputs/watch_predictions.csv")
    with open("outputs/watch_results.json", "w") as fh:
        json.dump(dict(scores=table.reset_index().to_dict(orient="records"),
                       calibration=calibration(preds["Causal watch model"].loc[common], y.loc[common]).reset_index().astype(str).to_dict(orient="records"),
                       episodes=eps.to_dict(orient="records"),
                       episodes_no_earnings=eps_ne.to_dict(orient="records"),
                       coefs=coefs.round(3).to_dict(), latest=latest,
                       oos=[str(common[0].date()), str(common[-1].date())],
                       peaks=[dict(peak=str(r.peak.date()), trough=str(r.trough.date()), decline=r.decline) for r in peaks.itertuples()]),
                  fh, indent=1, default=str)
