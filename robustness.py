"""Step 3b — Robustness: do simpler / more regularized versions of the watch
model do better? All variants are scored the same way, and ALL are reported
(picking the best one after seeing test results would be overfitting)."""
import json
import pandas as pd, numpy as np, watch_model as wm
from sklearn.metrics import roc_auc_score, brier_score_loss
from features import *
raw=load_shiller(); f=add_targets(build_features(raw),horizons=(24,))
# theory-signed equal-weight composite (signs from THEORY: + means raises crash risk)
signs={"log_cape":1,"boom_3y":1,"rate_chg_12":1,"vol_12":1,"calm_years":1,"eps_growth":-1,"inflation":1}
z=pd.DataFrame({c:s*(f[c]-f[c].expanding(120).mean())/f[c].expanding(120).std() for c,s in signs.items()})
f["composite"]=z.mean(axis=1,skipna=False)
full=wm.MODELS["Causal watch model"]
variants={"full C=0.5":(full,0.5),"full C=0.02":(full,0.02),
 "no rate_level C=0.02":([c for c in full if c!="rate_level"],0.02),
 "theory composite (1 weight)":(["composite"],0.5),"CAPE only":(["log_cape"],0.5)}
res=[]
for start in ["1925-01-01","1950-01-01"]:
  wm.OOS_START=start; base=wm.base_rate_forecast(f); print("OOS from",start)
  for n,(cols,C) in variants.items():
    p,_=wm.walk_forward(f,cols,C)
    m=pd.concat([p.rename('p'),base.rename('b'),f['y_24m'].rename('y')],axis=1).dropna()
    bs=brier_score_loss(m.y,m.p); bb=brier_score_loss(m.y,m.b)
    res.append(dict(oos_start=start,variant=n,auc=round(roc_auc_score(m.y,m.p),3),brier_skill=round(1-bs/bb,3),n=len(m)))
    print(f"  {n:32s} AUC {roc_auc_score(m.y,m.p):.3f}  BrierSkill {1-bs/bb:+.3f}  n={len(m)}")

json.dump(res,open("outputs/robustness.json","w"),indent=1)
