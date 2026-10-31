"""P16 calibration on the full plan-A sample (all first author-parent choices, not the 10% modelling sample).

The 10% calibration with author FE was too imprecise to tell a zero match effect from a small one (SE 0.007).
Coefficients stay those of the 10% plan-A conditional logit (as in p16_welfare.py); only the chosen rows of the
full choice data are needed to compute the chosen parent's utility indices x_k = theta_k . x_chosen.
The fit loss (needs the sampled candidates) stays the 10% one and is converted with the full-sample beta.

usage: python scripts/p16_calib_full.py
Writes processed/choices/main_strat/p16_calib_full.parquet and results/tables/p16_calib_full*.md / .csv.
"""
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.features.match import build, node_attrs
from p16_welfare import P15, P5, add_outcomes, calibrate

if __name__ == "__main__":
    t0 = time.time()
    # 1. coefficients: identical to p16_welfare.py
    rows, ch = P15.load()
    r, g = P15.sample(rows, ch, "planA")
    feats = P15.drop_constant(r, P15.ALL)
    b = dict(zip(feats, P15.fit(r, ch, feats, g, cluster=False)["coef"][: len(feats)]))
    del rows, r
    print(f"coefficients in {time.time() - t0:.0f}s", flush=True)

    # 2. chosen rows of all main choices, plan A
    rows, ch, final_deg, nodes = P5.load("main_strat", filters=[("chosen", "==", 1)])
    assert rows.choice_id.is_unique
    ch["first"] = P15.first_flag(ch, nodes)
    keep = ch.index[ch["first"]]
    rows = rows[rows.choice_id.isin(keep).to_numpy() & (rows.author_reuse.to_numpy() == 0)].reset_index(drop=True)
    ch = ch.loc[rows.choice_id]
    print(f"{len(rows):,} plan-A choices in {time.time() - t0:.0f}s", flush=True)
    nn = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                         columns=["model_id", "pipeline_tag", "params_b", "license_cat", "major_org", "author_code",
                                  "birth", "likes"])
    meta = pd.read_parquet(paths.INTERIM / "meta.parquet", columns=["model_id", "languages", "datasets"]).set_index("model_id")
    meta = meta.reindex(nn.model_id)
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent", "t"])
    M = build(rows, ch, node_attrs(nn, meta.languages.tolist(), meta.datasets.tolist()), full, nn.author_code.to_numpy())
    for c in M.columns:
        rows[c] = M[c].to_numpy(np.float32)
    P15.add_features(rows, ch, final_deg, nodes)
    print(f"features in {time.time() - t0:.0f}s", flush=True)

    # 3. indices, outcomes, calibration
    A = pd.DataFrame(index=pd.Index(rows.choice_id.to_numpy(), name="choice_id"))
    for k, fs in P15.GROUPS.items():
        A[f"x_{k}"] = sum(b.get(f, 0) * rows[f].to_numpy(float) for f in fs)
    kid = ch.child.to_numpy()
    add_outcomes(A, kid, ch.t.to_numpy(), nn)
    A["modality"] = ch.modality_name.to_numpy()
    A["author"] = nn.author_code.to_numpy()[kid]
    A["parent"] = ch.parent.to_numpy()
    A["child"] = kid
    A["hist_has"], A["ds_has"] = rows.hist_has.to_numpy(), rows.ds_has.to_numpy()
    A.to_parquet(paths.PROCESSED / "choices" / "main_strat" / "p16_calib_full.parquet")
    cal = calibrate(A)
    # subsample with a demand signal (the match index is 0 by construction elsewhere)
    sig = calibrate(A[(A.hist_has > 0) | (A.ds_has > 0)], outcomes=("y_children180", "y_likes"))
    cal = pd.concat([cal.assign(sample="all plan A"), sig.assign(sample="with demand signal")], ignore_index=True)
    P5.write(cal, "p16_calib_full", "P16 calibration on all plan-A first choices (chosen parent's utility indices; SE clustered by parent)")

    # 4. fit loss in outcome units: 10% fit gaps, full-sample beta
    W = pd.read_parquet(paths.PROCESSED / "choices" / P15.SPEC / "p16_welfare.parquet")
    conv = []
    for (smp, fes), sel in cal[(cal.outcome == "y_children180") & (cal.x == "x_match")].groupby(["sample", "fe"]):
        beta, se = float(sel.coef.iloc[0]), float(sel.se.iloc[0])
        for t, gap in W.groupby("type").fit_gap.mean().items():
            conv.append(dict(sample=smp, calibration=fes, beta_match=beta, beta_se=se, type=t, fit_gap_util=gap,
                             fit_loss_pct_children180=np.expm1(beta * gap) * 100,
                             ci95_lo=np.expm1((beta - 1.96 * se) * gap) * 100, ci95_hi=np.expm1((beta + 1.96 * se) * gap) * 100))
    P5.write(pd.DataFrame(conv), "p16_calib_full_fitloss", "P16 fit loss in outcome units with the full-sample calibration")
    print(f"done in {time.time() - t0:.0f}s", flush=True)
