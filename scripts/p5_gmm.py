"""Stage P5 (main): preferential attachment with node fixed effects on the real HF network.

Estimator: over-identified quasi-differenced GMM (hfpa.models.panel.qd_gmm2), validated in P0b-P0d.
Main spec (user decision 2026-10-29, "plan A"):
  outcome  = choices (finetune + adapter + unspecified, child after T0, same modality as parent),
             counting only each author's FIRST choice of a parent;
  degree   = every main-type child, including those before T0;
  period effects = modality x period (stands in for the modality-restricted choice sets).
Variants: all choices as outcome, period 7 / 14 / 30 days, log(1 + degree) only, start at robust_start,
per-modality fits; round 2 adds the non-linear age profile (age bins / log age; P0e) and drops
self-choices (child and parent by the same author) from the outcome.

usage: python scripts/p5_gmm.py [--spec main_strat]
Writes results/tables/p5_gmm.csv and p5_gmm.md.
"""
import argparse
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.data.build import T0
from hfpa.models.panel import build_panel, qd_gmm2

DAY = 86400
MAIN = ["finetune", "adapter", "unspecified"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="main_strat")
    a = ap.parse_args()
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code", "modality"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    ch = pd.read_parquet(paths.PROCESSED / "choices" / a.spec / "choices.parquet", columns=["child", "parent", "t"])
    birth = nodes.birth.to_numpy()
    author = nodes.author_code.to_numpy()
    mod = nodes.modality.to_numpy()

    m = E[E.rel_type.isin(MAIN)]
    deg_t, deg_p = birth[m.child_idx.to_numpy()], m.parent_idx.to_numpy()
    # first choice of each (author, parent) pair over the whole history, in birth order
    hist = pd.DataFrame({"a": author[m.child_idx.to_numpy()], "p": deg_p, "t": deg_t, "c": m.child_idx.to_numpy()})
    hist = hist.sort_values(["t", "c"], kind="stable")
    hist["first"] = ~hist.duplicated(["a", "p"])
    first = ch.merge(hist[["c", "p", "first"]], left_on=["child", "parent"], right_on=["c", "p"], how="left")["first"]
    assert first.notna().all()
    ch["first"] = first.to_numpy().astype(float)
    ch["self"] = author[ch.child.to_numpy()] == author[ch.parent.to_numpy()]
    ch["first_noself"] = ch["first"] * ~ch["self"]
    print(f"choices {len(ch):,}; first-choice share {ch['first'].mean():.3f}; self-choice share "
          f"{ch['self'].mean():.3f}; first & not self {ch['first_noself'].mean():.3f}", flush=True)

    starts = {"T0": int(T0.timestamp()), "robust_start": int(pd.Timestamp(paths.CFG["replay"]["robust_start"], tz="UTC").timestamp())}
    BASE = ("has_deg", "log_deg")
    AGE = BASE + ("age_lt30", "age_30_90", "age_90_365")
    runs = [dict(y="first", period=7, start="T0", feats=BASE, modality="all")]  # round-1 main spec first
    runs += [dict(y="all", period=7, start="T0", feats=("has_deg", "log_deg"), modality="all")]
    runs += [dict(y="first", period=p, start="T0", feats=("has_deg", "log_deg"), modality="all") for p in (14, 30)]
    runs += [dict(y="first", period=7, start="T0", feats=("log1p_deg",), modality="all")]
    runs += [dict(y="first", period=7, start="robust_start", feats=("has_deg", "log_deg"), modality="all")]
    runs += [dict(y="first", period=7, start="T0", feats=("has_deg", "log_deg"), modality=md)
             for md in ch.parent.map(pd.Series(mod)).value_counts().index]
    # round 2: life-cycle controls and self-choices
    runs += [dict(y=y, period=p, start="T0", feats=f, modality="all")
             for y in ("first", "first_noself") for f in (AGE, BASE + ("log_age",)) for p in (7, 14)]
    runs += [dict(y="first_noself", period=7, start="robust_start", feats=AGE, modality="all"),
             dict(y="all", period=7, start="T0", feats=AGE, modality="all")]
    runs += [dict(y="first_noself", period=7, start="T0", feats=AGE, modality=md) for md in ("text", "image_gen")]

    out = []
    for r in runs:
        t = time.time()
        t0 = starts[r["start"]]
        c = ch[ch.t >= t0]
        if r["modality"] != "all":
            c = c[mod[c.parent.to_numpy()] == r["modality"]]
        if len(c) < 1000:
            continue
        P = build_panel(birth, c.t.to_numpy(), c.parent.to_numpy(), r["period"] * DAY, deg_t=deg_t, deg_parent=deg_p,
                        y_weight=None if r["y"] == "all" else c[r["y"]].to_numpy(), t0=t0, age_unit=DAY)
        est, se, Jst, df = qd_gmm2(P, names=r["feats"], group=mod[P["nodes"]])
        for k in r["feats"]:
            out.append({**{kk: v for kk, v in r.items() if kk != "feats"}, "spec": "+".join(r["feats"][2:]) or "base", "param": k, "est": float(est[k]),
                        "se": float(se[k]), "z": float(est[k] / se[k]), "converged": est["converged"],
                        "hansen_J": Jst, "J_df": df, "n_nodes": P["y"].shape[0], "n_periods": P["T"],
                        "n_choices": len(c), "y_total": float(P["y"].sum())})
        print(r, {k: f"{est[k]:.3f} ({se[k]:.3f})" for k in r["feats"]}, f"J={Jst:.1f}/{df}",
              f"conv={est['converged']} [{time.time() - t:.0f}s]", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(paths.TABLES / "p5_gmm.csv", index=False)
    md = ["# P5: preferential attachment with node FE (quasi-differenced GMM, real data)", "",
          "Main spec = first row: outcome = first author-parent choice, period 7 days, start T0, all modalities.", "",
          df.drop(columns=["y_total"]).to_markdown(index=False, floatfmt=".3f")]
    (paths.TABLES / "p5_gmm.md").write_text("\n".join(md) + "\n")
    print("done")
