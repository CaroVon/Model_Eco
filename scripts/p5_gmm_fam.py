"""Stage P5, round 3: family x period effects (generational replacement) and finer age profile.

Round 2 (p5_gmm.py) still rejected the moment conditions (Hansen J 150-250 on 17 df). Candidate omitted
mechanism: a new release in a family (Llama-2 -> Llama-3) moves attention between members of that family
over time. Period effects are therefore made family-specific for the largest root families (others:
modality x period), and the age profile uses bins 0-7 / 7-30 / 30-90 / 90-180 days relative to >= 180.
Outcome: first author-parent choice, self-choices (same author) removed (plan A + round 2).

For the main family spec, Hansen J is decomposed by family (J_f = n * s_f' W s_f with s_f the family's
share of the mean moment) to show which families violate the moments.

usage: python scripts/p5_gmm_fam.py [--spec main_strat]
Writes results/tables/p5_gmm_fam.csv, p5_gmm_fam.md, p5_gmm_fam_Jbyfamily.csv.
"""
import argparse
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.data.build import T0
from hfpa.models.panel import build_panel, family_groups, qd_gmm2

DAY = 86400
MAIN = ["finetune", "adapter", "unspecified"]
BASE = ("has_deg", "log_deg")
AGE6 = BASE + ("a0_7", "a7_30", "a30_90", "a90_180")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="main_strat")
    a = ap.parse_args()
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["model_id", "birth", "author_code", "modality", "modality_code", "root"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    ch = pd.read_parquet(paths.PROCESSED / "choices" / a.spec / "choices.parquet", columns=["child", "parent", "t"])
    birth, author = nodes.birth.to_numpy(), nodes.author_code.to_numpy()
    mod, modc, root = nodes.modality.to_numpy(), nodes.modality_code.to_numpy(), nodes.root.to_numpy()

    m = E[E.rel_type.isin(MAIN)]
    deg_t, deg_p = birth[m.child_idx.to_numpy()], m.parent_idx.to_numpy()
    hist = pd.DataFrame({"a": author[m.child_idx.to_numpy()], "p": deg_p, "t": deg_t, "c": m.child_idx.to_numpy()})
    hist = hist.sort_values(["t", "c"], kind="stable")
    hist["first"] = ~hist.duplicated(["a", "p"])
    ch = ch.merge(hist[["c", "p", "first"]], left_on=["child", "parent"], right_on=["c", "p"], how="left")
    assert ch["first"].notna().all()
    self_ = author[ch.child.to_numpy()] == author[ch.parent.to_numpy()]
    ch["first_noself"] = ch["first"].astype(float) * ~self_
    ch["first"] = ch["first"].astype(float)

    starts = {"T0": int(T0.timestamp()),
              "robust_start": int(pd.Timestamp(paths.CFG["replay"]["robust_start"], tz="UTC").timestamp())}
    R = lambda **kw: {**dict(y="first_noself", period=7, start="T0", feats=AGE6, groups="modality", modality="all"), **kw}
    runs = [R(),  # round-2 spec with finer age bins
            R(groups="fam100"), R(groups="fam300"), R(groups="fam1000"),
            R(groups="fam100", period=14), R(groups="fam100", start="robust_start"),
            R(groups="fam100", y="first"), R(groups="fam100", feats=BASE),
            R(groups="fam100", modality="text"), R(groups="fam100", modality="image_gen")]
    out, diag = [], None
    for i, r in enumerate(runs):
        t = time.time()
        t0 = starts[r["start"]]
        c = ch[ch.t >= t0]
        if r["modality"] != "all":
            c = c[mod[c.parent.to_numpy()] == r["modality"]]
        P = build_panel(birth, c.t.to_numpy(), c.parent.to_numpy(), r["period"] * DAY, deg_t=deg_t, deg_parent=deg_p,
                        y_weight=None if r["y"] == "all" else c[r["y"]].to_numpy(), t0=t0, age_unit=DAY)
        nd = P["nodes"]
        if r["groups"] == "modality":
            grp = modc[nd]
        else:
            grp = family_groups(nd, root, modc[nd], P["y"].sum(1), top=int(r["groups"][3:]))
        est, se, Jst, df = qd_gmm2(P, names=r["feats"], group=grp, return_moments=(i == 1))
        for k in r["feats"]:
            out.append({**{kk: v for kk, v in r.items() if kk != "feats"}, "spec": "+".join(r["feats"][2:]) or "base",
                        "param": k, "est": float(est[k]), "se": float(se[k]), "z": float(est[k] / se[k]),
                        "converged": est["converged"], "hansen_J": Jst, "J_df": df, "n_nodes": len(nd),
                        "n_groups": len(np.unique(grp)), "n_periods": P["T"], "y_total": float(P["y"].sum())})
        print({kk: v for kk, v in r.items() if kk != "feats"}, {k: f"{est[k]:.3f} ({se[k]:.3f})" for k in r["feats"]},
              f"J={Jst:.1f}/{df} conv={est['converged']} [{time.time() - t:.0f}s]", flush=True)
        if i == 1:  # Hansen J by family for the main family spec
            M, W = est["_M"], est["_W"]
            n = M.shape[0]
            fam = root[nd]
            s = pd.DataFrame(M).groupby(fam).sum() / n
            jf = n * np.einsum("fl,lk,fk->f", s.to_numpy(), W, s.to_numpy())
            diag = pd.DataFrame({"root": s.index, "family": nodes.model_id.to_numpy()[s.index], "J_f": jf,
                                 "n_nodes": pd.Series(fam).value_counts().reindex(s.index).to_numpy(),
                                 "y": pd.Series(P["y"].sum(1)).groupby(fam).sum().reindex(s.index).to_numpy()})
            diag = diag.sort_values("J_f", ascending=False)
            diag.to_csv(paths.TABLES / "p5_gmm_fam_Jbyfamily.csv", index=False)
            print(diag.head(15).to_string(index=False), flush=True)
    df = pd.DataFrame(out)
    df.to_csv(paths.TABLES / "p5_gmm_fam.csv", index=False)
    md = ["# P5 round 3: family x period effects and finer age bins (quasi-differenced GMM, real data)", "",
          df[df.param == "log_deg"].drop(columns=["param", "y_total"]).to_markdown(index=False, floatfmt=".3f"), "",
          "## Hansen J by family (main family spec, top 20)", "",
          diag.head(20).to_markdown(index=False, floatfmt=".1f") if diag is not None else ""]
    (paths.TABLES / "p5_gmm_fam.md").write_text("\n".join(md) + "\n")
    print("done")
