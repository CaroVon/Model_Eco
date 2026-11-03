"""P21: general quality from independent sources, to narrow the gap between the quality lower and upper bound.

External data (robustness only; data/external/quality/):
  open_llm_leaderboard_v2_contents.parquet  HF dataset open-llm-leaderboard/contents (v2 leaderboard, 4.6k models;
                                            six benchmarks, 'Average'); matched to hubs by exact repo id
  lmarena_text_full.parquet                 HF dataset lmarena-ai/leaderboard-dataset, config text/full; category
                                            'overall', latest rating per model; matched by normalised repo name
--match    How much of the hub fixed effect phi (P17: P15 plan-A model with hub FE; hubs with SE < 1) do the
           quality measures explain? Noise-corrected R^2 = explained variance / (Var(phi) - mean SE^2), among the
           hubs each source covers; with the unsloth publisher premium (P17) and log parameters as controls.
           Also within-family size ladders (phi vs log parameters within the same lineage root).
--expert   "expert-revealed" quality: hub FE estimated separately from choices of experienced authors (>= 5
           earlier main choices) and of the others; noise-corrected correlation of the two.
usage: python scripts/p21_quality_ext.py --match ; sbatch slurm/gpu.sbatch scripts/p21_quality_ext.py --expert
"""
import argparse
import re

import numpy as np
import pandas as pd

from hfpa import paths
from p15_decomp import ALL, P5, drop_constant, fit, load, sample

EXT = paths.EXTERNAL / "quality"
SE_MAX = 1.0
EXPERT_MIN = 5


def norm_name(s):
    s = s.split("/")[-1].lower().replace("_", "-").replace(" ", "-")
    s = re.sub(r"^meta-(?=llama)", "", s)
    return re.sub(r"-hf$", "", s)


def hubs():
    F = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat_frac0.1" / "p17_hub_fe.parquet")
    n = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["root", "params_b", "author", "modality"])
    F = F.join(n, on="node")
    F["unsloth"] = (F.author == "unsloth").astype(float)
    F["lparams"] = np.log(F.params_b)
    v2 = pd.read_parquet(EXT / "open_llm_leaderboard_v2_contents.parquet")
    v2 = v2.sort_values("Submission Date").drop_duplicates("fullname", keep="last").set_index("fullname")
    F["v2_avg"] = v2["Average ⬆️"].reindex(F.model_id).to_numpy()
    for b in ("IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"):
        F[f"v2_{b}"] = v2[b].reindex(F.model_id).to_numpy()
    ar = pd.read_parquet(EXT / "lmarena_text_full.parquet")
    ar = ar[ar.category == "overall"].sort_values("leaderboard_publish_date").drop_duplicates("model_name", keep="last")
    ar["key"] = ar.model_name.map(norm_name)
    ar = ar.drop_duplicates("key", keep="last").set_index("key")
    F["arena"] = ar.rating.reindex(F.model_id.map(norm_name)).to_numpy()
    F["arena_name"] = ar.model_name.reindex(F.model_id.map(norm_name)).to_numpy()
    B = pd.read_parquet(paths.PROCESSED / "nodes" / "bench.parquet").set_index("node")
    F["native_bench"] = B.bench_z.reindex(F.node).to_numpy()
    return F[F.se < SE_MAX].reset_index(drop=True)


def ols_r2(y, X, se):
    """OLS of phi on X (with constant); noise-corrected R^2 (explained / (var(y) - mean se^2))."""
    A = np.column_stack([np.ones(len(y))] + [X[:, k] for k in range(X.shape[1])])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ b
    v, noise = y.var(), np.mean(se ** 2)
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A.T @ A) @ A.T)  # leverages
    press = np.mean(((y - fitted) / (1 - h)) ** 2)
    r2_loo = 1 - (press - noise) / max(v - noise, 1e-9)  # leave-one-out, noise-corrected
    return b, fitted.var() / v, fitted.var() / max(v - noise, 1e-9), v, noise, r2_loo


def part_match():
    F = hubs()
    out = []
    srcs = [("v2 average", ["v2_avg"]), ("v2 six benchmarks", [c for c in F.columns if c.startswith("v2_") and c != "v2_avg"]),
            ("arena rating", ["arena"]), ("native model-index", ["native_bench"])]
    for name, cols in srcs:
        sub = F.dropna(subset=cols)
        if len(sub) < 8:
            out.append(dict(source=name, hubs=len(sub)))
            continue
        for ctrl in ([], ["unsloth"], ["unsloth", "lparams"]):
            s2 = sub.dropna(subset=ctrl)
            X = s2[cols + ctrl].to_numpy(float)
            keep = [k for k in range(X.shape[1]) if X[:, k].std() > 0]
            b, r2, r2c, v, noise, r2_loo = ols_r2(s2.phi.to_numpy(), X[:, keep], s2.se.to_numpy())
            # quality-only part: variance of the quality terms' fitted values
            q = X[:, : len(cols)] @ b[1: 1 + len(cols)] if len(keep) == X.shape[1] else np.zeros(len(s2))
            out.append(dict(source=name, controls="+".join(ctrl) or "-", hubs=len(s2), r2=r2, r2_noise_corrected=r2c, r2_loo=r2_loo,
                            quality_part_share=q.var() / max(v - noise, 1e-9), var_phi=v, mean_se2=noise,
                            corr=np.corrcoef(s2.phi, s2[cols[0]])[0, 1]))
    R = pd.DataFrame(out)
    # implied shares: the hub-FE part of the Shapley quality share split by the measured-quality share q (as P17
    # splits it by s; non-quality part to popularity), q = quality_part_share with both controls
    P17 = pd.read_csv(paths.TABLES / "p17_mirror_quality.csv").set_index("quantity").value
    S = pd.read_csv(paths.TABLES / "p15_shapley_fe.csv")
    S = S[S.partition == "C: visibility and availability separate"].set_index("group").share
    S0 = pd.read_csv(paths.TABLES / "p15_shapley.csv")
    S0 = S0[(S0["sample"] == "planA") & (S0.partition == "C: visibility and availability separate")].set_index("group").share
    fe_part = P17["hub-FE part of quality share"]
    imp = []
    for _, x in R[R.controls == "unsloth+lparams"].dropna(subset=["quality_part_share"]).iterrows():
        q = float(np.clip(x.quality_part_share, 0, 1))
        fit_ = S["fit"] - (1 - q) * fe_part
        pop_ = S["popularity"] + (1 - q) * fe_part
        imp.append(dict(source=x.source, hubs=x.hubs, q=q, fit=fit_, popularity=pop_, ratio=fit_ / pop_))
    imp.append(dict(source="lower bound (no FE)", fit=S0["fit"], popularity=S0["popularity"], ratio=S0["fit"] / S0["popularity"]))
    imp.append(dict(source="P17-adjusted upper bound", q=1 - P17["s_total = s_sys + s"], fit=P17["adjusted upper bound (s_total): fit"],
                    popularity=P17["adjusted upper bound (s_total): popularity"], ratio=P17["adjusted upper bound (s_total): ratio"]))
    P5.write(pd.DataFrame(imp), "p21_implied", "P21 fit share and fit / popularity if only the measured-quality part of the hub FE is quality")
    P5.write(R, "p21_quality_ext", "P21 share of the hub FE (P17, SE < 1) explained by external and native quality measures")

    # within-family size ladders: same publisher and name up to the size token (Qwen2.5-{0.5,1.5,7}B-Instruct)
    F["root"] = F.model_id.str.split("/").str[0] + "/" + F.model_id.map(norm_name).str.replace(r"\d+(\.\d+)?[bm](?![a-z])", "#", regex=True)
    L = F.dropna(subset=["lparams"])
    L = L[L.groupby("root").node.transform("size") >= 3]
    lad = []
    for root, g in L.groupby("root"):
        if g.lparams.nunique() < 3:
            continue
        lad.append(dict(family=root, hubs=len(g), spearman=g[["phi", "lparams"]].corr("spearman").iloc[0, 1]))
    lad = pd.DataFrame(lad)
    if len(L):
        dm = L.assign(p=L.phi - L.groupby("root").phi.transform("mean"), x=L.lparams - L.groupby("root").lparams.transform("mean"))
        slope = (dm.p * dm.x).sum() / (dm.x ** 2).sum()
        within = dm.p.var()
        summ = pd.DataFrame([dict(families=len(lad), hubs=len(L), within_family_slope=slope,
                                  within_share_of_var_phi=within / F.phi.var(),
                                  size_share_of_within=(slope ** 2 * dm.x.var()) / within,
                                  mean_spearman=lad.spearman.mean() if len(lad) else np.nan)])
        P5.write(summ, "p21_size_ladder", "P21 hub FE vs log parameters within lineage families (reliable hubs)")
        lad.to_csv(paths.TABLES / "p21_size_ladder_families.csv", index=False)
    F.to_csv(paths.TABLES / "p21_hubs_quality.csv", index=False)
    print(F[["model_id", "phi", "v2_avg", "arena_name", "arena"]].dropna(subset=["arena"]).to_string())


def part_expert():
    rows, ch = load()
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "t"])
    a = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["author_code"]).author_code.to_numpy()
    full = full.assign(a=a[full.child]).sort_values("t", kind="stable")
    full["prior"] = full.groupby("a").cumcount()
    prior = full.drop_duplicates("child").set_index("child").prior  # earlier main choices of the author
    r, g = sample(rows, ch, "planA")
    del rows
    cid = np.unique(r.choice_id.to_numpy())
    exp_ = prior.reindex(ch.child.loc[cid]).fillna(0).to_numpy() >= EXPERT_MIN
    feats = drop_constant(r, ALL)
    phis = {}
    for name, ids in (("expert", cid[exp_]), ("other", cid[~exp_])):
        sub = r[r.choice_id.isin(ids).to_numpy()]
        fs = drop_constant(sub, feats)
        res = fit(sub, ch, fs, g, fe=True, cluster=False)
        n_fe = len(res["coef"]) - len(fs)
        phis[name] = pd.DataFrame({"phi": res["coef"][len(fs):], "se": res["se"][len(fs):]}, index=np.arange(1, n_fe + 1))
        print(name, len(ids), flush=True)
    E = phis["expert"].join(phis["other"], lsuffix="_e", rsuffix="_o")
    E = E[(E.se_e < SE_MAX) & (E.se_o < SE_MAX)]
    cov = np.cov(E.phi_e, E.phi_o)[0, 1]
    ve, vo = E.phi_e.var() - (E.se_e ** 2).mean(), E.phi_o.var() - (E.se_o ** 2).mean()
    out = pd.DataFrame([dict(hubs=len(E), choices_expert=int(exp_.sum()), choices_other=int((~exp_).sum()),
                             corr_raw=np.corrcoef(E.phi_e, E.phi_o)[0, 1], corr_noise_corrected=cov / np.sqrt(max(ve, 1e-9) * max(vo, 1e-9)),
                             var_expert=ve, var_other=vo)])
    P5.write(out, "p21_expert", f"P21 hub FE from experienced authors (>= {EXPERT_MIN} earlier choices) vs the others (SE < 1 in both)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--expert", action="store_true")
    a = ap.parse_args()
    if a.match:
        part_match()
    if a.expert:
        part_expert()
