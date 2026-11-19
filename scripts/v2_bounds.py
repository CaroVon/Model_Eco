"""v2 stage 4 (B.4, G): quality upper bounds for the four-group decomposition.

--ext      how much of the 500 ex-ante FE (v2 plan-A model with FE) external quality explains (P21 method):
           OLS of phi on the quality measure (+ redistributor dummy, log parameters), among the FE nodes each source
           covers; noise-corrected with the FE sampling variance from the multiplier bootstrap (G; analytic SE if the
           bootstrap is missing). quality_part_share q = Var(quality part of the fit) / (Var(phi) - mean noise).
           External-quality upper bound: the FE player's Shapley share (2^5 fits, v2_main --part shapley) is split
           q -> quality, 1 - q -> popularity. Also writes q_ext.npy (FE id -> quality part of phi, 0 if not covered)
           for the QF network (v2_networks.py).
--placebo  placebo demand (P20 design on v2): Shapley shares with the donor's demand-match features (v2_features
           --placebo) in place of the real ones; residual placebo match share should be near 0 after fix A.
usage: python scripts/v2_bounds.py --ext ; sbatch slurm/gpu.sbatch scripts/v2_bounds.py --placebo
"""
import argparse
import glob
import pickle
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hfpa import paths
from hfpa.features.groups import ALL, GROUPS, NAMES
from hfpa.models.clogit import to_buckets
from hfpa.v2data import cdir, load_rows, shapley_fits, shares, split, write

SPEC = "v2_pps"
EXT = paths.EXTERNAL / "quality"
N_FE = 500


def norm_name(s):
    s = s.split("/")[-1].lower().replace("_", "-").replace(" ", "-")
    s = re.sub(r"^meta-(?=llama)", "", s)
    return re.sub(r"-hf$", "", s)


def fe_nodes(spec=SPEC):
    """FE id -> node (ids are the same in every row file)."""
    out = {}
    for f in sorted(glob.glob(str(cdir(spec) / "feats" / "*.parquet"))):
        if f.endswith(".notext.parquet") or not pq.read_schema(f).names:
            continue
        x = pd.read_parquet(f, columns=["cand_id", "fe"])
        x = x[x.fe > 0].drop_duplicates()
        out.update(dict(zip(x.fe.to_numpy(), x.cand_id.to_numpy())))
    s = pd.Series(out).sort_index()
    return s


def fe_table():
    C = pickle.load(open(cdir(SPEC) / "v2_est" / "clogit.pkl", "rb"))
    F = len(C["feats"])
    phi = C["fe"]["coef"][F:]
    vb = cdir(SPEC) / "v2_est" / "fe_boot_var.npy"
    var = np.load(vb) if vb.exists() else C["fe"]["se"][F:] ** 2
    nodes = fe_nodes()
    T = pd.DataFrame({"fe": np.arange(1, N_FE + 1), "phi": phi, "var": var})
    T["node"] = nodes.reindex(T.fe).to_numpy()
    n = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "params_b", "author", "modality"])
    T = T.join(n, on="node")
    red = pd.read_csv(paths.INTERIM / "v2_idx" / "redistributor_orgs.csv").iloc[:, 0]
    T["redistributor"] = T.model_id.str.split("/").str[0].isin(red).astype(float)
    T["lparams"] = np.log(T.params_b.where(T.params_b > 0))
    v2 = pd.read_parquet(EXT / "open_llm_leaderboard_v2_contents.parquet")
    v2 = v2.sort_values("Submission Date").drop_duplicates("fullname", keep="last").set_index("fullname")
    T["v2_avg"] = v2["Average ⬆️"].reindex(T.model_id).to_numpy()
    ar = pd.read_parquet(EXT / "lmarena_text_full.parquet")
    ar = ar[ar.category == "overall"].sort_values("leaderboard_publish_date").drop_duplicates("model_name", keep="last")
    ar["key"] = ar.model_name.map(norm_name)
    ar = ar.drop_duplicates("key", keep="last").set_index("key")
    T["arena"] = ar.rating.reindex(T.model_id.map(norm_name)).to_numpy()
    B = pd.read_parquet(paths.PROCESSED / "nodes" / "bench.parquet").set_index("node")
    T["native_bench"] = B.bench_z.reindex(T.node).to_numpy()
    return T


def part_ext():
    T = fe_table()
    rel = T[np.sqrt(T["var"]) < 1.0]
    out, q_main, qext = [], None, np.zeros(N_FE)
    for name, cols in (("v2 average", ["v2_avg"]), ("arena rating", ["arena"]), ("native model-index", ["native_bench"]),
                       ("v2 average + arena (both covered)", ["v2_avg", "arena"])):
        for ctrl in ([], ["redistributor"], ["redistributor", "lparams"]):
            s = rel.dropna(subset=cols + ctrl)
            if len(s) < 8:
                out.append(dict(source=name, controls="+".join(ctrl) or "-", hubs=len(s)))
                continue
            X = s[cols + ctrl].to_numpy(float)
            keep = [k for k in range(X.shape[1]) if X[:, k].std() > 0]
            X = X[:, keep]
            A = np.column_stack([np.ones(len(s)), X])
            b, *_ = np.linalg.lstsq(A, s.phi.to_numpy(), rcond=None)
            v, noise = s.phi.var(), s["var"].mean()
            nq = sum(1 for k in keep if k < len(cols))
            qpart = X[:, :nq] @ b[1:1 + nq]
            qs = float(np.clip(qpart.var() / max(v - noise, 1e-9), 0, 1))
            r2c = (A @ b).var() / max(v - noise, 1e-9)
            out.append(dict(source=name, controls="+".join(ctrl) or "-", hubs=len(s), share_of_fe=len(s) / len(rel),
                            r2_noise_corrected=r2c, quality_part_share=qs, var_phi=v, mean_noise=noise))
            if name == "v2 average" and ctrl == ["redistributor", "lparams"]:
                q_main = qs
                qext[s.fe.to_numpy() - 1] = qpart - qpart.mean()
    R = pd.DataFrame(out)
    write(R, "v2_quality_ext", "Share of the ex-ante FE explained by external quality (FE with bootstrap SE < 1)",
          f"Reliable FE: {len(rel)} of {N_FE}. Noise = mean bootstrap variance of phi (G).")
    np.save(cdir(SPEC) / "v2_est" / "q_ext.npy", qext)
    # implied bounds from the 5-player Shapley
    S = pickle.load(open(cdir(SPEC) / "v2_est" / "shapley.pkl", "rb"))
    P5 = list(GROUPS) + ["fe"]
    imp = []
    sh0, _ = shares(S["fits"], list(GROUPS), {"popularity": ["pop"], "fit": ["qual", "match"], "inertia": ["inertia"]})
    imp.append(dict(bound="lower (no FE; quality = bench_z only)", fit=sh0["fit"]["share"],
                    popularity=sh0["popularity"]["share"]))
    for key, lab in (("ll", "in-sample"), ("hold_ll", "holdout")):
        sh, _ = shares(S["fits_fe"], P5, {"popularity": ["pop"], "fit": ["qual", "match"], "inertia": ["inertia"],
                                          "FE": ["fe"]}, key)
        fe_s = sh["FE"]["share"]
        imp.append(dict(bound=f"FE all quality ({lab}; old-style upper bound)", fit=sh["fit"]["share"] + max(fe_s, 0),
                        popularity=sh["popularity"]["share"], fe_share=fe_s))
        for q, nm in ((q_main, "external quality (v2 average, controls)"),):
            if q is None:
                continue
            imp.append(dict(bound=f"{nm}, {lab}", q=q, fe_share=fe_s, fit=sh["fit"]["share"] + q * max(fe_s, 0),
                            popularity=sh["popularity"]["share"] + (1 - q) * max(fe_s, 0)))
    I = pd.DataFrame(imp)
    I["ratio_fit_pop"] = I.fit / I.popularity
    write(I, "v2_bounds", "Fit share and fit / popularity under the quality bounds (B.4)",
          "Out-of-sample bound: the FE player's holdout Shapley share (negative = the FE do not generalise -> 0).")


def part_placebo():
    rows, ch = load_rows(SPEC, "planA")
    tr, _ = split(rows, ch)
    feats = [f for f in ALL if f in tr and tr[f].std() > 0]
    P = []
    for f in sorted(glob.glob(str(cdir(SPEC) / "feats_placebo" / "*.parquet"))):
        if pq.read_schema(f).names:
            P.append(pd.read_parquet(f))
    P = pd.concat(P, ignore_index=True)
    trp = tr[["choice_id", "cand_id"]].merge(P, on=["choice_id", "cand_id"], how="left")
    keep = trp.has_donor.fillna(0).to_numpy() > 0
    real = tr[keep].reset_index(drop=True)
    plac = real.copy()
    for c in GROUPS["match"]:
        plac[c] = trp.loc[keep, c].to_numpy(np.float32)
    out = []
    for lab, R in (("real", real), ("placebo", plac)):
        D = to_buckets(R, feats)
        fits = shapley_fits(D, feats, log=None)
        sh, tot = shares(fits, list(GROUPS))
        for g, v in sh.items():
            out.append(dict(demand=lab, group=NAMES[g], share=v["share"], total_gain=tot, n=D.n))
        print(lab, {g: round(v["share"], 4) for g, v in sh.items()}, flush=True)
    write(pd.DataFrame(out), "v2_placebo", "Placebo demand (donor of another author, same month x modality): Shapley "
          "shares with the donor's demand-match features in place of the real ones (fix A check)")


def part_placebo_joint():
    """Own and donor (placebo) demand-match features in one model: the donor's match is a fifth Shapley player, so
    the own match share is net of what any same-month same-modality demand would explain (common trends)."""
    from hfpa.models.clogit import CLogit
    rows, ch = load_rows(SPEC, "planA")
    tr, _ = split(rows, ch)
    P = []
    for f in sorted(glob.glob(str(cdir(SPEC) / "feats_placebo" / "*.parquet"))):
        if pq.read_schema(f).names:
            P.append(pd.read_parquet(f))
    P = pd.concat(P, ignore_index=True)
    trp = tr[["choice_id", "cand_id"]].merge(P, on=["choice_id", "cand_id"], how="left")
    keep = trp.has_donor.fillna(0).to_numpy() > 0
    R = tr[keep].reset_index(drop=True)
    pl = [f"pl_{c}" for c in GROUPS["match"]]
    for c, q in zip(GROUPS["match"], pl):
        R[q] = trp.loc[keep, c].to_numpy(np.float32)
    groups = {**GROUPS, "placebo": pl}
    feats = [f for f in ALL if f in R and R[f].std() > 0] + [q for q in pl if R[q].std() > 0]
    D = to_buckets(R, feats)
    fits = shapley_fits(D, feats, groups=groups, log=None)
    sh, tot = shares(fits, list(groups))
    out = [dict(group=NAMES.get(g, "placebo (donor) match"), share=v["share"], value=v["value"], total_gain=tot, n=D.n)
           for g, v in sh.items()]
    author = ch.author.loc[D.choice_ids].to_numpy()
    res = CLogit(len(feats)).fit(D, cluster=author)
    co = pd.DataFrame({"param": feats, "coef": res["coef"], "se_cl": res["se_cl"]})
    co = co[co.param.isin(GROUPS["match"] + pl)]
    write(pd.DataFrame(out), "v2_placebo_joint", "Own and donor demand match in one model: Shapley shares with the "
          "donor's match as a fifth player (own match share net of common same-month demand)",
          "Donor = another author's plan-A choice in the same month x modality (P20 design); "
          "choices with a donor only.")
    write(co.reset_index(drop=True), "v2_placebo_joint_coef", "Own vs donor demand-match coefficients in one model "
          "(SE clustered by author)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", action="store_true")
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--placebo-joint", action="store_true")
    a = ap.parse_args()
    if a.ext:
        part_ext()
    if a.placebo:
        part_placebo()
    if a.placebo_joint:
        part_placebo_joint()
