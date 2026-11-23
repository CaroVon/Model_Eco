"""v2.1 stage 1: sample-selection robustness of the decomposition (A, B, C and the few-authors question).

One configuration = sample x groups x weights x author subset (CONFIGS below). For each:
  --part fit    full clogit (no FE; cluster-robust SE by author; holdout mean ll / top-1 / top-10), all 2^G Shapley
                fits (in-sample weighted ll gain over uniform attachment and holdout gain); main config also the
                500-FE model and the FE as an extra player. Saved to processed/choices/v2_pps/v2_samples/<tag>/.
  --part boot   multiplier bootstrap by author (xi ~ Exp(1), multiplied into the config's base weights), REPS
                replicates of all Shapley fits + the full model, warm-started; --task i writes boot_<i>.pkl.
  --part collect  tables v2_samples_<tag>.md (coefficients, shares with 95% intervals).
Samples (hfpa.v2data.load_rows): planA (first adoption of each author-parent pair, main), decisions (all 24h
decisions incl. repeats, author history kept, pi = 1), uploads (all choices, no session merge).
selfdeg: log_deg / has_deg / lrec30 net of the author's own earlier children (scripts/v2_selfdeg.py).
weights: choice (1 per choice) or author (each author's choices sum to the same total, mean weight 1).
usage: sbatch slurm/gpu.sbatch scripts/v2_samples.py --tag P5 --part fit
       sbatch --array=0-9 slurm/gpu.sbatch scripts/v2_samples.py --tag P5 --part boot --reps 5
"""
import argparse
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from hfpa import paths
from hfpa.features.groups import GROUPS, GROUPS5, NAMES
from hfpa.models.clogit import CLogit, multiplier_weights, to_buckets, uniform_ll
from hfpa.v2data import cdir, drop_constant, load_rows, shapley_fits, shares, split, write

SPEC = "v2_pps"
N_FE = 500
TYPES = ["org_major", "org_redistributor", "org_other", "indiv_heavy", "indiv_mid", "indiv_light"]
BASE = {
    "P4": dict(sample="planA", groups="four"),  # regression check against v2_pps_clogit / v2_pps_shapley
    "P5": dict(sample="planA", fe=True),  # main
    "D5": dict(sample="decisions", selfdeg=True),
    "D5raw": dict(sample="decisions"),
    "U5": dict(sample="uploads", selfdeg=True),
    "P5a": dict(sample="planA", weight="author"),
    "D5a": dict(sample="decisions", selfdeg=True, weight="author"),
    "P5top": dict(sample="planA", drop_top=0.01),
    "D5top": dict(sample="decisions", selfdeg=True, drop_top=0.01),
}
for t in TYPES:
    BASE[f"P5_{t}"] = dict(sample="planA", atype=t)
    BASE[f"D5_{t}"] = dict(sample="decisions", selfdeg=True, atype=t)
DEFAULT = dict(sample="planA", groups="five", weight="choice", selfdeg=False, atype=None, drop_top=0.0, fe=False)
CONFIGS = {k: {**DEFAULT, **v} for k, v in BASE.items()}
PART = {"four": {"popularity": ["pop"], "fit (quality + match)": ["qual", "match"], "inertia": ["inertia"]},
        "five": {"popularity": ["pop"], "fit (quality + match)": ["qual", "match"], "self-supply": ["self"],
                 "habit": ["habit"]}}
T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def odir(tag):
    p = cdir(SPEC) / "v2_samples" / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def author_types():
    T = pd.read_parquet(paths.INTERIM / "v2_author_types.parquet")
    return T.set_index("author_code").atype


def base_weights(author, how):
    if how == "choice":
        return np.ones(len(author))
    _, a, n = np.unique(author, return_inverse=True, return_counts=True)
    return (len(author) / len(n)) / n[a]


def data(cfg):
    groups = GROUPS if cfg["groups"] == "four" else GROUPS5
    allf = sum(groups.values(), [])
    rows, ch = load_rows(SPEC, cfg["sample"], selfdeg=cfg["selfdeg"])
    keep = None
    if cfg["atype"]:
        at = ch.author.map(author_types())
        keep = ch.index[(at == cfg["atype"]).to_numpy()]
    if cfg["drop_top"]:
        ids = rows.choice_id.unique()
        n_a = ch.author.reindex(ids).value_counts()
        top = n_a.index[: int(np.ceil(len(n_a) * cfg["drop_top"]))]
        keep = ch.index[~ch.author.isin(top).to_numpy()] if keep is None else keep[~ch.author.reindex(keep).isin(top)]
    if keep is not None:
        rows = rows[rows.choice_id.isin(keep)].reset_index(drop=True)
    tr, ho = split(rows, ch)
    feats = drop_constant(tr, allf)
    author_tr = ch.author.reindex(np.unique(tr.choice_id)).to_numpy()  # to_buckets orders choices by id
    w = base_weights(author_tr, cfg["weight"])
    D = to_buckets(tr, feats, fe_col="fe" if cfg["fe"] else None, weight=w)
    Dh = to_buckets(ho, feats, fe_col="fe" if cfg["fe"] else None)
    assert np.array_equal(D.choice_ids, np.unique(tr.choice_id))
    info = dict(n_train=D.n, n_hold=Dh.n, rows_train=len(tr), authors=len(np.unique(author_tr)),
                hist_rows_share=float((tr.author_reuse == 1).mean()),
                dropped=sorted(set(allf) - set(feats)))
    log(f"{cfg}: {info}; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB" if torch.cuda.is_available() else str(info))
    groups = {g: [f for f in v if f in feats] for g, v in groups.items()}
    return D, Dh, feats, groups, author_tr, w, info


def part_fit(tag):
    cfg = CONFIGS[tag]
    D, Dh, feats, groups, author, w, info = data(cfg)
    u0, u0h = float((w * uniform_ll(D)).sum()), float(uniform_ll(Dh).mean())
    m = CLogit(len(feats))
    res = m.fit(D.subset_feats(feats), cluster=author)
    ev = m.evaluate(Dh.subset_feats(feats))
    log(f"no FE: ll {res['ll']:.1f} hold {ev}")
    S = dict(cfg=cfg, info=info, feats=feats, groups=groups, u0=u0, u0h=u0h,
             nofe=dict(coef=res["coef"], se=res["se"], se_cl=res["se_cl"], ll=res["ll"], hold=ev))
    S["fits"] = shapley_fits(D, feats, groups=groups, Dh=Dh, log=log)
    pickle.dump(S, open(odir(tag) / "fit.pkl", "wb"))
    if cfg["fe"]:
        mf = CLogit(len(feats), n_fe=N_FE, l2_fe=1e-4)
        mf.init(np.r_[res["coef"], np.zeros(N_FE)])
        rf = mf.fit(D, cluster=author)
        evf = mf.evaluate(Dh)
        log(f"FE: ll {rf['ll']:.1f} hold {evf}")
        S["fe"] = dict(coef=rf["coef"], se=rf["se"], se_cl=rf["se_cl"], ll=rf["ll"], hold=evf)
        S["fits_fe"] = shapley_fits(D, feats, groups=groups, fe_player=True, n_fe=N_FE, Dh=Dh, log=log)
    pickle.dump(S, open(odir(tag) / "fit.pkl", "wb"))
    tables(tag)


def part_boot(tag, task, reps):
    cfg = CONFIGS[tag]
    S = pickle.load(open(odir(tag) / "fit.pkl", "rb"))
    D, Dh, feats, groups, author, w, info = data(cfg)
    assert feats == S["feats"]
    warm = {k: v["coef"] for k, v in S["fits"].items()}
    rng = np.random.default_rng(20_000 + task)
    out = dict(shapley=[], full=[])
    for r in range(reps):
        D.set_weights(multiplier_weights(author, w, rng))
        fits = shapley_fits(D, feats, groups=groups, warm=warm, log=None)
        out["shapley"].append({"|".join(sorted(k)): v["ll"] for k, v in fits.items()})
        out["full"].append(fits[frozenset(groups)]["coef"])
        log(f"{tag} task {task} replicate {r + 1}/{reps}")
    pickle.dump(out, open(odir(tag) / f"boot_{task}.pkl", "wb"))


def share_rows(fits, players, part, key):
    rows = []
    for pname, p in (("groups", None), ("pop / fit / ...", part)):
        sh, tot = shares(fits, players, p, key)
        for g, v in sh.items():
            rows.append(dict(partition=pname, group=NAMES.get(g, g), value=v["value"], share=v["share"], total=tot))
    return rows


def tables(tag):
    S = pickle.load(open(odir(tag) / "fit.pkl", "rb"))
    cfg, feats, groups = S["cfg"], S["feats"], S["groups"]
    players = list(groups)
    part = PART[cfg["groups"]]
    gof = {f: g for g, v in groups.items() for f in v}
    C = pd.DataFrame({"param": feats, "group": [gof[f] for f in feats], "coef": S["nofe"]["coef"],
                      "se_cl": S["nofe"]["se_cl"]})
    C["z"] = C.coef / C.se_cl
    if "fe" in S:
        C["coef_fe"] = S["fe"]["coef"][: len(feats)]
        C["se_cl_fe"] = S["fe"]["se_cl"][: len(feats)]
    # bootstrap
    B = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(odir(tag) / "boot_*.pkl")))]
    reps = sum((b["shapley"] for b in B), [])
    if reps:
        full = np.array(sum((b["full"] for b in B), []))
        C["lo"], C["hi"] = np.percentile(full, 2.5, 0), np.percentile(full, 97.5, 0)
    out = []
    for key, lab in (("ll", "in-sample"), ("hold_ll", "holdout")):
        for r in share_rows(S["fits"], players, part, key):
            out.append(dict(measure=lab, variant="no FE", **r))
        if "fits_fe" in S:
            for r in share_rows(S["fits_fe"], players + ["fe"], {**part, "FE": ["fe"]}, key):
                out.append(dict(measure=lab, variant="FE as a player", **r))
    O = pd.DataFrame(out)
    if reps:
        draws = []
        for r in reps:
            fits = {frozenset(k.split("|")) if k else frozenset(): dict(ll=v) for k, v in r.items()}
            d = {}
            for x in share_rows(fits, players, part, "ll"):
                d[(x["partition"], x["group"])] = x["share"]
            d[("pop / fit / ...", "ratio fit / popularity")] = (d[("pop / fit / ...", "fit (quality + match)")]
                                                               / d[("pop / fit / ...", "popularity")])
            draws.append(d)
        Dd = pd.DataFrame(draws)
        sel = (O.measure == "in-sample") & (O.variant == "no FE")
        pt = O[sel & (O.partition == "pop / fit / ...")].set_index("group").share
        O = pd.concat([O, pd.DataFrame([dict(measure="in-sample", variant="no FE", partition="pop / fit / ...",
                                             group="ratio fit / popularity",
                                             share=pt["fit (quality + match)"] / pt["popularity"])])],
                      ignore_index=True)
        sel = (O.measure == "in-sample") & (O.variant == "no FE")
        O.loc[sel, "lo"] = [Dd[(p, g)].quantile(0.025) for p, g in zip(O.partition[sel], O.group[sel])]
        O.loc[sel, "hi"] = [Dd[(p, g)].quantile(0.975) for p, g in zip(O.partition[sel], O.group[sel])]
        O.loc[sel, "reps"] = len(Dd)
    info = {**S["info"], "ll": S["nofe"]["ll"], "u0": S["u0"], "hold_mean_ll": S["nofe"]["hold"]["mean_ll"],
            "hold_gain": S["nofe"]["hold"]["mean_ll"] - S["u0h"], "hold_top1": S["nofe"]["hold"]["top1"],
            "hold_top10": S["nofe"]["hold"]["top10"]}
    note = (f"config {tag}: {cfg}. " + "; ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                                                 for k, v in info.items()))
    write(C, f"v2s_{tag}_clogit", f"v2.1 {tag}: conditional logit (SE clustered by author)", note)
    write(O, f"v2s_{tag}_shapley", f"v2.1 {tag}: Shapley shares of the log-likelihood gain over uniform attachment"
          + (f", {len(reps)} bootstrap replicates" if reps else ""), note)
    pickle.dump(dict(info=info, cfg=cfg, shares=O, coef=C), open(odir(tag) / "summary.pkl", "wb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(CONFIGS))
    ap.add_argument("--part", required=True, choices=["fit", "boot", "collect"])
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    {"fit": lambda: part_fit(a.tag), "boot": lambda: part_boot(a.tag, a.task, a.reps),
     "collect": lambda: tables(a.tag)}[a.part]()
