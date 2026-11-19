"""v2 stage 4: first-choice decomposition on the full plan-A sample (part 1 of the v2 plan; q comparison).

Four players (hfpa.features.groups): popularity (incl. visibility, local search, availability), general quality,
demand match, inertia. Conditional logit with log q correction; train = choices before the cut, holdout after.

--part clogit   full model without / with the 500 ex-ante FE; coefficients with cluster-robust (author) SEs;
                holdout mean log-likelihood and top-1 / top-10; drop-one-group LR tests.
--part shapley  2^4 fits (in-sample and holdout log-likelihood gain over uniform attachment), and 2^5 fits with the
                FE as a fifth player (B.4: out-of-sample FE share). Fits saved for warm-started bootstraps.
--part boot     multiplier (Bayesian) bootstrap by author, xi ~ Exp(1): per task REPS replicates, each = the 16
                Shapley fits (warm-started from the point estimates) + 3 extra full fits; with --fe also 2 FE-model fits
                per replicate (FE sampling variances for G). Writes <spec>_boot/task_<i>.npz.
--part collect  percentile intervals; tables v2/<spec>_*.md.
usage: sbatch slurm/gpu.sbatch scripts/v2_main.py --spec v2_pps --part clogit
       sbatch --array=0-9 slurm/gpu.sbatch scripts/v2_main.py --spec v2_pps --part boot --reps 5 --fe
"""
import argparse
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from hfpa.features.groups import ALL, GROUPS, NAMES, group_of
from hfpa.models.clogit import CLogit, fit_subset, lr_test, multiplier_weights, to_buckets, uniform_ll
from hfpa.v2data import TABLES, cdir, drop_constant, load_rows, shapley_fits, shares, split, write

N_FE = 500
PLAYERS = list(GROUPS)
PART_FIT = {"popularity": ["pop"], "fit (quality + match)": ["qual", "match"], "inertia": ["inertia"]}
T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def data(spec, fe=True):
    rows, ch = load_rows(spec, "planA")
    tr, ho = split(rows, ch)
    feats = drop_constant(tr, ALL)
    log(f"{spec}: train {tr.choice_id.nunique():,} choices / {len(tr):,} rows; holdout {ho.choice_id.nunique():,}; "
        f"features {len(feats)} (dropped {sorted(set(ALL) - set(feats))})")
    D = to_buckets(tr, feats, fe_col="fe" if fe else None)
    Dh = to_buckets(ho, feats, fe_col="fe" if fe else None)
    author = ch.author.reindex(D.choice_ids).to_numpy()
    log(f"buckets: train {len(D.batches)}, holdout {len(Dh.batches)}; GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB"
        if torch.cuda.is_available() else "cpu")
    return D, Dh, feats, author, ch


def out_path(spec, name):
    p = cdir(spec) / "v2_est"
    p.mkdir(exist_ok=True)
    return p / name


# ---------------------------------------------------------------- clogit
def part_clogit(spec):
    D, Dh, feats, author, ch = data(spec)
    u0, u0h = uniform_ll(D).sum(), uniform_ll(Dh).mean()
    rec, fit_rows, saved = [], [], {}
    m = CLogit(len(feats))
    res = m.fit(D.subset_feats(feats), cluster=author)
    ev = m.evaluate(Dh.subset_feats(feats))
    log(f"no FE: ll {res['ll']:.1f} hold {ev}")
    saved["nofe"] = dict(coef=res["coef"], cov_cl=res["cov_cl"], cov=res["cov"])
    fit_rows.append(dict(model="full", ll=res["ll"], k=res["k"], n=res["n"], ll0=u0, pseudo_r2=1 - res["ll"] / u0,
                         hold_mean_ll=ev["mean_ll"], hold_gain=ev["mean_ll"] - u0h, hold_top1=ev["top1"],
                         hold_top10=ev["top10"]))
    for i, f in enumerate(feats):
        rec.append(dict(model="full", group=group_of(f), param=f, coef=res["coef"][i], se=res["se"][i],
                        se_cl=res["se_cl"][i], z=res["coef"][i] / res["se_cl"][i]))
    for g in PLAYERS:
        sub = [f for f in feats if f not in GROUPS[g]]
        init = [c for f, c in zip(feats, res["coef"]) if f in sub]
        _, rs = fit_subset(D, sub, init=init)
        st, p = lr_test(rs["ll"], res["ll"], len(feats) - len(sub))
        fit_rows.append(dict(model=f"drop {g}", ll=rs["ll"], k=len(sub), lr_stat=st, p=p))
    # with the 500 ex-ante FE
    mf = CLogit(len(feats), n_fe=N_FE, l2_fe=1e-4)
    mf.init(np.r_[res["coef"], np.zeros(N_FE)])
    rf = mf.fit(D, cluster=author)
    evf = mf.evaluate(Dh)
    log(f"FE: ll {rf['ll']:.1f} hold {evf}")
    saved["fe"] = dict(coef=rf["coef"], se=rf["se"], cov_cl=rf["cov_cl"])
    fit_rows.append(dict(model="full + FE", ll=rf["ll"], k=rf["k"], n=rf["n"], ll0=u0, pseudo_r2=1 - rf["ll"] / u0,
                         hold_mean_ll=evf["mean_ll"], hold_gain=evf["mean_ll"] - u0h, hold_top1=evf["top1"],
                         hold_top10=evf["top10"]))
    for i, f in enumerate(feats):
        rec.append(dict(model="full + FE", group=group_of(f), param=f, coef=rf["coef"][i], se=rf["se"][i],
                        se_cl=rf["se_cl"][i], z=rf["coef"][i] / rf["se_cl"][i]))
    saved["feats"] = feats
    saved["n_train"], saved["n_hold"] = D.n, Dh.n
    pickle.dump(saved, open(out_path(spec, "clogit.pkl"), "wb"))
    write(pd.DataFrame(rec), f"{spec}_clogit", f"{spec}: plan-A conditional logit, four groups (SE clustered by author)")
    write(pd.DataFrame(fit_rows), f"{spec}_clogit_fit", f"{spec}: fit, holdout and drop-one-group LR tests")


# ---------------------------------------------------------------- shapley
def share_table(fits, tag, players, extra_parts=()):
    out = []
    for key, lab in (("ll", "in-sample"), ("hold_ll", "holdout")):
        for pname, part in [("four groups", None), ("pop / fit / inertia", PART_FIT)] + list(extra_parts):
            if pname == "pop / fit / inertia" and "fe" in players:
                continue
            sh, tot = shares(fits, players, part, key)
            for g, v in sh.items():
                out.append(dict(variant=tag, measure=lab, partition=pname, group=NAMES.get(g, g), value=v["value"],
                                share=v["share"], total_gain=tot))
    return out


def part_shapley(spec, variant="both"):
    """variant: nofe (2^4 fits), fe (2^5 fits with the FE player) or both; results merged into shapley.pkl."""
    D, Dh, feats, author, ch = data(spec)
    f = out_path(spec, "shapley.pkl")
    saved = pickle.load(open(f, "rb")) if f.exists() else {}
    if saved.get("feats") not in (None, feats):
        saved = {}
    saved["feats"] = feats
    if variant in ("nofe", "both"):
        saved["fits"] = shapley_fits(D, feats, Dh=Dh, log=log)
        pickle.dump(saved, open(f, "wb"))
    if variant in ("fe", "both"):
        saved["fits_fe"] = shapley_fits(D, feats, fe_player=True, n_fe=N_FE, Dh=Dh, log=log)
        pickle.dump(saved, open(f, "wb"))
    shapley_tables(spec, saved)


def shapley_tables(spec, saved):
    out = share_table(saved["fits"], "no FE", PLAYERS) if "fits" in saved else []
    if "fits_fe" in saved:
        ff = saved["fits_fe"]
        P5 = PLAYERS + ["fe"]
        out += share_table(ff, "FE as fifth player", P5,
                           [("pop / fit / inertia / FE", {"popularity": ["pop"], "fit (quality + match)": ["qual", "match"],
                                                           "inertia": ["inertia"], "FE": ["fe"]})])
        full, full_fe, none = ff[frozenset(PLAYERS)], ff[frozenset(P5)], ff[frozenset()]
        oos = (full_fe["hold_ll"] - full["hold_ll"]) / (full_fe["hold_ll"] - none["hold_ll"])
        out.append(dict(variant="FE as fifth player", measure="holdout", partition="B.4 out-of-sample FE gain",
                        group="FE gain / full+FE holdout gain", value=full_fe["hold_ll"] - full["hold_ll"],
                        share=max(oos, 0.0), total_gain=full_fe["hold_ll"] - none["hold_ll"]))
    write(pd.DataFrame(out), f"{spec}_shapley", f"{spec}: Shapley shares of the log-likelihood gain over uniform attachment",
          "in-sample = training log-likelihood (sum); holdout = mean holdout log-likelihood per choice.")


# ---------------------------------------------------------------- bootstrap
def part_boot(spec, task, reps, fe):
    D, Dh, feats, author, ch = data(spec)
    S = pickle.load(open(out_path(spec, "shapley.pkl"), "rb"))
    C = pickle.load(open(out_path(spec, "clogit.pkl"), "rb"))
    assert S["feats"] == feats
    warm = {k: v["coef"] for k, v in S["fits"].items()}
    rng = np.random.default_rng(10_000 + task)
    res = dict(shapley=[], full=[], fe=[])
    for r in range(reps):
        w = multiplier_weights(author, np.ones(D.n), rng)
        D.set_weights(w)
        fits = shapley_fits(D, feats, warm=warm, log=None)
        res["shapley"].append({"|".join(sorted(k)): v["ll"] for k, v in fits.items()})
        res["full"].append(fits[frozenset(PLAYERS)]["coef"])
        for _ in range(3):
            D.set_weights(multiplier_weights(author, np.ones(D.n), rng))
            _, rr = fit_subset(D, feats, init=C["nofe"]["coef"])
            res["full"].append(rr["coef"])
        if fe:
            for _ in range(2):
                D.set_weights(multiplier_weights(author, np.ones(D.n), rng))
                _, rr = fit_subset(D, feats, fe=True, n_fe=N_FE, init=C["fe"]["coef"])
                res["fe"].append(rr["coef"])
        log(f"task {task} replicate {r + 1}/{reps} done")
    d = cdir(spec) / "v2_est" / "boot"
    d.mkdir(exist_ok=True, parents=True)
    pickle.dump(res, open(d / f"task_{task}.pkl", "wb"))


def part_collect(spec):
    C = pickle.load(open(out_path(spec, "clogit.pkl"), "rb"))
    S = pickle.load(open(out_path(spec, "shapley.pkl"), "rb"))
    feats = C["feats"]
    B = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(cdir(spec) / "v2_est" / "boot" / "task_*.pkl")))]
    if not B:
        print("no bootstrap replicates")
        return
    full = np.array(sum((b["full"] for b in B), []))
    tab = pd.DataFrame({"param": feats, "group": [group_of(f) for f in feats], "coef": C["nofe"]["coef"],
                        "se_cl": np.sqrt(np.diag(C["nofe"]["cov_cl"])), "se_boot": full.std(0, ddof=1),
                        "lo": np.percentile(full, 2.5, 0), "hi": np.percentile(full, 97.5, 0)})
    tab["reps"] = len(full)
    write(tab, f"{spec}_clogit_boot", f"{spec}: full model, multiplier bootstrap by author ({len(full)} replicates)")
    out = []
    point = {}
    for pname, part in [("four groups", None), ("pop / fit / inertia", PART_FIT)]:
        sh, _ = shares(S["fits"], PLAYERS, part)
        point[pname] = {g: v["share"] for g, v in sh.items()}
    reps = sum((b["shapley"] for b in B), [])
    draws = {pname: [] for pname in point}
    for r in reps:
        fits = {frozenset(k.split("|")) if k else frozenset(): dict(ll=v) for k, v in r.items()}
        for pname, part in [("four groups", None), ("pop / fit / inertia", PART_FIT)]:
            sh, _ = shares(fits, PLAYERS, part)
            draws[pname].append({g: v["share"] for g, v in sh.items()})
    for pname in point:
        Dd = pd.DataFrame(draws[pname])
        for g in point[pname]:
            out.append(dict(partition=pname, group=NAMES.get(g, g), share=point[pname][g], lo=Dd[g].quantile(0.025),
                            hi=Dd[g].quantile(0.975), reps=len(Dd)))
        if pname == "pop / fit / inertia":
            ratio = Dd["fit (quality + match)"] / Dd["popularity"]
            pr = point[pname]["fit (quality + match)"] / point[pname]["popularity"]
            out.append(dict(partition=pname, group="ratio fit / popularity", share=pr, lo=ratio.quantile(0.025),
                            hi=ratio.quantile(0.975), reps=len(Dd)))
    write(pd.DataFrame(out), f"{spec}_shapley_boot", f"{spec}: Shapley shares with multiplier-bootstrap 95% intervals")
    fe = sum((b["fe"] for b in B), [])
    if fe:
        F = np.array(fe)[:, len(feats):]
        np.save(cdir(spec) / "v2_est" / "fe_boot_var.npy", F.var(0, ddof=1))
        log(f"FE bootstrap variances from {len(F)} replicates saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v2_pps")
    ap.add_argument("--part", required=True, choices=["clogit", "shapley", "boot", "collect"])
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fe", action="store_true")
    ap.add_argument("--variant", default="both", choices=["both", "nofe", "fe"], help="shapley fits to run")
    a = ap.parse_args()
    if a.part == "clogit":
        part_clogit(a.spec)
    elif a.part == "shapley":
        part_shapley(a.spec, a.variant)
    elif a.part == "boot":
        part_boot(a.spec, a.task, a.reps, a.fe)
    else:
        part_collect(a.spec)
