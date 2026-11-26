"""Exploratory directions for the paper's framing, run on author subsamples of increasing size before any full run.

A  popsplit   popularity split into extensive (has_deg), cumulative (log_deg), momentum (lrec30), lifecycle
              (log_age, legacy), visibility & brand, local co-use; the other four groups form one player (7 players,
              128 fits). Which kind of popularity carries the gain?
B  margin     intensive margin: candidate sets restricted to models with degree >= m (m = 1, 10); valid under IIA
              (a pre-choice attribute defines the subset; choices whose parent is outside are dropped). Five-group
              Shapley. Does fit matter more among used models?
C  pioneers   (CPU) early adopters: for parents with >= K first adoptions, the type and the match / popularity
              index of the first K adopters; expert share by adoption rank; does early composition predict later
              adoptions (birth-quarter and modality FE)?
D  sfvar      (CPU) full-market popularity growth with feedback (v2_networks.SF), coefficients ablated: full, no
              momentum, no age, cumulative only; concentration and quarterly leader turnover vs the real network.
E  platform   (CPU) SIR expected adoption counts on real candidate sets with match x {1,2,4} and popularity x
              {1,0.5}: concentration, mean match / popularity index of the chosen, implied retention change.
Subsamples: authors drawn at fraction f (seed s); all their first adoptions are kept (train and holdout).
Outputs: processed/choices/v2_pps/v2_explore/<part>_<f>_<s>.pkl, tables results/tables/v2/v2x_*.md (--collect).
usage: sbatch slurm/gpu.sbatch scripts/v2_explore.py --part gpu --fracs 0.1 0.25 0.5 --seeds 0 1
       sbatch slurm/cpu.sbatch scripts/v2_explore.py --part cpu --fracs 0.1 0.25 0.5 --seeds 0 1
"""
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.features.groups import GROUPS5, LIB
from hfpa.v2data import cdir, drop_constant, load_rows, shapley_fits, shares, split, write

SPEC = "v2_pps"
OUT = cdir(SPEC) / "v2_explore"
T0 = time.time()
POP_SPLIT = {
    "extensive": ["has_deg"],
    "cumulative": ["log_deg"],
    "momentum": ["lrec30"],
    "lifecycle": ["log_age", "legacy"],
    "visibility": ["major_org", "redistributor", "lquant", "size_na", "txt_na", "bench_na"] + [f"lib_{x}" for x in LIB],
    "local": ["lco2", "ds_couse"],
    "nonpop": GROUPS5["qual"] + GROUPS5["match"] + GROUPS5["self"] + GROUPS5["habit"],
}
assert sorted(sum(POP_SPLIT.values(), [])) == sorted(sum(GROUPS5.values(), []))
EXPERT = {"indiv_heavy", "org_other", "org_redistributor", "org_major"}


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def author_subset(ch, frac, seed):
    if frac >= 1:
        return None
    a = np.unique(ch.author.to_numpy())
    keep = np.random.default_rng(1000 + seed).random(len(a)) < frac
    return set(a[keep].tolist())


def subsample(rows, ch, frac, seed):
    A = author_subset(ch, frac, seed)
    if A is None:
        return rows
    ok = ch.author.reindex(rows.choice_id).isin(A).to_numpy()
    return rows[ok].reset_index(drop=True)


def share_table(fits, groups, Dh=True):
    out = {}
    for key in ("ll", "hold_ll"):
        sh, tot = shares(fits, list(groups), None, key)
        out[key] = {g: v["share"] for g, v in sh.items()}
        out[key + "_total"] = tot
    return out


# ---------------------------------------------------------------- GPU parts: A and B
def run_gpu(fracs, seeds):
    from hfpa.models.clogit import to_buckets
    rows, ch = load_rows(SPEC, "planA", extra=("deg",))
    log(f"loaded {len(rows):,} rows")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in fracs:
        for s in (seeds if f < 1 else [0]):
            R = subsample(rows, ch, f, s)
            tr, ho = split(R, ch)
            # A: popularity split
            feats = drop_constant(tr, sum(POP_SPLIT.values(), []))
            groups = {g: [x for x in v if x in feats] for g, v in POP_SPLIT.items()}
            D, Dh = to_buckets(tr, feats), to_buckets(ho, feats)
            fits = shapley_fits(D, feats, groups=groups, Dh=Dh, log=None)
            res = dict(part="popsplit", frac=f, seed=s, n=D.n, **share_table(fits, groups))
            pickle.dump(res, open(OUT / f"popsplit_{f}_{s}.pkl", "wb"))
            log(f"A f={f} s={s} n={D.n:,}: {({g: round(v, 4) for g, v in res['ll'].items()})}")
            del D, Dh
            # B: intensive margins
            for m in (1, 10):
                keep_row = tr.deg.to_numpy() >= m
                trm = tr[keep_row]
                ok = trm.groupby("choice_id").chosen.transform("max").to_numpy() == 1
                trm = trm[ok].reset_index(drop=True)
                hk = ho[ho.deg.to_numpy() >= m]
                hk = hk[hk.groupby("choice_id").chosen.transform("max").to_numpy() == 1].reset_index(drop=True)
                feats5 = drop_constant(trm, sum(GROUPS5.values(), []))
                g5 = {g: [x for x in v if x in feats5] for g, v in GROUPS5.items()}
                D, Dh = to_buckets(trm, feats5), to_buckets(hk, feats5)
                fits = shapley_fits(D, feats5, groups=g5, Dh=Dh, log=None)
                part = {"popularity": ["pop"], "fit": ["qual", "match"], "self": ["self"], "habit": ["habit"]}
                sh, tot = shares(fits, list(g5), part)
                shh, _ = shares(fits, list(g5), part, "hold_ll")
                res = dict(part="margin", min_deg=m, frac=f, seed=s, n=D.n, kept_share=D.n / tr.choice_id.nunique(),
                           set_size=len(trm) / max(D.n, 1), ll={g: v["share"] for g, v in sh.items()},
                           hold_ll={g: v["share"] for g, v in shh.items()}, dropped=sorted(set(sum(GROUPS5.values(), [])) - set(feats5)))
                pickle.dump(res, open(OUT / f"margin{m}_{f}_{s}.pkl", "wb"))
                log(f"B m={m} f={f} s={s} n={D.n:,} kept {res['kept_share']:.3f} set {res['set_size']:.0f}: "
                    f"{({g: round(v, 4) for g, v in res['ll'].items()})}")
                del D, Dh


# ---------------------------------------------------------------- CPU parts: C, D, E
def run_pioneers(fracs, seeds, K=10):
    sys.path.insert(0, "scripts")
    from v2_retention import chosen_indices
    import statsmodels.formula.api as smf
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet")
    d = ch[ch.pair_first].sort_values("t", kind="stable").copy()
    T = pd.read_parquet(paths.INTERIM / "v2_author_types.parquet").set_index("author_code").atype
    d["atype"] = d.author.map(T)
    d["expert"] = d.atype.isin(EXPERT).astype(float)
    idx = chosen_indices()
    d = d.join(idx[["I_pop", "I_match", "signal"]], on="choice_id")
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "major_org", "author_code"])
    d["rank"] = d.groupby("parent").cumcount() + 1
    n_tot = d.groupby("parent").size()
    rows = []
    for f in fracs:
        for s in (seeds if f < 1 else [0]):
            par = n_tot.index[n_tot >= K].to_numpy()
            if f < 1:
                par = par[np.random.default_rng(2000 + s).random(len(par)) < f]
            e = d[d.parent.isin(par)]
            early = e[e["rank"] <= K].groupby("parent").agg(expert=("expert", "mean"), I_match=("I_match", "mean"),
                                                            I_pop=("I_pop", "mean"), t10=("t", "max"), modality=("modality", "first"))
            early["later"] = (n_tot.reindex(early.index) - K).to_numpy()
            b = nodes.birth.to_numpy()[early.index.to_numpy()]
            early["days_to_K"] = np.log1p((early.t10 - b) / 86400)
            early["major"] = nodes.major_org.to_numpy()[early.index.to_numpy()].astype(float)
            early["bq"] = pd.to_datetime(b, unit="s").to_period("Q").astype(str)
            early["ly"] = np.log1p(early.later)
            m = smf.ols("ly ~ expert + I_match + I_pop + days_to_K + major + C(bq) + C(modality)", data=early.dropna()).fit(cov_type="HC1")
            prof = e[e["rank"] <= 50].groupby(pd.cut(e["rank"], [0, 1, 3, 10, 20, 50]), observed=True).agg(
                expert=("expert", "mean"), I_match=("I_match", "mean"), light=("atype", lambda x: (x == "indiv_light").mean()))
            top = early.later >= early.later.quantile(0.9)
            rows.append(dict(part="pioneers", frac=f, seed=s, parents=len(early),
                             coef_expert=m.params["expert"], se_expert=m.bse["expert"],
                             coef_match=m.params["I_match"], se_match=m.bse["I_match"],
                             coef_pop=m.params["I_pop"], se_pop=m.bse["I_pop"],
                             expert_early_top10pct=early.expert[top].mean(), expert_early_rest=early.expert[~top].mean(),
                             match_early_top10pct=early.I_match[top].mean(), match_early_rest=early.I_match[~top].mean(),
                             profile=prof.reset_index().astype({"rank": str}).to_dict("records")))
            log(f"C f={f} s={s}: parents {len(early):,}, expert {m.params['expert']:.3f} ({m.bse['expert']:.3f}), "
                f"match {m.params['I_match']:.3f} ({m.bse['I_match']:.3f})")
    pickle.dump(rows, open(OUT / "pioneers.pkl", "wb"))


def run_sfvar(seeds):
    sys.path.insert(0, "scripts")
    import v2_networks as N
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet")
    ev = ch[ch.pair_first].sort_values("t", kind="stable")[["t", "modality", "author", "parent"]].reset_index(drop=True)
    feats, b, _ = N.model_coefs()
    base = dict(zip(feats, b))
    variants = {"SF full": {}, "no momentum": {"lrec30": 0.0}, "no age decay": {"log_age": 0.0},
                "cumulative only": {"lrec30": 0.0, "log_age": 0.0}}
    res = [dict(variant="REAL", seed=0, **N.metrics(ev.parent.to_numpy(), ev.t.to_numpy()))]
    for name, mod in variants.items():
        sim = N.SF("sf", {**base, **mod})
        for s in seeds:
            par = sim.run(ev, s)
            res.append(dict(variant=name, seed=s, **N.metrics(par, ev.t.to_numpy())))
            log(f"D {res[-1]}")
    pickle.dump(res, open(OUT / "sfvar.pkl", "wb"))


def run_platform(fracs, seeds):
    sys.path.insert(0, "scripts")
    import v2_networks as N
    S = pickle.load(open(cdir(SPEC) / "v2_samples" / "P5" / "fit.pkl", "rb"))
    feats, bcoef, groups = S["feats"], S["nofe"]["coef"], S["groups"]
    rows0, ch = load_rows(SPEC, "planA")
    out = []
    beta_ret, pbar = 0.2209, 0.2825  # author-FE retention coefficient and mean stay rate, individuals (v2s_retention_welfare)
    for f in fracs:
        for s in (seeds if f < 1 else [0]):
            rows = subsample(rows0, ch, f, s).sort_values("choice_id", kind="stable").reset_index(drop=True)
            X = rows[feats].to_numpy(np.float64)
            I = {g: X[:, [feats.index(x) for x in v]] @ bcoef[[feats.index(x) for x in v]] if v else np.zeros(len(X))
                 for g, v in groups.items()}
            cand = rows.cand_id.to_numpy()
            for name, (kp, km) in {"base": (1, 1), "match x2": (1, 2), "match x4": (1, 4), "pop x0.5": (0.5, 1),
                                   "pop x0.5, match x2": (0.5, 2)}.items():
                u = kp * I["pop"] + km * I["match"] + I["qual"] + I["self"] + I["habit"]
                pr, st = N.sir_probs(rows, u)
                ec = pd.Series(pr).groupby(cand).sum()
                p = np.sort(ec.to_numpy() / ec.sum())[::-1]
                p = p[p > 0]
                ch_match = float(np.add.reduceat(pr * I["match"], st).mean())
                ch_pop = float(np.add.reduceat(pr * I["pop"], st).mean())
                out.append(dict(part="platform", frac=f, seed=s, variant=name, top10=p[:10].sum(),
                                eff=float(np.exp(-(p * np.log(p)).sum())), chosen_match=ch_match, chosen_pop=ch_pop))
            base_m = [r for r in out if r["frac"] == f and r["seed"] == s and r["variant"] == "base"][0]["chosen_match"]
            for r in out:
                if r["frac"] == f and r["seed"] == s:
                    r["d_retention_pp"] = 100 * beta_ret * (r["chosen_match"] - base_m) * pbar * (1 - pbar)
            log(f"E f={f} s={s}: " + "; ".join(f"{r['variant']} top10 {r['top10']:.3f} eff {r['eff']:.0f}"
                                                for r in out if r["frac"] == f and r["seed"] == s))
    pickle.dump(out, open(OUT / "platform.pkl", "wb"))


# ---------------------------------------------------------------- full-scale follow-ups
def full_train(groups_def):
    from hfpa.models.clogit import to_buckets
    rows, ch = load_rows(SPEC, "planA")
    tr, ho = split(rows, ch)
    feats = drop_constant(tr, sum(groups_def.values(), []))
    groups = {g: [x for x in v if x in feats] for g, v in groups_def.items()}
    D, Dh = to_buckets(tr, feats), to_buckets(ho, feats)
    author = ch.author.reindex(D.choice_ids).to_numpy()
    return D, Dh, feats, groups, author


def run_popboot(task, reps):
    """A: multiplier bootstrap (by author) of the 7-player popularity split; point fits re-done in each task as warm
    starts."""
    from hfpa.models.clogit import multiplier_weights
    D, Dh, feats, groups, author = full_train(POP_SPLIT)
    point = shapley_fits(D, feats, groups=groups, log=None)
    warm = {k: v["coef"] for k, v in point.items()}
    rng = np.random.default_rng(30_000 + task)
    out = []
    for r in range(reps):
        D.set_weights(multiplier_weights(author, np.ones(D.n), rng))
        fits = shapley_fits(D, feats, groups=groups, warm=warm, log=None)
        sh, _ = shares(fits, list(groups))
        out.append({g: v["share"] for g, v in sh.items()})
        log(f"popboot task {task} rep {r + 1}/{reps}: {({g: round(v, 4) for g, v in out[-1].items()})}")
    pickle.dump(out, open(OUT / f"popboot_{task}.pkl", "wb"))


REFIT = {"SF full (refit)": [], "no momentum (refit)": ["lrec30"], "no age decay (refit)": ["log_age"],
         "cumulative only (refit)": ["lrec30", "log_age"]}


def run_refit():
    """D: re-estimate the full five-group model without the momentum / age terms (no FE), so the remaining
    coefficients adjust; the popularity coefficients then drive the full-market growth simulation."""
    from hfpa.models.clogit import fit_subset
    D, Dh, feats, groups, author = full_train(GROUPS5)
    out = {}
    for name, drop in REFIT.items():
        cols = [f for f in feats if f not in drop]
        m, r = fit_subset(D, cols)
        ev = m.evaluate(Dh.subset_feats(cols))
        out[name] = dict(coef=dict(zip(cols, np.asarray(r["coef"]))), ll=r["ll"], hold=ev)
        log(f"refit {name}: ll {r['ll']:.1f}; log_deg {out[name]['coef'].get('log_deg'):.3f}, "
            f"has_deg {out[name]['coef'].get('has_deg'):.3f}, lrec30 {out[name]['coef'].get('lrec30', 0):.3f}, "
            f"log_age {out[name]['coef'].get('log_age', 0):.3f}; hold {ev}")
    pickle.dump(out, open(OUT / "refit.pkl", "wb"))


def run_sfrefit(seeds):
    sys.path.insert(0, "scripts")
    import v2_networks as N
    R = pickle.load(open(OUT / "refit.pkl", "rb"))
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet")
    ev = ch[ch.pair_first].sort_values("t", kind="stable")[["t", "modality", "author", "parent"]].reset_index(drop=True)
    res = [dict(variant="REAL", seed=0, **N.metrics(ev.parent.to_numpy(), ev.t.to_numpy()))]
    for name, r in R.items():
        coef = {k: r["coef"].get(k, 0.0) for k in set(r["coef"]) | {"lrec30", "log_age"}}
        sim = N.SF("sf", coef)
        for s in seeds:
            par = sim.run(ev, s)
            res.append(dict(variant=name, seed=s, ll=r["ll"], **N.metrics(par, ev.t.to_numpy())))
            log(f"D refit {res[-1]}")
    pickle.dump(res, open(OUT / "sfrefit.pkl", "wb"))


PLATFORM = {"base": (1, 1), "match x2": (1, 2), "match x4": (1, 4), "pop x0.5": (0.5, 1), "pop x0.5, match x2": (0.5, 2)}


def platform_metrics(rows, X, feats, groups, bcoef, N):
    I = {g: X[:, [feats.index(x) for x in v]] @ bcoef[[feats.index(x) for x in v]] if v else np.zeros(len(X))
         for g, v in groups.items()}
    cand = rows.cand_id.to_numpy()
    out = {}
    for name, (kp, km) in PLATFORM.items():
        u = kp * I["pop"] + km * I["match"] + I["qual"] + I["self"] + I["habit"]
        pr, st = N.sir_probs(rows, u)
        ec = np.bincount(np.searchsorted(np.unique(cand), cand), weights=pr)
        p = np.sort(ec / ec.sum())[::-1]
        p = p[p > 0]
        out[name] = dict(top10=p[:10].sum(), eff=float(np.exp(-(p * np.log(p)).sum())),
                         chosen_match=float(np.add.reduceat(pr * I["match"], st).mean()),
                         chosen_pop=float(np.add.reduceat(pr * I["pop"], st).mean()))
    return out


def run_platform_unc():
    """E on all first adoptions: point estimate and 50 bootstrap coefficient draws of the P5 model."""
    import glob
    sys.path.insert(0, "scripts")
    import v2_networks as N
    S = pickle.load(open(cdir(SPEC) / "v2_samples" / "P5" / "fit.pkl", "rb"))
    feats, groups = S["feats"], S["groups"]
    draws = sum((pickle.load(open(f, "rb"))["full"] for f in sorted(glob.glob(str(cdir(SPEC) / "v2_samples" / "P5" / "boot_*.pkl")))), [])
    rows, ch = load_rows(SPEC, "planA")
    rows = rows.sort_values("choice_id", kind="stable").reset_index(drop=True)
    X = rows[feats].to_numpy(np.float64)
    beta_ret, pbar = 0.2209, 0.2825
    res = []
    for d, b in enumerate([S["nofe"]["coef"]] + list(draws)):
        m = platform_metrics(rows, X, feats, groups, np.asarray(b), N)
        for name, v in m.items():
            res.append(dict(draw=d, variant=name, d_retention_pp=100 * beta_ret * (v["chosen_match"] - m["base"]["chosen_match"])
                            * pbar * (1 - pbar), **v))
        log(f"E draw {d}: " + "; ".join(f"{k} eff {v['eff']:.0f}" for k, v in m.items()))
    pickle.dump(res, open(OUT / "platform_unc.pkl", "wb"))


# ---------------------------------------------------------------- collect
def collect():
    import glob
    PB = sum((pickle.load(open(p, "rb")) for p in sorted(glob.glob(str(OUT / "popboot_*.pkl")))), [])
    if PB and (OUT / "popsplit_1.0_0.pkl").exists():
        pt = pickle.load(open(OUT / "popsplit_1.0_0.pkl", "rb"))
        Bd = pd.DataFrame(PB)
        write(pd.DataFrame([dict(group=g, share=pt["ll"][g], hold_share=pt["hold_ll"][g], lo=Bd[g].quantile(.025),
                                 hi=Bd[g].quantile(.975), reps=len(Bd)) for g in pt["ll"]]),
              "v2x_popsplit_full", "A (full data): popularity split, Shapley shares with multiplier-bootstrap 95% intervals")
    if (OUT / "refit.pkl").exists():
        R = pickle.load(open(OUT / "refit.pkl", "rb"))
        write(pd.DataFrame([dict(variant=k, ll=v["ll"], hold_mean_ll=v["hold"]["mean_ll"], hold_top10=v["hold"]["top10"],
                                 **{f"coef_{c}": v["coef"].get(c, 0.0) for c in ("has_deg", "log_deg", "lrec30", "log_age")})
                            for k, v in R.items()]), "v2x_refit", "D: models re-estimated without momentum / age (no FE, full data)")
    if (OUT / "sfrefit.pkl").exists():
        S = pd.DataFrame(pickle.load(open(OUT / "sfrefit.pkl", "rb")))
        agg = S.groupby("variant")[["top10", "eff_parents", "gini", "turnover", "n_parents"]].agg(["mean", "min", "max"])
        agg.columns = ["_".join(c) for c in agg.columns]
        write(agg.reset_index(), "v2x_sfrefit", "D: full-market popularity growth with feedback, re-estimated ablations (5 seeds)")
    if (OUT / "platform_unc.pkl").exists():
        P = pd.DataFrame(pickle.load(open(OUT / "platform_unc.pkl", "rb")))
        pt = P[P.draw == 0].set_index("variant")
        bs = P[P.draw > 0].groupby("variant")
        tab_ = pt[["top10", "eff", "chosen_match", "d_retention_pp"]].copy()
        for c in ("top10", "eff", "d_retention_pp"):
            tab_[c + "_lo"], tab_[c + "_hi"] = bs[c].quantile(.025), bs[c].quantile(.975)
        tab_["draws"] = bs.size()
        write(tab_.reset_index(), "v2x_platform_full", "E (all first adoptions): platform counterfactuals with bootstrap coefficient draws")
    A = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(str(OUT / "popsplit_*.pkl")))]
    if A:
        T = pd.DataFrame([dict(frac=a["frac"], seed=a["seed"], n=a["n"], measure=m, **a[m]) for a in A for m in ("ll", "hold_ll")])
        write(T.sort_values(["measure", "frac", "seed"]), "v2x_popsplit", "A: popularity split into its components (Shapley shares; 7 players)")
    B = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(str(OUT / "margin*_*.pkl")))]
    if B:
        T = pd.DataFrame([dict(min_deg=b["min_deg"], frac=b["frac"], seed=b["seed"], n=b["n"], kept=b["kept_share"],
                               set_size=b["set_size"], measure=m, **b[m]) for b in B for m in ("ll", "hold_ll")])
        write(T.sort_values(["measure", "min_deg", "frac", "seed"]), "v2x_margin", "B: intensive margin (candidates with degree >= m)")
    if (OUT / "pioneers.pkl").exists():
        P = pickle.load(open(OUT / "pioneers.pkl", "rb"))
        write(pd.DataFrame([{k: v for k, v in r.items() if k != "profile"} for r in P]), "v2x_pioneers",
              "C: early adopters (first 10 first adoptions) and later adoptions")
        write(pd.DataFrame([dict(frac=r["frac"], seed=r["seed"], **p) for r in P for p in r["profile"]]), "v2x_pioneers_profile",
              "C: adopter profile by adoption rank (parents with >= 10 first adoptions)")
    if (OUT / "sfvar.pkl").exists():
        write(pd.DataFrame(pickle.load(open(OUT / "sfvar.pkl", "rb"))), "v2x_sfvar", "D: popularity growth with feedback, ablations")
    if (OUT / "platform.pkl").exists():
        write(pd.DataFrame(pickle.load(open(OUT / "platform.pkl", "rb"))), "v2x_platform", "E: platform counterfactuals (SIR expected counts)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["gpu", "cpu", "pioneers", "sfvar", "platform", "collect", "popboot",
                                                     "refit", "sfrefit", "platform_unc"])
    ap.add_argument("--task", type=int, default=int(__import__("os").environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.1, 0.25, 0.5])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "gpu":
        run_gpu(a.fracs, a.seeds)
    if a.part in ("cpu", "pioneers"):
        run_pioneers(a.fracs, a.seeds)
    if a.part in ("cpu", "platform"):
        run_platform(a.fracs, a.seeds)
    if a.part == "sfvar" or (a.part == "cpu" and a.seeds):
        run_sfvar(a.seeds if a.part == "sfvar" else [0, 1, 2])
    if a.part == "popboot":
        run_popboot(a.task, a.reps)
    if a.part == "refit":
        run_refit()
    if a.part == "sfrefit":
        run_sfrefit(a.seeds)
    if a.part == "platform_unc":
        run_platform_unc()
    collect()
