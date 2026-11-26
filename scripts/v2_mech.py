"""Concentration mechanisms (plan: note5). Small-scale first; full runs only after the user approves.

--part p0   Phase 0, from saved outputs only (no new model runs):
  networks  one table of the growth / SIR networks (v2_networks, v2_grow) -> v2m_networks; degree CCDF data
            (first adoptions per parent, seed 0 of each growth network and REAL) -> v2_mech/ccdf.parquet
  rerun     "rerun history" (Salganik et al. 2006): with the same event sequence and choice model, do the same
            parents win in every simulated world? Per network: overlap of each world's top-10 with the real top-10,
            mean pairwise overlap of top-10 sets across worlds, share of the real top-10 parents' adoptions kept,
            unpredictability U = mean over parents of the across-world variance of the market share (only parents
            with a positive share in some world), and Spearman correlation of parent counts with the real counts
  levels    concentration at three levels of aggregation: parent model, lineage root (nodes.root: the topmost
            ancestor) and the root's author (the releasing organization) -> v2m_levels
  corr      within-set correlation of the match and popularity indices (P5 coefficients, training first
            adoptions): is the popular candidate also the better-fitting one? All candidates and d > 0 only;
            mean match index by degree bin -> v2m_corr, v2m_match_by_deg
--part p2   Phase 2 small-scale tier (author subsamples --fracs x --seeds): candidate "brake" features computed from
            node metadata and the real edges, added to the five-group model one at a time and together:
  org_eco     log(1 + first-adoption-relevant adoptions, before t, of OTHER models whose lineage root has the same
              author as the candidate's root): the installed base of the releasing organization
  superseded  1 if the candidate's uploader released, between the candidate's birth and t, another model of the same
              modality, the same root status (base vs derived) and the same size class (round(log2 params); unknown
              = its own class); log_newer = log(1 + number of such models)
            Reported: coefficient (SE clustered by author), likelihood-ratio statistic, holdout gain per choice
            -> v2m_brake_fit
--part s5   network-level Shapley (Phase 1): networks grown by the five-group model re-estimated on each subset of
            the groups (v2_grow.py --s5-task, grow/mech/S5_<bits>__<until>__<seed>.parquet). Value of a subset =
            concentration of its network minus that of the empty subset (uniform attachment); concentration measured
            as -log(effective number of parents), top-10 share and the root-organization top-10 share. Shapley values
            per horizon, mean over seeds, next to the likelihood Shapley shares -> v2m_s5_nets, v2m_s5_shapley
usage: sbatch --exclude=<ds nodes> slurm/cpu.sbatch scripts/v2_mech.py --part p0
       sbatch --exclude=<ds nodes> slurm/gpu.sbatch scripts/v2_mech.py --part p2 --fracs 0.1 0.25 0.5 --seeds 0 1
"""
import argparse
import glob
import os
import pickle
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.v2data import cdir, load_rows, split, write

SPEC = "v2_pps"
CDIR = cdir(SPEC)
OUTD = CDIR / "v2_est" / "grow"
OUT = CDIR / "v2_mech"
T = paths.TABLES / "v2"
sys.path.insert(0, str(paths.REPO / "scripts"))


def grown():
    """{network: [DataFrame(choice_id, sim_parent, real_parent)] over seeds}."""
    G = {}
    for f in sorted(glob.glob(str(OUTD / "*_[0-9].parquet"))):
        net, _ = os.path.basename(f)[:-8].rsplit("_", 1)
        G.setdefault(net, []).append(pd.read_parquet(f, columns=["choice_id", "sim_parent", "real_parent"]))
    return G


def part_networks(G):
    A = pd.read_csv(T / "v2_networks.csv").assign(simulator="replay, no feedback (SIR) / simplified growth")
    B = pd.read_csv(T / "v2_grow.csv").assign(simulator="growth with feedback (v2_grow)")
    cols = ["simulator", "network", "top10_mean", "top10_std", "eff_parents_mean", "eff_parents_std", "gini_mean",
            "tail_alpha_mean", "turnover_mean", "n_parents_mean"]
    write(pd.concat([A, B], ignore_index=True)[cols], "v2m_networks", "Phase 0: all simulated networks in one table",
          "v2_networks: SF = simplified full-market growth (popularity terms, degree from first adoptions); SF-SIR / "
          "FULL / QF = importance resampling on the real candidate sets, no feedback. v2_grow: PPS sets drawn on "
          "simulated degrees, all features recomputed, all main choices counted.")
    rows = []
    r0 = next(iter(G.values()))[0]
    rows.append(pd.DataFrame({"network": "REAL", "count": r0.real_parent.value_counts().to_numpy()}))
    for net, L in G.items():
        rows.append(pd.DataFrame({"network": net, "count": L[0].sim_parent.value_counts().to_numpy()}))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(OUT / "ccdf.parquet", index=False)


def top(s, k=10):
    return set(s.value_counts().index[:k])


def part_rerun(G):
    r0 = next(iter(G.values()))[0]
    real = r0.real_parent.value_counts()
    real_top = set(real.index[:10])
    out = []
    for net, L in G.items():
        counts = [l.sim_parent.value_counts() for l in L]
        tops = [set(c.index[:10]) for c in counts]
        S = pd.concat([c / c.sum() for c in counts], axis=1).fillna(0)
        rt = real.reindex(S.index).fillna(0)
        out.append(dict(network=net, worlds=len(L),
                        overlap_real_top10=np.mean([len(t & real_top) / 10 for t in tops]),
                        overlap_between_worlds=np.mean([len(a & b) / 10 for a, b in combinations(tops, 2)])
                        if len(tops) > 1 else np.nan,
                        real_top10_share_kept=np.mean([c.reindex(list(real_top)).fillna(0).sum() / c.sum()
                                                       for c in counts]) / (real.iloc[:10].sum() / real.sum()),
                        unpredictability_U=float(S.var(axis=1, ddof=1).mean()) if len(L) > 1 else np.nan,
                        U_top100=float(S.loc[S.mean(axis=1).nlargest(100).index].var(axis=1, ddof=1).mean())
                        if len(L) > 1 else np.nan,
                        spearman_real=float(np.mean([pd.concat([c, real], axis=1).fillna(0).corr("spearman").iloc[0, 1]
                                                     for c in counts])),
                        same_parent_as_real=float(np.mean([(l.sim_parent == l.real_parent).mean() for l in L]))))
    write(pd.DataFrame(out), "v2m_rerun", "Phase 0: rerun history - do the same parents win in every simulated world?",
          "Worlds = seeds of v2_grow (same event sequence and model). real_top10_share_kept = adoptions of the real "
          "top-10 parents in a world relative to their real share. U = mean over parents of the across-world variance "
          "of the market share; U_top100 = the same over the 100 parents with the largest mean share.")


def part_levels(G):
    from v2_networks import metrics
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["root", "author"])
    root = nodes.root.to_numpy()
    org = pd.factorize(nodes.author)[0][root]  # author of the lineage root
    ch = pd.read_parquet(CDIR / "decisions.parquet", columns=["choice_id", "t"]).set_index("choice_id").t
    r0 = next(iter(G.values()))[0]
    t = ch.loc[r0.choice_id].to_numpy()
    out = []
    for net, par in [("REAL", [r0.real_parent.to_numpy()])] + [(n, [l.sim_parent.to_numpy() for l in L])
                                                               for n, L in G.items()]:
        for lev, f in (("parent model", lambda p: p), ("lineage root", lambda p: root[p]),
                       ("root organization", lambda p: org[p])):
            M = pd.DataFrame([metrics(f(p), t) for p in par]).mean()
            out.append(dict(network=net, level=lev, worlds=len(par), **M.to_dict()))
    write(pd.DataFrame(out), "v2m_levels", "Phase 0: concentration at three levels of aggregation (first adoptions)",
          "lineage root = topmost ancestor of the parent (nodes.root); root organization = author of that root. "
          "Turnover = mean quarterly share of the top-10 replaced. Simulated networks: mean over worlds.")


def part_corr():
    P = pickle.load(open(CDIR / "v2_samples" / "P5" / "fit.pkl", "rb"))
    feats, b, groups = P["feats"], P["nofe"]["coef"], P["groups"]
    rows, ch = load_rows(SPEC, "planA")
    tr, _ = split(rows, ch)
    del rows
    tr = tr.sort_values("choice_id", kind="stable").reset_index(drop=True)
    I = {}
    for g in ("pop", "match"):
        idx = [feats.index(f) for f in groups[g]]
        I[g] = tr[[feats[i] for i in idx]].to_numpy(np.float64) @ b[idx]
    used = tr.has_deg.to_numpy() > 0
    out = []
    for lab, keep in (("all candidates", np.ones(len(tr), bool)), ("candidates with d > 0", used)):
        cid = tr.choice_id.to_numpy()[keep]
        x, y = I["pop"][keep], I["match"][keep]
        D = pd.DataFrame({"c": cid, "x": x, "y": y, "xx": x * x, "yy": y * y, "xy": x * y, "n": 1})
        S = D.groupby("c").sum()
        n = S.n
        cov = S.xy / n - S.x * S.y / n ** 2
        vx, vy = S.xx / n - (S.x / n) ** 2, S.yy / n - (S.y / n) ** 2
        ok = (vx > 1e-12) & (vy > 1e-12) & (n >= 3)
        r = (cov / np.sqrt(vx * vy))[ok]
        pooled = np.corrcoef(x - np.repeat((S.x / n).to_numpy(), n.to_numpy()),
                             y - np.repeat((S.y / n).to_numpy(), n.to_numpy()))[0, 1]
        out.append(dict(sample=lab, choices=int(ok.sum()), share_choices_defined=float(ok.mean()),
                        corr_median=float(r.median()), corr_mean=float(r.mean()), share_positive=float((r > 0).mean()),
                        corr_pooled_within=float(pooled)))
    write(pd.DataFrame(out), "v2m_corr", "Phase 0: within-set correlation of the popularity and match indices "
          "(P5 coefficients, training first adoptions)",
          "Per choice: Pearson correlation across the candidates of its PPS set (sets with variation in both "
          "indices); pooled = correlation of the within-set deviations over all rows.")
    d = np.where(used, np.exp(tr.log_deg.to_numpy()), 0)
    bins = [-1, 0, 1, 3, 10, 30, 100, 300, 1000, 1e9]
    lab = ["0", "1", "2-3", "4-10", "11-30", "31-100", "101-300", "301-1000", ">1000"]
    B = pd.DataFrame({"deg": pd.cut(d, bins, labels=lab), "match": I["match"], "chosen": tr.chosen.to_numpy()})
    M = B.groupby("deg", observed=True).agg(rows=("match", "size"), match_mean=("match", "mean"),
                                            chosen_rate=("chosen", "mean")).reset_index()
    write(M, "v2m_match_by_deg", "Phase 0: mean match index of candidates by degree bin (training first adoptions)")


# ---------------------------------------------------------------- Phase 2: brake features (estimation tier)
BIG = 10 ** 10  # > any unix time


def count_before(keys_sorted, key, t):
    """number of events with the same key and time < t; keys_sorted = sorted(key * BIG + time)."""
    q = key.astype(np.int64) * BIG
    return np.searchsorted(keys_sorted, q + t, "left") - np.searchsorted(keys_sorted, q, "left")


def brake_features(rows, t_of_choice):
    from hfpa.features.groups import MAIN
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["birth", "author_code", "root", "is_root", "modality_code", "params_b"])
    birth = nodes.birth.to_numpy(np.int64)
    org = nodes.author_code.to_numpy(np.int64)[nodes.root.to_numpy()]
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    E = E[E.rel_type.isin(MAIN)]
    et = birth[E.child_idx.to_numpy()]
    ep = E.parent_idx.to_numpy(np.int64)
    k_par = np.sort(ep * BIG + et)
    k_org = np.sort(org[ep] * BIG + et)
    c = rows.cand_id.to_numpy(np.int64)
    t = t_of_choice
    d_own = count_before(k_par, c, t)
    d_org = count_before(k_org, org[c], t)
    F = {"org_eco": np.log1p(np.maximum(d_org - d_own, 0)).astype(np.float32)}
    pb = nodes.params_b.to_numpy(np.float64)
    size = np.where(np.isfinite(pb) & (pb > 0), np.round(np.log2(np.where(pb > 0, pb, 1))), -99).astype(np.int64)
    grp = pd.factorize(pd.MultiIndex.from_arrays([nodes.author_code.to_numpy(), nodes.modality_code.to_numpy(),
                                                  nodes.is_root.to_numpy(), size]))[0].astype(np.int64)
    k_grp = np.sort(grp * BIG + birth)
    n_newer = np.maximum(count_before(k_grp, grp[c], t) - count_before(k_grp, grp[c], birth[c] + 1), 0)
    F["superseded"] = (n_newer > 0).astype(np.float32)
    F["log_newer"] = np.log1p(n_newer).astype(np.float32)
    # controls (p2 round 2): a prolific uploader vs replacement by a substitute
    au = nodes.author_code.to_numpy(np.int64)
    k_au = np.sort(au * BIG + birth)
    n_any = np.maximum(count_before(k_au, au[c], t) - count_before(k_au, au[c], birth[c] + 1), 0)
    F["log_newer_other"] = np.log1p(np.maximum(n_any - n_newer, 0)).astype(np.float32)
    F["log_up_before"] = np.log1p(count_before(k_au, au[c], birth[c])).astype(np.float32)
    return F


BRAKES = {"org_eco": ["org_eco"], "succession": ["superseded", "log_newer"], "both": ["org_eco", "superseded", "log_newer"]}
# round 2: succession with the placebo (newer models of a different type by the same uploader) and prolificness
BRAKES2 = {"succession": ["superseded", "log_newer"],
           "succession + controls": ["superseded", "log_newer", "log_newer_other", "log_up_before"]}


def part_p2(fracs, seeds, brakes=None, table="v2m_brake_fit"):
    brakes = brakes or BRAKES
    from hfpa.features.groups import GROUPS5
    from hfpa.models.clogit import CLogit, to_buckets, uniform_ll
    from v2_explore import subsample
    from hfpa.v2data import drop_constant
    rows, ch = load_rows(SPEC, "planA")
    t = pd.read_parquet(CDIR / "decisions.parquet", columns=["choice_id", "t"]).set_index("choice_id").t
    B = brake_features(rows, t.reindex(rows.choice_id).to_numpy(np.int64))
    for k, v in B.items():
        rows[k] = v
    ch_r = rows[rows.chosen == 1]
    print({k: (float(rows[k].mean()), float(ch_r[k].mean())) for k in B}, flush=True)
    base = sum(GROUPS5.values(), [])
    out = []
    for f in fracs:
        for s in (seeds if f < 1 else [0]):
            R = subsample(rows, ch, f, s)
            tr, ho = split(R, ch)
            for name, extra in [("base", [])] + list(brakes.items()):
                feats = drop_constant(tr, base + extra)
                D, Dh = to_buckets(tr, feats), to_buckets(ho, feats)
                m = CLogit(len(feats))
                res = m.fit(D, cluster=ch.author.reindex(D.choice_ids).to_numpy())
                hold = float(m.ll_per_choice(Dh).mean() - uniform_ll(Dh).mean())
                rec = dict(frac=f, seed=s, model=name, n=D.n, ll=res["ll"], hold_gain=hold)
                for x in ["org_eco", "superseded", "log_newer", "log_newer_other", "log_up_before", "log_deg", "lrec30"]:
                    if x in feats:
                        i = feats.index(x)
                        rec[f"b_{x}"], rec[f"se_{x}"] = float(res["coef"][i]), float(res["se_cl"][i])
                out.append(rec)
                print(rec, flush=True)
                del D, Dh
            base_ll = next(r["ll"] for r in out if r["frac"] == f and r["seed"] == s and r["model"] == "base")
            for r in out:
                if r["frac"] == f and r["seed"] == s:
                    r["LR"] = 2 * (r["ll"] - base_ll)
    write(pd.DataFrame(out), table, "Phase 2 (small scale): brake features added to the five-group model, "
          "author subsamples", "org_eco = log(1 + earlier adoptions of other models of the same root organization); "
          "superseded / log_newer = the uploader released a newer model of the same modality, root status and size "
          "class. LR against the base model on the same subsample; hold_gain = holdout mean log-likelihood over "
          "uniform (nats per choice).")


# ---------------------------------------------------------------- Phase 1: network-level Shapley
P5_ORDER = ["pop", "qual", "match", "self", "habit"]


def shapley_from_values(v, n=5):
    """v: {bitmask int: value}; bit i (from the left of the 5-char string) = player i. Returns list of values."""
    from math import factorial
    phi = np.zeros(n)
    for S in range(2 ** n):
        for i in range(n):
            bit = 1 << (n - 1 - i)
            if S & bit:
                continue
            k = bin(S).count("1")
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            phi[i] += w * (v[S | bit] - v[S])
    return phi


def part_s5():
    from v2_networks import metrics
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["root", "author"])
    org = pd.factorize(nodes.author)[0][nodes.root.to_numpy()]
    ch = pd.read_parquet(CDIR / "decisions.parquet", columns=["choice_id", "t"]).set_index("choice_id").t
    rec = []
    for f in sorted(glob.glob(str(OUTD / "mech" / "S5_*__*__*.parquet"))):
        net, until, seed = os.path.basename(f)[:-8].split("__")
        r = pd.read_parquet(f)
        t = ch.loc[r.choice_id].to_numpy()
        m = metrics(r.sim_parent.to_numpy(), t)
        mo = metrics(org[r.sim_parent.to_numpy()], t)
        real_top = set(r.real_parent.value_counts().index[:10])
        rec.append(dict(net=net, bits=net[3:], until=until, seed=int(seed), n=len(r), top10=m["top10"],
                        eff=m["eff_parents"], conc=-np.log(m["eff_parents"]), gini=m["gini"], turnover=m["turnover"],
                        org_top10=mo["top10"], org_turnover=mo["turnover"],
                        overlap_real_top10=len(set(r.sim_parent.value_counts().index[:10]) & real_top) / 10))
        if net == "S5_00000" and seed == "0":
            mr = metrics(r.real_parent.to_numpy(), t)
            mro = metrics(org[r.real_parent.to_numpy()], t)
            rec.append(dict(net="REAL", bits="REAL", until=until, seed=0, n=len(r), top10=mr["top10"],
                            eff=mr["eff_parents"], conc=-np.log(mr["eff_parents"]), gini=mr["gini"],
                            turnover=mr["turnover"], org_top10=mro["top10"], org_turnover=mro["turnover"],
                            overlap_real_top10=1.0))
    N = pd.DataFrame(rec)
    write(N.sort_values(["until", "bits", "seed"]), "v2m_s5_nets", "Phase 1: networks grown by the subset models "
          "(five-group model re-estimated on each subset of the groups)")
    L = pd.read_csv(T / "v2s_P5_shapley.csv")
    L = L[(L.variant == "no FE") & (L.partition == "groups") & (L.measure == "in-sample")]
    lab = {"pop": "popularity", "qual": "general quality", "match": "demand match", "self": "self-supply", "habit": "habit"}
    out = []
    for until, G in N[N.net != "REAL"].groupby("until"):
        M = G.groupby("bits")[["conc", "top10", "org_top10", "overlap_real_top10"]].mean()
        if len(M) < 32:
            print(f"{until}: only {len(M)} subsets", flush=True)
            continue
        for metric in ["conc", "top10", "org_top10", "overlap_real_top10"]:
            v = {int(b, 2): M.loc[b, metric] for b in M.index}
            phi = shapley_from_values(v)
            tot = v[31] - v[0]
            for g, x in zip(P5_ORDER, phi):
                out.append(dict(until=until, metric=metric, group=lab[g], value=x, share=x / tot if tot else np.nan,
                                total=tot, likelihood_share=float(L.set_index("group").share.get(lab[g], np.nan)),
                                seeds=int(G.seed.nunique())))
    write(pd.DataFrame(out), "v2m_s5_shapley", "Phase 1: network-level Shapley - each group's contribution to the "
          "concentration of the grown network, next to its likelihood share",
          "conc = -log(effective number of parents); value of a subset = metric of its network minus that of the "
          "uniform network (empty subset). Shares can be negative (a dispersing group). Mean over seeds.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["p0", "p2", "p2b", "s5"])
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.1, 0.25, 0.5])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "p0":
        G = grown()
        part_networks(G)
        part_rerun(G)
        part_levels(G)
        part_corr()
    if a.part == "s5":
        part_s5()
    if a.part == "p2":
        part_p2(a.fracs, a.seeds)
    if a.part == "p2b":
        part_p2(a.fracs, a.seeds, BRAKES2, "v2m_brake_fit2")
