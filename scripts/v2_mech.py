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


class SuccIndex:
    """Succession features from node metadata only (shared by the estimation here and the growth simulator):
    n_newer(l, t) = models of the candidate's uploader born in (birth_l, t) with the same modality, root status and
    size class (round(log2 params_b); unknown = its own class)."""
    def __init__(self, nodes=None):
        if nodes is None:
            nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                                    columns=["birth", "author_code", "is_root", "modality_code", "params_b"])
        self.birth = nodes.birth.to_numpy(np.int64)
        pb = nodes.params_b.to_numpy(np.float64)
        size = np.where(np.isfinite(pb) & (pb > 0), np.round(np.log2(np.where(pb > 0, pb, 1))), -99).astype(np.int64)
        self.grp = pd.factorize(pd.MultiIndex.from_arrays([nodes.author_code.to_numpy(), nodes.modality_code.to_numpy(),
                                                           nodes.is_root.to_numpy(), size]))[0].astype(np.int64)
        self.k_grp = np.sort(self.grp * BIG + self.birth)

    def n_newer(self, c, t):
        g = self.grp[c]
        return np.maximum(count_before(self.k_grp, g, t) - count_before(self.k_grp, g, self.birth[c] + 1), 0)

    def matrix(self, c, t):
        """columns SUCC = (superseded, log_newer)."""
        n = self.n_newer(np.asarray(c, np.int64), np.asarray(t, np.int64))
        return np.column_stack([(n > 0).astype(np.float64), np.log1p(n)])


SUCC = ["superseded", "log_newer"]


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
    n_newer = SuccIndex(nodes).n_newer(c, t)
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


def part_succ_full():
    """A, step 1: full-data fits with the succession features (five groups + succession; popularity + succession),
    coefficients saved for the growth simulator (v2_grow.py SUCC_FULL / SUCC_POP) -> v2_mech/succ_fit.pkl,
    v2m_succ_full."""
    from hfpa.features.groups import GROUPS5
    from hfpa.models.clogit import CLogit, to_buckets, uniform_ll
    from hfpa.v2data import drop_constant
    rows, ch = load_rows(SPEC, "planA")
    t = pd.read_parquet(CDIR / "decisions.parquet", columns=["choice_id", "t"]).set_index("choice_id").t
    X = SuccIndex().matrix(rows.cand_id.to_numpy(), t.reindex(rows.choice_id).to_numpy(np.int64))
    for i, c in enumerate(SUCC):
        rows[c] = X[:, i].astype(np.float32)
    tr, ho = split(rows, ch)
    del rows
    P = pickle.load(open(CDIR / "v2_samples" / "P5" / "fit.pkl", "rb"))
    out, save = [], {}
    for name, base_key, groups in (("SUCC_FULL", frozenset(GROUPS5), list(GROUPS5)), ("SUCC_POP", frozenset(["pop"]), ["pop"])):
        feats = drop_constant(tr, sum((GROUPS5[g] for g in groups), []) + SUCC)
        D, Dh = to_buckets(tr, feats), to_buckets(ho, feats)
        m = CLogit(len(feats))
        res = m.fit(D, cluster=ch.author.reindex(D.choice_ids).to_numpy())
        hold = float(m.ll_per_choice(Dh).mean())
        base = P["fits"][base_key]
        save[name] = dict(feats=feats, coef=np.asarray(res["coef"]), se_cl=np.asarray(res["se_cl"]), ll=res["ll"],
                          hold_ll=hold)
        rec = dict(model=name, n=D.n, ll=res["ll"], LR=2 * (res["ll"] - base["ll"]),
                   hold_gain=hold - float(uniform_ll(Dh).mean()), hold_gain_base=base["hold_ll"] - P["u0h"])
        for x in SUCC + ["log_deg", "lrec30", "has_deg", "task_hist"]:
            if x in feats:
                i = feats.index(x)
                rec[f"b_{x}"], rec[f"se_{x}"] = float(res["coef"][i]), float(res["se_cl"][i])
        out.append(rec)
        print(rec, flush=True)
        del D, Dh
    pickle.dump(save, open(OUT / "succ_fit.pkl", "wb"))
    write(pd.DataFrame(out), "v2m_succ_full", "A: full-data fits with the succession features (training first adoptions)",
          "LR against the same model without succession (P5 subset fits); hold_gain = holdout mean log-likelihood "
          "over uniform, nats per choice (base = without succession).")


# ---------------------------------------------------------------- three diagnostics of the feedback gap
HORIZONS = [("<= 2024-03", "2024-03"), ("<= 2024-09", "2024-09"), ("full", "9999-12")]


def part_support():
    """1. Support check: how far does feedback push the popularity state beyond the real data? Chosen parents and
    all candidates, log_deg and lrec30, simulated (v2_grow EST / SF, seed 0, full run with feedback) vs real; by
    horizon (cumulative windows) -> v2m_support."""
    cols = ["choice_id", "chosen", "log_deg", "lrec30"]
    real, _ = load_rows(SPEC, "planA", cols=["log_deg", "lrec30"])
    real = real[cols]
    mon = pd.read_parquet(CDIR / "decisions.parquet", columns=["choice_id", "t"]).set_index("choice_id").t
    mon = pd.to_datetime(mon, unit="s").dt.strftime("%Y-%m")
    S = {"REAL": real}
    for net in ("EST", "SF"):
        S[net] = pd.read_parquet(OUTD / f"{net}_0_rows.parquet", columns=cols)
    out = []
    for hlab, hmax in HORIZONS:
        R = S["REAL"][mon.reindex(S["REAL"].choice_id).to_numpy() <= hmax]
        rmax = {v: float(R[v].max()) for v in ("log_deg", "lrec30")}
        rq = {v: float(R.loc[R.chosen == 1, v].quantile(0.99)) for v in ("log_deg", "lrec30")}
        for net, X in S.items():
            X = X[mon.reindex(X.choice_id).to_numpy() <= hmax]
            C = X[X.chosen == 1]
            for v in ("log_deg", "lrec30"):
                q = C[v].quantile([0.5, 0.9, 0.99]).to_numpy()
                out.append(dict(horizon=hlab, network=net, var=v, choices=len(C), chosen_p50=q[0], chosen_p90=q[1],
                                chosen_p99=q[2], chosen_max=float(C[v].max()), cand_max=float(X[v].max()),
                                chosen_above_real_max=float((C[v] > rmax[v]).mean()),
                                chosen_above_real_p99=float((C[v] > rq[v]).mean()),
                                cand_above_real_max=float((X[v] > rmax[v]).mean())))
    write(pd.DataFrame(out), "v2m_support", "Diagnostic 1: popularity state of chosen parents and candidates, "
          "simulated with feedback (seed 0) vs real, cumulative horizons",
          "log_deg = log max(d, 1); lrec30 = log(1 + adoptions in the last 30 days). above_real_max / p99 = share "
          "of simulated chosen parents (or candidate rows) beyond the real maximum over all candidates / the real 99th "
          "percentile of chosen parents in the same window.")


# saturation: piecewise-linear (hinge) terms on top of log_deg and lrec30
KNOTS = {"log_deg": [np.log(10), np.log(100), np.log(1000)], "lrec30": [np.log1p(3), np.log1p(10), np.log1p(30), np.log1p(100)]}
HINGE = [f"{v}_h{i}" for v, ks in KNOTS.items() for i in range(len(ks))]


def hinge(log_deg, lrec30):
    """columns HINGE: max(0, x - knot)."""
    x = {"log_deg": np.asarray(log_deg, np.float64), "lrec30": np.asarray(lrec30, np.float64)}
    return np.column_stack([np.maximum(x[v] - k, 0) for v, ks in KNOTS.items() for k in ks])


def part_sat(fracs, seeds):
    """2. Saturation, estimation tier: five-group model with hinge terms on log_deg and lrec30 (author
    subsamples); segment slopes, LR, holdout gain -> v2m_sat_fit; coefficients of every fit -> v2_mech/sat_fit.pkl
    (the growth tier uses frac 0.5 seed 0, base vs spline)."""
    from hfpa.features.groups import GROUPS5
    from hfpa.models.clogit import CLogit, to_buckets, uniform_ll
    from v2_explore import subsample
    from hfpa.v2data import drop_constant
    rows, ch = load_rows(SPEC, "planA")
    H = hinge(rows.log_deg.to_numpy(), rows.lrec30.to_numpy())
    for i, c in enumerate(HINGE):
        rows[c] = H[:, i].astype(np.float32)
    ch_r = rows[rows.chosen == 1]
    print("chosen parents above each knot:", {c: float((ch_r[c] > 0).mean()) for c in HINGE}, flush=True)
    base = sum(GROUPS5.values(), [])
    out, save = [], {}
    for f in fracs:
        for s in (seeds if f < 1 else [0]):
            R = subsample(rows, ch, f, s)
            tr, ho = split(R, ch)
            res_b = None
            for name, feats0 in (("base", base), ("spline", base + HINGE)):
                feats = drop_constant(tr, feats0)
                D, Dh = to_buckets(tr, feats), to_buckets(ho, feats)
                m = CLogit(len(feats))
                res = m.fit(D, cluster=ch.author.reindex(D.choice_ids).to_numpy())
                hold = float(m.ll_per_choice(Dh).mean() - uniform_ll(Dh).mean())
                c = dict(zip(feats, res["coef"]))
                se = dict(zip(feats, res["se_cl"]))
                save[(f, s, name)] = dict(feats=feats, coef=np.asarray(res["coef"]))
                rec = dict(frac=f, seed=s, model=name, n=D.n, ll=res["ll"], hold_gain=hold)
                if name == "base":
                    res_b = res
                else:
                    rec["LR"] = 2 * (res["ll"] - res_b["ll"])
                for v, ks in KNOTS.items():  # slope in each segment = base slope + sum of hinge coefficients so far
                    sl = c.get(v, 0.0)
                    rec[f"{v}_slope_0"] = sl
                    for i in range(len(ks)):
                        sl += c.get(f"{v}_h{i}", 0.0)
                        rec[f"{v}_slope_{i + 1}"] = sl
                        if f"{v}_h{i}" in se:
                            rec[f"{v}_h{i}_se"] = float(se[f"{v}_h{i}"])
                out.append(rec)
                print(rec, flush=True)
                del D, Dh
    pickle.dump(save, open(OUT / "sat_fit.pkl", "wb"))
    write(pd.DataFrame(out), "v2m_sat_fit", "Diagnostic 2: saturation of the popularity response, five-group model "
          "with piecewise-linear log_deg and lrec30 (author subsamples)",
          "Knots: log_deg at d = 10, 100, 1000; lrec30 at 3, 10, 30, 100 recent adoptions. slope_i = utility slope in "
          "segment i (0 = below the first knot). LR = spline against base on the same subsample.")


def part_xi(fracs, seeds):
    """3. Persistence of the hub premium xi: five-group model + hub FE estimated separately on the first and the
    second half (in time) of the training choices of each author subsample; correlation of the FE across halves
    for hubs chosen at least MIN_CH times in both halves, raw and corrected for estimation noise -> v2m_xi,
    v2_mech/xi_<f>_<s>.parquet (per-hub estimates)."""
    from hfpa.features.groups import GROUPS5
    from hfpa.models.clogit import CLogit, to_buckets
    from v2_explore import subsample
    from hfpa.v2data import drop_constant
    MIN_CH, N_FE = 5, 500
    rows, ch = load_rows(SPEC, "planA")
    base = sum(GROUPS5.values(), [])
    out = []
    for f in fracs:
        for s in (seeds if f < 1 else [0]):
            R = subsample(rows, ch, f, s)
            tr, _ = split(R, ch)
            t = ch.t.reindex(tr.choice_id).to_numpy()
            cut = np.median(ch.t.reindex(np.unique(tr.choice_id)).to_numpy())
            est = {}
            for half, mask in (("first", t < cut), ("second", t >= cut)):
                X = tr[mask].reset_index(drop=True)
                feats = drop_constant(X, base)
                D = to_buckets(X, feats, fe_col="fe")
                m = CLogit(len(feats), n_fe=N_FE, l2_fe=1e-4)
                res = m.fit(D, cluster=None, se=True)
                nch = np.bincount(X.fe.to_numpy()[X.chosen.to_numpy() == 1], minlength=N_FE + 1)[1:]
                est[half] = pd.DataFrame({"hub": np.arange(1, N_FE + 1), "phi": res["coef"][len(feats):],
                                          "se": res["se"][len(feats):], "n_chosen": nch})
            E = est["first"].merge(est["second"], on="hub", suffixes=("_1", "_2"))
            E.to_parquet(OUT / f"xi_{f}_{s}.parquet", index=False)
            K = E[(E.n_chosen_1 >= MIN_CH) & (E.n_chosen_2 >= MIN_CH)]
            r = float(np.corrcoef(K.phi_1, K.phi_2)[0, 1]) if len(K) > 2 else np.nan
            rel = lambda p, e: max(1 - np.mean(e ** 2) / np.var(p), 1e-6)
            r1, r2 = rel(K.phi_1, K.se_1), rel(K.phi_2, K.se_2)
            out.append(dict(frac=f, seed=s, hubs_both=len(K), corr=r,
                            spearman=float(K[["phi_1", "phi_2"]].corr("spearman").iloc[0, 1]) if len(K) > 2 else np.nan,
                            reliability_1=r1, reliability_2=r2, corr_noise_corrected=r / np.sqrt(r1 * r2),
                            sd_phi_1=float(K.phi_1.std()), sd_phi_2=float(K.phi_2.std())))
            print(out[-1], flush=True)
    write(pd.DataFrame(out), "v2m_xi", "Diagnostic 3: persistence of the hub premium - FE estimated on the two time "
          "halves of the training choices (author subsamples)",
          f"Hubs chosen >= {MIN_CH} times in both halves. Noise-corrected correlation = raw / sqrt(rel_1 rel_2), "
          "rel = 1 - mean(se^2) / var(phi) (analytic SE).")


# ---------------------------------------------------------------- shape of the popularity response (kernel)
# step functions (sieve): degree d >= 1 (reference [1, 2); d = 0 is has_deg), recent adoptions r (reference 0)
DEG_FINE, DEG_COARSE = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 3000], [2, 10, 100, 1000]
REC_FINE, REC_COARSE = [1, 2, 3, 5, 10, 20, 50, 100, 300], [1, 5, 20, 100]


def counts(log_deg, lrec30):
    d = np.rint(np.exp(np.asarray(log_deg, np.float64)))  # log max(d, 1): d = 0 and 1 both give 1 (has_deg separates)
    r = np.rint(np.expm1(np.asarray(lrec30, np.float64)))
    return d, r


def step_basis(log_deg, lrec30, deg_edges, rec_edges):
    d, r = counts(log_deg, lrec30)
    cols, names = [], []
    for v, x, ed in (("d", d, deg_edges), ("r", r, rec_edges)):
        e = list(ed) + [np.inf]
        for lo, hi in zip(e[:-1], e[1:]):
            cols.append(((x >= lo) & (x < hi)).astype(np.float64))
            names.append(f"{v}_bin_{int(lo)}")
    return np.column_stack(cols), names


BASES = {"steps_fine": lambda a, b: step_basis(a, b, DEG_FINE, REC_FINE),
         "steps_coarse": lambda a, b: step_basis(a, b, DEG_COARSE, REC_COARSE),
         "hinge": lambda a, b: (hinge(a, b), HINGE)}


def boxcox(x, lam):
    x = np.asarray(x, np.float64)
    return np.log(x) if abs(lam) < 1e-9 else (x ** lam - 1) / lam


def fit_one(tr, ho, feats, ch):
    from hfpa.models.clogit import CLogit, to_buckets, uniform_ll
    D, Dh = to_buckets(tr, feats), to_buckets(ho, feats)
    m = CLogit(len(feats))
    res = m.fit(D, cluster=ch.author.reindex(D.choice_ids).to_numpy())
    hold = float(m.ll_per_choice(Dh).mean() - uniform_ll(Dh).mean())
    n = D.n
    del D, Dh
    return dict(ll=res["ll"], hold_gain=hold, n=n, coef=dict(zip(feats, res["coef"])), se=dict(zip(feats, res["se_cl"])))


def slopes(c):
    out = {}
    for v, ks in KNOTS.items():
        sl = c.get(v, 0.0)
        out[f"{v}_slope_0"] = sl
        for i in range(len(ks)):
            sl += c.get(f"{v}_h{i}", 0.0)
            out[f"{v}_slope_{i + 1}"] = sl
    return out


def rank_features(rows, t):
    """Rank of the candidate by degree within its modality at the start of the choice's month (1 = most adopted),
    from the real first-adoption-relevant edges: log(rank), top-10 and top-100 indicators."""
    from hfpa.features.groups import MAIN
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "modality_code"])
    birth, mod = nodes.birth.to_numpy(np.int64), nodes.modality_code.to_numpy(np.int64)
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    E = E[E.rel_type.isin(MAIN)]
    et, ep = birth[E.child_idx.to_numpy()], E.parent_idx.to_numpy(np.int64)
    mon = pd.to_datetime(t, unit="s").to_period("M")
    starts = {m: int(m.to_timestamp().timestamp()) for m in mon.unique()}
    c = rows.cand_id.to_numpy(np.int64)
    rank = np.full(len(rows), np.nan)
    M = mod.max() + 1
    for m, t0 in starts.items():
        sel = np.flatnonzero(mon == m)
        deg = np.bincount(ep[et < t0], minlength=len(birth))
        alive = np.flatnonzero((birth < t0) & (deg > 0))
        for mm in range(M):
            a = alive[mod[alive] == mm]
            if not len(a):
                continue
            srt = np.sort(deg[a])
            s2 = sel[mod[c[sel]] == mm]
            rank[s2] = len(srt) - np.searchsorted(srt, deg[c[s2]], "right") + 1
    rank = np.where(np.isnan(rank), 1e6, rank)
    used = (np.exp(rows.log_deg.to_numpy()) >= 1) & (rows.has_deg.to_numpy() > 0)
    return {"log_rank": np.where(used, np.log(rank), np.log(1e6)).astype(np.float32),
            "top10_mod": (used & (rank <= 10)).astype(np.float32),
            "top100_mod": (used & (rank <= 100)).astype(np.float32)}


AGE_KNOTS = [np.log1p(30), np.log1p(90), np.log1p(180), np.log1p(365)]


def part_shape(test, fracs, seeds):
    from hfpa.features.groups import GROUPS5
    from v2_explore import subsample
    from hfpa.v2data import drop_constant
    rows, ch = load_rows(SPEC, "planA")
    t_all = ch.t.reindex(rows.choice_id).to_numpy(np.int64)
    base = sum(GROUPS5.values(), [])
    H = hinge(rows.log_deg.to_numpy(), rows.lrec30.to_numpy())
    for i, c in enumerate(HINGE):
        rows[c] = H[:, i].astype(np.float32)
    models = {}
    if test == "np":
        models["linear"] = base
        for nm in ("steps_coarse", "steps_fine"):
            B, names = BASES[nm](rows.log_deg.to_numpy(), rows.lrec30.to_numpy())
            for i, c in enumerate(names):
                rows[c] = B[:, i].astype(np.float32)
            models[nm] = [f for f in base if f not in ("log_deg", "lrec30")] + names
        models["hinge"] = base + HINGE
    elif test == "bc":
        d, r = counts(rows.log_deg.to_numpy(), rows.lrec30.to_numpy())
        grid = [-0.6, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
        for lam in grid:
            rows[f"bcd_{lam}"] = boxcox(np.maximum(d, 1), lam).astype(np.float32)
            rows[f"bcr_{lam}"] = boxcox(1 + r, lam).astype(np.float32)
        for lam in grid:  # one-dimensional profiles, the other variable at its log form
            models[f"deg lambda={lam}"] = [f if f != "log_deg" else f"bcd_{lam}" for f in base]
            models[f"rec lambda={lam}"] = [f if f != "lrec30" else f"bcr_{lam}" for f in base]
    elif test == "rank":
        Rf = rank_features(rows, t_all)
        for k, v in Rf.items():
            rows[k] = v
        models["hinge"] = base + HINGE
        models["hinge + rank"] = base + HINGE + list(Rf)
        models["linear + rank"] = base + list(Rf)
    elif test == "conf":
        Xs = SuccIndex().matrix(rows.cand_id.to_numpy(), t_all)
        for i, c in enumerate(SUCC):
            rows[c] = Xs[:, i].astype(np.float32)
        la = rows.log_age.to_numpy(np.float64)
        ages = [f"log_age_h{i}" for i in range(len(AGE_KNOTS))]
        for i, k in enumerate(AGE_KNOTS):
            rows[ages[i]] = np.maximum(la - k, 0).astype(np.float32)
        models["hinge"] = base + HINGE
        models["hinge + age shape"] = base + HINGE + ages
        models["hinge + age shape + succession"] = base + HINGE + ages + SUCC
    elif test == "type":
        at = ch.author.map(__import__("v2_samples").author_types())
        grp = at.map({"indiv_light": "novice individuals", "indiv_mid": "experienced individuals",
                      "indiv_heavy": "experienced individuals", "org_other": "organizations",
                      "org_redistributor": "organizations", "org_major": "organizations"})
        for g in ("novice individuals", "experienced individuals", "organizations"):
            models[g] = (base + HINGE, set(ch.index[(grp == g).to_numpy()]))
    out, save = [], {}
    for f in fracs:
        for s in (seeds if f < 1 else [0]):
            R = subsample(rows, ch, f, s)
            tr0, ho0 = split(R, ch)
            ref = None
            for name, spec in models.items():
                feats0, keep = (spec if isinstance(spec, tuple) else (spec, None))
                tr, ho = (tr0, ho0) if keep is None else (tr0[tr0.choice_id.isin(keep)], ho0[ho0.choice_id.isin(keep)])
                feats = drop_constant(tr, feats0)
                r = fit_one(tr.reset_index(drop=True), ho.reset_index(drop=True), feats, ch)
                save[(test, name, f, s)] = dict(feats=feats, coef=np.array([r["coef"][x] for x in feats]))
                rec = dict(test=test, model=name, frac=f, seed=s, n=r["n"], ll=r["ll"], hold_gain=r["hold_gain"],
                           k=len(feats))
                if keep is None:
                    ref = ref if ref is not None else r["ll"]
                    rec["LR_vs_first"] = 2 * (r["ll"] - ref)
                rec.update(slopes(r["coef"]))
                for x, v in r["coef"].items():
                    if x.startswith(("d_bin", "r_bin", "log_rank", "top10", "top100", "superseded", "log_newer",
                                     "log_age", "has_deg")):
                        rec[f"b_{x}"], rec[f"se_{x}"] = float(v), float(r["se"][x])
                out.append(rec)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                       if not k.startswith("se_")}, flush=True)
    pickle.dump(save, open(OUT / f"shape_fit_{test}.pkl", "wb"))  # one file per test (jobs run in parallel)
    write(pd.DataFrame(out), f"v2m_shape_{test}", f"Shape of the popularity response, test '{test}' (author "
          "subsamples)", "slope_i = utility slope of the hinge model in segment i (log_deg knots d = 10, 100, 1000; "
          "lrec30 knots 3, 10, 30, 100). Step models: b_d_bin_<lo> / b_r_bin_<lo> = utility of the degree / recent bin "
          "relative to d = 1 / r = 0. LR_vs_first = against the first model of the test on the same subsample.")


# ---------------------------------------------------------------- the top of the kernel: superstar bonus or xi?
DEG_TOP = DEG_FINE + [10000]
KNOTS_TOP = KNOTS["log_deg"] + [np.log(3000), np.log(10000)]
HINGE_TOP = [f"log_deg_t{i}" for i in range(len(KNOTS_TOP))] + [f"lrec30_h{i}" for i in range(len(KNOTS["lrec30"]))]


def fit_fe(tr, ho, feats, ch, fe):
    from hfpa.models.clogit import CLogit, to_buckets, uniform_ll
    D = to_buckets(tr, feats, fe_col="fe" if fe else None)
    Dh = to_buckets(ho, feats, fe_col="fe" if fe else None)
    m = CLogit(len(feats), n_fe=500 if fe else 0, l2_fe=1e-4 if fe else 0.0)
    res = m.fit(D, cluster=None if fe else ch.author.reindex(D.choice_ids).to_numpy())
    hold = float(m.ll_per_choice(Dh).mean() - uniform_ll(Dh).mean())
    se = res["se"] if fe else res["se_cl"]
    n = D.n
    del D, Dh
    return dict(ll=res["ll"], hold_gain=hold, n=n, coef=dict(zip(feats, res["coef"][: len(feats)])),
                se=dict(zip(feats, se[: len(feats)])))


def part_top(fracs, seeds):
    from hfpa.features.groups import GROUPS5
    from v2_explore import subsample
    from hfpa.v2data import drop_constant
    rows, ch = load_rows(SPEC, "planA")
    base = sum(GROUPS5.values(), [])
    d, r = counts(rows.log_deg.to_numpy(), rows.lrec30.to_numpy())
    B, bnames = step_basis(rows.log_deg.to_numpy(), rows.lrec30.to_numpy(), DEG_TOP, REC_FINE)
    for i, c in enumerate(bnames):
        rows[c] = B[:, i].astype(np.float32)
    ld = rows.log_deg.to_numpy(np.float64)
    for i, k in enumerate(KNOTS_TOP):
        rows[f"log_deg_t{i}"] = np.maximum(ld - k, 0).astype(np.float32)
    H = hinge(rows.log_deg.to_numpy(), rows.lrec30.to_numpy())
    for i, c in enumerate(HINGE):
        if c.startswith("lrec30"):
            rows[c] = H[:, i].astype(np.float32)
    # descriptive: who sits in the top bins (training choices)
    tr_all, _ = split(rows, ch)
    C = tr_all[tr_all.chosen == 1]
    dc = np.rint(np.exp(C.log_deg.to_numpy()))
    desc = []
    for lo, hi in ((1000, 3000), (3000, 10000), (10000, np.inf)):
        k = (dc >= lo) & (dc < hi) & (C.has_deg.to_numpy() > 0)
        desc.append(dict(bin=f"[{lo}, {hi})", chosen=int(k.sum()), share_of_choices=float(k.mean()),
                         distinct_parents=int(C.cand_id[k].nunique()),
                         top5_parents_share_of_bin=float(C.cand_id[k].value_counts().iloc[:5].sum() / max(k.sum(), 1))))
    write(pd.DataFrame(desc), "v2m_top_desc", "Top of the kernel: chosen parents in the highest degree bins "
          "(training first adoptions)")
    print(desc, flush=True)
    steps = [f for f in base if f not in ("log_deg", "lrec30")] + bnames
    hin = base + HINGE_TOP
    out = []
    for f in fracs:
        for s in (seeds if f < 1 else [0]):
            R = subsample(rows, ch, f, s)
            tr, ho = split(R, ch)
            top10 = tr[tr.chosen == 1].cand_id.value_counts().index[:10]
            drop = lambda X: X[~X.cand_id.isin(top10)].pipe(
                lambda Y: Y[Y.groupby("choice_id").chosen.transform("max") == 1]).reset_index(drop=True)
            for name, feats0, fe, dt in (("steps", steps, False, False), ("steps + hub FE", steps, True, False),
                                         ("hinge", hin, False, False), ("hinge + hub FE", hin, True, False),
                                         ("hinge, top-10 hubs dropped", hin, False, True)):
                a, b = (drop(tr), drop(ho)) if dt else (tr, ho)
                feats = drop_constant(a, feats0)
                res = fit_fe(a, b, feats, ch, fe)
                c, se = res["coef"], res["se"]
                rec = dict(model=name, frac=f, seed=s, n=res["n"], ll=res["ll"], hold_gain=res["hold_gain"])
                if name.startswith("steps"):
                    for x in bnames:
                        if x.startswith("d_bin") and int(x.split("_")[-1]) >= 500:
                            rec[f"b_{x}"], rec[f"se_{x}"] = float(c.get(x, np.nan)), float(se.get(x, np.nan))
                else:
                    sl = c.get("log_deg", 0.0)
                    edges = ["<10", "10-100", "100-1000", "1000-3000", "3000-10000", ">10000"]
                    rec[f"deg_slope_{edges[0]}"] = sl
                    for i in range(len(KNOTS_TOP)):
                        sl += c.get(f"log_deg_t{i}", 0.0)
                        rec[f"deg_slope_{edges[i + 1]}"] = sl
                    rs = c.get("lrec30", 0.0) + sum(c.get(f"lrec30_h{i}", 0.0) for i in range(len(KNOTS["lrec30"])))
                    rec["rec_slope_top"] = rs
                    rec["kappa_top(>10000)"] = sl + rs
                out.append(rec)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if not k.startswith("se_")},
                      flush=True)
    write(pd.DataFrame(out), "v2m_top", "Top of the kernel (author subsamples): highest degree bins and hinge "
          "slopes, without / with hub FE, and with the 10 most-chosen hubs dropped",
          "Steps: b_d_bin_<lo> = utility relative to d = 1 (d_bin_10000 = d >= 10000). Hinge knots for log_deg at "
          "d = 10, 100, 1000, 3000, 10000. kappa_top = degree slope above 10000 + momentum slope above 100 recent "
          "adoptions. Hub FE: 500 ex-ante hubs, ridge 1e-4, analytic SE. Dropping hubs removes them from every "
          "candidate set (valid under IIA) and drops the choices of them.")


# ---------------------------------------------------------------- kernel, full data (main specification)
KERNEL_MODELS = {"K_LIN": ("linear", False), "K_HINGE": ("hinge", False), "K_HINGE_FE": ("hinge", True)}


def part_kernel_full():
    """Full-data fits of the three kernel specifications used by the growth simulator (v2_grow.py K_*): linear
    (log-linear kernel, the five-group model), hinge (knots log_deg d = 10, 100, 1000; lrec30 r = 3, 10, 30, 100)
    and hinge + hub FE -> v2_mech/kernel_fit.pkl, v2m_kernel_full."""
    from hfpa.features.groups import GROUPS5
    from hfpa.models.clogit import CLogit, to_buckets, uniform_ll
    from hfpa.v2data import drop_constant
    rows, ch = load_rows(SPEC, "planA")
    H = hinge(rows.log_deg.to_numpy(), rows.lrec30.to_numpy())
    for i, c in enumerate(HINGE):
        rows[c] = H[:, i].astype(np.float32)
    tr, ho = split(rows, ch)
    del rows
    base = sum(GROUPS5.values(), [])
    out, save = [], {}
    for net, (form, fe) in KERNEL_MODELS.items():
        feats = drop_constant(tr, base + (HINGE if form == "hinge" else []))
        D = to_buckets(tr, feats, fe_col="fe" if fe else None)
        Dh = to_buckets(ho, feats, fe_col="fe" if fe else None)
        m = CLogit(len(feats), n_fe=500 if fe else 0, l2_fe=1e-4 if fe else 0.0)
        res = m.fit(D, cluster=ch.author.reindex(D.choice_ids).to_numpy())
        hold = float(m.ll_per_choice(Dh).mean() - uniform_ll(Dh).mean())
        coef = np.asarray(res["coef"])
        save[net] = dict(feats=feats, coef=coef[: len(feats)], fe=np.r_[0.0, coef[len(feats):]] if fe else np.zeros(501))
        c, se = dict(zip(feats, coef)), dict(zip(feats, res["se_cl"]))
        rec = dict(model=net, form=form, hub_fe=fe, n=D.n, ll=res["ll"], hold_gain=hold, k=len(coef))
        rec.update(slopes(c))
        for x in ["has_deg", "log_deg", "lrec30"] + HINGE:
            if x in se:
                rec[f"se_{x}"] = float(se[x])
        out.append(rec)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if not k.startswith("se_")}, flush=True)
        del D, Dh
    pickle.dump(save, open(OUT / "kernel_fit.pkl", "wb"))
    write(pd.DataFrame(out), "v2m_kernel_full", "Kernel specifications, full data (training first adoptions): "
          "linear vs hinge vs hinge + hub FE", "slope_i = utility slope in segment i (log_deg: d < 10, 10-100, "
          "100-1000, > 1000; lrec30: r < 3, 3-10, 10-30, 30-100, > 100). SE clustered by author.")


def part_mechnets():
    """Metrics of every network in grow/mech/ (<net>__<until>__<seed>.parquet) and of the real first adoptions of
    the same choices, per horizon -> v2m_mech_nets (one row per network x horizon x seed)."""
    from v2_networks import metrics
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["root", "author"])
    org = pd.factorize(nodes.author)[0][nodes.root.to_numpy()]
    ch = pd.read_parquet(CDIR / "decisions.parquet", columns=["choice_id", "t"]).set_index("choice_id").t
    rec, done_real = [], set()
    for f in sorted(glob.glob(str(OUTD / "mech" / "*__*__*.parquet"))):
        net, until, seed = os.path.basename(f)[:-8].split("__")
        r = pd.read_parquet(f)
        t = ch.loc[r.choice_id].to_numpy()
        for lab, par in [(net, r.sim_parent.to_numpy())] + ([("REAL", r.real_parent.to_numpy())] if until not in done_real else []):
            m, mo = metrics(par, t), metrics(org[par], t)
            real_top = set(r.real_parent.value_counts().index[:10])
            rec.append(dict(network=lab, until=until, seed=int(seed) if lab != "REAL" else 0, n=len(r), top10=m["top10"],
                            eff_parents=m["eff_parents"], gini=m["gini"], tail_alpha=m["tail_alpha"],
                            turnover=m["turnover"], n_parents=m["n_parents"], org_top10=mo["top10"],
                            org_turnover=mo["turnover"],
                            overlap_real_top10=len(set(pd.Series(par).value_counts().index[:10]) & real_top) / 10))
        done_real.add(until)
    write(pd.DataFrame(rec), "v2m_mech_nets", "All growth networks of the mechanism round (grow/mech) with the real "
          "network of the same choices, by horizon",
          "SAT_* / NP_*: 50% author-subsample fits; S5_*: five-group subset fits on the full data. Metrics as in "
          "v2_networks.metrics; org = author of the lineage root.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["p0", "p2", "p2b", "s5", "succ_full", "support", "sat", "xi", "shape", "top", "kernel_full", "mechnets"])
    ap.add_argument("--test", choices=["np", "bc", "rank", "conf", "type"])
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
    if a.part == "succ_full":
        part_succ_full()
    if a.part == "support":
        part_support()
    if a.part == "sat":
        part_sat(a.fracs, a.seeds)
    if a.part == "xi":
        part_xi(a.fracs, a.seeds)
    if a.part == "mechnets":
        part_mechnets()
    if a.part == "kernel_full":
        part_kernel_full()
    if a.part == "top":
        part_top(a.fracs, a.seeds)
    if a.part == "shape":
        part_shape(a.test, a.fracs, a.seeds)
    if a.part == "p2":
        part_p2(a.fracs, a.seeds)
    if a.part == "p2b":
        part_p2(a.fracs, a.seeds, BRAKES2, "v2m_brake_fit2")
