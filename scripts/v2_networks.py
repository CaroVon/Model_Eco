"""v2 stage 4b: three networks -- scale-free growth (popularity only), quality-fit growth (quality + demand match only)
and the real network -- plus the E.2 concentration counterfactual.

The real sequence of plan-A first decisions (who adopts a parent for the first time, when, in which modality) is kept;
only the parent is redrawn. The network compared is the set of these first-adoption edges (133k), as in the P16
ecosystem baseline.

  SF       full-market simulation with feedback. Every alive node of the modality is a candidate; utility = the
           popularity coefficients of the v2 plan-A model that can be evaluated for any node in a simulation:
           has_deg, log_deg, lrec30 (simulated state: pre-T0 real degree + simulated first adoptions), log_age, and the
           static node terms major_org, legacy, lib_*, redistributor, bench_na. Chooser-specific popularity terms
           (lco2, ds_couse, size_na, txt_na) and lquant are left out. Weights use the state at the start of each day.
           Zero-degree nodes enter through (static profile x birth day) cells. Also BA: weight d + 1.
  SF-SIR   popularity group only, no feedback: importance resampling on the real PPS candidate set D_k,
           P(j) ∝ exp(V_pop(j) - log pi_j) (same estimator as QF, separates feedback from the utility itself).
  QF       quality + demand match only: V = beta_qual bench_z + beta_match . match + q_ext(j), q_ext = the part of the
           ex-ante FE explained by external quality (v2_bounds.py --ext; 0 for non-FE nodes). SIR on D_k (no network
           feedback; author histories are the real ones).
  FULL     full plan-A model (with FE), SIR on D_k: model-implied real network (fit check).
  REAL     observed first adoptions.
Metrics (5 seeds each): Top-10 share, effective number of parents exp(H), Gini over parents with >= 1 adoption,
tail exponent (Hill, adoptions >= 10), quarterly top-10 leader turnover.
Share recovery: 4-group Shapley re-estimated on the SIR choices (SF-SIR, QF, FULL; real candidate sets, chosen
replaced). E.2: popularity coefficients x {0.5, 0.75, 1} in the full model, expected adoption counts by SIR
probabilities (no draw), Top-10 share and effective number of parents.

usage: sbatch slurm/gpu.sbatch scripts/v2_networks.py --part sir      (SIR networks, E.2, share recovery)
       sbatch slurm/cpu.sbatch scripts/v2_networks.py --part sf       (SF and BA simulations)
       python scripts/v2_networks.py --part collect
"""
import argparse
import glob
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from hfpa import paths
from hfpa.features.groups import GROUPS, LIB, MAIN, NAMES
from hfpa.models.clogit import to_buckets
from hfpa.v2data import TABLES, cdir, load_rows, shapley_fits, shares, split, write

SPEC = "v2_pps"
DAY = 86400
SEEDS = range(5)
OUTD = paths.PROCESSED / "choices" / SPEC / "v2_est" / "networks"
T0 = time.time()
STATIC = ["major_org", "legacy", "redistributor", "bench_na"] + [f"lib_{x}" for x in LIB]


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


# ---------------------------------------------------------------- metrics
def metrics(parents, t):
    """parents: adopted parent per event; t: event times."""
    c = pd.Series(parents).value_counts()
    p = c.to_numpy() / c.sum()
    top10 = p[:10].sum()
    eff = float(np.exp(-(p * np.log(p)).sum()))
    x = np.sort(c.to_numpy())
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))
    tail = x[x >= 10]
    hill = float(1 + len(tail) / np.log(tail / 9.5).sum()) if len(tail) > 10 else np.nan  # CCDF exponent + 1
    q = pd.to_datetime(t, unit="s").to_period("Q")
    D = pd.DataFrame({"q": q, "p": parents})
    tops = D.groupby("q").p.agg(lambda s: set(s.value_counts().index[:10]))
    turn = [1 - len(a & b) / 10 for a, b in zip(tops.iloc[:-1], tops.iloc[1:])]
    return dict(top10=top10, eff_parents=eff, gini=gini, tail_alpha=hill, turnover=float(np.mean(turn)),
                n_parents=n)


# ---------------------------------------------------------------- SIR networks
def model_coefs():
    C = pickle.load(open(cdir(SPEC) / "v2_est" / "clogit.pkl", "rb"))
    feats = C["feats"]
    return feats, C["nofe"]["coef"], C["fe"]["coef"]


def q_ext():
    f = cdir(SPEC) / "v2_est" / "q_ext.npy"  # written by v2_bounds.py --ext (FE id -> external-quality part)
    return np.load(f) if f.exists() else None


def sir_probs(rows, u):
    """Within-choice probabilities exp(u - log q) / sum (rows sorted by choice_id)."""
    v = u - rows.log_q.to_numpy()
    cid = rows.choice_id.to_numpy()
    st = np.r_[0, np.flatnonzero(np.diff(cid)) + 1]
    mx = np.maximum.reduceat(v, st)
    e = np.exp(v - np.repeat(mx, np.diff(np.r_[st, len(v)])))
    s = np.add.reduceat(e, st)
    return e / np.repeat(s, np.diff(np.r_[st, len(v)])), st


def draw(rows, pr, st, rng):
    """One draw per choice: row positions."""
    ends = np.r_[st[1:], len(pr)]
    cum = np.cumsum(pr)
    base = np.r_[0.0, cum[st[1:] - 1]] if len(st) > 1 else np.zeros(1)
    u = base + rng.random(len(st)) * (cum[ends - 1] - base)
    pos = np.searchsorted(cum, u, "right")
    return np.minimum(pos, ends - 1)


def part_sir():
    rows, ch = load_rows(SPEC, "planA")
    feats, b, bfe = model_coefs()
    X = rows[feats].to_numpy(np.float64)
    fe = rows.fe.to_numpy()
    phi = np.r_[0.0, bfe[len(feats):]]
    qx = q_ext()
    qx = np.zeros(len(phi)) if qx is None else np.r_[0.0, qx]
    gi = {g: [i for i, f in enumerate(feats) if f in GROUPS[g]] for g in GROUPS}
    idx = lambda g, coef: X[:, gi[g]] @ coef[gi[g]]
    U = {"SF-SIR": idx("pop", b),
         "QF": idx("qual", b) + idx("match", b) + qx[fe],
         "FULL": X @ bfe[: len(feats)] + phi[fe]}
    t = ch.t.reindex(rows.choice_id).to_numpy()
    cid_t = ch.t.loc[np.unique(rows.choice_id)].to_numpy()
    real = rows.cand_id.to_numpy()[rows.chosen.to_numpy() == 1]
    res = [dict(network="REAL", seed=0, **metrics(real, cid_t))]
    OUTD.mkdir(parents=True, exist_ok=True)
    draws = {}
    for name, u in U.items():
        pr, st = sir_probs(rows, u)
        for s in SEEDS:
            pos = draw(rows, pr, st, np.random.default_rng(100 + s))
            res.append(dict(network=name, seed=s, **metrics(rows.cand_id.to_numpy()[pos], cid_t)))
            if s == 0:
                draws[name] = pos
        log(f"{name}: {res[-1]}")
    pd.DataFrame(res).to_csv(OUTD / "sir_metrics.csv", index=False)
    # E.2: popularity coefficients scaled, expected adoption counts
    e2 = []
    for k in (0.0, 0.5, 0.75, 1.0):
        bb = bfe.copy()
        bb[gi["pop"]] *= k
        pr, st = sir_probs(rows, X @ bb[: len(feats)] + phi[fe])
        cnt = pd.Series(pr).groupby(rows.cand_id.to_numpy()).sum().sort_values(ascending=False)
        p = cnt.to_numpy() / cnt.sum()
        p = p[p > 0]
        e2.append(dict(pop_scale=k, top10=p[:10].sum(), eff_parents=float(np.exp(-(p * np.log(p)).sum()))))
        log(f"E.2 scale {k}: {e2[-1]}")
    write(pd.DataFrame(e2), "v2_concentration_cf", "E.2 concentration when the popularity coefficients are scaled "
          "(full plan-A model with FE; expected adoption counts from SIR probabilities on the sampled sets, no feedback)")
    # share recovery on SIR choices (training part)
    rec = []
    tr_mask = ~ch.holdout.reindex(rows.choice_id).to_numpy()
    for name, pos in draws.items():
        R = rows.loc[tr_mask, ["choice_id", "log_q", "fe"] + feats].copy()
        newc = np.zeros(len(rows), np.int8)
        newc[pos] = 1
        R["chosen"] = newc[tr_mask]
        D = to_buckets(R.reset_index(drop=True), feats)
        fits = shapley_fits(D, feats, log=None)
        for pname, part in [("four groups", None),
                            ("pop / fit / inertia", {"popularity": ["pop"], "fit (quality + match)": ["qual", "match"],
                                                     "inertia": ["inertia"]})]:
            sh, tot = shares(fits, list(GROUPS), part)
            for g, v in sh.items():
                rec.append(dict(network=name, partition=pname, group=NAMES.get(g, g), share=v["share"], total_gain=tot))
        log(f"recovery {name} done")
        del D
        torch.cuda.empty_cache()
    pd.DataFrame(rec).to_csv(OUTD / "recovery.csv", index=False)


# ---------------------------------------------------------------- SF full-market simulation
class SF:
    def __init__(self, kind, coef):
        self.kind = kind  # "sf" (fitted popularity terms) or "ba" (d + 1)
        nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                                columns=["model_id", "birth", "modality_code", "legacy", "major_org", "library_cat",
                                         "author_code"])
        E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
        from hfpa.features.groups import redistributors
        _, red = redistributors(nodes, E)
        B = pd.read_parquet(paths.PROCESSED / "nodes" / "bench.parquet")
        bench_na = np.ones(len(nodes), bool)
        bench_na[B.node.to_numpy()] = ~np.isfinite(B.bench_z.to_numpy())
        S = pd.DataFrame({"major_org": nodes.major_org.astype(float), "legacy": nodes.legacy.astype(float),
                          "redistributor": red.astype(float), "bench_na": bench_na.astype(float),
                          **{f"lib_{x}": (nodes.library_cat == x).astype(float) for x in LIB}})
        self.N = len(nodes)
        self.birth = nodes.birth.to_numpy(np.int64)
        self.bday = self.birth // DAY
        self.mod = nodes.modality_code.to_numpy()
        c = coef
        g = lambda k: c.get(k, 0.0)
        self.b = dict(has_deg=g("has_deg"), log_deg=g("log_deg"), lrec30=g("lrec30"), log_age=g("log_age"))
        stat = S[STATIC].to_numpy() @ np.array([g(k) for k in STATIC]) if kind == "sf" else np.zeros(self.N)
        codes, self.code = np.unique(stat.round(10), return_inverse=True)
        self.stat_code = codes  # static utility per code
        self.stat = stat
        self.K = len(codes)
        self.d0 = int(self.bday.min())
        self.nd = int(self.bday.max()) - self.d0 + 2
        # nodes sorted by (modality, code, birth day) for uniform draws inside a cell
        o = np.lexsort((self.bday, self.code, self.mod))
        self.sorted = o
        key = (self.mod[o].astype(np.int64) * self.K + self.code[o]) * self.nd + (self.bday[o] - self.d0)
        self.cell_key = key
        self.M = int(self.mod.max()) + 1
        # initial state: real edges with child born at or before T0
        from hfpa.data.build import T0 as T0_
        t0 = int(T0_.timestamp())
        m = E[E.rel_type.isin(MAIN)]
        pre = self.birth[m.child_idx.to_numpy()] <= t0
        self.deg = np.bincount(m.parent_idx.to_numpy()[pre], minlength=self.N).astype(np.int64)
        self.t0 = t0

    def run(self, ev, seed):
        """ev: DataFrame t, modality, author (time-sorted plan-A first decisions). Returns adopted parents."""
        rng = np.random.default_rng(seed)
        deg = self.deg.copy()
        rec_log = defaultdict(list)
        rec = np.zeros(self.N, np.int64)
        # zero-degree alive counts per (modality, code, birth day) cell
        zc = np.zeros((self.M, self.K, self.nd), np.int64)
        alive_ptr = np.zeros(self.M, np.int64)
        mod_nodes = [np.flatnonzero(self.mod == m) for m in range(self.M)]  # sorted by birth
        mod_bday = [self.bday[x] for x in mod_nodes]
        pool = [set(np.flatnonzero((deg > 0) & (self.mod == m)).tolist()) for m in range(self.M)]
        hist = defaultdict(set)
        out = np.empty(len(ev), np.int64)
        evd = (ev.t.to_numpy() // DAY)
        days = np.unique(evd)
        ptr = 0
        for d in days:
            # retire recent-window entries older than 30 days
            for old in [k for k in rec_log if k <= d - 30]:
                np.subtract.at(rec, rec_log.pop(old), 1)
            sel = np.flatnonzero(evd == d)
            for m in np.unique(ev.modality.to_numpy()[sel]):
                k = int(np.searchsorted(mod_bday[m], d, "left"))  # born before the day starts
                if k > alive_ptr[m]:
                    new = mod_nodes[m][alive_ptr[m]:k]
                    z = new[deg[new] == 0]
                    np.add.at(zc[m], (self.code[z], self.bday[z] - self.d0), 1)
                    alive_ptr[m] = k
                P = np.fromiter(pool[m], np.int64, len(pool[m]))
                P = P[self.bday[P] < d] if len(P) else P
                age = np.maximum(d - self.bday[P], 0) + 0.5
                if self.kind == "ba":
                    wp = deg[P] + 1.0
                    wz = zc[m].astype(float)
                else:
                    b = self.b
                    up = (b["has_deg"] * (deg[P] > 0) + b["log_deg"] * np.log(np.maximum(deg[P], 1))
                          + b["lrec30"] * np.log1p(rec[P]) + b["log_age"] * np.log1p(age) + self.stat[P])
                    ages = np.maximum(d - (self.d0 + np.arange(self.nd)), 0) + 0.5
                    uz = self.stat_code[:, None] + b["log_age"] * np.log1p(ages)[None, :]
                    mx = max(up.max() if len(up) else -np.inf, uz.max())
                    wp = np.exp(up - mx)
                    wz = zc[m] * np.exp(uz - mx)
                wzf = wz.ravel()
                tot_p, tot = wp.sum(), wp.sum() + wzf.sum()
                cp, cz = np.cumsum(wp), np.cumsum(wzf)
                ids = sel[ev.modality.to_numpy()[sel] == m]
                for e in ids:
                    a = int(ev.author.iat[e])
                    for _ in range(20):  # a first adoption: redraw a parent the author already adopted
                        r = rng.random() * tot
                        if r < tot_p:
                            j = int(P[min(np.searchsorted(cp, r, "right"), len(P) - 1)])
                        else:
                            cell = min(int(np.searchsorted(cz, r - tot_p, "right")), len(cz) - 1)
                            j = self._node_in_cell(m, cell, deg, rng)
                            if j < 0:
                                continue
                        if j not in hist[a]:
                            break
                    hist[a].add(j)
                    out[e] = j
                # update state after the day
                for e in ids:
                    j = int(out[e])
                    if deg[j] == 0:
                        if self.bday[j] < d:
                            zc[m, self.code[j], self.bday[j] - self.d0] -= 1
                        pool[m].add(j)
                    deg[j] += 1
                    rec[j] += 1
                    rec_log[d].append(j)
            ptr += len(sel)
        return out

    def _node_in_cell(self, m, cell, deg, rng):
        c, bd = divmod(cell, self.nd)
        key = (m * self.K + c) * self.nd + bd
        lo, hi = np.searchsorted(self.cell_key, key, "left"), np.searchsorted(self.cell_key, key, "right")
        if hi <= lo:
            return -1
        for _ in range(200):
            j = int(self.sorted[rng.integers(lo, hi)])
            if deg[j] == 0:
                return j
        cand = self.sorted[lo:hi]
        cand = cand[deg[cand] == 0]
        return int(cand[rng.integers(len(cand))]) if len(cand) else -1


def part_sf():
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet")
    ev = ch[ch.pair_first].sort_values("t", kind="stable")[["t", "modality", "author"]].reset_index(drop=True)
    feats, b, _ = model_coefs()
    coef = dict(zip(feats, b))
    OUTD.mkdir(parents=True, exist_ok=True)
    res = []
    for kind in ("sf", "ba"):
        sim = SF(kind, coef)
        for s in SEEDS:
            par = sim.run(ev, s)
            res.append(dict(network="SF" if kind == "sf" else "BA (d+1)", seed=s, **metrics(par, ev.t.to_numpy())))
            log(res[-1])
    pd.DataFrame(res).to_csv(OUTD / "sf_metrics.csv", index=False)


def part_collect():
    M = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUTD / "*_metrics.csv")))])
    order = ["SF", "BA (d+1)", "SF-SIR", "FULL", "REAL", "QF"]
    agg = M.groupby("network").agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.drop(columns=[c for c in agg.columns if c.startswith("seed")]).reindex([o for o in order if o in agg.index])
    write(agg.reset_index(), "v2_networks", "Three networks: scale-free growth (SF), quality-fit growth (QF) and the real "
          "network (plan-A first adoptions; mean and SD over 5 seeds)",
          "SF = full-market simulation with feedback, popularity terms of the v2 model; BA = weight d+1; SF-SIR / QF / "
          "FULL = importance resampling on the real candidate sets (no feedback). Expected ordering of concentration: "
          "SF > REAL > QF.")
    rf = OUTD / "recovery.csv"
    if rf.exists():
        R = pd.read_csv(rf)
        write(R, "v2_networks_recovery", "Share recovery: four-group Shapley re-estimated on choices drawn from SF-SIR "
              "(popularity only), QF (quality + match only) and FULL (full model)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["sir", "sf", "collect"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    SEEDS = range(a.seeds)
    {"sir": part_sir, "sf": part_sf, "collect": part_collect}[a.part]()
