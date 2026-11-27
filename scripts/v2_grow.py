"""v2 growth networks with feedback: SF / QF / EST grown on the full market (follow-up of v2_networks.py).

The real event sequence is kept (every edge, its time, the child, its author and modality); only the parents of
the main choices are re-drawn. Time runs in groups of equal child birth time, as in hfpa.replay: the choices of a
group see the network state before the group, then the state is updated with every edge of the group.

  plan-A first adoptions (the estimation sample): a fresh (d+1) Poisson-PPS candidate set is drawn from the
      SIMULATED degrees (E|D| = 300, alive nodes of the child's modality), the author's earlier parents are removed
      (plan A), all 26 features are computed against the simulated state and one parent is drawn with weight
      exp(V - log pi) (importance resampling on the PPS set, i.e. an approximate full-market draw).
  repeat choices of an (author, real parent) pair follow the simulated parent of the pair's first choice, so the
      simulated degrees carry all 423k main choices and not only first adoptions.
  everything else (non-choice edges, quantized children, node attributes, card text, benchmarks) stays real.

State that follows the simulation: degree, 30-day adoptions, the PPS degree histogram, the author's parents and
families (author_family, plan-A exclusion), the author's choice history (size / task / language history), the
(dataset, parent) co-use index (ds_couse and the A4 own-children removal) and the two-hop co-user index (lco2,
rebuilt at each month start from the simulated edges, as in v2_features.twohop_month).

Utilities (v2_pps plan-A coefficients, clogit.pkl):
  SF      popularity group (no-FE fit)                       EST     all four groups (no-FE fit)
  QF      quality + match (no-FE fit) + externally explained FE part (q_ext.npy)
  SF_FE   popularity group + hub FE (FE fit)                 EST_FE  all four groups + hub FE (FE fit)

--check MONTHS  validation: every choice takes its real parent and features are recomputed on the REAL candidate
                sets of the given months; compared with processed/choices/v2_pps/feats (must agree).
--net NET --seed S [--save-rows]  one simulated network -> v2_est/grow/NET_S.parquet (+ rows for share recovery)
--collect       metrics of every saved network + REAL -> results/tables/v2/v2_grow.md
usage: sbatch --array=0-24 slurm/cpu_grow.sbatch      (task id -> NET x seed)
"""
import argparse
import glob
import os
import pickle
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import scipy.sparse as sp

from hfpa import paths
from hfpa.features.groups import GROUPS, LIB, MAIN, MIN_TOKENS, redistributors
from hfpa.features.match import build_v2
from hfpa.features.text import row_cosine

SPEC = "v2_pps"
DAY = 86400
S_PPS = 300
NETS = ["SF", "QF", "EST", "SF_FE", "EST_FE", "SF_NOMOM", "SF_CUM", "SUCC_FULL", "SUCC_POP", "SAT_BASE", "SAT_SPL"]
# SAT_BASE / SAT_SPL: five-group model without / with hinge terms on log_deg and lrec30, both estimated on the same
# 50% author subsample (v2_mech.py --part sat -> v2_mech/sat_fit.pkl, key (0.5, 0, base|spline))
# SUCC_FULL / SUCC_POP: five-group model / popularity group re-estimated with the succession features
# (v2_mech.py --part succ_full -> v2_mech/succ_fit.pkl; features from v2_mech.SuccIndex, static given t)
# growth ablations (v2_explore.py --part refit): popularity group of the five-group model re-estimated without
# momentum (lrec30) or with cumulative degree only (no lrec30, no log_age); SF itself equals "SF full (refit)"
REFIT_NETS = {"SF_NOMOM": "no momentum (refit)", "SF_CUM": "cumulative only (refit)"}
# network-level Shapley (v2_mech.py): S5_<bits> = the five-group model re-estimated on a subset of the groups
# (coefficients of v2_samples/P5/fit.pkl fits[subset]); bits over P5_ORDER, e.g. S5_10100 = popularity + match
P5_ORDER = ["pop", "qual", "match", "self", "habit"]
S5_NETS = ["S5_" + format(i, "05b") for i in range(32)]
CDIR = paths.PROCESSED / "choices" / SPEC
OUTD = CDIR / "v2_est" / "grow"
MECHD = OUTD / "mech"  # subset networks and truncated horizons (kept out of --collect)
IDX = paths.INTERIM / "v2_idx"
T_START = time.time()


def log(s):
    print(f"[{time.time() - T_START:7.0f}s] {s}", flush=True)


def pps_c(cnt, n0, S=S_PPS):
    """c with sum_j min(1, c (d_j + 1)) + n0 min(1, c) = S; cnt[d] = number of alive nodes of degree d > 0.
    Same closed form as hfpa.replay.Replay._pps_c; inf if the market is not larger than S."""
    d = np.flatnonzero(cnt)[::-1]
    d = d[d > 0]
    n = cnt[d].astype(np.float64)
    if n0 + n.sum() <= S:
        return np.inf
    w = n * (d + 1)
    nsat = np.r_[0.0, np.cumsum(n)]
    W = n0 + w.sum() - np.r_[0.0, np.cumsum(w)]
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (S - nsat) / W
    K = len(d)
    ok = (c > 0) & np.isfinite(c)
    ok[1:] &= c[1:] * (d + 1) >= 1
    ok[:K] &= c[:K] * (d + 1) < 1
    ok &= c <= 1
    return float(c[int(np.flatnonzero(ok)[0])])


class Hist:
    """author -> (times, parents) of main choices, appended during the simulation (build_v2 interface)."""

    def __init__(self):
        self.t, self.p, self.cache = defaultdict(list), defaultdict(list), {}

    def add(self, a, t, p):
        self.t[a].append(t)
        self.p[a].append(p)
        self.cache.pop(a, None)

    def get(self, a, default=None):
        if a not in self.t:
            return default
        r = self.cache.get(a)
        if r is None:
            r = self.cache[a] = (np.array(self.t[a], np.int64), np.array(self.p[a], np.int64))
        return r


class Couse:
    """(dataset, parent) -> (child births, child authors, child ids) of main choices (build_v2 cidx interface)."""

    def __init__(self):
        self.v, self.cache = defaultdict(list), {}

    def add(self, d, p, t, a, c):
        self.v[(d, p)].append((t, a, c))
        self.cache.pop((d, p), None)

    def get(self, k, default=None):
        if k not in self.v:
            return default
        r = self.cache.get(k)
        if r is None:
            v = self.v[k]  # appended in time order
            r = self.cache[k] = tuple(np.array([x[i] for x in v]) for i in range(3))
        return r


class Grow:
    def __init__(self):
        t0 = time.time()
        nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                                columns=["model_id", "birth", "modality_code", "author_code", "root", "legacy",
                                         "major_org", "library_cat"])
        E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
        self.N = N = len(nodes)
        self.birth = nodes.birth.to_numpy(np.int64)
        self.mod = nodes.modality_code.to_numpy(np.int64)
        self.author = nodes.author_code.to_numpy(np.int64)
        self.root = nodes.root.to_numpy(np.int64)
        self.M = int(self.mod.max()) + 1
        self.mod_nodes = [np.flatnonzero(self.mod == m) for m in range(self.M)]  # birth-sorted (nodes are)
        self.mod_birth = [self.birth[x] for x in self.mod_nodes]
        # static node features
        lib = nodes.library_cat.to_numpy()
        self.static = {"legacy": nodes.legacy.to_numpy(np.float32), "major_org": nodes.major_org.to_numpy(np.float32),
                       "redistributor": np.load(IDX / "redistributor.npy").astype(np.float32),
                       **{f"lib_{x}": (lib == x).astype(np.float32) for x in LIB}}
        B = pd.read_parquet(paths.PROCESSED / "nodes" / "bench.parquet")
        bz = np.full(N, np.nan, np.float32)
        bz[B.node.to_numpy()] = B.bench_z.to_numpy()
        self.static["bench_na"] = np.isnan(bz).astype(np.float32)
        self.static["bench_z"] = np.nan_to_num(bz).astype(np.float32)
        ld = lambda n: pickle.load(open(IDX / f"{n}.pkl", "rb"))
        self.A, self.prof = ld("A"), ld("prof")
        self.X = sp.load_npz(paths.INTERIM / "v2_text_tfidf.npz").tocsr()
        T = np.load(paths.INTERIM / "v2_text_rows.npz")
        self.crow, self.prow, self.ntc, self.ntp = T["crow"], T["prow"], T["ntok_child"], T["ntok_cand"]
        # choices
        ch = pd.read_parquet(CDIR / "decisions.parquet",
                             columns=["choice_id", "child", "parent", "t", "modality", "author", "pair_first", "holdout"])
        self.ch = ch.set_index("choice_id", drop=False)
        # plan-A estimation sample: first choices whose chosen parent is not an earlier parent of the author
        P = []
        for f in sorted(glob.glob(str(CDIR / "feats" / "*.parquet"))):
            if f.endswith(".notext.parquet"):
                continue
            x = pd.read_parquet(f, columns=["choice_id", "chosen", "author_reuse"])
            if len(x):
                P.append(x.choice_id[(x.chosen == 1) & (x.author_reuse == 0)].to_numpy())
        okc = np.zeros(int(ch.choice_id.max()) + 1, bool)
        okc[np.concatenate(P)] = True
        self.sim_choice = okc & np.isin(np.arange(len(okc)), ch.choice_id[ch.pair_first].to_numpy())
        # pair first of every choice
        o = np.argsort(ch.t.to_numpy(), kind="stable")
        key = pd.Series(list(zip(ch.author.to_numpy()[o], ch.parent.to_numpy()[o])))
        firsts = pd.Series(ch.choice_id.to_numpy()[o]).groupby(key.values).transform("first").to_numpy()
        self.pair_first_of = np.empty(len(okc), np.int64)
        self.pair_first_of[ch.choice_id.to_numpy()[o]] = firsts
        # edges with their choice id (cumcount aligns duplicated child-parent edges)
        E = E[E.rel_type.isin(MAIN + ["quantized"])].copy()
        E["t"] = self.birth[E.child_idx.to_numpy()]
        E = E.sort_values("t", kind="stable").reset_index(drop=True)
        E["k"] = E.groupby(["child_idx", "parent_idx"]).cumcount()
        C = pd.DataFrame({"child_idx": ch.child.to_numpy(), "parent_idx": ch.parent.to_numpy(),
                          "choice_id": ch.choice_id.to_numpy()}).sort_values("choice_id")
        C["k"] = C.groupby(["child_idx", "parent_idx"]).cumcount()
        E = E.merge(C, on=["child_idx", "parent_idx", "k"], how="left")
        E["choice_id"] = E.choice_id.fillna(-1).astype(np.int64)
        E.loc[~E.rel_type.isin(MAIN), "choice_id"] = -1
        assert (E.choice_id >= 0).sum() == len(ch), "every choice must map to one edge"
        self.e_child, self.e_par, self.e_t = E.child_idx.to_numpy(np.int64), E.parent_idx.to_numpy(np.int64), E.t.to_numpy()
        self.e_main, self.e_cid = E.rel_type.isin(MAIN).to_numpy(), E.choice_id.to_numpy()
        from hfpa.data.build import T0
        self.t_start = int(T0.timestamp())
        # FE ids (ex-ante: top 500 by degree before the train / holdout cut), checked against the rows
        t_h = int(ch.t[ch.holdout].min())
        de = self.e_main & (self.e_t < t_h)
        dh = np.bincount(self.e_par[de], minlength=N)
        top = np.argsort(-dh, kind="stable")[:500]
        top = top[dh[top] > 0]
        self.fe = np.zeros(N, np.int64)
        self.fe[top] = np.arange(1, len(top) + 1)
        x = pd.read_parquet(sorted(glob.glob(str(CDIR / "feats" / "2024-*.parquet")))[0], columns=["cand_id", "fe"])
        assert np.array_equal(self.fe[x.cand_id.to_numpy()], x.fe.to_numpy()), "FE ids differ from the replay"
        # coefficients
        Cf = pickle.load(open(CDIR / "v2_est" / "clogit.pkl", "rb"))
        self.feats, self.b, self.bfe = Cf["feats"], np.asarray(Cf["nofe"]["coef"]), np.asarray(Cf["fe"]["coef"])
        q = CDIR / "v2_est" / "q_ext.npy"
        self.qext = np.r_[0.0, np.load(q)] if q.exists() else None
        rf = CDIR / "v2_explore" / "refit.pkl"
        self.refit = pickle.load(open(rf, "rb")) if rf.exists() else None
        p5 = CDIR / "v2_samples" / "P5" / "fit.pkl"
        self.p5 = pickle.load(open(p5, "rb")) if p5.exists() else None
        sys.path.insert(0, str(paths.REPO / "scripts"))
        from v2_mech import HINGE, SUCC, SuccIndex, hinge
        self.hinge, self.HINGE = hinge, HINGE
        sp_ = CDIR / "v2_mech" / "sat_fit.pkl"
        self.sat_fit = pickle.load(open(sp_, "rb")) if sp_.exists() else None
        self.hw = None  # weights on the hinge terms (SAT_SPL only)
        self.SUCC = SUCC
        self.succ = SuccIndex()
        sf = CDIR / "v2_mech" / "succ_fit.pkl"
        self.succ_fit = pickle.load(open(sf, "rb")) if sf.exists() else None
        self.wx = None  # weights on the succession features (SUCC nets only)
        log(f"loaded in {time.time() - t0:.0f}s: {N:,} nodes, {len(E):,} edges, {len(ch):,} choices, "
            f"{self.sim_choice.sum():,} simulated first choices")

    # ------------------------------------------------------------ utility of a network
    def utility(self, net):
        F = len(self.feats)
        gi = lambda gs: [i for i, f in enumerate(self.feats) if any(f in GROUPS[g] for g in gs)]
        w = np.zeros(F)
        fe = np.zeros(501)
        if net == "SF":
            w[gi(["pop"])] = self.b[gi(["pop"])]
        elif net == "QF":
            w[gi(["qual", "match"])] = self.b[gi(["qual", "match"])]
            fe = self.qext
        elif net == "EST":
            w = self.b.copy()
        elif net == "SF_FE":
            w[gi(["pop"])] = self.bfe[gi(["pop"])]
            fe = np.r_[0.0, self.bfe[F:]]
        elif net == "EST_FE":
            w = self.bfe[:F].copy()
            fe = np.r_[0.0, self.bfe[F:]]
        elif net in ("SAT_BASE", "SAT_SPL"):
            R = self.sat_fit[(0.5, 0, "base" if net == "SAT_BASE" else "spline")]
            c = dict(zip(R["feats"], R["coef"]))
            w = np.array([c.get(f, 0.0) for f in self.feats])
            if net == "SAT_SPL":
                self.hw = np.array([c.get(f, 0.0) for f in self.HINGE])
        elif net in ("SUCC_FULL", "SUCC_POP"):
            R = self.succ_fit[net]
            c = dict(zip(R["feats"], R["coef"]))
            w = np.array([c.get(f, 0.0) for f in self.feats])
            self.wx = np.array([c.get(f, 0.0) for f in self.SUCC])
        elif net.startswith("S5_"):
            S = [g for g, bit in zip(P5_ORDER, net[3:]) if bit == "1"]
            gr, f5 = self.p5["groups"], self.p5["feats"]
            cols = [f for f in f5 if any(f in gr[g] for g in S)]
            c = dict(zip(cols, self.p5["fits"][frozenset(S)]["coef"]))
            w = np.array([c.get(f, 0.0) for f in self.feats])
        elif net in REFIT_NETS:
            rc = self.refit[REFIT_NETS[net]]["coef"]
            ip = gi(["pop"])
            w[ip] = [rc.get(self.feats[i], 0.0) for i in ip]
        return w, fe

    # ------------------------------------------------------------ state
    def reset(self):
        N = self.N
        self.deg = np.zeros(N, np.int64)
        self.quant = np.zeros(N, np.int64)
        self.rec = np.zeros(N, np.int64)
        self.recq = deque()  # (t, parent) of degree edges, for the 30-day window
        self.dcnt = [np.zeros(20_000, np.int64) for _ in range(self.M)]
        self.pos = [np.empty(len(x), np.int64) for x in self.mod_nodes]
        self.npos = np.zeros(self.M, np.int64)
        self.parents = defaultdict(set)  # author -> parents (any modality) = author_reuse
        self.roots = defaultdict(set)  # author -> roots of its parents = author_family
        self.hist = Hist()
        self.cidx = Couse()
        self.sim_par = np.full(len(self.pair_first_of), -1, np.int64)
        self.done_t, self.done_a, self.done_p = [], [], []  # main edges so far (two-hop index)
        self.th_month, self.th_cache = None, {}

    def add_edge(self, i, p):
        """Apply edge i of the real sequence with parent p (simulated for choice edges)."""
        t, c = int(self.e_t[i]), int(self.e_child[i])
        if not self.e_main[i]:
            self.quant[p] += 1
            return
        m = self.mod[p]
        if self.deg[p] == 0:
            self.pos[m][self.npos[m]] = p
            self.npos[m] += 1
        else:
            self.dcnt[m][self.deg[p]] -= 1
        if self.deg[p] + 2 >= len(self.dcnt[m]):
            self.dcnt[m] = np.r_[self.dcnt[m], np.zeros(len(self.dcnt[m]), np.int64)]
        self.deg[p] += 1
        self.dcnt[m][self.deg[p]] += 1
        self.rec[p] += 1
        self.recq.append((t, p))
        a = int(self.author[c])
        self.parents[a].add(p)
        self.roots[a].add(int(self.root[p]))
        self.done_t.append(t)
        self.done_a.append(a)
        self.done_p.append(p)
        k = self.e_cid[i]
        if k >= 0:  # a choice: author history and co-use index (as author_history / couse_index_v2 on choices)
            self.hist.add(a, t, p)
            D = self.A["ds"][c]
            if D is not None:
                for d in set(D):
                    self.cidx.add(d, p, t, a, c)

    def twohop_index(self, t):
        """v2_features.twohop_month on the simulated edges before the month start of t."""
        mon = pd.Timestamp(t, unit="s").strftime("%Y-%m")
        if mon == self.th_month:
            return
        t0 = int(pd.Period(mon, "M").to_timestamp().tz_localize("UTC").timestamp())
        et, ea, ep = (np.asarray(x) for x in (self.done_t, self.done_a, self.done_p))
        pre = et < t0
        NA = int(self.author.max()) + 1
        dpre = np.bincount(ep[pre], minlength=self.N)
        hubs = np.argsort(-dpre, kind="stable")[:500]
        self.th_hub = np.zeros(self.N, bool)
        self.th_hub[hubs] = True
        A = sp.csr_matrix((np.ones(pre.sum()), (ea[pre], ep[pre])), shape=(NA, self.N))
        A.sum_duplicates()
        A.data[:] = 1.0
        niche = pre & ~self.th_hub[ep]
        Ant = sp.csr_matrix((np.ones(niche.sum()), (ep[niche], ea[niche])), shape=(self.N, NA))
        Ant.sum_duplicates()
        self.th_A, self.th_Ant, self.th_month, self.th_cache = A, Ant, mon, {}

    def twohop(self, a, J):
        r = self.th_cache.get(a)
        if r is None:
            S = self.th_A[a].indices
            S = S[~self.th_hub[S]]
            r = (np.empty(0, np.int64), np.empty(0, np.int64))
            if len(S):
                users = np.unique(self.th_Ant[S].indices)
                users = users[users != a]
                if len(users):
                    r = np.unique(self.th_A[users].indices, return_counts=True)
            self.th_cache[a] = r
        used, cnt = r
        co = np.zeros(len(J), np.float32)
        if len(used):
            pos = np.minimum(np.searchsorted(used, J), len(used) - 1)
            hit = used[pos] == J
            co[hit] = cnt[pos[hit]]
        return co

    # ------------------------------------------------------------ candidate sets
    def sample(self, m, t, a, rng, cache):
        """(d+1) Poisson PPS on the simulated degrees; the author's earlier parents removed (plan A)."""
        na = int(np.searchsorted(self.mod_birth[m], t, "left"))
        n1 = int(self.npos[m])
        if m not in cache:
            cache[m] = pps_c(self.dcnt[m], na - n1)
        c = cache[m]
        P = self.pos[m][:n1]
        pi1 = np.minimum(1.0, c * (self.deg[P] + 1.0))
        take = rng.random(n1) < pi1
        pool1, lq1 = P[take], np.log(pi1[take])
        n0 = na - n1
        pi0 = min(1.0, c)
        k0 = int(rng.binomial(n0, pi0)) if n0 > 0 else 0
        alive = self.mod_nodes[m][:na]
        if k0 == 0:
            pool0 = np.empty(0, np.int64)
        elif k0 > 0.3 * n0 or n0 < 0.1 * na:
            z = alive[self.deg[alive] == 0]
            pool0 = z[rng.choice(len(z), k0, replace=False)]
        else:
            got = {}
            while len(got) < k0:
                dd = alive[rng.integers(0, na, size=2 * k0 + 8)]
                for j in dd[self.deg[dd] == 0]:
                    got.setdefault(int(j), None)
            pool0 = np.fromiter(got, np.int64)[:k0]
        J = np.concatenate([pool1, pool0])
        lq = np.concatenate([lq1, np.full(len(pool0), np.log(pi0))])
        H = self.parents.get(a)
        if H:
            keep = ~np.isin(J, np.fromiter(H, np.int64, len(H)))
            J, lq = J[keep], lq[keep]
        return J, lq

    # ------------------------------------------------------------ features
    def features(self, ks, Js, lqs):
        """Feature matrix (rows of all choices ks, in order) in the order of self.feats + fe ids."""
        cid = np.concatenate([np.full(len(J), k) for k, J in zip(ks, Js)])
        J = np.concatenate(Js).astype(np.int64)
        rows = pd.DataFrame({"choice_id": cid, "cand_id": J})
        ch = self.ch
        t = ch.t.loc[cid].to_numpy()
        kid = ch.child.loc[cid].to_numpy()
        a = self.author[kid]
        f = {}
        d = self.deg[J]
        f["log_deg"] = np.log(np.maximum(d, 1))
        f["has_deg"] = (d > 0).astype(np.float32)
        f["log_age"] = np.log1p(np.maximum((t - self.birth[J]) / DAY, 0))
        f["lrec30"] = np.log1p(self.rec[J])
        f["lquant"] = np.log1p(self.quant[J])
        for k, v in self.static.items():
            f[k] = v[J]
        f["same_author"] = (self.author[J] == a).astype(np.float32)
        fam = np.zeros(len(J), np.float32)
        lco = np.zeros(len(J), np.float32)
        s = 0
        for k, Jk in zip(ks, Js):
            e = s + len(Jk)
            ak = int(self.author[ch.child.at[k]])
            R = self.roots.get(ak)
            if R:
                fam[s:e] = np.isin(self.root[Jk], np.fromiter(R, np.int64, len(R)))
            lco[s:e] = self.twohop(ak, Jk)
            s = e
        f["author_family"] = fam
        f["lco2"] = np.log1p(lco)
        M = build_v2(rows, ch, self.A, None, self.author, self.prof, self.cidx, hist=self.hist)
        for k in M.columns:
            f[k] = M[k].to_numpy(np.float32)
        ra, rb = self.crow[kid], self.prow[J]
        nt = np.minimum(self.ntc[kid], self.ntp[J])
        nt[(ra < 0) | (rb < 0)] = 0
        ok = nt >= 1
        sim = np.zeros(len(J), np.float32)
        if ok.any():
            sim[ok] = row_cosine(self.X, ra[ok], rb[ok])
        na_ = nt < MIN_TOKENS
        f["txt_na"] = na_.astype(np.float32)
        f["txt_sim"] = np.where(na_, 0, sim).astype(np.float32)
        Xf = np.column_stack([np.asarray(f[k], np.float64) for k in self.feats])
        return Xf, self.fe[J], rows

    # ------------------------------------------------------------ main loop
    def run(self, net=None, seed=0, check_months=None, save_rows=False, until=None):
        """net=None with check_months: validation (real parents, real candidate sets)."""
        self.reset()
        rng = np.random.default_rng(1000 + seed)
        if net is not None:
            w, fe_u = self.utility(net)
        starts = np.r_[0, np.flatnonzero(np.diff(self.e_t)) + 1]
        ends = np.r_[starts[1:], len(self.e_t)]
        out, saved, checks = [], [], []
        real_rows, real_mon = None, None
        nsim = 0
        last_log = time.time()
        for g0, g1 in zip(starts, ends):
            t = int(self.e_t[g0])
            if until is not None and pd.Timestamp(t, unit="s").strftime("%Y-%m") > until:
                break
            while self.recq and self.recq[0][0] <= t - 30 * DAY:
                _, p = self.recq.popleft()
                self.rec[p] -= 1
            sims = [i for i in range(g0, g1) if self.e_cid[i] >= 0 and self.sim_choice[self.e_cid[i]]] \
                if t > self.t_start else []
            if check_months is not None:
                mon = pd.Timestamp(t, unit="s").strftime("%Y-%m")
                if mon > max(check_months):
                    break
                sims = sims if mon in check_months else []
            if sims:
                self.twohop_index(t)
                ks = [int(self.e_cid[i]) for i in sims]
                if check_months is not None:
                    if real_mon != mon:
                        real_rows = pd.read_parquet(CDIR / "feats" / f"{mon}.parquet")
                        real_rows = real_rows[real_rows.author_reuse == 0]
                        real_rows = {k: g for k, g in real_rows.groupby("choice_id")}
                        real_mon = mon
                    Js = [real_rows[k].cand_id.to_numpy() for k in ks]
                    lqs = [real_rows[k].log_q.to_numpy() for k in ks]
                else:
                    cache = {}
                    Js, lqs = [], []
                    for k in ks:
                        J, lq = self.sample(int(self.ch.modality.at[k]), t, int(self.ch.author.at[k]), rng, cache)
                        Js.append(J)
                        lqs.append(lq)
                ok = [len(J) > 0 for J in Js]
                ks2 = [k for k, o in zip(ks, ok) if o]
                Js2 = [J for J, o in zip(Js, ok) if o]
                lq2 = [l for l, o in zip(lqs, ok) if o]
                if ks2:
                    Xf, fe, rows = self.features(ks2, Js2, lq2)
                    if check_months is not None:
                        R = pd.concat([real_rows[k] for k in ks2], ignore_index=True)
                        checks.append((R, pd.DataFrame(Xf, columns=self.feats)))
                        for k in ks2:
                            self.sim_par[k] = int(self.ch.parent.at[k])
                    else:
                        v = Xf @ w + fe_u[fe] - np.concatenate(lq2)
                        if self.hw is not None:
                            v = v + self.hinge(Xf[:, self.feats.index("log_deg")], Xf[:, self.feats.index("lrec30")]) @ self.hw
                        if self.wx is not None:
                            v = v + self.succ.matrix(rows.cand_id.to_numpy(), self.ch.t.loc[rows.choice_id].to_numpy()) @ self.wx
                        s = 0
                        for kk, (k, J) in enumerate(zip(ks2, Js2)):
                            e = s + len(J)
                            vv = v[s:e]
                            p = np.exp(vv - vv.max())
                            j = int(rng.choice(len(J), p=p / p.sum()))
                            self.sim_par[k] = int(J[j])
                            out.append((k, int(J[j]), int(self.ch.parent.at[k]), len(J)))
                            if save_rows:
                                saved.append(pd.DataFrame(
                                    {"choice_id": np.full(len(J), k, np.int32), "cand_id": J.astype(np.int32),
                                     "chosen": (np.arange(len(J)) == j).astype(np.int8),
                                     "log_q": lq2[kk].astype(np.float32), "fe": fe[s:e].astype(np.int16),
                                     **{c: Xf[s:e, i].astype(np.float32) for i, c in enumerate(self.feats)}}))
                            s = e
                        nsim += len(ks2)
                for k, o in zip(ks, ok):
                    if not o:  # no candidate left: keep the real parent
                        self.sim_par[k] = int(self.ch.parent.at[k])
            # apply the group
            for i in range(g0, g1):
                k = self.e_cid[i]
                if k < 0 or t <= self.t_start or check_months is not None:
                    p = int(self.e_par[i])
                else:
                    f0 = self.pair_first_of[k]
                    p = int(self.sim_par[f0]) if self.sim_par[f0] >= 0 else int(self.e_par[i])
                    if self.sim_par[k] < 0:
                        self.sim_par[k] = p
                self.add_edge(i, p)
            if time.time() - last_log > float(os.environ.get("GROW_LOG_S", 300)):
                log(f"{pd.Timestamp(t, unit='s').date()}: simulated {nsim:,} first choices")
                last_log = time.time()
        if check_months is not None:
            return checks
        rows = pd.concat(saved, ignore_index=True) if saved else None
        return pd.DataFrame(out, columns=["choice_id", "sim_parent", "real_parent", "n_cands"]), rows


def check(months):
    g = Grow()
    checks = g.run(check_months=months)
    R = pd.concat([c[0] for c in checks], ignore_index=True)
    S = pd.concat([c[1] for c in checks], ignore_index=True)
    res = []
    for f in g.feats:
        a, b = R[f].to_numpy(np.float64), S[f].to_numpy(np.float64)
        res.append(dict(feature=f, rows=len(a), agree=float(np.mean(np.isclose(a, b, atol=1e-4))),
                        max_abs_diff=float(np.max(np.abs(a - b))), mean_real=a.mean(), mean_sim=b.mean()))
    T = pd.DataFrame(res)
    print(T.to_string(), flush=True)
    (paths.TABLES / "v2").mkdir(parents=True, exist_ok=True)
    T.to_csv(paths.TABLES / "v2" / "v2_grow_check.csv", index=False)
    return T


def simulate(net, seed, save_rows, until=None):
    g = Grow()
    od = MECHD if (until or net.startswith("S5_")) else OUTD
    od.mkdir(parents=True, exist_ok=True)
    log(f"=== {net} seed {seed} until {until or 'end'}")
    res, rows = g.run(net=net, seed=seed, save_rows=save_rows, until=until)
    tag = f"{net}__{until or 'full'}__{seed}" if od == MECHD else f"{net}_{seed}"
    res.to_parquet(od / f"{tag}.parquet", index=False)
    if rows is not None:
        rows.to_parquet(OUTD / f"{net}_{seed}_rows.parquet", index=False)
    log(f"{net} seed {seed}: {len(res):,} simulated choices; same parent as real {np.mean(res.sim_parent == res.real_parent):.3f}")


def collect():
    sys.path.insert(0, str(paths.REPO / "scripts"))
    from v2_networks import metrics
    from hfpa.v2data import write
    ch = pd.read_parquet(CDIR / "decisions.parquet", columns=["choice_id", "t", "parent"]).set_index("choice_id")
    res = []
    files = sorted(glob.glob(str(OUTD / "*_[0-9].parquet")) + glob.glob(str(OUTD / "*_[0-9][0-9].parquet")))
    for f in files:
        net, seed = os.path.basename(f)[:-8].rsplit("_", 1)
        r = pd.read_parquet(f)
        res.append(dict(network=net, seed=int(seed), **metrics(r.sim_parent.to_numpy(), ch.t.loc[r.choice_id].to_numpy())))
    r = pd.read_parquet(files[0])
    res.append(dict(network="REAL", seed=0, **metrics(r.real_parent.to_numpy(), ch.t.loc[r.choice_id].to_numpy())))
    D = pd.DataFrame(res)
    order = {n: i for i, n in enumerate(["SF", "SF_NOMOM", "SF_CUM", "SUCC_POP", "SF_FE", "EST", "SUCC_FULL", "EST_FE",
                                         "REAL", "QF"])}
    S = D.groupby("network").agg(["mean", "std"])
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S = S.drop(columns=["seed_mean", "seed_std"]).reset_index()
    S["n_seeds"] = D.groupby("network").size().reindex(S.network).to_numpy()
    S = S.sort_values("network", key=lambda s: s.map(order)).reset_index(drop=True)
    write(S, "v2_grow", "Growth networks with feedback on the full market (PPS candidate sets drawn from the simulated "
          "degrees; plan-A first adoptions; mean and SD over seeds)",
          "SF = popularity group, EST = all four groups (no-FE fit), QF = quality + match + externally explained FE; "
          "_FE = with hub FE (FE fit); SF_NOMOM / SF_CUM = popularity group re-estimated without momentum / with "
          "cumulative degree only. REAL = the observed first adoptions of the same events.")
    print(S.to_string())


def recovery():
    """Four-group Shapley re-estimated on the simulated choices of seed 0 (training period): does the
    decomposition recover the mechanism that grew the network?"""
    import torch
    from hfpa.features.groups import NAMES
    from hfpa.models.clogit import to_buckets
    from hfpa.v2data import shapley_fits, shares, write
    hold = pd.read_parquet(CDIR / "decisions.parquet", columns=["choice_id", "holdout"]).set_index("choice_id").holdout
    rec = []
    for net in ("SF", "QF", "EST"):
        f = OUTD / f"{net}_0_rows.parquet"
        if not f.exists():
            continue
        R = pd.read_parquet(f)
        R = R[~hold.reindex(R.choice_id).to_numpy()].reset_index(drop=True)
        feats = [c for c in R.columns if c not in ("choice_id", "cand_id", "chosen", "log_q", "fe") and R[c].std() > 0]
        D = to_buckets(R, feats)
        fits = shapley_fits(D, feats, log=None)
        for pname, part in [("four groups", None),
                            ("pop / fit / inertia", {"popularity": ["pop"], "fit (quality + match)": ["qual", "match"],
                                                     "inertia": ["inertia"]})]:
            sh, tot = shares(fits, list(GROUPS), part)
            for g, v in sh.items():
                rec.append(dict(network=net, partition=pname, group=NAMES.get(g, g), share=v["share"], total_gain=tot))
        log(f"recovery {net} done")
        del D
        torch.cuda.empty_cache()
    write(pd.DataFrame(rec), "v2_grow_recovery", "Share recovery on the growth networks with feedback (seed 0, "
          "training period): four-group Shapley re-estimated on the simulated choices and their PPS candidate sets")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="*")
    ap.add_argument("--net", choices=NETS + S5_NETS)
    ap.add_argument("--until", help="stop after this month (YYYY-MM); output in grow/mech/")
    ap.add_argument("--s5-task", type=int, help="array id -> S5 subset id // n, seed id % n (n = --s5-nseeds)")
    ap.add_argument("--s5-nseeds", type=int, default=2)
    ap.add_argument("--nseeds", type=int, default=5, help="--task: seeds per network")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=int, help="array task id -> (net, seed): net = NETS[id // 5], seed = id % 5")
    ap.add_argument("--save-rows", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--recovery", action="store_true")
    a = ap.parse_args()
    if a.check is not None:
        check(a.check)
    elif a.collect:
        collect()
    elif a.recovery:
        recovery()
    else:
        if a.task is not None:
            a.net, a.seed = NETS[a.task // a.nseeds], a.task % a.nseeds
        if a.s5_task is not None:
            a.net, a.seed = S5_NETS[a.s5_task // a.s5_nseeds], a.s5_task % a.s5_nseeds
        simulate(a.net, a.seed, a.save_rows or (a.seed == 0 and a.net in ("SF", "QF", "EST") and not a.until), a.until)
