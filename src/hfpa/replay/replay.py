"""Time replay of the fine-tuning network and construction of choice data (plan 3.2, 5.3, 5.5, 6.5).

Inputs (shared by synthetic data in P0 and real data in P4):
  nodes: DataFrame sorted by `birth`; positional index = node idx.
         columns birth (int64 seconds), modality (int), author (int), root (int node idx)
  edges: DataFrame with child, parent (node idx), rel_type (str), weight (float).
         Parents must be born no later than children (cleaned in P2).

Every edge whose rel_type is in `choice_types` and whose child is born after `t_start`
becomes one choice. Edges are processed in groups of equal child birth time: all choices
of a group are recorded against the state *before* the group, then the state is updated.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400
HUB, POS, ZERO, UNIF = 0, 1, 2, 3

DEFAULT_SPEC = dict(
    deg_types=("finetune", "adapter", "unspecified"),
    choice_types=("finetune", "adapter", "unspecified"),
    quant_types=("quantized",),
    sampling="stratified",  # or "uniform"
    hub_k=500, s1=24, s0=24, s=50,
    modality_restrict=True,
    t_start=None,  # seconds; choices only for children born strictly after
    recent_windows=(7, 30, 90),
    fam_window=30,
    sample_frac=1.0,
    seed=42,
)

ROW_COLS = ["choice_id", "cand_id", "chosen", "log_q", "stratum", "fe", "deg", "log_deg", "has_deg",
            "age_days", "quant_deg", "fam_size", "fam_recent", "same_author", "author_reuse", "author_family"]


def _log(x):
    return float(np.log(x)) if x > 0 else 0.0


class Replay:
    def __init__(self, nodes, edges, spec=None):
        sp = {**DEFAULT_SPEC, **(spec or {})}
        self.sp = sp
        self.rng = np.random.default_rng(sp["seed"])
        self.birth = nodes["birth"].to_numpy(np.int64)
        assert np.all(np.diff(self.birth) >= 0), "nodes must be sorted by birth"
        N = self.N = len(self.birth)
        self.author = nodes["author"].to_numpy(np.int64)
        self.root = nodes["root"].to_numpy(np.int64)
        mod = nodes["modality"].to_numpy(np.int64)
        self.cmod = mod if sp["modality_restrict"] else np.zeros(N, np.int64)
        self.M = int(self.cmod.max()) + 1

        rel = edges["rel_type"].to_numpy().astype(str)
        is_deg = np.isin(rel, sp["deg_types"])
        is_quant = np.isin(rel, sp["quant_types"])
        is_choice = np.isin(rel, sp["choice_types"])
        keep = is_deg | is_quant | is_choice
        child = edges["child"].to_numpy(np.int64)[keep]
        parent = edges["parent"].to_numpy(np.int64)[keep]
        weight = edges["weight"].to_numpy(np.float64)[keep] if "weight" in edges else np.ones(keep.sum())
        t_e = self.birth[child]
        o = np.argsort(t_e, kind="stable")
        self.e_child, self.e_parent, self.e_t, self.e_w = child[o], parent[o], t_e[o], weight[o]
        self.e_rel = rel[keep][o]
        self.is_deg, self.is_quant, self.is_choice = is_deg[keep][o], is_quant[keep][o], is_choice[keep][o]
        assert np.all(self.birth[self.e_parent] <= self.e_t), "parent born after child"

        # hubs: top-K parents by final degree (plan 3.2 stratum 1)
        self.final_deg = np.bincount(self.e_parent[self.is_deg], minlength=N)
        hubs = np.argsort(-self.final_deg, kind="stable")[: sp["hub_k"]]
        hubs = np.sort(hubs[self.final_deg[hubs] > 0])
        self.hubs = hubs
        self.is_hub = np.zeros(N, bool)
        self.is_hub[hubs] = True
        self.fe_idx = np.zeros(N, np.int32)
        rank = np.argsort(-self.final_deg[hubs], kind="stable")
        self.fe_idx[hubs[rank]] = np.arange(1, len(hubs) + 1)
        self.hub_by_mod = [hubs[self.cmod[hubs] == m] for m in range(self.M)]  # sorted by idx == by birth
        self.hub_birth_by_mod = [self.birth[h] for h in self.hub_by_mod]
        self.mod_nodes = [np.flatnonzero(self.cmod == m) for m in range(self.M)]
        self.mod_birth = [self.birth[x] for x in self.mod_nodes]

    # ---------------------------------------------------------------- sampling
    def _sample_stratified(self, m, t, p):
        sp, rng, deg = self.sp, self.rng, self.deg
        hubs_t = self.hub_by_mod[m][: np.searchsorted(self.hub_birth_by_mod[m], t, "left")]
        n1 = int(self.pos_n[m])
        k1 = min(sp["s1"], n1)
        pool1 = self.pos_arr[m][:n1][rng.choice(n1, k1, replace=False)] if k1 else np.empty(0, np.int64)
        lq1 = _log(k1 / n1) if n1 else 0.0
        na = int(np.searchsorted(self.mod_birth[m], t, "left"))
        n0 = na - n1 - len(hubs_t)
        k0 = min(sp["s0"], n0)
        alive = self.mod_nodes[m][:na]
        if k0 <= 0:
            pool0 = np.empty(0, np.int64)
        elif n0 < 0.1 * na:  # few zero-degree nodes left: enumerate instead of rejection sampling
            z = alive[(deg[alive] == 0) & ~self.is_hub[alive]]
            pool0 = z[rng.choice(len(z), k0, replace=False)]
        else:
            got = {}
            while len(got) < k0:
                d = alive[rng.integers(0, na, size=2 * k0)]
                for j in d[(deg[d] == 0) & ~self.is_hub[d]]:
                    got.setdefault(int(j), None)
            pool0 = np.fromiter(got, np.int64)[:k0]
        lq0 = _log(k0 / n0) if n0 > 0 else 0.0
        cands = np.concatenate([hubs_t, pool1, pool0])
        lq = np.concatenate([np.zeros(len(hubs_t)), np.full(k1, lq1), np.full(len(pool0), lq0)])
        st = np.concatenate([np.full(len(hubs_t), HUB), np.full(k1, POS), np.full(len(pool0), ZERO)])
        if not (cands == p).any():  # add chosen parent with its stratum's log q
            pos = deg[p] > 0
            cands = np.append(cands, p)
            lq = np.append(lq, lq1 if pos else lq0)
            st = np.append(st, POS if pos else ZERO)
        return cands, lq, st

    def _sample_uniform(self, m, t, p):
        na = int(np.searchsorted(self.mod_birth[m], t, "left"))
        k = min(self.sp["s"], na)
        cands = self.mod_nodes[m][:na][self.rng.choice(na, k, replace=False)]
        if not (cands == p).any():
            cands = np.append(cands, p)
        lq = np.full(len(cands), _log(k / na))
        return cands, lq, np.full(len(cands), UNIF)

    # ---------------------------------------------------------------- features
    def _features(self, k, c, p, t, cands, lq, st):
        d = self.deg[cands]
        a = self.author[c]
        hist, roots = self.author_hist.get(a), self.author_roots.get(a)
        r = self.root[cands]
        f = {
            "choice_id": np.full(len(cands), k, np.int32), "cand_id": cands.astype(np.int32),
            "chosen": (cands == p).astype(np.int8), "log_q": lq.astype(np.float32), "stratum": st.astype(np.int8),
            "fe": self.fe_idx[cands].astype(np.int16), "deg": d.astype(np.int32),
            "log_deg": np.log(np.maximum(d, 1)).astype(np.float32), "has_deg": (d > 0).astype(np.int8),
            "age_days": ((t - self.birth[cands]) / DAY).astype(np.float32),
            "quant_deg": self.quant_deg[cands].astype(np.int32),
            "fam_size": self.fam_size[r].astype(np.int32), "fam_recent": self.fam_recent[r].astype(np.int32),
            "same_author": (self.author[cands] == a).astype(np.int8),
            "author_reuse": (np.isin(cands, list(hist)) if hist else np.zeros(len(cands), bool)).astype(np.int8),
            "author_family": (np.isin(r, list(roots)) if roots else np.zeros(len(cands), bool)).astype(np.int8),
        }
        for w in self.sp["recent_windows"]:
            f[f"recent{w}"] = self.recent[w][cands].astype(np.int32)
        return f

    # ---------------------------------------------------------------- main loop
    def run(self, out_dir=None, progress=False):
        sp, N = self.sp, self.N
        self.deg = np.zeros(N, np.int64)
        self.quant_deg = np.zeros(N, np.int64)
        self.recent = {w: np.zeros(N, np.int32) for w in sp["recent_windows"]}
        rptr = {w: 0 for w in sp["recent_windows"]}
        self.fam_size = np.zeros(N, np.int32)
        self.fam_recent = np.zeros(N, np.int32)
        bptr = fptr = 0
        self.author_hist, self.author_roots = defaultdict(set), defaultdict(set)
        self.pos_arr = [np.empty(len(x), np.int64) for x in self.mod_nodes]
        self.pos_n = np.zeros(self.M, np.int64)
        deg_e = np.flatnonzero(self.is_deg)
        deg_t, deg_p = self.e_t[deg_e], self.e_parent[deg_e]
        t_start = -np.inf if sp["t_start"] is None else sp["t_start"]
        sampler = self._sample_stratified if sp["sampling"] == "stratified" else self._sample_uniform

        E = len(self.e_t)
        starts = np.r_[0, np.flatnonzero(np.diff(self.e_t)) + 1]
        ends = np.r_[starts[1:], E]
        buf, choices, parts = [], [], []
        self.n_mod_mismatch = 0
        cur_month = None
        it = zip(starts, ends)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(starts), mininterval=30)
        for g0, g1 in it:
            t = int(self.e_t[g0])
            # births strictly before t; family recent window (t - W, t)
            nb = int(np.searchsorted(self.birth, t, "left"))
            if nb > bptr:
                np.add.at(self.fam_size, self.root[bptr:nb], 1)
                np.add.at(self.fam_recent, self.root[bptr:nb], 1)
                bptr = nb
            nf = int(np.searchsorted(self.birth, t - sp["fam_window"] * DAY, "right"))
            if nf > fptr:
                np.subtract.at(self.fam_recent, self.root[fptr:nf], 1)
                fptr = nf
            for w in sp["recent_windows"]:
                kk = int(np.searchsorted(deg_t, t - w * DAY, "right"))
                if kk > rptr[w]:
                    np.subtract.at(self.recent[w], deg_p[rptr[w]:kk], 1)
                    rptr[w] = kk

            if out_dir is not None:
                month = pd.Timestamp(t, unit="s").strftime("%Y-%m")
                if cur_month is not None and month != cur_month and buf:
                    parts.append(self._flush(buf, out_dir, cur_month))
                    buf = []
                cur_month = month

            # record choices against the state before this group
            for e in range(g0, g1):
                if not self.is_choice[e] or t <= t_start:
                    continue
                if sp["sample_frac"] < 1 and self.rng.random() >= sp["sample_frac"]:
                    continue
                c, p = int(self.e_child[e]), int(self.e_parent[e])
                m = int(self.cmod[c])
                if self.cmod[p] != m:
                    self.n_mod_mismatch += 1
                    continue
                k = len(choices)
                cands, lq, st = sampler(m, t, p)
                buf.append(self._features(k, c, p, t, cands, lq, st))
                choices.append((k, c, p, t, m, self.e_rel[e], self.e_w[e], len(cands), int(self.deg[p])))

            # update state with every edge of the group
            for e in range(g0, g1):
                p = int(self.e_parent[e])
                if self.is_deg[e]:
                    if self.deg[p] == 0 and not self.is_hub[p]:
                        m = self.cmod[p]
                        self.pos_arr[m][self.pos_n[m]] = p
                        self.pos_n[m] += 1
                    self.deg[p] += 1
                    for w in sp["recent_windows"]:
                        self.recent[w][p] += 1
                    a = int(self.author[self.e_child[e]])
                    self.author_hist[a].add(p)
                    self.author_roots[a].add(int(self.root[p]))
                if self.is_quant[e]:
                    self.quant_deg[p] += 1

        ch = pd.DataFrame(choices, columns=["choice_id", "child", "parent", "t", "modality", "rel_type",
                                            "weight", "n_cands", "parent_deg"])
        if out_dir is not None:
            if buf:
                parts.append(self._flush(buf, out_dir, cur_month))
            ch.to_parquet(Path(out_dir) / "choices.parquet")
            return ch, None
        return ch, _concat(buf)

    def _flush(self, buf, out_dir, month):
        out = Path(out_dir) / "rows"
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{month}.parquet"
        _concat(buf).to_parquet(f, index=False)
        return f


def _concat(buf):
    if not buf:
        return pd.DataFrame(columns=ROW_COLS)
    return pd.DataFrame({k: np.concatenate([b[k] for b in buf]) for k in buf[0]})
