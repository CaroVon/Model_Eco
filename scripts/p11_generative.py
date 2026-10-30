"""Generative validation and counterfactuals for the fitted choice models (P10 "generative check").

The real sequence of choice events after T0 is kept fixed: who chooses (child model and its author),
when, and in which modality. Only the parent is redrawn. Each choice first draws a mode of the
5-mode latent-class logit (p9_extensions.py --part mixture2), then a parent:
  pop       logit over all alive nodes of the modality: has_deg, log_deg, log(1+age), lrec30, optional
            hub fixed effects (K = 500; hubs enter the pool at birth so a big release can win at once)
  unif      uniform over alive nodes of the modality
  reuse     logit over parents the author used before (same modality): log_deg, lrec30
  self      logit over the author's own earlier models (same modality): log(1+age)
  template  the author's most recent parent (if in the same modality)
Modes whose set is empty for a choice (new author: nothing to reuse, no own model, no template) are
unavailable and the mode weights are renormalised over the available ones.

State starts from the real network at T0 (edges whose child is born at or before T0) and then evolves
with the simulated choices only. Popularity weights use the state at the start of each day; the author
history (personal, template) is updated after every choice. Families are the real root of each parent.

Scenarios: base (fitted mixture), alpha0 (pop log_deg = 0), norecent (lrec30 = 0), nofeedback
(has_deg = log_deg = lrec30 = 0 in pop: degree and recent heat have no effect), noquality (hub FE removed),
pop_only (no reuse / self / template mode), m3 (single-class M3 logit with alpha = 0.93, no other modes).

Fitness spec (--spec fitness, estimated by --fit_fitness): the popularity mode has no degree or heat
feedback; instead each node has a fixed quality q_j and all nodes share an age profile g(age):
u_j = q_j + g(age_j) [+ alpha * log(1 + deg_j) in the counterfactuals fit_a0.25 / fit_a0.5 / fit_a1].
q and g are the MLE of that logit (no degree term) on first author-parent choices without self-choices,
by modality, fitted by alternating updates (the denominator is a convolution over birth days). Nodes
never chosen get q = -inf. The other modes (reuse, self, template) and their weights come from g0_fe.
This asks: given the qualities and release dates, does the system reproduce concentration and turnover,
and how much more concentrated would HF be with preferential attachment of strength alpha?

Static decomposition (--static): no feedback through simulated state. For every sampled real choice
(10% sample, real state at the time of the choice), the probability each mode puts on the top-10 and
top-500 nodes by final degree (importance-sampling estimate over the sampled candidate set), averaged
over choices, under the fitted parameters and under each counterfactual (popularity-mode terms set to 0).

usage: python scripts/p11_generative.py --spec g0_fe --scenario base --seed 0
       python scripts/p11_generative.py --collect
       python scripts/p11_generative.py --static
Writes results/tables/p11_gen/<scenario>_<seed>.npz, then p11_generative.md / .csv and figures.
"""
import argparse
import glob
from collections import defaultdict

import numpy as np
import pandas as pd

from hfpa import paths

DAY = 86400
MAIN = ["finetune", "adapter", "unspecified"]
OUT = paths.TABLES / "p11_gen"
POPF = ["has_deg", "log_deg", "log_age", "lrec30", "legacy"]


MODES = ["pop", "unif", "reuse", "self", "template"]


def load_params(spec):
    z = np.load(paths.TABLES / f"p9_mixture2_{spec}.npz")
    pop = {f: 0.0 for f in POPF}
    pop.update(dict(zip(z["pop_feats"], z["pop_coef"])))
    P = dict(pop=pop, reuse=dict(zip(z["reuse_feats"], z["reuse_coef"])), self=dict(zip(z["self_feats"], z["self_coef"])),
             pi=np.asarray(z["pi"], float), fe=(z["pop_fe_node"], z["pop_fe"]) if "pop_fe" in z else None)
    return P


AGE_EDGES = np.array([0, 1, 3, 7, 14, 30, 60, 90, 180, 365, 730, 100000])


def fit_fitness(n_iter=60):
    n, pre, ch = load_data()
    birth, mod, author = n.birth.to_numpy(np.int64), n.modality_code.to_numpy(np.int64), n.author_code.to_numpy()
    c = ch.sort_values("t", kind="stable")
    first = ~pd.Series(list(zip(author[c.child], c.parent))).duplicated().to_numpy()
    noself = author[c.child.to_numpy()] != author[c.parent.to_numpy()]
    c = c[first & noself]
    d0 = int(birth.min() // DAY)
    T = int(birth.max() // DAY) - d0 + 1
    q = np.full(len(n), -np.inf)
    G = np.zeros(len(AGE_EDGES) - 1)
    lag = np.arange(T)
    abin = np.clip(np.searchsorted(AGE_EDGES, lag - 0.5, "right") - 1, 0, len(G) - 1)  # same rule as the simulator
    for m in np.unique(c.modality):
        cm = c[c.modality == m]
        nt = np.bincount(cm.t.to_numpy() // DAY - d0, minlength=T).astype(float)  # choices per day
        y = pd.Series(1, index=cm.parent.to_numpy()).groupby(level=0).size()
        J = y.index.to_numpy()
        bJ = birth[J] // DAY - d0
        yJ = y.to_numpy(float)
        tday = cm.t.to_numpy() // DAY - d0
        pb = birth[cm.parent.to_numpy()] // DAY - d0
        lg = tday - pb
        obs_bin = np.bincount(abin[lg[lg >= 1]], minlength=len(G)).astype(float)
        g = np.zeros(len(G))
        qJ = np.log(yJ)
        for it in range(n_iter):
            eg = np.exp(g[abin]) * (lag >= 1)  # by lag; only nodes born before the choice day
            S = np.bincount(bJ, np.exp(qJ), T)  # quality mass by birth day
            # Z_t = sum_b S_b eg[t - b] for b < t (alive strictly before the choice day)
            Z = np.convolve(S, eg)[:T]
            ratio = np.divide(nt, Z, out=np.zeros(T), where=Z > 0)
            H = np.correlate(ratio, eg, "full")[T - 1:]  # H_b = sum_t ratio_t eg[t - b]
            qJ = np.log(yJ) - np.log(np.maximum(H[bJ], 1e-300))
            S = np.bincount(bJ, np.exp(qJ), T)
            exp_bin = np.zeros(len(G))
            for k in range(len(G)):  # expected choices at each age bin
                ek = (abin == k) * eg
                exp_bin[k] = (ratio * np.convolve(S, ek)[:T]).sum()
            g = g + np.log(np.maximum(obs_bin, 1e-9) / np.maximum(exp_bin, 1e-300))
            g -= g[-1]
        q[J] = qJ
        G_m = g
        print(m, len(J), "age profile", np.round(G_m, 2), flush=True)
        np.save(OUT / f"fitness_g_{m}.npy", G_m)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "fitness_q.npy", q)


def scenario(spec, name):
    if spec == "fitness":
        P = load_params("g0_fe")
        P["fitness"] = dict(q=np.load(OUT / "fitness_q.npy"),
                            g={int(f.stem.split("_")[-1]): np.load(f) for f in OUT.glob("fitness_g_*.npy")},
                            alpha=float(name.split("_a")[1]) if name.startswith("fit_a") else 0.0)
        P["fe"] = None
        if name == "pop_only":
            pi = P["pi"]
            P["pi"] = np.array([pi[0], pi[1], 0, 0, 0]) / (pi[0] + pi[1])
        return P
    P = load_params(spec)
    pop, pi = P["pop"], P["pi"]
    if name == "alpha0":
        pop["log_deg"] = 0.0
    elif name == "norecent":
        pop["lrec30"] = 0.0
    elif name == "nofeedback":
        pop.update(has_deg=0.0, log_deg=0.0, lrec30=0.0)
    elif name == "noquality":
        P["fe"] = None
    elif name == "pop_only":
        P["pi"] = np.array([pi[0], pi[1], 0, 0, 0]) / (pi[0] + pi[1])
    elif name == "m3":
        lad = pd.read_csv(paths.TABLES / "p5_ladder.csv")
        P["pop"] = {f: 0.0 for f in POPF} | lad[lad.model == "M3"].set_index("param").coef.to_dict()
        P["pi"], P["fe"] = np.array([1.0, 0, 0, 0, 0]), None
    elif name != "base":
        raise ValueError(name)
    return P


def load_data():
    n = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "modality_code", "author_code", "root", "legacy"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    ch = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent", "t", "modality", "parent_deg"])
    E = E[E.rel_type.isin(MAIN)]
    birth = n.birth.to_numpy(np.int64)
    t0 = int(ch.t.min())
    pre = E[birth[E.child_idx.to_numpy()] < t0]
    pre = pre.assign(t=birth[pre.child_idx.to_numpy()]).sort_values("t", kind="stable")
    return n, pre, ch


class Sim:
    def __init__(self, n, pre, ch, P, seed):
        self.rng = np.random.default_rng(seed)
        self.pop, self.reuse_b, self.self_b, self.pi = P["pop"], P["reuse"], P["self"], P["pi"]
        self.fit = P.get("fitness")
        self.birth = n.birth.to_numpy(np.int64)
        self.mod = n.modality_code.to_numpy(np.int64)
        self.author = n.author_code.to_numpy(np.int64)
        self.legacy = n.legacy.to_numpy().astype(float)
        self.bday = self.birth // DAY
        N, self.M = len(n), int(self.mod.max()) + 1
        self.N = N
        self.mod_nodes = [np.flatnonzero(self.mod == m) for m in range(self.M)]  # sorted by birth
        self.mod_bday = [self.bday[x] for x in self.mod_nodes]
        self.d0 = int(self.bday.min())
        nd = int(self.bday.max()) - self.d0 + 2
        # bucket offsets: nodes of modality m born on day d are mod_nodes[m][off[m][d]:off[m][d+1]]
        self.off = [np.searchsorted(b, np.arange(self.d0, self.d0 + nd + 1)) for b in self.mod_bday]
        self.zc = [np.zeros(nd, np.int64) for _ in range(self.M)]  # alive zero-degree nodes by birth day
        self.alive_ptr = np.zeros(self.M, np.int64)
        self.deg = np.zeros(N, np.int64)
        self.rec = np.zeros(N, np.int64)
        self.pos = [[] for _ in range(self.M)]
        self.fe = np.zeros(N)
        self.listed = np.zeros(N, bool)  # node is in the explicit pool (degree > 0 or hub with FE)
        if P["fe"] is not None:
            self.fe[P["fe"][0]] = P["fe"][1]
        self.has_fe = self.fe != 0
        # author's own models, sorted by birth
        o = np.argsort(self.author, kind="stable")
        a_s = self.author[o]
        self.own_start = dict(zip(*np.unique(a_s, return_index=True)))
        self.own_sorted, self.a_s = o, a_s
        self.hist = defaultdict(set)
        self.last = {}
        self.recent_log = defaultdict(list)  # day -> parents that got a child that day
        # initial state: real network before T0
        pp, pc, pt = pre.parent_idx.to_numpy(), pre.child_idx.to_numpy(), pre.t.to_numpy()
        np.add.at(self.deg, pp, 1)
        t0 = int(ch.t.min())
        for d, g in pd.Series(pp[pt > t0 - 30 * DAY]).groupby(pt[pt > t0 - 30 * DAY] // DAY):
            self.recent_log[int(d)] = list(g.to_numpy())
            np.add.at(self.rec, g.to_numpy(), 1)
        for c, p in zip(pc, pp):
            a = int(self.author[c])
            self.hist[a].add(int(p))
            self.last[a] = int(p)
        for m in range(self.M):
            self.pos[m] = list(np.unique(pp[self.mod[pp] == m]))
        self.listed[pp] = True

    def _advance_births(self, day):
        for m in range(self.M):
            k = int(np.searchsorted(self.mod_bday[m], day, "left"))
            if k > self.alive_ptr[m]:
                new = self.mod_nodes[m][self.alive_ptr[m]:k]
                new = new[~self.listed[new]]
                hub = new[self.has_fe[new]]
                self.pos[m].extend(int(x) for x in hub)
                self.listed[hub] = True
                new = new[~self.has_fe[new]]
                np.add.at(self.zc[m], self.bday[new] - self.d0, 1)
                self.alive_ptr[m] = k

    def _pop_dist(self, m, day):
        if self.fit is not None:
            F = self.fit
            if not hasattr(self, "_fpool"):
                q = F["q"]
                self._fpool = [x[np.isfinite(q[x])] for x in self.mod_nodes]
            J = self._fpool[m]
            J = J[self.bday[J] < day]
            if len(J) == 0:
                return None
            age = day - self.bday[J]
            g = F["g"].get(m, np.zeros(len(AGE_EDGES) - 1))
            ab = np.clip(np.searchsorted(AGE_EDGES, age - 0.5, "right") - 1, 0, len(g) - 1)
            u = F["q"][J] + g[ab] + F["alpha"] * np.log1p(self.deg[J])
            return J, np.cumsum(np.exp(u - u.max())), np.zeros(0)
        b = self.pop
        P = np.asarray(self.pos[m], np.int64)
        age = (day * DAY - self.birth[P]) / DAY
        dP = self.deg[P]
        u = (b["has_deg"] * (dP > 0) + b["log_deg"] * np.log(np.maximum(dP, 1)) + b["log_age"] * np.log1p(np.maximum(age, 0))
             + b["lrec30"] * np.log1p(self.rec[P]) + b["legacy"] * self.legacy[P] + self.fe[P])
        nb = day - self.d0
        zc = self.zc[m][:nb]
        bd = np.arange(nb)
        bl = (bd + self.d0) * DAY <= self.t0
        uz = b["log_age"] * np.log1p(day - self.d0 - bd - 0.5) + b["legacy"] * bl
        mx = max(u.max() if len(u) else -np.inf, uz.max() if nb else -np.inf)
        w = np.exp(u - mx)
        wz = zc * np.exp(uz - mx)
        return P, np.cumsum(w), np.cumsum(wz)

    def _draw_pop(self, m, dist):
        P, cw, cz = dist
        tot = (cw[-1] if len(cw) else 0) + (cz[-1] if len(cz) else 0)
        x = self.rng.random() * tot
        if len(cw) and x < cw[-1]:
            return int(P[np.searchsorted(cw, x, "right")])
        x -= cw[-1] if len(cw) else 0
        bk = int(np.searchsorted(cz, x, "right"))
        bk = min(bk, len(cz) - 1)
        nodes = self.mod_nodes[m][self.off[m][bk]:self.off[m][bk + 1]]
        z = nodes[self.deg[nodes] == 0]
        if len(z) == 0:  # bucket emptied within the day by an earlier draw: fall back to any node of the day
            z = nodes
        return int(z[self.rng.integers(len(z))])

    def _personal_set(self, a, m, t):
        s = self.own_start.get(a)
        own = np.empty(0, np.int64)
        if s is not None:
            e = int(np.searchsorted(self.a_s, a, "right"))
            own = self.own_sorted[s:e]
            own = own[(self.birth[own] < t) & (self.mod[own] == m)]
        h = self.hist.get(a)
        reuse = np.fromiter((p for p in h if self.mod[p] == m), np.int64) if h else np.empty(0, np.int64)
        return own, reuse

    def run(self, ch):
        self.t0 = int(ch.t.min())
        child = ch.child.to_numpy(np.int64)
        tt = ch.t.to_numpy(np.int64)
        mm = ch.modality.to_numpy(np.int64)
        days = tt // DAY
        n = len(ch)
        par = np.empty(n, np.int64)
        mode = np.empty(n, np.int8)
        pdeg = np.empty(n, np.int64)
        reused = np.zeros(n, bool)
        starts = np.r_[0, np.flatnonzero(np.diff(days)) + 1]
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            day = int(days[s])
            self._advance_births(day)
            for d in [k for k in self.recent_log if k <= day - 30]:
                np.subtract.at(self.rec, np.asarray(self.recent_log.pop(d), np.int64), 1)
            dists = {}
            today = []
            for i in range(s, e):
                c, t, m = int(child[i]), int(tt[i]), int(mm[i])
                a = int(self.author[c])
                own, reuse = self._personal_set(a, m, t)
                lp = self.last.get(a)
                avail = np.array([True, True, len(reuse) > 0, len(own) > 0, lp is not None and self.mod[lp] == m])
                w = np.where(avail, self.pi, 0)
                k = int(self.rng.choice(5, p=w / w.sum()))
                if k == 0 and m not in dists:
                    dists[m] = self._pop_dist(m, day)
                if k == 0 and dists[m] is None:  # no node of this modality was ever chosen yet: uniform
                    k = 1
                if k == 0:
                    p = self._draw_pop(m, dists[m])
                elif k == 1:
                    p = int(self.mod_nodes[m][self.rng.integers(int(np.searchsorted(self.mod_bday[m], day, "left")))])
                elif k == 2:
                    b = self.reuse_b
                    u = b.get("log_deg", 0) * np.log(np.maximum(self.deg[reuse], 1)) + b.get("lrec30", 0) * np.log1p(self.rec[reuse])
                    pr = np.exp(u - u.max())
                    p = int(reuse[self.rng.choice(len(reuse), p=pr / pr.sum())])
                elif k == 3:
                    u = self.self_b.get("log_age", 0) * np.log1p(np.maximum(t - self.birth[own], 0) / DAY)
                    pr = np.exp(u - u.max())
                    p = int(own[self.rng.choice(len(own), p=pr / pr.sum())])
                else:
                    p = lp
                par[i], mode[i], pdeg[i] = p, k, self.deg[p]
                reused[i] = p in self.hist.get(a, ())
                self.hist[a].add(p)
                self.last[a] = p
                today.append(p)
            # end of day: degree update
            td = np.asarray(today, np.int64)
            newpos = np.unique(td[~self.listed[td]])
            np.add.at(self.deg, td, 1)
            np.add.at(self.rec, td, 1)
            self.recent_log[day] = today
            self.listed[newpos] = True
            for p in newpos:
                m = int(self.mod[p])
                self.pos[m].append(int(p))
                if self.birth[p] < day * DAY:  # was counted as an alive zero-degree node
                    self.zc[m][self.bday[p] - self.d0] -= 1
        return dict(parent=par, mode=mode, parent_deg=pdeg, reused=reused)


# ---------------------------------------------------------------- metrics
def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())) if n and x.sum() > 0 else np.nan


def metrics(parent, parent_deg, reused, child, author, root, deg0, t, birth=None):
    cnt = np.bincount(parent, minlength=len(deg0))
    final = deg0 + cnt
    top = np.sort(cnt)[::-1]
    fam = np.bincount(root[parent], minlength=len(deg0))
    ftop = np.sort(fam)[::-1]
    n = len(parent)
    out = dict(top1_share=top[0] / n, top10_share=top[:10].sum() / n, top100_share=top[:100].sum() / n,
               fam_top10_share=ftop[:10].sum() / n, fam_top100_share=ftop[:100].sum() / n,
               gini_final_deg=gini(final[final > 0]), n_parents=int((cnt > 0).sum()),
               share_new_parent=float((parent_deg == 0).mean()), share_reuse=float(reused.mean()),
               share_self=float((author[parent] == author[child]).mean()), max_final_deg=int(final.max()))
    if birth is not None:
        out["share_parent_age_lt180d"] = float(((t - birth[parent]) < 180 * 86400).mean())
        mon_ = pd.to_datetime(t, unit="s").to_period("M")
        top1 = pd.DataFrame({"m": mon_, "p": parent}).groupby(["m", "p"]).size().groupby(level=0).idxmax()
        out["n_distinct_monthly_top1"] = int(len({x[1] for x in top1}))
    # monthly top-10 family share
    mon = pd.to_datetime(t, unit="s").to_period("M")
    df = pd.DataFrame({"m": mon, "f": root[parent]})
    g = df.groupby(["m", "f"]).size()
    tot = g.groupby(level=0).sum()
    top10 = g.groupby(level=0, group_keys=False).apply(lambda s: s.nlargest(10).sum())
    return out, (top10 / tot), final, cnt


def real_reuse(child, parent, author):
    seen, out = set(), np.zeros(len(parent), bool)
    for i, (c, p) in enumerate(zip(child, parent)):
        k = (int(author[c]), int(p))
        out[i] = k in seen
        seen.add(k)
    return out


def collect():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hfpa.viz.descriptive import GRID, INK2, SERIES, _save
    n, pre, ch = load_data()
    author, root = n.author_code.to_numpy(), n.root.to_numpy()
    deg0 = np.bincount(pre.parent_idx.to_numpy(), minlength=len(n))
    child, t = ch.child.to_numpy(), ch.t.to_numpy()
    rows, series, finals = [], {}, {}
    rm, rs, rf, rc = metrics(ch.parent.to_numpy(), ch.parent_deg.to_numpy(), real_reuse(child, ch.parent.to_numpy(), author),
                             child, author, root, deg0, t, n.birth.to_numpy())
    rows.append(dict(scenario="real", seed=-1, **rm))
    series["real"], finals["real"] = rs, rf
    real_top = set(np.argsort(-rc)[:10])
    for f in sorted(glob.glob(str(OUT / "*.npz"))):
        z = np.load(f)
        sc, seed = f'{z["spec"]}/{z["scenario"]}', int(z["seed"])
        m, s, fin, cnt = metrics(z["parent"], z["parent_deg"], z["reused"], child, author, root, deg0, t, n.birth.to_numpy())
        m["real_top10_recovered"] = len(real_top & set(np.argsort(-cnt)[:10]))
        m.update({f"mode_{k}": float((z["mode"] == i).mean()) for i, k in enumerate(MODES)})
        rows.append(dict(scenario=sc, seed=seed, **m))
        if seed == 0:
            series[sc], finals[sc] = s, fin
    df = pd.DataFrame(rows)
    df.to_csv(paths.TABLES / "p11_generative.csv", index=False)
    sc_order = ["base", "noquality", "alpha0", "norecent", "nofeedback", "pop_only", "m3"]
    order = (["real"] + [f"fitness/{x}" for x in ("fit_a0", "fit_a0.25", "fit_a0.5", "fit_a1", "pop_only")]
             + [f"{sp}/{x}" for sp in ("g0_fe", "g0_nofe", "g0.5_fe", "g0.5_nofe") for x in sc_order])
    cols = ["top1_share", "top10_share", "top100_share", "fam_top10_share", "gini_final_deg", "n_parents",
            "share_new_parent", "share_reuse", "share_self", "max_final_deg", "real_top10_recovered",
            "share_parent_age_lt180d", "n_distinct_monthly_top1"]
    s = df.groupby("scenario")[cols].agg(["mean", "std"])
    mean = s.xs("mean", axis=1, level=1).reindex([o for o in order if o in s.index])
    sd = s.xs("std", axis=1, level=1).reindex(mean.index)
    md = ["# P11 generative check: real network vs networks regenerated from the fitted choice models", "",
          f"Same {len(ch):,} choice events after T0 (who, when, modality); parent redrawn. Mean over seeds (sd below).", "",
          mean.to_markdown(floatfmt=".3f"), "", "sd across seeds:", "", sd.to_markdown(floatfmt=".3f"), "",
          "Mode shares drawn:", "", df.groupby("scenario")[[c for c in df if c.startswith("mode_")]].mean().to_markdown(floatfmt=".3f")]
    (paths.TABLES / "p11_generative.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    lab = {"real": "real network", "fitness/fit_a0": "quality + lifecycle, α = 0", "fitness/fit_a0.5": "same, α = 0.5",
           "fitness/fit_a1": "same, α = 1", "fitness/pop_only": "α = 0, no personal modes", "g0_fe/base": "feedback logit (hub FE)"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, k in enumerate([o for o in lab if o in series]):
        st = dict(color="#222222", lw=2.4) if k == "real" else dict(color=SERIES[(i - 1) % len(SERIES)], lw=1.6)
        x = series[k].index.to_timestamp()
        ax[0].plot(x, series[k].to_numpy(), label=lab[k], **st)
        f = np.sort(finals[k][finals[k] > 0])
        ax[1].loglog(f, 1 - np.arange(len(f)) / len(f), label=lab[k], **st)
    ax[0].set(ylabel="top-10 families' share of new choices", title="Monthly concentration")
    ax[1].set(xlabel="final degree", ylabel="P(degree ≥ x)", title="Degree distribution (nodes with ≥ 1 child)")
    for a_ in ax:
        a_.grid(color=GRID, lw=0.6)
    ax[1].legend(frameon=False, fontsize=8)
    _save(fig, "p11_generative")


def static():
    import importlib.util
    sp = importlib.util.spec_from_file_location("p9", paths.REPO / "scripts" / "p9_extensions.py")
    P9 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(P9)
    rows, ch, final_deg, nodes = P9.P5.load()
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    nn = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code"])
    lp = P9.last_parent(ch.reset_index(drop=True), nn, E)
    cid = rows.choice_id.to_numpy()
    _, start, cnt = np.unique(cid, return_index=True, return_counts=True)
    C = len(cnt)
    g = np.repeat(np.arange(C), cnt)
    fe = rows.fe.to_numpy()
    top = {"top10": (fe >= 1) & (fe <= 10), "top500": fe >= 1}
    rank = pd.Series(fe[fe > 0], index=rows.cand_id.to_numpy()[fe > 0]).groupby(level=0).first()
    lp_rank = rank.reindex(lp).fillna(0).to_numpy()
    lq = rows.log_q.to_numpy(np.float64)
    chosen = rows.chosen.to_numpy() == 1
    t = pd.to_datetime(ch.t.to_numpy(), unit="s")
    out = []

    def mode_share(u, sel):
        """per choice: (available, share on top10, share on top500) for a logit on rows in sel."""
        w = np.where(sel, np.exp(np.clip(u - lq, -700, 700) - 0), 0.0)
        den = np.bincount(g, w, C)
        res = {"avail": den > 0}
        for k, m in top.items():
            res[k] = np.divide(np.bincount(g, w * m, C), den, out=np.zeros(C), where=den > 0)
        return res

    obs = {k: np.bincount(g, chosen & m, C) for k, m in top.items()}
    out.append(dict(spec="real", scenario="observed", **{k: v.mean() for k, v in obs.items()}))
    for spec in ("g0_fe", "g0_nofe", "g0.5_fe", "g0.5_nofe"):
        for sc in ("base", "alpha0", "norecent", "nofeedback", "noquality", "pop_only"):
            if sc == "noquality" and not spec.endswith("_fe"):
                continue
            P = scenario(spec, sc)
            b = P["pop"]
            u = sum(b.get(f, 0.0) * rows[f].to_numpy(np.float64) for f in ["has_deg", "log_deg", "log_age", "lrec30", "legacy"])
            if P["fe"] is not None:
                fev = np.zeros(int(fe.max()) + 1)
                node_fe = dict(zip(P["fe"][0], P["fe"][1]))
                fev[rank.index.map(rank).to_numpy()] = [node_fe.get(x, 0.0) for x in rank.index]
                u = u + fev[fe]
            pop = mode_share(u - u.max(), np.ones(len(rows), bool))
            br = P["reuse"]
            ur = sum(br.get(f, 0.0) * rows[f].to_numpy(np.float64) for f in ["log_deg", "lrec30"])
            reuse = mode_share(ur, rows.author_reuse.to_numpy() > 0)
            us = P["self"].get("log_age", 0.0) * rows.log_age.to_numpy(np.float64)
            self_ = mode_share(us, rows.same_author.to_numpy() > 0)
            tmpl = {"avail": np.isfinite(lp), "top10": ((lp_rank >= 1) & (lp_rank <= 10)).astype(float), "top500": (lp_rank >= 1).astype(float)}
            modes = [pop, {"avail": np.ones(C, bool), "top10": np.zeros(C), "top500": np.zeros(C)}, reuse, self_, tmpl]
            A = np.stack([m["avail"] for m in modes], 1) * P["pi"][None, :]
            A = A / A.sum(1, keepdims=True)
            row = dict(spec=spec, scenario=sc)
            for k in top:
                v = (A * np.stack([m[k] for m in modes], 1)).sum(1)
                row[k] = v.mean()
                row[f"{k}_pop_mode"] = pop[k].mean()
                for yr in (2023, 2024, 2025):
                    row[f"{k}_{yr}"] = v[t.year == yr].mean()
            row["mode_weight_pop"] = A[:, 0].mean()
            out.append(row)
            print(row, flush=True)
    for yr in (2023, 2024, 2025):
        for k in top:
            out[0][f"{k}_{yr}"] = obs[k][t.year == yr].mean()
    df = pd.DataFrame(out)
    df.to_csv(paths.TABLES / "p11_static.csv", index=False)
    md = ["# P11 static decomposition: expected share of choices on the largest nodes (real state, no feedback)", "",
          "top10 / top500 = nodes ranked 1-10 / 1-500 by final degree (the hub stratum, sampled with q = 1). "
          "Expected share under each fitted mixture, with popularity-mode terms switched off one at a time; "
          "other modes unchanged. 10% choice sample, all choices weighted equally.", "",
          df.to_markdown(index=False, floatfmt=".3f")]
    (paths.TABLES / "p11_static.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="g0_fe")
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--fit_fitness", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N choices (smoke test)")
    a = ap.parse_args()
    if a.collect:
        collect()
    elif a.static:
        static()
    elif a.fit_fitness:
        fit_fitness()
    else:
        n, pre, ch = load_data()
        if a.limit:
            ch = ch.iloc[: a.limit]
        P = scenario(a.spec, a.scenario)
        print(a.spec, a.scenario, P["pop"], P["reuse"], P["self"], P["pi"].round(4), "FE" if P["fe"] is not None else "", flush=True)
        import time
        t_ = time.time()
        r = Sim(n, pre, ch, P, a.seed).run(ch)
        print(f"done in {time.time() - t_:.0f}s", flush=True)
        if not a.limit:
            OUT.mkdir(exist_ok=True)
            np.savez_compressed(OUT / f"{a.spec}_{a.scenario}_{a.seed}.npz", spec=a.spec, scenario=a.scenario, seed=a.seed, **r)
