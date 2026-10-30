"""Follow-ups from Overgoor, Benson & Ugander (2020), on the 10% choice sample (same setup as p5_models.py).

parts:
  iia_sim  Hausman-McFadden IIA test on synthetic networks (P0 generator): pure PA should pass,
           pure fitness (omitted node quality) should fail. Validates the test before using it.
  iia      Same test on the real data, M3 specification: full candidate sets vs subsets of
           alternatives (hubs only, non-hubs only, degree > 0 only, outside the author's own history).
           Under a correct conditional logit any subset gives a consistent estimate.
  mixture  HF version of the paper's (r,p)-model (latent-class logit, EM): modes
             pop      global popularity: has_deg, log_deg, log_age, lrec30 over all candidates
             unif     global uniform over all candidates
             personal author's own consideration set (own models, parents and families used before):
                      author_reuse, same_author, log_deg
             template the author's most recent parent (point mass)
           fitted as {pop, unif}, {pop, unif, personal}, {pop, unif, personal, template}; responsibilities
           summarised by uploader type, modality and half-year.
  twohop   "friend-of-friend" analogue on the author-model bipartite graph: number of other authors who
           share a niche (non-hub) parent with the chooser's author and have used candidate j (monthly
           snapshots, edges strictly before the month). Added to M3 and M7.
Mode probabilities use the full-set estimate log P(j) = u_j - log sum_k exp(u_k - log q_k) (the
importance-sampling denominator), so modes with different candidate sets are comparable in the mixture.

usage: python scripts/p9_extensions.py --part iia
Writes results/tables/p9_<part>*.csv / .md.
"""
import argparse
import importlib.util

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import chi2

from hfpa import paths
from hfpa.models.clogit import CLogit, to_padded

_spec = importlib.util.spec_from_file_location("p5m", paths.REPO / "scripts" / "p5_models.py")
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)
MAIN = ["finetune", "adapter", "unspecified"]
HUB = 0


# ---------------------------------------------------------------- IIA
def hausman(full, restr, names):
    i_f = [full["names"].index(n) for n in names]
    i_r = [restr["names"].index(n) for n in names]
    d = restr["coef"][i_r] - full["coef"][i_f]
    Vf = full["cov_cl"][np.ix_(i_f, i_f)]
    Vr = restr["cov_cl"][np.ix_(i_r, i_r)]
    Vd = Vr - Vf
    ev = np.linalg.eigvalsh(Vd)
    H = float(d @ np.linalg.pinv(Vd) @ d)
    df = int((ev > 1e-12 * max(abs(ev).max(), 1e-300)).sum())
    Hc = float(d @ np.linalg.inv(Vr) @ d)  # conservative: ignores the efficiency gain of the full model
    return dict(H=H, df=df, p=chi2.sf(H, max(df, 1)), H_conservative=Hc, p_conservative=chi2.sf(Hc, len(names)),
                Vd_posdef=bool(ev.min() > 0), **{f"d_{n}": v for n, v in zip(names, d)})


def subsets(rows):
    """name -> (row mask, parameters that still vary in the subset)."""
    st = rows.stratum.to_numpy()
    out = {"hubs only": (st == HUB, ["log_deg", "log_age", "legacy"]),
           "non-hubs only": (st != HUB, ["has_deg", "log_deg", "log_age", "legacy"]),
           "degree > 0 only": (rows.deg.to_numpy() > 0, ["log_deg", "log_age", "legacy"])}
    if "author_reuse" in rows:
        own = (rows.author_reuse.to_numpy() + rows.author_family.to_numpy() + rows.same_author.to_numpy()) > 0
        out["outside author's own history"] = (~own, ["has_deg", "log_deg", "log_age", "legacy"])
    return out


def restrict(rows, mask, min_size=2):
    """Keep alternatives in mask, and only choices whose chosen alternative is kept (and >= min_size left)."""
    r = rows[mask]
    keep = r.loc[r.chosen == 1, "choice_id"]
    r = r[r.choice_id.isin(keep)]
    n = r.groupby("choice_id").size()
    return r[r.choice_id.isin(n.index[n >= min_size])]


def iia_table(rows, ch, base, label):
    out = []
    full = P5.fit(rows, ch, base)
    for name, (mask, names) in subsets(rows).items():
        r = restrict(rows, mask)
        feats = [f for f in base if f in names]
        res = P5.fit(r, ch, feats)
        h = hausman(full, res, feats)
        out.append(dict(data=label, subset=name, n_choices=r.choice_id.nunique(), share_of_choices=r.choice_id.nunique() / rows.choice_id.nunique(),
                        alpha_full=full["coef"][full["names"].index("log_deg")],
                        alpha_subset=res["coef"][res["names"].index("log_deg")],
                        se_subset=res["se_cl"][res["names"].index("log_deg")], **h))
        print(out[-1], flush=True)
    return out


def part_iia_sim(**_):
    from hfpa.replay.replay import Replay
    from hfpa.sim.synthetic import generate
    out = []
    for kind, kw in [("pure PA (alpha = 1)", dict(kind="pa", alpha=1.0)), ("pure fitness (alpha = 0)", dict(kind="fitness"))]:
        nodes, edges, _ = generate(n=50_000, seed=42, **kw)
        ch, rows = Replay(nodes, edges, {"hub_k": 200, "s1": 24, "s0": 24, "seed": 42}).run()
        rows["log_age"] = np.log1p(rows.age_days).astype(np.float32)
        rows["legacy"] = np.float32(0)
        chs = pd.DataFrame({"choice_id": ch.choice_id, "n_pair": 1, "author": ch.choice_id}).set_index("choice_id", drop=False)
        out += iia_table(rows.drop(columns=["author_reuse", "author_family", "same_author"]), chs,
                         ["has_deg", "log_deg", "log_age"], kind)
    for o in out:
        o.pop("d_legacy", None)
    P5.write(pd.DataFrame(out), "p9_iia_sim", "IIA (Hausman-McFadden) test on synthetic networks")


def part_iia(rows, ch, **_):
    tr, _ = P5.split(rows, ch)
    P5.write(pd.DataFrame(iia_table(tr, ch, P5.M["M3"], "HF, M3, gamma = 0.5")), "p9_iia",
             "IIA (Hausman-McFadden) test, real data: full candidate sets vs subsets of alternatives")


# ---------------------------------------------------------------- mixture
POP = ["has_deg", "log_deg", "log_age", "lrec30"]
PERS = ["author_reuse", "same_author", "log_deg"]


def last_parent(ch, nodes, E):
    """Most recent parent chosen by the same author strictly before the child's birth (any main edge)."""
    m = E[E.rel_type.isin(MAIN)]
    author, birth = nodes.author_code.to_numpy(), nodes.birth.to_numpy()
    h = pd.DataFrame({"author": author[m.child_idx], "t": birth[m.child_idx], "lp": m.parent_idx.to_numpy()})
    h = h.sort_values("t", kind="stable")
    q = pd.DataFrame({"choice_id": ch.choice_id.to_numpy(), "author": author[ch.child.to_numpy()], "t": ch.t.to_numpy()})
    q = q.sort_values("t", kind="stable")
    r = pd.merge_asof(q, h, on="t", by="author", allow_exact_matches=False, direction="backward")
    return r.set_index("choice_id").lp.reindex(ch.choice_id).to_numpy()


class Mode:
    """One latent class: a conditional logit on its own candidate set (or a point mass / uniform)."""

    def __init__(self, name, rows, feats, cid):
        self.name, self.feats, self.cid = name, feats, cid
        self.model, self.D, self.idx = None, None, None
        if rows is None:
            return
        D = to_padded(rows, feats if feats else ["chosen"], weight=None)
        if not feats:  # uniform: no parameters
            D["X"] = np.zeros(D["X"].shape[:2] + (0,))
        self.D = D
        ids = np.unique(rows.choice_id.to_numpy())
        self.idx = np.searchsorted(cid, ids)  # position of each of this mode's choices among all choices
        self.lq_ch = D["lq"][np.arange(len(D["chosen"])), D["chosen"]]
        if feats:
            self.model = CLogit(len(feats))

    def logp(self, n):
        """Full-set log probability of the chosen alternative for all n choices (-inf if not in the set)."""
        out = np.full(n, -np.inf)
        if self.D is None:
            return out
        if self.model is None:
            lq = self.D["lq"]
            lse = np.log(np.where(self.D["mask"], np.exp(-lq), 0).sum(1))
            out[self.idx] = -lse
            return out
        vals = []
        with torch.no_grad():
            for b in self.model._batches(self.D, 20000):
                vals.append(self.model._ll_batch(b).cpu().numpy())
        out[self.idx] = np.concatenate(vals) + self.lq_ch
        return out

    def mstep(self, w):
        if self.model is None:
            return
        self.D["w"] = w[self.idx]
        prev = self.model.p.detach().clone()
        self.model = CLogit(len(self.feats))
        with torch.no_grad():
            self.model.p.copy_(prev)
        self.res = self.model.fit(self.D, max_iter=50)


def em(modes, w, n, n_iter=25, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    M = len(modes)
    pi = np.full(M, 1 / M) if seed == 0 else rng.dirichlet(np.ones(M))
    hist = []
    for it in range(n_iter):
        L = np.stack([m.logp(n) for m in modes], 1) + np.log(pi)[None, :]
        lse = np.logaddexp.reduce(L, 1)
        ll = float((w * lse).sum())
        hist.append(ll)
        g = np.exp(L - lse[:, None])
        pi = (w[:, None] * g).sum(0) / w.sum()
        for k, m in enumerate(modes):
            m.mstep(w * g[:, k])
        if it > 2 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-1]):
            break
    L = np.stack([m.logp(n) for m in modes], 1) + np.log(pi)[None, :]
    lse = np.logaddexp.reduce(L, 1)
    return dict(pi=pi, ll=float((w * lse).sum()), gamma=np.exp(L - lse[:, None]), iters=len(hist), hist=hist)


def build_modes(rows, ch, cid, which):
    modes = []
    if "pop" in which:
        modes.append(Mode("pop", rows, POP, cid))
    if "unif" in which:
        modes.append(Mode("unif", rows, [], cid))
    if "personal" in which:
        own = (rows.author_reuse + rows.author_family + rows.same_author) > 0
        modes.append(Mode("personal", restrict(rows, own.to_numpy(), min_size=1), PERS, cid))
    if "template" in which:
        lp = ch.last_parent.loc[cid].to_numpy()
        chosen = rows.loc[rows.chosen == 1].set_index("choice_id").cand_id.loc[cid].to_numpy()
        hit = np.where(np.isfinite(lp) & (lp == chosen), 0.0, -np.inf)
        t = Mode("template", None, [], cid)
        t.logp = lambda n, hit=hit: hit
        modes.append(t)
    return modes


def part_mixture(rows, ch, nodes, **_):
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    nn = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code"])
    ch["last_parent"] = last_parent(ch.reset_index(drop=True), nn, E)
    tr, ho = P5.split(rows, ch)
    out, resp = [], None
    for sets in (("pop", "unif"), ("pop", "unif", "personal"), ("pop", "unif", "personal", "template")):
        best = None
        for seed in (0, 1):
            cid = np.unique(tr.choice_id.to_numpy())
            w = ch.n_pair.loc[cid].to_numpy(float) ** -P5.GAMMA
            modes = build_modes(tr, ch, cid, sets)
            for m in modes:  # initialise each conditional-logit mode on its own data
                m.mstep(np.ones(len(cid)))
            r = em(modes, w, len(cid), seed=seed)
            print(sets, seed, r["pi"].round(3), round(r["ll"], 1), r["iters"], flush=True)
            if best is None or r["ll"] > best[0]["ll"]:
                best = (r, modes, cid, w)
        r, modes, cid, w = best
        # holdout
        hcid = np.unique(ho.choice_id.to_numpy())
        hm = build_modes(ho, ch, hcid, sets)
        for a_, b_ in zip(hm, modes):
            if b_.model is not None:
                a_.model = b_.model
        Lh = np.stack([m.logp(len(hcid)) for m in hm], 1) + np.log(r["pi"])[None, :]
        hll = float(np.logaddexp.reduce(Lh, 1).mean())
        k = sum(len(m.feats) for m in modes) + len(modes) - 1
        row = dict(modes="+".join(sets), ll=r["ll"], k=k, bic=k * np.log(len(cid)) - 2 * r["ll"], holdout_mean_ll=hll,
                   em_iters=r["iters"], **{f"pi_{m.name}": p for m, p in zip(modes, r["pi"])})
        for m in modes:
            if m.model is not None:
                for f, c, s in zip(m.feats, m.res["coef"], m.res["se"]):
                    row[f"{m.name}:{f}"] = c
                    row[f"{m.name}:{f}_se"] = s
        out.append(row)
        if len(sets) == 4:
            resp = pd.DataFrame(r["gamma"], columns=[m.name for m in modes], index=cid)
            resp["w"] = w
    df = pd.DataFrame(out)
    P5.write(df, "p9_mixture", "HF (r,p)-style latent-class logit (EM, gamma = 0.5 weights)")
    # who follows popularity? weighted mean responsibilities by chooser characteristics
    c = ch.loc[resp.index]
    exp_t = pd.qcut(c.child_lexp, 3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
    dt = pd.to_datetime(c.t, unit="s")
    half = pd.Series(dt.dt.year.astype(str).to_numpy() + np.where(dt.dt.month <= 6, "H1", "H2"), index=c.index)
    rows_ = []
    for cut, key in [("all", pd.Series("all", index=c.index)), ("child author is major org", c.child_major.map({1.0: "yes", 0.0: "no"})),
                     ("author experience tercile", exp_t), ("modality", c.modality_name), ("half-year", half)]:
        for v, idx in key.groupby(key).groups.items():
            g = resp.loc[idx]
            ww = g.w.to_numpy()
            rows_.append(dict(cut=cut, value=v, n=len(g), **{m: float((g[m] * ww).sum() / ww.sum()) for m in resp.columns if m != "w"}))
    P5.write(pd.DataFrame(rows_), "p9_mixture_responsibilities", "Mean responsibility of each mode (4-mode model), by chooser group")


# ---------------------------------------------------------------- two-hop
def part_twohop(rows, ch, final_deg, **_):
    nn = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    m = E[E.rel_type.isin(MAIN)]
    author, birth = nn.author_code.to_numpy(), nn.birth.to_numpy()
    ea, ep, et = author[m.child_idx.to_numpy()], m.parent_idx.to_numpy(), birth[m.child_idx.to_numpy()]
    hubs = np.argsort(-final_deg, kind="stable")[:500]
    niche = ~np.isin(ep, hubs)
    NA, NN = int(author.max()) + 1, len(nn)
    ch_month = pd.to_datetime(ch.t, unit="s").dt.to_period("M")
    rows_month = ch_month.loc[rows.choice_id].to_numpy()
    rows_author = author[ch.child.loc[rows.choice_id].to_numpy()]
    co = np.zeros(len(rows), np.float32)
    for mon in np.unique(rows_month):
        t0 = int(mon.to_timestamp().tz_localize("UTC").timestamp())
        pre = et < t0
        A = sp.csr_matrix((np.ones(pre.sum()), (ea[pre], ep[pre])), shape=(NA, NN))
        A.data[:] = 1.0
        A.sum_duplicates()
        A.data[:] = 1.0
        Ant = sp.csr_matrix((np.ones((pre & niche).sum()), (ep[pre & niche], ea[pre & niche])), shape=(NN, NA))
        sel = np.flatnonzero(rows_month == mon)
        for a in np.unique(rows_author[sel]):
            S = A[a].indices
            S = S[~np.isin(S, hubs)]
            if len(S) == 0:
                continue
            users = np.unique(Ant[S].indices)
            users = users[users != a]
            if len(users) == 0:
                continue
            v = np.asarray(A[users].sum(0)).ravel()  # distinct co-users who used each node
            ii = sel[rows_author[sel] == a]
            co[ii] = v[rows.cand_id.to_numpy()[ii]]
        print(mon, flush=True)
    rows["lco2"] = np.log1p(co).astype(np.float32)
    rows["is_co2"] = (co > 0).astype(np.float32)
    tr, ho = P5.split(rows, ch)
    out = []
    for base in ("M3", "M7"):
        fe = P5.FE.get(base)
        out += P5.records(base, P5.fit(tr, ch, P5.M[base], fe, hold=ho), keep=["log_deg", "lrec30", "author_reuse", "author_family"])
        for extra in (["is_co2"], ["lco2"]):
            out += P5.records(f"{base} + {extra[0]}", P5.fit(tr, ch, P5.M[base] + extra, fe, hold=ho),
                              keep=["log_deg", "lrec30", "author_reuse", "author_family"] + extra)
    df = pd.DataFrame(out)
    df["share_rows_co2"] = float((co > 0).mean())
    P5.write(df, "p9_twohop", "Two-hop co-use feature on the author-model graph (friend-of-friend analogue)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["iia_sim", "iia", "mixture", "twohop"])
    a = ap.parse_args()
    if a.part == "iia_sim":
        part_iia_sim()
    else:
        rows, ch, final_deg, nodes = P5.load()
        globals()[f"part_{a.part}"](rows=rows, ch=ch, final_deg=final_deg, nodes=nodes)
    print("done")
