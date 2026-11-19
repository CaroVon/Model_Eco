"""v2 stage 6 (C): latent-class decomposition with availability renormalisation (hfpa.models.latent).

Modes (one per mechanism; local search is part of popularity in v2):
  pop       logit over all candidates, popularity group features
  qual      logit over all candidates, bench_z (--fe: plus the 500 ex-ante FE, upper bound)
  match     logit over all candidates, demand-match features; available only if the choice has a demand signal
  unif      uniform over all candidates
  self      uniform over the author's own models in the set (available if there is one)
  reuse     logit over the author's earlier parents H_k (decisions sample only; log_deg, lrec30)
  template  point mass on the author's last parent (decisions sample only)
Samples: planA (first decisions; H_k removed, so reuse / template never apply) and decisions (every 24h decision).
--fit       EM on the training choices; shares = mean responsibilities; coefficients per mode.
--sim       semi-synthetic check: choices redrawn from the fitted mixture (renormalised mode draw, then Gumbel-max
            on u - log q within the mode's set), EM re-run; absolute and relative errors of the recovered shares.
usage: sbatch slurm/gpu.sbatch scripts/v2_mixture.py --sample planA --sim --fit
"""
import argparse
import time

import numpy as np
import pandas as pd

from hfpa.features.groups import GROUPS, SIGNAL
from hfpa.models.clogit import to_buckets
from hfpa.models.latent import LogitMode, PointMode, draw_modes, em
from hfpa.v2data import load_rows, split, write

SPEC = "v2_pps"
T0 = time.time()
REUSE = ["log_deg", "lrec30"]


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def mode_list(sample):
    return ["pop", "qual", "match", "unif", "self"] + (["reuse", "template"] if sample == "decisions" else [])


def build(rows, sample, fe=False):
    """Modes and availability matrix for the training rows (sorted by choice_id)."""
    cid = np.unique(rows.choice_id.to_numpy())
    n = len(cid)
    sig = (rows.groupby("choice_id")[SIGNAL].max().max(1) > 0).reindex(cid).to_numpy()
    feats = lambda g: [f for f in GROUPS[g] if f in rows and rows[f].std() > 0]
    names = mode_list(sample)
    modes, avail = [], np.zeros((n, len(names)), bool)

    def restricted(mask):
        r = rows[mask]
        has = np.isin(cid, np.unique(r.choice_id))
        ok = r.groupby("choice_id").chosen.max()
        r = r[r.choice_id.isin(ok.index[ok == 1])].reset_index(drop=True)
        return r, has

    for k, name in enumerate(names):
        if name in ("pop", "qual", "unif"):
            f = {"pop": feats("pop"), "qual": feats("qual"), "unif": []}[name]
            D = to_buckets(rows, f, fe_col="fe" if (fe and name == "qual") else None)
            modes.append(LogitMode(name, D, np.arange(n), n_fe=500 if (fe and name == "qual") else 0))
            avail[:, k] = True
        elif name == "match":
            r = rows[np.isin(rows.choice_id.to_numpy(), cid[sig])].reset_index(drop=True)
            D = to_buckets(r, feats("match"))
            modes.append(LogitMode(name, D, np.searchsorted(cid, D.choice_ids)))
            avail[:, k] = sig
        elif name in ("self", "reuse"):
            col = "same_author" if name == "self" else "author_reuse"
            r, has = restricted(rows[col].to_numpy() > 0)
            avail[:, k] = has
            if len(r) == 0:
                modes.append(LogitMode(name, None, []))
                continue
            D = to_buckets(r, [] if name == "self" else [x for x in REUSE if r[x].std() > 0])
            modes.append(LogitMode(name, D, np.searchsorted(cid, D.choice_ids)))
        else:  # template
            hit = rows[rows.chosen == 1].set_index("choice_id").is_last.reindex(cid).fillna(0).to_numpy() > 0
            avail[:, k] = (rows.groupby("choice_id").is_last.max().reindex(cid).to_numpy() > 0)
            modes.append(PointMode(name, hit))
    log(f"modes built; availability {dict(zip(names, avail.mean(0).round(3)))}")
    return modes, avail, cid


def fit(rows, sample, fe, seeds=(0, 1)):
    best = None
    for s in seeds:
        modes, avail, cid = build(rows, sample, fe)
        r = em(modes, avail, np.ones(len(cid)), n_iter=40, seed=s, log=log)
        if best is None or r["ll"] > best[0]["ll"]:
            best = (r, modes, avail, cid)
    return best


def mode_u(mode, r):
    if mode.model is None:
        return -r.log_q.to_numpy(float)
    p = mode.model.p.detach().cpu().numpy()
    u = r[mode.feats].to_numpy(float) @ p[: len(mode.feats)] - r.log_q.to_numpy(float)
    if mode.n_fe:
        u = u + np.r_[0.0, p[len(mode.feats):]][r.fe.to_numpy()]
    return u


def simulate(rows, modes, avail, cid, pi, seed, sample):
    rng = np.random.default_rng(seed)
    names = [m.name for m in modes]
    pick = draw_modes(avail, pi, rng)
    sig = (rows.groupby("choice_id")[SIGNAL].max().max(1) > 0).reindex(cid).to_numpy()
    rid = rows.choice_id.to_numpy()
    pos_c = np.searchsorted(cid, rid)
    chosen_row = np.full(len(cid), -1)
    for k, m in enumerate(modes):
        want = pick == k
        if not want.any():
            continue
        mask = want[pos_c]
        if m.name == "match":
            mask &= sig[pos_c]
        elif m.name == "self":
            mask &= rows.same_author.to_numpy() > 0
        elif m.name == "reuse":
            mask &= rows.author_reuse.to_numpy() > 0
        elif m.name == "template":
            mask &= rows.is_last.to_numpy() > 0
        idx = np.flatnonzero(mask)
        r = rows.iloc[idx]
        if m.name == "template":
            u = np.zeros(len(r))
        elif m.name in ("unif", "self") or m.model is None:
            u = -r.log_q.to_numpy(float)
        else:
            u = mode_u(m, r)
        u = u + rng.gumbel(size=len(r))
        s = pd.Series(u, index=idx).groupby(rid[idx]).idxmax()
        chosen_row[np.searchsorted(cid, s.index.to_numpy())] = s.to_numpy()
    sim = rows.copy()
    sim["chosen"] = 0
    sim.iloc[chosen_row[chosen_row >= 0], sim.columns.get_loc("chosen")] = 1
    keep = np.isin(rid, cid[chosen_row >= 0])
    return sim[keep].reset_index(drop=True), pick[chosen_row >= 0]


def main(sample, do_sim, do_fit, fe):
    rows, ch = load_rows(SPEC, sample)
    tr, _ = split(rows, ch)
    del rows
    log(f"{sample}: {tr.choice_id.nunique():,} training choices, {len(tr):,} rows")
    tag = f"{sample}{'_fe' if fe else ''}"
    res, modes, avail, cid = fit(tr, sample, fe, seeds=(0,) if do_sim and not do_fit else (0, 1))
    names = [m.name for m in modes]
    if do_fit:
        row = dict(sample=sample, hub_fe=fe, ll=res["ll"], iters=res["iters"],
                   **{f"share_{k}": v for k, v in zip(names, res["share"])}, **{f"pi_{k}": v for k, v in zip(names, res["pi"])})
        for m in modes:
            for f, c in m.coef().items():
                row[f"{m.name}:{f}"] = c
        row["R_fit_over_pop"] = (res["share"][names.index("qual")] + res["share"][names.index("match")]) / res["share"][0]
        write(pd.DataFrame([row]).T.reset_index().rename(columns={"index": "quantity", 0: "value"}), f"v2_mixture_{tag}",
              f"Latent-class decomposition with availability renormalisation ({sample}; FE in quality mode: {fe})")
    if do_sim:
        out = []
        for lab, pi in (("fitted pi", res["pi"]),
                        ("more match", np.r_[res["pi"][:2] * 0.6, [0.3], res["pi"][3:] * 0.6])):
            pi = pi / pi.sum()
            for seed in (1, 2):
                sim, truth = simulate(tr, modes, avail, cid, pi, seed, sample)
                est, _, _, _ = fit(sim, sample, fe, seeds=(0,))
                for k, nm in enumerate(names):
                    real = float((truth == k).mean())
                    out.append(dict(dgp=lab, seed=seed, mode=nm, pi_dgp=pi[k], realised_share=real, est_share=est["share"][k],
                                    abs_error=est["share"][k] - real,
                                    rel_error=(est["share"][k] - real) / real if real > 0 else np.nan))
                log(pd.DataFrame(out).tail(len(names)).to_string())
        write(pd.DataFrame(out), f"v2_mixsim_{tag}", f"Semi-synthetic check of the renormalised latent-class decomposition "
              f"({sample}): absolute and relative errors of the recovered shares")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="planA", choices=["planA", "decisions"])
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--fe", action="store_true")
    a = ap.parse_args()
    main(a.sample, a.sim, a.fit, a.fe)
