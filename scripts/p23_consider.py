"""P23: does popularity work through attention (which candidates are considered) or through utility?
And is substitution stronger within a model family (nested logit)?  Plan A, hfpa.models.consider.

Models (training choices; holdout = last 10% in time, as in P15):
  logit        conditional logit, all P15 groups in utility (= P15 full model)
  logit_nopop  conditional logit without popularity and visibility
  consider     attention phi_k = sigmoid(g0 + Z_k g), Z = popularity + visibility; utility = quality, match,
               inertia, local search, availability
  consider_both  popularity and visibility in attention and in utility
  nested       all groups in utility, nests = model family (root of the candidate's lineage), one lambda
Reported: log-likelihood, AIC, holdout mean log-likelihood, coefficients; Vuong test logit vs consider; for
consider: consideration probabilities of hubs / non-hubs / the chosen parent.
--sim: identification check on the real candidate sets: choices redrawn from the fitted logit and from the fitted
consider model, both models re-estimated on each; the true model should win on likelihood and the true
parameters should be recovered.
usage: sbatch slurm/gpu.sbatch scripts/p23_consider.py [--sim]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from hfpa import paths
from hfpa.models.clogit import CLogit, to_padded
from hfpa.models.consider import Model, add_padded
from p15_decomp import GROUPS, P5, P9, drop_constant, load, sample

ATT = GROUPS["pop"] + GROUPS["base"]
REST = sum((GROUPS[g] for g in ("qual", "match", "inertia", "local", "avail")), [])


def family(rows):
    """Nest id within each choice: the candidate's root (lineage), renumbered 0..K-1 per choice."""
    root = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["root"]).root.to_numpy()
    r = root[rows.cand_id.to_numpy()]
    key = pd.DataFrame({"c": rows.choice_id.to_numpy(), "r": r})
    return key.groupby("c").r.transform(lambda s: pd.factorize(s)[0]).to_numpy(np.int64)


def padded(r, X, Z=None, nest=False):
    D = to_padded(r, X)
    if Z is not None:
        add_padded(D, r, Z, "Z")
    if nest:
        add_padded(D, r.assign(nest=family(r)), "nest", "nest", np.int64)
    return D


def fit_all(tr, ho):
    feats = drop_constant(tr, ATT + REST)
    att, rest = [f for f in ATT if f in feats], [f for f in REST if f in feats]
    out, coefs, per = [], [], {}
    specs = {"logit": ("clogit", feats, None), "logit_nopop": ("clogit", rest, None),
             "consider": ("consider", rest, att), "consider_both": ("consider", rest + att, att),
             "nested": ("nested", feats, None)}
    for name, (kind, X, Z) in specs.items():
        t0 = time.time()
        D, H = padded(tr, X, Z, kind == "nested"), padded(ho, X, Z, kind == "nested")
        if kind == "clogit":
            m = CLogit(len(X))
            res = m.fit(D)
            hold = m.evaluate(H)["mean_ll"]
            with torch.no_grad():
                per[name] = np.concatenate([m._ll_batch(b).cpu().numpy() for b in m._batches(D, 20000)])
            names = X
        else:
            m = Model(kind, len(X), len(Z) if Z else 0)
            res = m.fit(D)
            hold = m.evaluate(H)
            with torch.no_grad():
                per[name] = np.concatenate([m.ll_rows(b).cpu().numpy() for b in m.batches(D)])
            names = X + (["att:const"] + [f"att:{z}" for z in Z] if kind == "consider" else ["lambda_logit"])
        rec = dict(model=name, ll=res["ll"], k=res["k"], aic=2 * res["k"] - 2 * res["ll"], hold_mean_ll=hold,
                   seconds=time.time() - t0)
        if kind == "nested":
            rec["lambda"] = 1 / (1 + np.exp(-res["coef"][len(X)]))
        if kind == "consider":
            phi = m.phi(D)
            st = np.full(phi.shape, -1)
            st[D["mask"]] = tr.stratum.to_numpy()
            rec.update(phi_hub=np.nanmean(np.where(st == P9.HUB, phi, np.nan)),
                       phi_used_nonhub=np.nanmean(np.where(st == 1, phi, np.nan)),
                       phi_unused=np.nanmean(np.where(st == 2, phi, np.nan)),
                       phi_chosen=float(np.mean(phi[np.arange(len(phi)), D["chosen"]])),
                       exp_considered=float(np.median(np.nansum(phi * np.exp(np.where(D["mask"], D["lq"], 0) * -1) * D["mask"], 1))))
        out.append(rec)
        coefs += [dict(model=name, param=n, coef=c, se=s) for n, c, s in zip(names, res["coef"], res["se"])]
        print(rec, flush=True)
    for a, b in [("logit", "consider"), ("logit", "consider_both"), ("logit", "nested")]:
        d = per[b] - per[a]
        z = d.sum() / (d.std() * np.sqrt(len(d)))
        out.append(dict(model=f"Vuong {b} vs {a}", ll=d.sum(), vuong_z=z, p_two_sided=2 * norm.sf(abs(z))))
    return pd.DataFrame(out), pd.DataFrame(coefs), specs


def simulate(tr, kind, X, Z, coef, seed):
    """Redraw the chosen candidate of every training choice from a fitted model (sampled set, log q shift)."""
    rng = np.random.default_rng(seed)
    v = tr[X].to_numpy(float) @ coef[: len(X)] - tr.log_q.to_numpy(float)
    if kind == "consider":
        eta = coef[len(X)] + tr[Z].to_numpy(float) @ coef[len(X) + 1:]
        B = rng.random(len(tr)) < 1 / (1 + np.exp(-eta))
        v = np.where(B, v, -np.inf)
    u = v + rng.gumbel(size=len(tr))
    s = pd.Series(u, index=tr.index)
    best = s.groupby(tr.choice_id.to_numpy()).idxmax()
    ok = np.isfinite(s.loc[best.to_numpy()].to_numpy())  # some candidate considered
    sim = tr[tr.choice_id.isin(best.index[ok]).to_numpy()].copy()
    sim["chosen"] = 0
    sim.loc[best.to_numpy()[ok], "chosen"] = 1
    return sim.reset_index(drop=True)


def part_sim(tr, coefs, specs):
    out = []
    feats = drop_constant(tr, ATT + REST)
    att, rest = [f for f in ATT if f in feats], [f for f in REST if f in feats]
    for truth in ("logit", "consider"):
        kind, X, Z = specs[truth]
        kind = "consider" if truth == "consider" else "clogit"
        coef = coefs[coefs.model == truth].coef.to_numpy()
        for seed in (1, 2):
            sim = simulate(tr, kind, X, Z, coef, seed)
            ll = {}
            for est, (k2, X2, Z2) in (("logit", ("clogit", feats, None)), ("consider", ("consider", rest, att))):
                D = padded(sim, X2, Z2)
                res = CLogit(len(X2)).fit(D) if k2 == "clogit" else Model("consider", len(X2), len(Z2)).fit(D)
                ll[est] = res["ll"]
                if est == truth:
                    names = X2 + (["att:const"] + [f"att:{z}" for z in Z2] if k2 == "consider" else [])
                    tv = dict(zip(names, coef))
                    for n, c, s in zip(names, res["coef"], res["se"]):
                        out.append(dict(truth=truth, seed=seed, kind="param", param=n, true=tv[n], est=c, se=s,
                                        z=(c - tv[n]) / s if s > 0 else np.nan))
            out.append(dict(truth=truth, seed=seed, kind="selection", param="ll consider - ll logit",
                            est=ll["consider"] - ll["logit"], n=sim.choice_id.nunique()))
            print(out[-1], flush=True)
    P5.write(pd.DataFrame(out), "p23_sim", "P23 identification check: choices redrawn from the fitted logit / consideration model")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    a = ap.parse_args()
    rows, ch = load()
    r, _ = sample(rows, ch, "planA")
    tr, ho = P5.split(r, ch)
    tr, ho = tr.reset_index(drop=True), ho.reset_index(drop=True)
    del rows, r
    fits, coefs, specs = fit_all(tr, ho)
    P5.write(fits, "p23_consider", "P23 attention (consideration) vs utility channel of popularity; nested logit by family (plan A)")
    P5.write(coefs, "p23_consider_coef", "P23 coefficients (attention terms prefixed att:)")
    if a.sim:
        part_sim(tr, coefs, specs)
