"""Can the two-nest logit recover lambda when the new nest is a (d+1) Poisson-PPS sample of a LARGE market?

The real v2 estimate has lambda_new = 0.02 and lambda_old = 0.13. With ~300 sampled candidates out of ~10^5-10^6
models the Horvitz-Thompson inclusive value log sum exp(V / lam) / pi is noisy; a small lambda makes the sum
dominated by the pi = 1 hubs, which could pull lambda towards 0. This simulates exact nested-logit choices on a
heavy-tailed market, samples the new nest as in the replay (pi = min(1, c (d + 1)), E|D| = 300, chosen added with
its own pi; old nest = the author's earlier parents, all included) and re-estimates with hfpa.models.nested2.
Scenarios: true lambdas x market size (300 = no sampling, 50,000 = PPS). Writes results/tables/v2/v2_nested_check.md.
usage: python scripts/v2_nested_check.py [--C 3000]
"""
import argparse

import numpy as np
import pandas as pd
import torch

from hfpa.models.clogit import CLogit, to_buckets
from hfpa.models.nested2 import Nested2
from hfpa.v2data import write

BETA = np.array([0.35, 1.0])  # log(d + 1), x
DELTA = 1.0  # old-nest constant


def pps_pi(d, S):
    w = d + 1.0
    lo, hi = 1e-12, 1.0
    for _ in range(80):
        c = np.sqrt(lo * hi)
        lo, hi = (c, hi) if np.minimum(1, c * w).sum() < S else (lo, c)
    return np.minimum(1.0, hi * w)


def simulate(C, N, lam, seed, S=300):
    rng = np.random.default_rng(seed)
    d = np.floor(rng.pareto(1.1, N) * (rng.random(N) < 0.2) * 3).astype(np.int64)  # 80% zero degree, heavy tail
    ld = np.log(d + 1.0)
    pi_mkt = pps_pi(d, S) if N > S else np.ones(N)
    lam = np.asarray(lam)
    out = []
    for k in range(C):
        x = rng.normal(size=N)
        n_old = int(rng.integers(1, 6)) if k % 3 else 0
        old = rng.choice(N, n_old, replace=False, p=(d + 1.0) / (d + 1.0).sum()) if n_old else np.empty(0, int)
        nest = np.zeros(N, int)
        nest[old] = 1
        V = BETA[0] * ld + BETA[1] * x + DELTA * nest
        y = V / lam[nest]
        iv = np.array([np.logaddexp.reduce(y[nest == b]) if (nest == b).any() else -np.inf for b in (0, 1)])
        z = lam * iv
        pb = np.exp(z - np.logaddexp.reduce(z))
        b = rng.choice(2, p=pb)
        inb = np.flatnonzero(nest == b)
        p = np.exp(y[inb] - iv[b])
        j = int(inb[rng.choice(len(inb), p=p / p.sum())])
        pi = np.where(nest == 1, 1.0, pi_mkt)
        keep = (rng.random(N) < pi) | (np.arange(N) == j) | (nest == 1)
        idx = np.flatnonzero(keep)
        out.append(pd.DataFrame({"choice_id": k, "ld": ld[idx], "x": x[idx], "old": nest[idx].astype(float),
                                 "nest": nest[idx], "log_q": np.log(pi[idx]), "chosen": (idx == j).astype(int)}))
    return pd.concat(out, ignore_index=True)


def main(C):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = []
    for lam in [(1.0, 1.0), (0.7, 0.5), (0.3, 0.3)]:
        for N in (300, 50_000):
            for seed in range(2):
                R = simulate(C, N, lam, seed)
                D = to_buckets(R, ["ld", "x", "old"], device=dev, extra_int={"nest": "nest"})
                cl = CLogit(3, device=dev).fit(D, se=False)
                m = Nested2(3, device=dev)
                m.init(cl["coef"], [0.9, 0.9])
                r = m.fit(D, cluster=np.arange(D.n))
                lam_hat = r["lam"]
                se_lam = lam_hat * (1 - lam_hat) * r["se"][3:5]
                rec = dict(lam_new_true=lam[0], lam_old_true=lam[1], market=N, seed=seed,
                           mean_set=R.groupby("choice_id").size().mean(),
                           lam_new=lam_hat[0], se_new=se_lam[0], lam_old=lam_hat[1], se_old=se_lam[1],
                           b_ld=r["coef"][0], b_x=r["coef"][1], delta=r["coef"][2])
                res.append(rec)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    write(pd.DataFrame(res), "v2_nested_check", "Synthetic recovery of the two-nest lambdas: exact choices on a "
          "heavy-tailed market, new nest PPS-sampled (E|D| = 300) as in the replay",
          f"True beta: log(d+1) {BETA[0]}, x {BETA[1]}, delta {DELTA} (nested2 parameterisation: V = X beta, "
          "y = V / lambda within the nest, so the estimates should equal these values).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=3000)
    main(ap.parse_args().C)
