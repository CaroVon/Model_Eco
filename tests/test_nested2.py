import numpy as np
import pandas as pd
import torch

from hfpa.models.clogit import CLogit, to_buckets
from hfpa.models.nested2 import Nested2


def simulate(C=3000, lam=(0.7, 0.4), beta=(1.0, -0.5), delta=0.5, seed=0, sample=True):
    """Exact nested-logit draws on the full set; then the new nest is Poisson-sampled (pi from x0) and the
    chosen alternative added with its own pi."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(C):
        n_old = int(rng.integers(0, 5)) if k % 3 else 0  # a third are first decisions
        n_new = int(rng.integers(10, 300))  # varying market size identifies lam_new apart from delta
        n = n_new + n_old
        X = rng.normal(size=(n, 2))
        nest = np.r_[np.zeros(n_new, int), np.ones(n_old, int)]
        V = X @ np.array(beta) + delta * nest
        lamv = np.array(lam)[nest]
        iv = np.array([np.log(np.exp(V[nest == b] / lam[b]).sum()) if (nest == b).any() else -np.inf for b in (0, 1)])
        pb = np.exp(np.array(lam) * iv - np.logaddexp.reduce(np.array(lam) * iv))
        b = rng.choice(2, p=pb)
        pj = np.exp(V / lamv - iv[nest]) * (nest == b)
        j = rng.choice(n, p=pj / pj.sum())
        pi = np.where(nest == 1, 1.0, np.clip(0.15 * np.exp(0.5 * X[:, 0]), 0.02, 1)) if sample else np.ones(n)
        keep = (rng.random(n) < pi) | (np.arange(n) == j)
        ch = np.zeros(n, int)
        ch[j] = 1
        out.append(pd.DataFrame({"choice_id": k, "x0": X[keep, 0], "x1": X[keep, 1], "old": nest[keep].astype(float),
                                 "nest": nest[keep], "log_q": np.log(pi[keep]), "chosen": ch[keep]}))
    return pd.concat(out, ignore_index=True)


def test_lambda_one_is_clogit():
    R = simulate(C=400, lam=(1.0, 1.0), seed=2)
    D = to_buckets(R, ["x0", "x1", "old"], device="cpu", extra_int={"nest": "nest"})
    cl = CLogit(3, device="cpu").fit(D, se=False)
    m = Nested2(3, device="cpu")
    m.init(cl["coef"], [1 - 1e-9, 1 - 1e-9])
    with torch.no_grad():
        ll = sum(m.ll_rows(b).sum().item() for b in D.batches)
    assert abs(ll - cl["ll"]) < 1e-3 * abs(ll)


def test_recovers_lambda_with_sampling():
    R = simulate(C=6000, seed=1)
    D = to_buckets(R, ["x0", "x1", "old"], device="cpu", extra_int={"nest": "nest"})
    res = Nested2(3, device="cpu").fit(D, cluster=np.arange(D.n) // 4)
    truth = np.r_[1.0, -0.5, 0.5, np.log(0.7 / 0.3), np.log(0.4 / 0.6)]  # lambdas on the logit scale
    z = (res["coef"] - truth) / res["se"]
    assert np.all(np.abs(z) < 3.5), (res["coef"], res["se"])
    assert np.all(np.isfinite(res["se_cl"]))
