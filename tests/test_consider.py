import numpy as np
import pandas as pd
import torch

from hfpa.models.clogit import CLogit, to_padded
from hfpa.models.consider import Model, add_padded


def _rows(C=3000, S=30, seed=0, b=(1.0, -0.5), g0=-0.5, g=1.5, lam=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(C, S, 2))
    Z = rng.normal(size=(C, S))
    nest = rng.integers(0, 5, size=(C, S))
    v = X @ np.array(b)
    ch = np.empty(C, int)
    for c in range(C):
        if lam is None:  # independent consideration, logit among the considered
            while True:
                B = rng.random(S) < 1 / (1 + np.exp(-(g0 + g * Z[c])))
                if B.any():
                    break
            u = np.where(B, v[c] + rng.gumbel(size=S), -np.inf)
        else:  # nested logit: nest by inclusive values, then logit within the nest
            iv = np.array([np.log(np.exp(v[c][nest[c] == k] / lam).sum()) if (nest[c] == k).any() else -np.inf for k in range(5)])
            pk = np.exp(lam * iv - np.logaddexp.reduce(lam * iv))
            k = rng.choice(5, p=pk)
            idx = np.flatnonzero(nest[c] == k)
            pj = np.exp(v[c][idx] / lam - iv[k])
            ch[c] = idx[rng.choice(len(idx), p=pj / pj.sum())]
            continue
        ch[c] = int(np.argmax(u))
    R = pd.DataFrame({"choice_id": np.repeat(np.arange(C), S), "x1": X[..., 0].ravel(), "x2": X[..., 1].ravel(),
                      "z": Z.ravel(), "nest": nest.ravel(), "log_q": 0.0,
                      "chosen": (np.arange(S)[None, :] == ch[:, None]).ravel().astype(int)})
    return R


def test_consider_recovers_parameters():
    R = _rows()
    D = add_padded(to_padded(R, ["x1", "x2"]), R, ["z"], "Z")
    res = Model("consider", 2, 1, device="cpu").fit(D)
    est = res["coef"]
    assert np.allclose(est, [1.0, -0.5, -0.5, 1.5], atol=0.25), est


def test_consider_and_nested_nest_the_logit():
    R = _rows(C=500)
    D = to_padded(R, ["x1", "x2"])
    cl = CLogit(2, device="cpu").fit(D)
    D = add_padded(add_padded(D, R, ["z"], "Z"), R, "nest", "nest", np.int64)
    m = Model("consider", 2, 1, device="cpu")
    with torch.no_grad():
        m.p[:] = torch.tensor([*cl["coef"], 40.0, 0.0], dtype=torch.float64)  # phi = 1
    assert abs(float(m.total(m.batches(D))) - cl["ll"]) < 1e-3
    n = Model("nested", 2, device="cpu")
    with torch.no_grad():
        n.p[:] = torch.tensor([*cl["coef"], 40.0], dtype=torch.float64)  # lambda = 1
    assert abs(float(n.total(n.batches(D))) - cl["ll"]) < 1e-3


def test_nested_recovers_lambda():
    R = _rows(C=3000, lam=0.5, seed=1)
    D = add_padded(to_padded(R, ["x1", "x2"]), R, "nest", "nest", np.int64)
    res = Model("nested", 2, device="cpu").fit(D)
    lam = 1 / (1 + np.exp(-res["coef"][2]))
    assert np.allclose(res["coef"][:2], [1.0, -0.5], atol=0.2), res["coef"]
    assert abs(lam - 0.5) < 0.1, lam
