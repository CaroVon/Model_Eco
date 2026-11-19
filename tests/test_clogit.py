import numpy as np
import pandas as pd

from hfpa.models.clogit import CLogit, fit_subset, multiplier_weights, to_buckets, to_padded, uniform_ll


def fake_rows(C=600, seed=0):
    """Choices with varying set sizes (5..80), 3 features, FE on 10 'hubs', log q shifts."""
    rng = np.random.default_rng(seed)
    beta, phi = np.array([1.0, -0.5, 0.3]), np.r_[0.0, rng.normal(0, 1, 10)]
    out = []
    for k in range(C):
        S = int(rng.integers(5, 80))
        X = rng.normal(size=(S, 3))
        fe = np.where(rng.random(S) < 0.2, rng.integers(1, 11, S), 0)
        lq = np.log(rng.uniform(0.05, 1, S))
        u = X @ beta + phi[fe] - lq + rng.gumbel(size=S)
        ch = np.zeros(S, int)
        ch[u.argmax()] = 1
        out.append(pd.DataFrame({"choice_id": k, "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2], "fe": fe,
                                 "log_q": lq, "chosen": ch}))
    return pd.concat(out, ignore_index=True)


def test_buckets_equal_padded():
    rows = fake_rows()
    feats = ["x0", "x1", "x2"]
    C = rows.choice_id.nunique()
    w = np.random.default_rng(1).uniform(0.5, 2, C)
    cl = np.arange(C) // 7
    a = CLogit(3, n_fe=10, l2_fe=1e-4, device="cpu").fit(to_padded(rows, feats, fe_col="fe", weight=w), cluster=cl)
    D = to_buckets(rows, feats, fe_col="fe", weight=w, device="cpu", max_cells=5000)
    assert len(D.batches) > 3
    b = CLogit(3, n_fe=10, l2_fe=1e-4, device="cpu").fit(D, cluster=cl)
    assert np.allclose(a["coef"], b["coef"], atol=1e-5)
    assert np.allclose(a["se"], b["se"], rtol=1e-4)
    assert np.allclose(a["se_cl"][:3], b["se_cl"][:3], rtol=1e-4)
    assert abs(a["ll"] - b["ll"]) < 1e-6 * abs(a["ll"])


def test_uniform_and_subset_and_bootstrap():
    rows = fake_rows(300)
    D = to_buckets(rows, ["x0", "x1", "x2"], fe_col="fe", device="cpu", max_cells=4000)
    m0 = uniform_ll(D)
    g = rows.groupby("choice_id")
    ref = -rows[rows.chosen == 1].log_q.to_numpy() - g.log_q.apply(lambda s: np.log(np.exp(-s).sum())).to_numpy()
    assert np.allclose(m0, ref)
    _, r1 = fit_subset(D, ["x0"])
    _, r2 = fit_subset(D, ["x0"], init=r1["coef"])  # warm start converges at once to the same point
    assert np.allclose(r1["coef"], r2["coef"], atol=1e-6)
    m, full = fit_subset(D, ["x0", "x1", "x2"])
    assert np.allclose(m.ll_per_choice(D.subset_feats(["x0", "x1", "x2"])).sum(), full["ll"])
    w = multiplier_weights(np.arange(D.n) // 3, np.ones(D.n), np.random.default_rng(0))
    assert np.all(w[0:3] == w[0]) and abs(w.mean() - 1) < 0.3
    D.set_weights(w)
    _, rb = fit_subset(D, ["x0", "x1", "x2"], init=full["coef"])
    assert not np.allclose(rb["coef"], full["coef"])
