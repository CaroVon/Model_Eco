import numpy as np
import pandas as pd

from hfpa.models.clogit import to_buckets
from hfpa.models.latent import LogitMode, PointMode, draw_modes, em


def simulate(C=4000, pi=(0.5, 0.2, 0.2, 0.1), seed=0):
    """Modes: pop (logit on x over all), unif (all), sub (logit on z over a subset, empty for half the choices),
    point (one marked candidate, present for 40% of choices)."""
    rng = np.random.default_rng(seed)
    pi = np.array(pi)
    recs, avail, pt = [], np.zeros((C, 4), bool), np.zeros(C, int) - 1
    for k in range(C):
        S = int(rng.integers(10, 40))
        x, z = rng.normal(size=S), rng.normal(size=S)
        insub = (rng.random(S) < 0.3) & (k % 2 == 0)
        mark = int(rng.integers(S)) if rng.random() < 0.4 else -1
        avail[k] = [True, True, insub.any(), mark >= 0]
        recs.append((S, x, z, insub, mark))
    modes = draw_modes(avail, pi, rng)
    rows = []
    for k, ((S, x, z, insub, mark), m) in enumerate(zip(recs, modes)):
        if m == 0:
            u = 1.5 * x + rng.gumbel(size=S)
        elif m == 1:
            u = rng.gumbel(size=S)
        elif m == 2:
            u = np.where(insub, 1.0 * z + rng.gumbel(size=S), -np.inf)
        else:
            u = np.where(np.arange(S) == mark, 0.0, -np.inf)
        ch = np.zeros(S, int)
        ch[np.argmax(u)] = 1
        rows.append(pd.DataFrame({"choice_id": k, "x": x, "z": z, "sub": insub, "mark": np.arange(S) == mark,
                                  "chosen": ch, "log_q": 0.0}))
    return pd.concat(rows, ignore_index=True), avail, modes


def build(R, avail):
    cid = np.unique(R.choice_id)
    D_all = to_buckets(R, ["x"], device="cpu")
    D_unif = to_buckets(R, [], device="cpu")
    sub = R[R["sub"]]
    ok = sub.groupby("choice_id").chosen.max()
    sub = sub[sub.choice_id.isin(ok.index[ok == 1])]
    D_sub = to_buckets(sub, ["z"], device="cpu")
    hit = R[R.chosen == 1].set_index("choice_id").mark.reindex(cid).to_numpy()
    return [LogitMode("pop", D_all, np.arange(len(cid))), LogitMode("unif", D_unif, np.arange(len(cid))),
            LogitMode("sub", D_sub, np.searchsorted(cid, D_sub.choice_ids)), PointMode("point", hit)]


def test_em_renormalised_recovers_pi():
    pi = np.array([0.5, 0.2, 0.2, 0.1])
    R, avail, modes = simulate(pi=pi)
    res = em(build(R, avail), avail, np.ones(len(avail)), n_iter=60)
    real = np.bincount(modes, minlength=4) / len(modes)
    assert np.allclose(res["pi"], pi, atol=0.05), res["pi"]
    assert np.allclose(res["share"], real, atol=0.04), (res["share"], real)
