import numpy as np

from hfpa.models.clogit import to_buckets
from hfpa.v2data import shapley_fits, shares
from test_clogit import fake_rows


def test_shapley_fits_and_shares():
    R = fake_rows(400)
    groups = {"a": ["x0"], "b": ["x1"], "c": ["x2"]}
    D = to_buckets(R, ["x0", "x1", "x2"], fe_col="fe", device="cpu")
    Dh = to_buckets(fake_rows(100, seed=5), ["x0", "x1", "x2"], fe_col="fe", device="cpu")
    fits = shapley_fits(D, ["x0", "x1", "x2"], groups=groups, Dh=Dh, log=None)
    assert len(fits) == 8
    sh, tot = shares(fits, list(groups))
    assert abs(sum(v["share"] for v in sh.values()) - 1) < 1e-9 and tot > 0
    assert sh["a"]["share"] > sh["c"]["share"]  # |beta| 1.0 vs 0.3
    ff = shapley_fits(D, ["x0", "x1", "x2"], groups=groups, fe_player=True, n_fe=10, log=None)
    assert len(ff) == 16
    sh5, _ = shares(ff, list(groups) + ["fe"])
    assert sh5["fe"]["share"] > 0
    warm = {k: v["coef"] for k, v in fits.items()}
    again = shapley_fits(D, ["x0", "x1", "x2"], groups=groups, warm=warm, log=None)
    assert all(abs(again[k]["ll"] - fits[k]["ll"]) < 1e-6 for k in fits)
