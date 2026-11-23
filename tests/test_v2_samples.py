import sys

import numpy as np

sys.path.insert(0, "scripts")


def test_author_weights_equalise_authors():
    from v2_samples import base_weights
    a = np.array([1, 1, 1, 1, 2, 3, 3])
    w = base_weights(a, "author")
    tot = {k: w[a == k].sum() for k in np.unique(a)}
    assert np.allclose(list(tot.values()), tot[1])
    assert np.isclose(w.mean(), 1.0)
    assert np.allclose(base_weights(a, "choice"), 1.0)


def test_own_counts():
    from v2_selfdeg import DAY, SHIFT, own_counts
    N = np.int64(10)
    # author 2 -> parent 5 children born at t = 100, 200, 40*DAY; author 3 -> parent 5 at 150
    a = np.array([2, 2, 2, 3]); p = np.array([5, 5, 5, 5]); te = np.array([100, 200, 40 * DAY, 150])
    key = a * N + p
    ukey, kid = np.unique(key, return_inverse=True)
    comb = np.sort(kid.astype(np.int64) * SHIFT + te)
    prior, d30 = own_counts(ukey, comb, N, np.array([2, 2, 2, 3, 4]), np.array([5, 5, 5, 5, 5]),
                            np.array([200, 201, 41 * DAY, 151, 1000]))
    assert prior.tolist() == [1, 2, 3, 1, 0]  # strictly before t
    assert d30.tolist() == [1, 2, 1, 1, 0]   # within (t - 30d, t)
