"""P0 synthetic growth models (plan 6.1).

pa:      each new node attaches to an existing node j with weight d_j^alpha (d_j > 0) or e^beta0 (d_j = 0).
fitness: weight exp(phi_j), phi_j ~ N(0, sd^2), independent of degree (alpha = 0).
Output has the same nodes/edges layout the replay expects; one node per day.
"""
import numpy as np
import pandas as pd

from hfpa.replay.replay import DAY


def generate(n=50_000, kind="pa", alpha=1.0, beta0=-1.0, fitness_sd=1.0, seed=42):
    rng = np.random.default_rng(seed)
    deg = np.zeros(n)
    w = np.zeros(n)
    phi = rng.normal(0, fitness_sd, n) if kind == "fitness" else None
    base = (lambda i: np.exp(phi[i])) if kind == "fitness" else (lambda i: np.exp(beta0))
    parent = np.full(n, -1)
    w[0] = base(0)
    for i in range(1, n):
        cw = np.cumsum(w[:i])
        j = min(int(np.searchsorted(cw, rng.random() * cw[-1], "right")), i - 1)
        parent[i] = j
        deg[j] += 1
        if kind == "pa":
            w[j] = deg[j] ** alpha
        w[i] = base(i)
    root = np.arange(n)
    for i in range(1, n):
        root[i] = root[parent[i]]
    nodes = pd.DataFrame({"birth": np.arange(n, dtype=np.int64) * DAY, "modality": 0,
                          "author": np.arange(n), "root": root})
    edges = pd.DataFrame({"child": np.arange(1, n), "parent": parent[1:], "rel_type": "finetune", "weight": 1.0})
    truth = dict(kind=kind, alpha=alpha if kind == "pa" else 0.0,
                 has_deg=-beta0 if kind == "pa" else 0.0, phi=phi)
    return nodes, edges, truth


class _Fenwick:
    """Prefix sums over node weights: O(log n) update and weighted draw."""

    def __init__(self, n):
        self.n, self.t, self.top = n, np.zeros(n + 1), 1 << int(np.log2(n))

    def add(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def draw(self, u):
        """Smallest idx whose prefix sum exceeds u."""
        pos, step = 0, self.top
        while step:
            nxt = pos + step
            if nxt <= self.n and self.t[nxt] <= u:
                pos, u = nxt, u - self.t[nxt]
            step >>= 1
        return pos


def generate_mixed(n=50_000, alpha=0.5, h=1.0, fitness_sd=1.0, decay=0.0,
                   bulk_frac=0.0, n_bulk=50, repeat_p=0.9, hype=(), seed=42):
    """P0b generator mixing the mechanisms the real-data estimator must tell apart.

    Node j's utility for a newcomer at day t:
        u_j = 1[d_j > 0] (h + alpha log d_j) + phi_j - decay * age_j(days)
    phi_j ~ N(0, fitness_sd^2) is latent quality. Linear age decay is equivalent to a static
    per-node shift (+decay * birth_j), so sampling stays O(log n) per step.
    Bulk uploaders: a share `bulk_frac` of nodes belongs to `n_bulk` authors; an author's first
    upload chooses by u, later uploads repeat that parent with prob `repeat_p` (habit, not PA).
    Other nodes are their own author.
    hype: ((age_limit_days, boost), ...) ascending; a node gets +boost while age < limit (first matching
    bin): a non-linear launch-hype life cycle that node and period effects cannot absorb.
    """
    rng = np.random.default_rng(seed)
    phi = rng.normal(0, fitness_sd, n) if fitness_sd > 0 else np.zeros(n)
    base = phi + decay * np.arange(n)  # log weight at degree 0 (age shift folded in)
    deg = np.zeros(n, np.int64)
    w = np.zeros(n)
    F = _Fenwick(n)
    bulk = rng.random(n) < bulk_frac
    author = np.where(bulk, rng.integers(0, n_bulk, n), n_bulk + np.arange(n))
    fav = {}
    parent = np.full(n, -1)
    habit = np.zeros(n, bool)

    cur = [0]

    def boost(j):
        age = cur[0] - j
        for lim, b in hype:
            if age < lim:
                return b
        return 0.0

    def setw(j):
        lw = base[j] + boost(j) + ((h + alpha * np.log(deg[j])) if deg[j] > 0 else 0.0)
        nw = np.exp(lw)
        F.add(j, nw - w[j])
        w[j] = nw

    setw(0)
    for i in range(1, n):
        cur[0] = i
        for lim, _ in hype:  # nodes crossing an age-bin boundary today
            if i - lim >= 0:
                setw(i - lim)
        a = author[i]
        if a in fav and rng.random() < repeat_p:
            j, habit[i] = fav[a], True
        else:
            j = min(F.draw(rng.random() * _prefix(F, i)), i - 1)  # nodes >= i still have weight 0
            fav.setdefault(a, j)
        parent[i] = j
        deg[j] += 1
        setw(j)
        setw(i)
    root = np.arange(n)
    for i in range(1, n):
        root[i] = root[parent[i]]
    nodes = pd.DataFrame({"birth": np.arange(n, dtype=np.int64) * DAY, "modality": 0, "author": author, "root": root})
    edges = pd.DataFrame({"child": np.arange(1, n), "parent": parent[1:], "rel_type": "finetune", "weight": 1.0})
    truth = dict(alpha=alpha, has_deg=h, age_k=-decay * 1000, phi=phi, habit=habit[1:], hype=hype)
    return nodes, edges, truth


def _prefix(F, i):
    """Sum of weights of nodes [0, i)."""
    s, k = 0.0, i
    while k > 0:
        s += F.t[k]
        k -= k & -k
    return s
