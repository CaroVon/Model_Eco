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
