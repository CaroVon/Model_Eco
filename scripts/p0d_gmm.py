"""Stage P0d: quasi-differenced GMM, just-identified (P0c) vs over-identified two-step (qd_gmm2).

Larger synthetic networks (default 150,000 nodes, close to the ~420k real choices), period lengths
100/200 days (50 was worse in the smoke test). Outcome counts in bulk scenarios: all choices, n^-0.5 weights, or the first choice of
each author-parent pair only ("first"); degree always counts every child.

usage: python scripts/p0d_gmm.py --scenario mix [--seed 42] [--n 150000]
Writes results/tables/p0d/<scenario>_s<seed>.csv; summary via scripts/p0d_collect.py.
"""
import argparse
import importlib.util
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.models.panel import build_panel, qd_gmm, qd_gmm2
from hfpa.sim.synthetic import generate_mixed

_spec = importlib.util.spec_from_file_location("p0b", paths.REPO / "scripts" / "p0b_identification.py")
_p0b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p0b)
SCENARIOS = _p0b.SCENARIOS
PERIODS = (100, 200)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=paths.CFG["seed"])
    ap.add_argument("--n", type=int, default=150_000)
    a = ap.parse_args()
    sc = SCENARIOS[a.scenario]
    nodes, edges, truth = generate_mixed(n=a.n, seed=a.seed, **sc)
    birth = nodes.birth.to_numpy() // 86400
    child, parent = edges.child.to_numpy(), edges.parent.to_numpy()
    ct = birth[child]
    pairs = pd.DataFrame({"a": nodes.author.to_numpy()[child], "p": parent})
    n_pair = pairs.groupby(["a", "p"]).p.transform("size").to_numpy().astype(float)
    first = (~pairs.duplicated(["a", "p"])).to_numpy().astype(float)  # edges are in birth order
    variants = {"all": None}
    if sc.get("bulk_frac", 0) > 0:
        variants.update({"g=0.5": n_pair ** -0.5, "first": first})
    tv = {"has_deg": sc["h"], "log_deg": sc["alpha"]}
    out = []
    for per in PERIODS:
        for v, w in variants.items():
            t = time.time()
            P = build_panel(birth, ct, parent, per, y_weight=w)
            est1, _ = qd_gmm(P)
            est2, se2, Jst, df = qd_gmm2(P)
            for method, est, se in (("QD_GMM", est1, {}), ("QD_GMM2", est2, se2)):
                for k in (x for x in est if x != "converged"):
                    out.append(dict(scenario=a.scenario, seed=a.seed, n=a.n, period=per, y=v, method=method,
                                    param=k, truth=tv[k], est=float(est[k]), se=float(se.get(k, np.nan)),
                                    J=Jst if method == "QD_GMM2" else np.nan, J_df=df,
                                    converged=est.get("converged", np.nan)))
            print(f"per={per} y={v} J={P['y'].shape[0]} T={P['T']} GMM a={est1['log_deg']:.3f} | GMM2 a={est2['log_deg']:.3f} "
                  f"(se {se2['log_deg']:.3f}) h={est2['has_deg']:.3f} Hansen J={Jst:.1f} df={df} [{time.time() - t:.0f}s]", flush=True)
    d = paths.TABLES / "p0d"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(d / f"{a.scenario}_s{a.seed}.csv", index=False)
    print("done")
