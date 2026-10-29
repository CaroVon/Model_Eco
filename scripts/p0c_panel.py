"""Stage P0c: node fixed effects for every node via the count-panel form (see hfpa/models/panel.py).

Same synthetic scenarios as P0b. For each period length: Poisson FE MLE, its split-panel jackknife,
and quasi-differenced GMM. In bulk scenarios counts are weighted by n^-gamma (n = choices of the
same author-parent pair); degree is never weighted.

usage: python scripts/p0c_panel.py --scenario mix [--seed 42] [--n 50000]
Writes results/tables/p0c/<scenario>_s<seed>.csv; summary via scripts/p0c_collect.py.
"""
import argparse
import importlib.util
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.models.panel import build_panel, jackknife, poisson_fe, qd_gmm
from hfpa.sim.synthetic import generate_mixed

_spec = importlib.util.spec_from_file_location("p0b", paths.REPO / "scripts" / "p0b_identification.py")
_p0b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p0b)
SCENARIOS = _p0b.SCENARIOS
PERIODS = (25, 100, 400)  # days; one new node per day

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=paths.CFG["seed"])
    ap.add_argument("--n", type=int, default=50_000)
    a = ap.parse_args()
    sc = SCENARIOS[a.scenario]
    nodes, edges, truth = generate_mixed(n=a.n, seed=a.seed, **sc)
    birth = nodes.birth.to_numpy() // 86400
    child, parent = edges.child.to_numpy(), edges.parent.to_numpy()
    ct = birth[child]
    au = nodes.author.to_numpy()[child]
    n_pair = pd.Series(1, index=pd.MultiIndex.from_arrays([au, parent])).groupby(level=[0, 1]).transform("size").to_numpy()
    gammas = (0.0, 0.5, 1.0) if sc.get("bulk_frac", 0) > 0 else (0.0,)
    tv = {"has_deg": sc["h"], "log_deg": sc["alpha"]}
    out = []
    for per in PERIODS:
        for g in gammas:
            t = time.time()
            P = build_panel(birth, ct, parent, per, y_weight=n_pair.astype(float) ** -g)
            jk, fe = jackknife(P)
            gm, resid = qd_gmm(P)
            for method, est in (("PFE", fe), ("PFE_JK", jk), ("QD_GMM", gm)):
                for k, v in est.items():
                    out.append(dict(scenario=a.scenario, seed=a.seed, period=per, gamma=g, method=method,
                                    param=k, truth=tv[k], est=float(v)))
            print(f"per={per} g={g} J={P['y'].shape[0]} T={P['T']} FE a={fe['log_deg']:.3f} JK a={jk['log_deg']:.3f} "
                  f"GMM a={gm['log_deg']:.3f} h={gm['has_deg']:.3f} |m|={resid:.1e} [{time.time() - t:.0f}s]", flush=True)
    d = paths.TABLES / "p0c"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(d / f"{a.scenario}_s{a.seed}.csv", index=False)
    print("done")
