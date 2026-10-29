"""Stage P0e: can node-FE GMM separate preferential attachment from a launch-hype life cycle?

Real-data P5 gave negative alpha with a rejected Hansen J; a likely cause is that attention peaks right
after release and then fades non-linearly, which node + period effects cannot absorb. Here the synthetic
network adds that life cycle (boost 2.0 for age < 30 days, 1.0 for 30-90 days) and runs QD_GMM2 with
no age terms, age bins (<30, 30-90, 90-365 days) and log(1 + age).
Time scale: 20 new nodes per day (150,000 nodes ~ 20 years), period 10 days.

usage: python scripts/p0e_hype.py --scenario mix_hype [--seed 42]
Writes results/tables/p0e/<scenario>_s<seed>.csv.
"""
import argparse
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.models.panel import build_panel, qd_gmm2
from hfpa.sim.synthetic import generate_mixed

PER_DAY, PERIOD = 20, 10
HYPE = ((30 * PER_DAY, 2.0), (90 * PER_DAY, 1.0))
SCENARIOS = {
    "pa_hype": dict(alpha=0.5, h=1.0, fitness_sd=0.0, hype=HYPE),
    "fit_hype": dict(alpha=0.0, h=0.0, fitness_sd=1.0, hype=HYPE),
    "mix_hype": dict(alpha=0.5, h=1.0, fitness_sd=1.0, hype=HYPE),
    "mix": dict(alpha=0.5, h=1.0, fitness_sd=1.0),
}
AGE_BINS = ("age_lt30", "age_30_90", "age_90_365")
SPECS = {"no_age": ("has_deg", "log_deg"), "age_bins": ("has_deg", "log_deg") + AGE_BINS,
         "log_age": ("has_deg", "log_deg", "log_age")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=paths.CFG["seed"])
    ap.add_argument("--n", type=int, default=150_000)
    a = ap.parse_args()
    sc = SCENARIOS[a.scenario]
    nodes, edges, truth = generate_mixed(n=a.n, seed=a.seed, **sc)
    birth = np.arange(a.n) / PER_DAY  # days
    ct = birth[edges.child.to_numpy()]
    P = build_panel(birth, ct, edges.parent.to_numpy(), PERIOD)
    tv = {"has_deg": sc["h"], "log_deg": sc["alpha"]}
    out = []
    for name, feats in SPECS.items():
        t = time.time()
        est, se, Jst, df = qd_gmm2(P, names=feats)
        for k in feats:
            out.append(dict(scenario=a.scenario, seed=a.seed, spec=name, param=k, truth=tv.get(k, np.nan),
                            est=float(est[k]), se=float(se[k]), hansen_J=Jst, J_df=df, converged=est["converged"]))
        print(name, {k: f"{est[k]:.3f} ({se[k]:.3f})" for k in feats}, f"J={Jst:.1f}/{df}",
              f"conv={est['converged']} [{time.time() - t:.0f}s]", flush=True)
    d = paths.TABLES / "p0e"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(d / f"{a.scenario}_s{a.seed}.csv", index=False)
    print("done")
