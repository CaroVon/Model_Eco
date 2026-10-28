"""Stage P0 diagnostic: where does the spurious alpha under pure fitness come from? (plan 5.6, 6.1 step 4)

Fits on the pure-fitness network: (a) alpha split into hub / non-hub candidates with hub FE;
(b) fixed effects for the top-K' nodes by final degree, K' in {200, 1000, 5000}; the candidate
sampling (K=200 hub stratum) is unchanged, only the FE index is extended.
"""
import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.models.clogit import CLogit, to_padded
from hfpa.replay.replay import Replay
from hfpa.sim.synthetic import generate

N, K, SEED = 50_000, 200, paths.CFG["seed"]

if __name__ == "__main__":
    nodes, edges, truth = generate(n=N, kind="fitness", fitness_sd=1.0, seed=SEED)
    r = Replay(nodes, edges, {"hub_k": K, "s1": 24, "s0": 24, "seed": SEED})
    _, rows = r.run()
    hub = (rows.fe > 0).astype(float)
    rows["log_deg_hub"], rows["log_deg_nonhub"] = rows.log_deg * hub, rows.log_deg * (1 - hub)
    rows["has_deg_nonhub"] = rows.has_deg * (1 - hub)
    out = []

    def run(label, feats, fe_col=None):
        n_fe = int(rows[fe_col].max()) if fe_col else 0
        D = to_padded(rows, feats, fe_col=fe_col)
        res = CLogit(len(feats), n_fe=n_fe, l2_fe=1e-4 if n_fe else 0.0).fit(D)
        for f in feats:
            i = res["names"].index(f)
            out.append(dict(model=label, n_fe=n_fe, param=f, est=res["coef"][i], se=res["se"][i], ll=res["ll"]))
            print(out[-1], flush=True)

    run("M5 hubFE, alpha split", ["has_deg_nonhub", "log_deg_hub", "log_deg_nonhub"], "fe")
    order = np.argsort(-r.final_deg, kind="stable")
    for k2 in (200, 1000, 5000):
        rank = np.zeros(N, np.int64)
        top = order[:k2][r.final_deg[order[:k2]] > 0]
        rank[top] = np.arange(1, len(top) + 1)
        rows["fe2"] = rank[rows.cand_id.to_numpy()]
        run(f"M5 FE top{k2}", ["has_deg", "log_deg"], "fe2")
    rows["fe_true"] = 0
    out_true = rows.cand_id.map(pd.Series(truth["phi"]))
    rows["phi"] = out_true
    run("oracle: true phi as covariate", ["has_deg", "log_deg", "phi"])
    pd.DataFrame(out).to_csv(paths.TABLES / "p0_fitness_diag.csv", index=False)
