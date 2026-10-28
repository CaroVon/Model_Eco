"""Stage P0: recover known parameters on synthetic growth models (plan 6.1)."""
import pandas as pd

from hfpa import paths
from hfpa.models.clogit import CLogit, to_padded
from hfpa.replay.replay import Replay
from hfpa.sim.synthetic import generate

N, K, SEED = 50_000, 200, paths.CFG["seed"]
STRAT = {"hub_k": K, "s1": 24, "s0": 24, "seed": SEED}
UNIF = {"sampling": "uniform", "s": 50, "hub_k": K, "seed": SEED}


def fit(rows, feats, fe=False, use_log_q=True):
    D = to_padded(rows, feats, fe_col="fe" if fe else None, use_log_q=use_log_q)
    n_fe = int(rows.fe.max()) if fe else 0
    return CLogit(len(feats), n_fe=n_fe, l2_fe=1e-4 if fe else 0.0).fit(D)


def record(out, net, spec, model, res, truth):
    for name in ("has_deg", "log_deg"):
        i = res["names"].index(name)
        t = truth["alpha"] if name == "log_deg" else truth["has_deg"]
        c, s = res["coef"][i], res["se"][i]
        out.append(dict(network=net, sampling=spec, model=model, param=name, truth=t, est=c, se=s,
                        z_vs_truth=(c - t) / s, within_2se=abs(c - t) <= 2 * s, ll=res["ll"], n=res["n"]))
        print(out[-1], flush=True)


if __name__ == "__main__":
    out = []
    for alpha in (0.0, 0.5, 1.0):
        nodes, edges, truth = generate(n=N, kind="pa", alpha=alpha, beta0=-1.0, seed=SEED)
        net = f"pa_alpha{alpha}"
        _, rs = Replay(nodes, edges, STRAT).run()
        _, ru = Replay(nodes, edges, UNIF).run()
        record(out, net, "stratified", "M2", fit(rs, ["has_deg", "log_deg"]), truth)
        record(out, net, "stratified_no_logq", "M2", fit(rs, ["has_deg", "log_deg"], use_log_q=False), truth)
        record(out, net, "uniform", "M2", fit(ru, ["has_deg", "log_deg"]), truth)
    nodes, edges, truth = generate(n=N, kind="fitness", fitness_sd=1.0, seed=SEED)
    _, rs = Replay(nodes, edges, STRAT).run()
    record(out, "fitness_sd1", "stratified", "M2", fit(rs, ["has_deg", "log_deg"]), truth)
    record(out, "fitness_sd1", "stratified", "M5_hubFE", fit(rs, ["has_deg", "log_deg"], fe=True), truth)
    paths.ensure()
    df = pd.DataFrame(out)
    df.to_csv(paths.TABLES / "p0_synth.csv", index=False)
    print(df.to_string())
