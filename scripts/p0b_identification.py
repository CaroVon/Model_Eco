"""Stage P0b: which estimator separates preferential attachment from latent quality? (follow-up to P0)

P0 showed that hub FE (M5) leaves a spurious alpha on a pure-fitness network and that alpha moves
with the number of FE. Here every candidate estimator is run on synthetic networks whose truth is
known, including the confounders seen in the real data (quality decay, bulk uploaders).

usage: python scripts/p0b_identification.py --scenario mix [--seed 42] [--n 50000]
Writes results/tables/p0b/<scenario>_s<seed>.csv; scripts/p0b_collect.py builds the summary.

Estimators (all include has_deg, log_deg, age_k = age_days / 1000):
  M2          no fixed effects
  M5          FE for the K hubs (chosen by final degree), all sampled candidates
  M5_EXANTE   FE for the top-K nodes by degree at the midpoint (no look-ahead), second half only
  HUB         only choices of a hub, candidate set = hubs alive at t, hub FE (identified within hub)
  SPLIT_OFF   FE for the top-K' nodes by midpoint degree, estimated on the first half, used as a fixed
              offset when estimating on the second half
  SPLIT_COV   same FE estimates, used as a covariate with a free coefficient
  RECENT      M5 + log(1 + recent30), log(1 + recent90) as quality proxies
Choice weights w = n^-gamma, n = choices of the same (author, parent) pair; SEs clustered by author.
"""
import argparse

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.models.clogit import CLogit, to_padded
from hfpa.replay.replay import HUB, Replay
from hfpa.sim.synthetic import generate_mixed

SCENARIOS = {
    "pa": dict(alpha=0.5, h=1.0, fitness_sd=0.0),
    "fit": dict(alpha=0.0, h=0.0, fitness_sd=1.0),
    "mix": dict(alpha=0.5, h=1.0, fitness_sd=1.0),
    "fit_decay": dict(alpha=0.0, h=0.0, fitness_sd=1.0, decay=1e-3),
    "mix_decay": dict(alpha=0.5, h=1.0, fitness_sd=1.0, decay=1e-3),
    "fit_bulk": dict(alpha=0.0, h=0.0, fitness_sd=1.0, bulk_frac=0.5),
    "mix_bulk": dict(alpha=0.5, h=1.0, fitness_sd=1.0, bulk_frac=0.5),
}
K, K_SPLIT = 200, 1000
BASE = ["has_deg", "log_deg", "age_k"]


def fit(rows, feats, fe_col=None, w=None, cluster=None, offset_col=None):
    n_fe = int(rows[fe_col].max()) if fe_col else 0
    D = to_padded(rows, feats, fe_col=fe_col, weight=w, offset_col=offset_col)
    return CLogit(len(feats), n_fe=n_fe, l2_fe=1e-4 if n_fe else 0.0).fit(D, cluster=cluster)


def per_choice(ch, rows_cid, col):
    """Map a per-choice column onto the choices present in rows (sorted choice ids)."""
    return ch.set_index("choice_id")[col].loc[np.unique(rows_cid)].to_numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=paths.CFG["seed"])
    ap.add_argument("--n", type=int, default=50_000)
    a = ap.parse_args()
    sc = SCENARIOS[a.scenario]
    nodes, edges, truth = generate_mixed(n=a.n, seed=a.seed, **sc)
    r = Replay(nodes, edges, {"hub_k": K, "s1": 24, "s0": 24, "seed": a.seed})
    ch, rows = r.run()
    rows["age_k"] = rows.age_days / 1000
    rows["lrec30"], rows["lrec90"] = np.log1p(rows.recent30), np.log1p(rows.recent90)
    ch["author"] = nodes.author.to_numpy()[ch.child]
    ch["n_pair"] = ch.groupby(["author", "parent"]).choice_id.transform("size")
    ch["first"] = ~ch.duplicated(["author", "parent"])
    bulk = sc.get("bulk_frac", 0) > 0
    gammas = (0.0, 0.5, 1.0) if bulk else (0.0,)
    out = []

    def run(method, sub, feats, gamma=0.0, **kw):
        w = per_choice(ch, sub.choice_id, "n_pair").astype(float) ** -gamma
        res = fit(sub, feats, w=w, cluster=per_choice(ch, sub.choice_id, "author"), **kw)
        tv = {"has_deg": sc["h"], "log_deg": sc["alpha"], "age_k": -sc.get("decay", 0) * 1000}
        for f in feats:
            i = res["names"].index(f)
            c, s, s_cl = res["coef"][i], res["se"][i], res["se_cl"][i]
            t = tv.get(f, np.nan)
            out.append(dict(scenario=a.scenario, seed=a.seed, method=method, gamma=gamma, param=f, truth=t,
                            est=c, se=s, se_cl=s_cl, z_cl=(c - t) / s_cl, n_choices=res["n"]))
        print(method, gamma, {f: round(res["coef"][res["names"].index(f)], 3) for f in feats}, flush=True)

    hub_choice = rows.loc[(rows.chosen == 1) & (rows.stratum == HUB), "choice_id"]
    hub_rows = rows[rows.choice_id.isin(hub_choice) & (rows.stratum == HUB)]
    # ex-ante node sets: rank by degree at the midpoint, so selection does not use later choices
    half = ch.choice_id.iloc[len(ch) // 2]
    t_half = ch.t.iloc[len(ch) // 2]
    birth = nodes.birth.to_numpy()
    deg_half = np.bincount(edges.parent[birth[edges.child] < t_half], minlength=len(nodes))

    def ex_ante_rank(k):
        order = np.argsort(-deg_half, kind="stable")[:k]
        order = order[deg_half[order] > 0]
        rank = np.zeros(len(nodes), np.int64)
        rank[order] = np.arange(1, len(order) + 1)
        return rank[rows.cand_id.to_numpy()]

    rows["fe2"], rows["fe3"] = ex_ante_rank(K_SPLIT), ex_ante_rank(K)
    early, late = rows[rows.choice_id < half], rows[rows.choice_id >= half].copy()
    r1 = fit(early, BASE, fe_col="fe2")
    phi_hat = np.r_[0.0, r1["coef"][len(BASE):]]
    late["phi_hat"] = phi_hat[late.fe2.to_numpy()]

    for g in gammas:
        run("M2", rows, BASE, g)
        run("M5", rows, BASE, g, fe_col="fe")
        run("M5_EXANTE", late, BASE, g, fe_col="fe3")
        run("HUB", hub_rows, BASE, g, fe_col="fe")
        run("RECENT", rows, BASE + ["lrec30", "lrec90"], g, fe_col="fe")
        run("SPLIT_OFF", late, BASE, g, offset_col="phi_hat")
        run("SPLIT_COV", late, BASE + ["phi_hat"], g)
    if bulk:
        firsts = rows[rows.choice_id.isin(ch.choice_id[ch["first"]])]
        run("M5_first", firsts, BASE, fe_col="fe")
        run("HUB_first", hub_rows[hub_rows.choice_id.isin(firsts.choice_id)], BASE, fe_col="fe")
        run("M5_reuse", rows, BASE + ["author_reuse"], fe_col="fe")
        run("HUB_reuse", hub_rows, BASE + ["author_reuse"], fe_col="fe")
    d = paths.TABLES / "p0b"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(d / f"{a.scenario}_s{a.seed}.csv", index=False)
    print(f"habit share of choices: {truth['habit'].mean():.3f}; n_choices {len(ch)}; hub choices {len(hub_choice)}")
