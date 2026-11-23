"""v2.1 stage 2 (D): which utility terms disperse demand -- five groups, self-supply vs habit, by author type.

SIR networks on the real PPS candidate sets (no feedback): each choice redrawn with probabilities
exp(u - log q) / sum within its set, u = sum of the chosen group indices (no-FE coefficients of a v2_samples
config). Variants switch groups on / off; the 'full minus self for <type>' variants remove self-supply only for the
choices of one author type, to locate where the dispersion comes from. Metrics (scripts/v2_networks.metrics): top-10
share, effective number of parents, Gini (3 draws each) and the expected-count versions (no draw noise).
usage: sbatch slurm/cpu.sbatch scripts/v2_conc5.py --tag P5      (and --tag D5)
"""
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import v2_networks as N  # noqa: E402
from v2_samples import CONFIGS, TYPES, author_types, odir  # noqa: E402
from hfpa.v2data import load_rows, write  # noqa: E402

T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def main(tag):
    cfg = CONFIGS[tag]
    S = pickle.load(open(odir(tag) / "fit.pkl", "rb"))
    feats, b, groups = S["feats"], S["nofe"]["coef"], S["groups"]
    rows, ch = load_rows("v2_pps", cfg["sample"], selfdeg=cfg["selfdeg"])
    rows = rows.sort_values("choice_id", kind="stable").reset_index(drop=True)
    X = rows[feats].to_numpy(np.float64)
    I = {g: X[:, [feats.index(f) for f in v]] @ b[[feats.index(f) for f in v]] if v else np.zeros(len(X))
         for g, v in groups.items()}
    cid = rows.choice_id.to_numpy()
    st = np.r_[0, np.flatnonzero(np.diff(cid)) + 1]
    cand, chosen = rows.cand_id.to_numpy(), rows.chosen.to_numpy() == 1
    t = ch.t.reindex(cid[st]).to_numpy()
    at = ch.author.reindex(cid).map(author_types()).to_numpy()
    full = sum(I.values())
    V = {"pop": I["pop"], "pop + self": I["pop"] + I["self"], "pop + habit": I["pop"] + I["habit"],
         "pop + self + habit": I["pop"] + I["self"] + I["habit"], "pop + match": I["pop"] + I["match"],
         "full (no FE)": full, "full - self": full - I["self"], "full - habit": full - I["habit"]}
    for ty in TYPES:
        V[f"full - self for {ty}"] = full - I["self"] * (at == ty)
    res = [dict(model="REAL", **N.metrics(cand[chosen], t))]
    top_real = pd.Series(cand[chosen]).value_counts().index[:10]
    for name, u in V.items():
        pr, _ = N.sir_probs(rows, u)
        ec = pd.Series(pr).groupby(cand).sum()
        p = np.sort(ec.to_numpy() / ec.sum())[::-1]
        p = p[p > 0]
        ms = pd.DataFrame([N.metrics(cand[N.draw(rows, pr, st, np.random.default_rng(100 + s))], t)
                           for s in range(3)]).mean().to_dict()
        res.append(dict(model=name, ll=float(np.log(pr[chosen]).sum()), top10_expected=float(p[:10].sum()),
                        eff_expected=float(np.exp(-(p * np.log(p)).sum())),
                        mass_on_real_top10=float(ec[ec.index.isin(top_real)].sum() / ec.sum()), **ms))
        log(res[-1])
    R = pd.DataFrame(res)
    cols = ["model", "ll", "top10", "eff_parents", "gini", "n_parents", "top10_expected", "eff_expected",
            "mass_on_real_top10"]
    sh = pd.Series(at[st]).value_counts(normalize=True).round(4).to_dict()
    write(R[[c for c in cols if c in R]], f"v2s_conc_{tag}",
          f"D: concentration of SIR networks by utility terms ({tag}: {cfg['sample']}, five groups, no FE, "
          "no feedback; 3 draws)", f"choices by author type: {sh}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="P5")
    main(ap.parse_args().tag)
