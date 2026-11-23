"""v2.1 stage 4: final sample-selection table.

(1) weight ladder: share of choices from the top-1% authors / author types in uploads (U), decisions (D), first
    adoptions (P);
(2) robustness: for every v2_samples config that has been fitted, Shapley shares (popularity / fit / self-supply /
    habit, in-sample, no FE) with bootstrap 95% intervals, fit / popularity ratio, holdout top-10, n, authors;
(3) heterogeneity by author type: LR test sum_type ll_type vs the pooled ll of the same sample;
(4) the pre-registered rule: robust if fit/pop is within a factor of 3 of the main config (same order) and the
    popularity share differs by <= 10 points.
usage: python scripts/v2_sample_summary.py
"""
import pickle

import numpy as np
import pandas as pd
from scipy.stats import chi2

from hfpa import paths
from hfpa.v2data import cdir, write

import sys
sys.path.insert(0, "scripts")
from v2_samples import CONFIGS, TYPES, odir  # noqa: E402

MAIN = "P5"
GROUPS = ["popularity", "fit (quality + match)", "self-supply", "habit", "ratio fit / popularity"]


def ladder():
    ch = pd.read_parquet(cdir("v2_pps") / "decisions.parquet", columns=["author", "dfirst24", "pair_first"])
    T = pd.read_parquet(paths.INTERIM / "v2_author_types.parquet").set_index("author_code").atype
    ch["atype"] = ch.author.map(T).fillna("unclassified")
    out = []
    for lab, s in (("U: all uploads", ch), ("D: 24h decisions", ch[ch.dfirst24]), ("P: first adoptions", ch[ch.pair_first])):
        n_a = s.author.value_counts()
        top = n_a.index[: int(np.ceil(len(n_a) * 0.01))]
        r = dict(sample=lab, choices=len(s), authors=len(n_a), top1pct_share=s.author.isin(top).mean(),
                 choices_per_author=len(s) / len(n_a))
        for t in TYPES + ["unclassified"]:
            r[t] = (s.atype == t).mean()
        out.append(r)
    return pd.DataFrame(out)


def robustness():
    rows, ll = [], {}
    for tag, cfg in CONFIGS.items():
        f = odir(tag) / "summary.pkl"
        if not f.exists():
            continue
        S = pickle.load(open(f, "rb"))
        O = S["shares"]
        O = O[(O.measure == "in-sample") & (O.variant == "no FE") & (O.partition == "pop / fit / ...")].set_index("group")
        r = dict(config=tag, sample=cfg["sample"], groups=cfg["groups"], weight=cfg["weight"], selfdeg=cfg["selfdeg"],
                 subset=cfg["atype"] or (f"drop top {cfg['drop_top']:.0%}" if cfg["drop_top"] else "all"),
                 n_train=S["info"]["n_train"], authors=S["info"]["authors"], hold_top10=S["info"]["hold_top10"])
        for g in GROUPS:
            if g in O.index:
                r[g] = O.share[g]
                if "lo" in O:
                    r[f"{g} lo"], r[f"{g} hi"] = O.lo[g], O.hi[g]
        r["reps"] = O.reps.max() if "reps" in O else 0
        rows.append(r)
        ll[tag] = (S["info"]["ll"], len(S["coef"]))
    return pd.DataFrame(rows), ll


def hetero(ll):
    out = []
    for smp, pooled in (("P", "P5"), ("D", "D5")):
        parts = [f"{smp}5_{t}" for t in TYPES if f"{smp}5_{t}" in ll]
        if pooled not in ll or len(parts) < 2:
            continue
        l_sep = sum(ll[p][0] for p in parts)
        k = ll[pooled][1]
        stat = 2 * (l_sep - ll[pooled][0])
        df = sum(ll[p][1] for p in parts) - k
        out.append(dict(sample=smp, types=len(parts), ll_pooled=ll[pooled][0], ll_by_type=l_sep, lr=stat, df=df,
                        p=chi2.sf(stat, df), note="all types fitted" if len(parts) == len(TYPES) else "partial"))
    return pd.DataFrame(out)


def rule(R):
    if MAIN not in set(R.config):
        return R
    m = R.set_index("config").loc[MAIN]
    R["d_pop_pts"] = 100 * (R["popularity"] - m["popularity"])
    R["ratio_vs_main"] = R["ratio fit / popularity"] / m["ratio fit / popularity"]
    R["robust"] = (R.d_pop_pts.abs() <= 10) & (R.ratio_vs_main.between(1 / 3, 3))
    return R


if __name__ == "__main__":
    write(ladder(), "v2s_ladder", "Weight of few authors along the sample ladder (U -> D -> P), by author type")
    R, ll = robustness()
    write(rule(R), "v2s_robustness", "v2.1 final: Shapley shares by sample, weights and author subset (in-sample, no FE)",
          f"Main = {MAIN} (first adoptions, five groups, one weight per choice). Rule fixed before the runs: robust if "
          "|popularity share - main| <= 10 points and the fit/popularity ratio is within a factor 3 of the main.")
    H = hetero(ll)
    if len(H):
        write(H, "v2s_hetero", "Heterogeneity by author type: LR test of separate type models vs the pooled model")
