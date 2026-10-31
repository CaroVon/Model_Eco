"""P16: welfare of the current choices (status quo; no counterfactual yet).

Uses the full plan-A conditional logit of P15 (utility u_j = theta . x_j, random-utility interpretation).
Per choice k with sampled set C (importance weights exp(-log q)):
  logsum      W_k = log sum_C exp(u_j - log q_j), the full-set expected maximum utility up to a constant;
              reported relative to random choice: W_k - log sum_C exp(-log q_j) = log of the mean of exp(u) over
              the market.
  advantage   u_chosen - E_market[u] by feature group: which mechanism the realised utility comes from.
  fit loss    max match index among the sampled hub candidates minus the chosen parent's (P15 classify).
Calibration to ecosystem units: the child's later success, regressed on the indices of the parent it chose.
  main        log(1 + children within 180 days of the child's birth), children born >= 180 days before the
              snapshot
  robustness  log(1 + likes) of the child; child's leaderboard bench_z (P14, text models only)
  controls    month FE, modality FE, with / without author FE (alternating projections); SE clustered by parent
The match coefficient converts the fit loss into outcome units. Ecosystem status: concentration and the
effective number of parents (exp entropy) over all first author-parent choices and by choice type.

usage: python scripts/p16_welfare.py
Writes results/tables/p16_welfare_*.md / .csv.
"""
import importlib.util

import numpy as np
import pandas as pd
import statsmodels.api as sm

from hfpa import paths

_s = importlib.util.spec_from_file_location("p15", paths.REPO / "scripts" / "p15_decomp.py")
P15 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P15)
P5 = P15.P5
WIN = 180 * 86400


def demean(df, cols, fes, n_iter=50, tol=1e-8):
    """Partial out several sets of fixed effects by alternating projections."""
    X = df[cols].astype(float).copy()
    for _ in range(n_iter):
        prev = X.to_numpy().copy()
        for f in fes:
            X = X - X.groupby(df[f].to_numpy()).transform("mean")
        if np.abs(X.to_numpy() - prev).max() < tol:
            break
    return X


def regress(df, y, xs, fes, cluster):
    d = df.dropna(subset=[y] + xs)
    Z = demean(d, [y] + xs, fes)
    r = sm.OLS(Z[y], Z[xs]).fit(cov_type="cluster", cov_kwds={"groups": d[cluster].to_numpy()})
    return r, len(d)


XS = ["x_match", "x_pop", "x_base", "x_qual", "x_inertia", "x_local", "x_avail"]


def add_outcomes(A, kid, t, nodes):
    """Outcomes of the child models kid (row-aligned with A) and the month of the choice."""
    birth = nodes.birth.to_numpy()
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    m = E[E.rel_type.isin(P5.MAIN)]
    tmax = birth.max()
    lag = birth[m.child_idx.to_numpy()] - birth[m.parent_idx.to_numpy()]
    early = m[(lag >= 0) & (lag < WIN)].parent_idx.value_counts()
    A["y_children180"] = np.where(birth[kid] <= tmax - WIN, np.log1p(early.reindex(kid).fillna(0).to_numpy()), np.nan)
    A["y_likes"] = np.log1p(nodes.likes.to_numpy(float)[kid])
    B = pd.read_parquet(paths.PROCESSED / "nodes" / "bench.parquet")
    A["y_bench"] = B.set_index("node").bench_z.reindex(kid).to_numpy()
    A["month"] = (t // (30 * 86400)).astype(int)


def calibrate(A, outcomes=("y_children180", "y_likes", "y_bench")):
    cal = []
    for y in outcomes:
        for fes in (["month", "modality"], ["month", "modality", "author"]):
            try:
                rr, n = regress(A, y, XS, fes, "parent")
            except Exception as e:  # noqa: BLE001  (too few observations for y_bench with author FE)
                print(y, fes, e)
                continue
            for x in XS:
                cal.append(dict(outcome=y, fe="+".join(fes), x=x, coef=rr.params[x], se=rr.bse[x], n=n))
    return pd.DataFrame(cal)


if __name__ == "__main__":
    rows, ch = P15.load()
    r, g = P15.sample(rows, ch, "planA")
    feats = P15.drop_constant(r, P15.ALL)
    res = P15.fit(r, ch, feats, g, cluster=False)
    b = dict(zip(feats, res["coef"][: len(feats)]))
    groups = list(P15.GROUPS)
    U = {gname: sum(b.get(f, 0) * r[f].to_numpy(float) for f in P15.GROUPS[gname]) for gname in groups}
    u = sum(U.values())
    lq = r.log_q.to_numpy(float)
    cid = r.choice_id.to_numpy()
    D = pd.DataFrame({"choice_id": cid, "chosen": r.chosen.to_numpy(), "u": u, "lq": lq,
                      **{f"u_{k}": v for k, v in U.items()}})
    D["w"] = np.exp(-D.lq)  # importance weight of each sampled candidate for the full market
    gb = D.groupby("choice_id")
    W = gb.apply(lambda x: np.log(np.sum(np.exp(x.u - x.lq))) - np.log(np.sum(x.w)))
    wsum = gb.w.sum()
    adv = {}
    chosen = D[D.chosen == 1].set_index("choice_id")
    for k in ["u"] + [f"u_{x}" for x in groups]:
        mean_mkt = (D[k] * D.w).groupby(D.choice_id).sum() / wsum
        adv[k] = chosen[k] - mean_mkt.loc[chosen.index]
    A = pd.DataFrame(adv)
    A["logsum_rel"] = W.loc[A.index]
    C = pd.read_parquet(paths.PROCESSED / "choices" / P15.SPEC / "p15_classify.parquet")
    A = A.join(C[["type", "fit_gap", "pop_gap"]])
    A["modality"] = ch.modality_name.loc[A.index].to_numpy()
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent", "t"])
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code", "likes", "major_org"])
    a = nodes.author_code.to_numpy()
    nb = full.assign(a=a[full.child]).groupby("a").size()
    au = a[ch.child.loc[A.index].to_numpy()]
    A["author_type"] = np.where(nodes.major_org.to_numpy()[ch.child.loc[A.index].to_numpy()], "major org",
                                np.where(nb.reindex(au).to_numpy() >= 100, "bulk (>= 100 choices)", "other"))
    A.to_parquet(paths.PROCESSED / "choices" / P15.SPEC / "p16_welfare.parquet")

    # 1. welfare level and its composition by mechanism
    comp = []
    for cut, key in [("all", pd.Series("all", index=A.index)), ("author type", A.author_type),
                     ("modality", A.modality), ("choice type", A["type"])]:
        for v, idx in key.groupby(key).groups.items():
            x = A.loc[idx]
            row = dict(cut=cut, value=v, n=len(x), logsum_rel=x.logsum_rel.mean(), realised_adv=x.u.mean(),
                       fit_gap=x.fit_gap.mean())
            for gname in groups:
                row[f"adv_{gname}"] = x[f"u_{gname}"].mean()
                row[f"share_{gname}"] = x[f"u_{gname}"].mean() / x.u.mean()
            comp.append(row)
    P5.write(pd.DataFrame(comp), "p16_welfare_composition",
             "P16 welfare (utility units): logsum vs random choice, realised utility advantage of the chosen parent by mechanism, fit loss")

    # 2. calibration: child's later success on the chosen parent's indices
    kid = ch.child.loc[A.index].to_numpy()
    add_outcomes(A, kid, ch.t.loc[A.index].to_numpy(), nodes)
    A["author"] = au
    A["parent"] = ch.parent.loc[A.index].to_numpy()
    # indices of the chosen parent (levels, not advantages)
    for k in groups:
        A[f"x_{k}"] = chosen.loc[A.index, f"u_{k}"].to_numpy()
    cal = calibrate(A)
    P5.write(cal, "p16_welfare_calibration", "P16 child success on the chosen parent's utility indices (per utility unit; SE clustered by parent)")
    # fit loss in outcome units, under both calibrations (user decision: report both; with author FE the match
    # effect is about zero, so the without-FE number is an upper bound)
    conv = []
    for fes in ("month+modality", "month+modality+author"):
        sel = cal[(cal.outcome == "y_children180") & (cal.fe == fes) & (cal.x == "x_match")]
        if not len(sel):
            continue
        beta, se = float(sel.coef.iloc[0]), float(sel.se.iloc[0])
        for t, gap in A.groupby("type").fit_gap.mean().items():
            conv.append(dict(calibration=fes, beta_match=beta, beta_se=se, type=t, fit_gap_util=gap,
                             fit_loss_pct_children180=np.expm1(beta * gap) * 100))
    P5.write(pd.DataFrame(conv), "p16_welfare_fitloss_outcome",
             "P16 fit loss in outcome units (% of children within 180 days), with and without author FE")

    # 3. ecosystem status: concentration and effective number of parents
    f = full.assign(a=a[full.child]).sort_values("t", kind="stable")
    f = f[~f.duplicated(["a", "parent"])]
    eco = []

    def stats(p):
        s = p.value_counts(normalize=True)
        return dict(n=len(p), top10_share=s.head(10).sum(), hhi=float((s ** 2).sum()), eff_parents=float(np.exp(-(s * np.log(s)).sum())))
    eco.append(dict(set="all first author-parent choices (full data)", **stats(f.parent)))
    for t, idx in A.groupby("type").groups.items():
        eco.append(dict(set=f"10% sample, {t}", **stats(A.loc[idx, "parent"])))
    P5.write(pd.DataFrame(eco), "p16_welfare_ecosystem", "P16 ecosystem status: concentration and effective number of parents")
