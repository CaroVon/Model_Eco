"""v2 stage 6 (D, E.1, H): per-choice classification, welfare of the status quo, rarefied concentration.

--classify  plan-A choices of v2_pps. Comparison set = the sampled candidates that already have a child at the time of
            the choice (certainty stratum pi = 1 and the positive-degree Poisson draws, ~50 per choice; with pi ∝ d + 1
            only ~10 nodes reach pi = 1) plus the chosen parent. Group indices I_g = theta_g . x (v2 model without FE). Ties (H): the chosen
            parent is "best" on match with probability 1[I_match = max] / (number tied at the max) (random tie-breaking),
            same for popularity; type probabilities fit-type / popularity-type / both / neither assume independent
            tie-breaking; number of ties reported. Best-match candidate j* (ties broken at random) and its indices are
            stored (D3). Choices need a demand signal and variation in I_match, otherwise "no demand signal".
--welfare   D: loss from not choosing the best match, all groups changed together:
              Delta_k = sum_g beta_g (I^g_{k,j*} - I^g_{k,j}),
            (i) main: retention (v2_retention.py --part binary, logit coefficients on the four indices, 24h decisions):
                Delta Pr(stay) ~ Delta_k p(1-p); match-only version beta_match * fit_gap * p(1-p) as in plan 2.5;
            (ii) auxiliary: log(1 + children by OTHER authors within 180 days of the child's birth), calibrated on the
                indices of the chosen parent with the child's own card (txt_has) as control, month + modality FE, with /
                without author FE (alternating projections), SE clustered by parent.
            Two units: per mismatched choice (fit_gap > 0) and per choice (all plan-A choices).
--rarefy    E.1: equal-size random subsets of each choice type (size = the smallest type), 1,000 draws; Top-10 share
            and effective number of parents of the chosen parents.
usage: sbatch slurm/cpu.sbatch scripts/v2_welfare.py --classify ; python scripts/v2_welfare.py --welfare --rarefy
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm

from hfpa import paths
from hfpa.features.groups import GROUPS, MAIN, SIGNAL
from hfpa.v2data import cdir, load_rows, write

SPEC = "v2_pps"
WIN = 180 * 86400
HUB, POS = 0, 1


def part_classify():
    rows, ch = load_rows(SPEC, "planA")
    C = pickle.load(open(cdir(SPEC) / "v2_est" / "clogit.pkl", "rb"))
    feats, b = C["feats"], C["nofe"]["coef"]
    I = {}
    for g in GROUPS:
        cols = [f for f in feats if f in GROUPS[g]]
        I[g] = rows[cols].to_numpy(np.float64) @ np.array([b[feats.index(c)] for c in cols]) if cols else np.zeros(len(rows))
    sig = (rows.groupby("choice_id")[SIGNAL].transform("max").max(1) > 0).to_numpy()
    D = pd.DataFrame({"choice_id": rows.choice_id.to_numpy(), "cand": rows.cand_id.to_numpy(),
                      "chosen": rows.chosen.to_numpy(), "hub": np.isin(rows.stratum.to_numpy(), [HUB, POS]), "sig": sig,
                      **{f"I_{g}": v for g, v in I.items()}})
    del rows
    D = D[D.hub | (D.chosen == 1)].reset_index(drop=True)
    rng = np.random.default_rng(0)
    D["tb"] = rng.random(len(D))  # tie-breaker for j*
    g = D.groupby("choice_id")
    n = g.cand.transform("size")
    D = D[n.to_numpy() >= 2]
    g = D.groupby("choice_id")
    eps = 1e-9
    mx_m, mn_m, mx_p = g.I_match.transform("max"), g.I_match.transform("min"), g.I_pop.transform("max")
    D["at_max_m"] = D.I_match >= mx_m - eps
    D["at_max_p"] = D.I_pop >= mx_p - eps
    ties_m = D.groupby("choice_id").at_max_m.sum()
    ties_p = D.groupby("choice_id").at_max_p.sum()
    c = D[D.chosen == 1].set_index("choice_id")
    js = D[D.at_max_m].sort_values("tb").drop_duplicates("choice_id").set_index("choice_id")
    out = pd.DataFrame(index=c.index)
    out["n"] = g.size()
    out["signal"] = c.sig & ((mx_m - mn_m).groupby(D.choice_id).first() > eps)
    out["ties_match"] = ties_m
    out["ties_pop"] = ties_p
    out["p_fit"] = c.at_max_m / ties_m
    out["p_pop"] = c.at_max_p / ties_p
    out["fit_gap"] = mx_m.groupby(D.choice_id).first() - c.I_match
    out["pop_gap"] = mx_p.groupby(D.choice_id).first() - c.I_pop
    out["parent"] = c.cand
    out["jstar"] = js.cand.reindex(out.index)
    for gname in GROUPS:
        out[f"I_{gname}"] = c[f"I_{gname}"]
        out[f"Istar_{gname}"] = js[f"I_{gname}"].reindex(out.index)
    s = out.signal
    out["P_both"] = np.where(s, out.p_fit * out.p_pop, 0)
    out["P_fit"] = np.where(s, out.p_fit * (1 - out.p_pop), 0)
    out["P_pop"] = np.where(s, (1 - out.p_fit) * out.p_pop, 0)
    out["P_neither"] = np.where(s, (1 - out.p_fit) * (1 - out.p_pop), 0)
    out["P_nosignal"] = (~s).astype(float)
    out["type"] = np.where(~s, "no demand signal", np.select(
        [out.p_fit * out.p_pop >= 0.5, out.p_fit >= 0.5, out.p_pop >= 0.5], ["both", "fit-type", "popularity-type"],
        "neither"))
    out.reset_index().to_parquet(cdir(SPEC) / "v2_est" / "classify.parquet", index=False)
    tab = []
    tot = len(out)
    nsig = s.sum()
    for k, lab in (("P_fit", "fit-type"), ("P_pop", "popularity-type"), ("P_both", "both"), ("P_neither", "neither"),
                   ("P_nosignal", "no demand signal")):
        w = out[k]
        tab.append(dict(type=lab, expected_n=w.sum(), share_all=w.sum() / tot,
                        share_with_signal=w.sum() / nsig if lab != "no demand signal" else np.nan,
                        mean_fit_gap=np.average(out.fit_gap, weights=w) if w.sum() > 0 else np.nan))
    tab.append(dict(type="(ties) choices where the chosen parent ties others at the max match index",
                    expected_n=float(((out.ties_match > 1) & out.p_fit.gt(0)).sum())))
    tab.append(dict(type="(ties) mean number tied at the max match index, choices with signal",
                    expected_n=float(out.loc[s, "ties_match"].mean())))
    write(pd.DataFrame(tab), "v2_classify", "Per-choice classification among the sampled candidates with degree > 0 "
          "(plan A, v2 model; expected shares under random tie-breaking)")


def other_author_children(kid, nodes):
    birth, author = nodes.birth.to_numpy(), nodes.author_code.to_numpy()
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    m = E[E.rel_type.isin(MAIN)]
    c, p = m.child_idx.to_numpy(), m.parent_idx.to_numpy()
    lag = birth[c] - birth[p]
    ok = (lag >= 0) & (lag < WIN) & (author[c] != author[p])
    cnt = pd.Series(p[ok]).value_counts()
    y = np.log1p(cnt.reindex(kid).fillna(0).to_numpy())
    return np.where(birth[kid] <= birth.max() - WIN, y, np.nan)


def demean(df, cols, fes, n_iter=50, tol=1e-8):
    X = df[cols].astype(float).copy()
    for _ in range(n_iter):
        prev = X.to_numpy().copy()
        for f in fes:
            X = X - X.groupby(df[f].to_numpy()).transform("mean")
        if np.abs(X.to_numpy() - prev).max() < tol:
            break
    return X


def part_welfare():
    K = pd.read_parquet(cdir(SPEC) / "v2_est" / "classify.parquet").set_index("choice_id")
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet").set_index("choice_id")
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code"])
    K = K.join(ch[["child", "author", "modality", "t"]])
    gs = list(GROUPS)
    dI = np.column_stack([K[f"Istar_{g}"] - K[f"I_{g}"] for g in gs])
    mism = K.fit_gap.to_numpy() > 1e-9
    out = []
    # (i) retention
    RB = pd.read_csv(paths.TABLES / "v2" / "v2_retention_binary.csv")
    RB = RB[(RB.window_h == 24) & (RB.model == "logit")].set_index("var")
    RS = pd.read_parquet(cdir(SPEC) / "v2_est" / "retention_sample.parquet")
    pbar = RS.R.mean()
    beta = np.array([RB.coef.get(f"I_{g}", 0.0) for g in gs])
    se = np.array([RB.se.get(f"I_{g}", 0.0) for g in gs])
    full = dI @ beta * pbar * (1 - pbar)
    match_only = beta[gs.index("match")] * K.fit_gap.to_numpy() * pbar * (1 - pbar)
    for lab, v in (("retention: all groups changed (Delta_k)", full), ("retention: match only (beta_fit fit_gap)", match_only)):
        out.append(dict(outcome="Pr(stay with the parent at the next decision)", measure=lab, mean_baseline=pbar,
                        per_mismatched_choice=np.nanmean(v[mism]), per_choice=np.nanmean(v), n_mismatched=int(mism.sum()),
                        n=len(v), beta_match=beta[gs.index("match")], se_beta_match=se[gs.index("match")]))
    # (ii) grandchildren by other authors
    tr = np.load(paths.INTERIM / "v2_text_rows.npz")
    kid = K.child.to_numpy()
    K["y"] = other_author_children(kid, nodes)
    K["txt_has_child"] = (tr["ntok_child"][kid] >= 5).astype(float)
    K["month"] = (K.t // (30 * 86400)).astype(int)
    xs = [f"I_{g}" for g in gs if K[f"I_{g}"].std() > 0]
    for fes in (["month", "modality"], ["month", "modality", "author"]):
        d = K.dropna(subset=["y"] + xs)
        Z = demean(d, ["y"] + xs + ["txt_has_child"], fes)
        r = sm.OLS(Z.y, Z[xs + ["txt_has_child"]]).fit(cov_type="cluster", cov_kwds={"groups": d.parent.to_numpy()})
        bb = np.array([r.params.get(f"I_{g}", 0.0) for g in gs])
        V = r.cov_params().reindex(index=[f"I_{g}" for g in gs], columns=[f"I_{g}" for g in gs]).fillna(0).to_numpy()
        for lab, rows_ in (("per mismatched choice", mism), ("per choice", np.ones(len(K), bool))):
            m_dI = np.nanmean(dI[rows_], 0)
            val = m_dI @ bb
            sev = float(np.sqrt(m_dI @ V @ m_dI))
            out.append(dict(outcome="log(1 + other authors' children within 180d)", measure=f"Delta_k, FE {'+'.join(fes)}, "
                            f"{lab}", per_choice=val, se=sev, pct=np.expm1(val), n=int(rows_.sum()), n_calib=len(d),
                            beta_match=r.params.get("I_match", np.nan), se_beta_match=r.bse.get("I_match", np.nan),
                            beta_txt_has=r.params["txt_has_child"]))
    write(pd.DataFrame(out), "v2_welfare", "Welfare of the status quo: loss from not choosing the best-matched candidate "
          "(all group indices changed together, D3), retention (main) and other authors' children (auxiliary)")


def part_rarefy(n_draw=1000):
    K = pd.read_parquet(cdir(SPEC) / "v2_est" / "classify.parquet")
    K = K[K["type"] != "no demand signal"]
    m = K.groupby("type").size().min()
    rng = np.random.default_rng(0)
    out = []
    for t, g in K.groupby("type"):
        par = g.parent.to_numpy()
        tops, effs = [], []
        for _ in range(n_draw):
            s = par[rng.choice(len(par), m, replace=False)]
            c = pd.Series(s).value_counts().to_numpy() / m
            tops.append(c[:10].sum())
            effs.append(np.exp(-(c * np.log(c)).sum()))
        out.append(dict(type=t, n_type=len(g), subsample=m, top10=np.mean(tops), top10_lo=np.percentile(tops, 2.5),
                        top10_hi=np.percentile(tops, 97.5), eff_parents=np.mean(effs),
                        eff_lo=np.percentile(effs, 2.5), eff_hi=np.percentile(effs, 97.5)))
    write(pd.DataFrame(out), "v2_rarefied_concentration", f"E.1 concentration by choice type, equal subsamples of {m} "
          f"choices ({n_draw} draws)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--welfare", action="store_true")
    ap.add_argument("--rarefy", action="store_true")
    a = ap.parse_args()
    if a.classify:
        part_classify()
    if a.welfare:
        part_welfare()
    if a.rarefy:
        part_rarefy()
