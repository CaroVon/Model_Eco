"""v2 stage 5 (part 2 of the v2 plan): repeated choices -- stay or switch.

Decisions: consecutive uploads of one author on one parent with gaps <= 24h are one decision (v2_features --prep;
6h / 72h for sensitivity). Unit of analysis = decision.

--part sessions  choices vs decisions under the three windows; uploads per decision.
--part binary    (2.3) sample: first decision of each author-parent pair (plan A) with a later decision by the author.
                 R = 1 if the next decision uses the same parent (also: again within 180 days).
                 Regressors: the parent's group indices at the first decision (I_g = theta_g . x_chosen, v2 plan-A
                 model without FE). Controls: log gap to the next decision, next decision has the same task (child
                 pipeline_tag), a newer version of the parent's family released in between (same organisation and
                 family token, final degree >= 20), author experience (log 1 + earlier decisions), modality, half-year.
                 Models: logit (SE clustered by author); conditional logit with author FE (statsmodels
                 ConditionalLogit); same-task subsample; stay / upgrade / switch multinomial logit; satisfaction test
                 adding the outcome of the child made with p, as seen at the next decision (other authors' children of
                 that child born before the next decision) and its likes. Also 6h / 72h windows.
--part nested    (2.4) two-nest logit (hfpa.models.nested2) on every decision of v2_pps (first and repeat): nest 1 =
                 parents the author used before (H_k, pi = 1), nest 0 = the PPS sample of all other models. Utility =
                 the four groups (shared beta) + delta 1[j in H] + kappa' (1[j in H] r_kj), r_kj = outcome of the author's
                 last child on j as seen at t (log 1 + other authors' children before t), log days since last use, a newer
                 family version released since the last use. Reports lambda_new, lambda_old, delta (also in log-degree
                 units), kappa; P(old nest) by author type.
usage: python scripts/v2_retention.py --part sessions|binary ; sbatch slurm/gpu.sbatch scripts/v2_retention.py --part nested
"""
import argparse
import glob
import pickle
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hfpa import paths
from hfpa.features.groups import ALL, GROUPS, MAIN, NAMES
from hfpa.v2data import TABLES, cdir, write

SPEC = "v2_pps"
DAY = 86400
T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def node_info():
    n = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                        columns=["model_id", "birth", "author_code", "pipeline_tag", "major_org"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    m = E[E.rel_type.isin(MAIN)]
    n["org"] = n.model_id.str.split("/").str[0]
    n["fam"] = n.model_id.str.split("/").str[-1].str.lower().str.extract(r"^([a-z]+)", expand=False).fillna("")
    n["deg"] = np.bincount(m.parent_idx, minlength=len(n))
    likes = pd.read_parquet(paths.INTERIM / "meta.parquet", columns=["model_id", "likes"]).set_index("model_id").likes
    n["likes"] = likes.reindex(n.model_id).fillna(0).to_numpy()
    return n, m


def kids_before(m, n, child, t):
    """Number of other-author main children of `child` born strictly before time t (vectorised)."""
    birth, author = n.birth.to_numpy(), n.author_code.to_numpy()
    e = pd.DataFrame({"p": m.parent_idx.to_numpy(), "b": birth[m.child_idx.to_numpy()],
                      "oa": author[m.child_idx.to_numpy()] != author[m.parent_idx.to_numpy()]})
    e = e[e.oa].sort_values(["p", "b"])
    pk = e.p.to_numpy()
    bk = e.b.to_numpy()
    lo = np.searchsorted(pk, child, "left")
    hi = np.searchsorted(pk, child, "right")
    out = np.zeros(len(child), np.int64)
    for i in np.flatnonzero(hi > lo):
        out[i] = np.searchsorted(bk[lo[i]:hi[i]], t[i], "left")
    return out


def decisions(w=24):
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet")
    d = ch[ch[f"dfirst{w}"]].sort_values(["author", "t"], kind="stable").reset_index(drop=True)
    g = d.groupby("author")
    d["next_parent"] = g.parent.shift(-1)
    d["next_t"] = g.t.shift(-1)
    d["next_child"] = g.child.shift(-1)
    d["n_prev"] = g.cumcount()
    return d, ch


# ---------------------------------------------------------------- sessions
def part_sessions():
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet")
    out = []
    for w in (6, 24, 72):
        nd = int(ch[f"dfirst{w}"].sum())
        per = ch.groupby(f"sess{w}").size()
        out.append(dict(window_h=w, choices=len(ch), decisions=nd, pairs=int(ch.pair_first.sum()),
                        repeat_decisions=nd - int(ch.pair_first.sum()), uploads_per_decision=per.mean(),
                        p99_uploads=per.quantile(0.99), max_uploads=per.max()))
    write(pd.DataFrame(out), "v2_sessions", "Choices merged into decisions (same author, same parent, gaps <= window)")


# ---------------------------------------------------------------- indices of the chosen parent
def chosen_indices():
    """Group indices theta_g . x of the chosen parent at each featurised choice (v2 plan-A model without FE)."""
    C = pickle.load(open(cdir(SPEC) / "v2_est" / "clogit.pkl", "rb"))
    feats, b = C["feats"], C["nofe"]["coef"]
    parts = []
    for f in sorted(glob.glob(str(cdir(SPEC) / "feats" / "*.parquet"))):
        if f.endswith(".notext.parquet") or not pq.read_schema(f).names:
            continue
        x = pd.read_parquet(f, columns=["choice_id", "chosen", "txt_has", "hist_has", "ds_has"] + feats,
                            filters=[("chosen", "==", 1)])
        parts.append(x)
    X = pd.concat(parts, ignore_index=True).set_index("choice_id")
    out = pd.DataFrame(index=X.index)
    for g in GROUPS:
        cols = [f for f in feats if f in GROUPS[g]]
        out[f"I_{g}"] = X[cols].to_numpy(float) @ np.array([b[feats.index(c)] for c in cols]) if cols else 0.0
    out["signal"] = (X[["txt_has", "hist_has", "ds_has"]].max(1) > 0).astype(float)
    return out


# ---------------------------------------------------------------- binary
def binary_sample(w, n, m, idx):
    d, _ = decisions(w)
    s = d[d.pair_first & d.next_parent.notna()].copy()
    s["next_parent"] = s.next_parent.astype(np.int64)
    s["next_child"] = s.next_child.astype(np.int64)
    s["R"] = (s.next_parent == s.parent).astype(float)
    # again within 180 days (any later decision of the author on p)
    later = d[["author", "parent", "t"]].rename(columns={"t": "t2"})
    mm = s[["choice_id", "author", "parent", "t"]].merge(later, on=["author", "parent"])
    mm = mm[(mm.t2 > mm.t) & (mm.t2 <= mm.t + 180 * DAY)]
    s["R180"] = s.choice_id.isin(mm.choice_id).astype(float)
    tag = n.pipeline_tag.to_numpy()
    s["same_task"] = (tag[s.child.to_numpy()] == tag[s.next_child.to_numpy()]).astype(float)
    s["lgap"] = np.log1p((s.next_t - s.t) / DAY)
    s["lexp"] = np.log1p(s.n_prev)
    # newer family version between t and the next decision: same org and family token, final degree >= 20
    org, fam, birth, deg = n.org.to_numpy(), n.fam.to_numpy(), n.birth.to_numpy(), n.deg.to_numpy()
    key = pd.Series(org + "|" + fam)
    big = np.flatnonzero(deg >= 20)
    rel = pd.DataFrame({"k": key.to_numpy()[big], "b": birth[big], "node": big}).sort_values(["k", "b"])
    groups = {k: (g.b.to_numpy(), g.node.to_numpy()) for k, g in rel.groupby("k")}
    nv, upg = np.zeros(len(s)), np.zeros(len(s))
    for i, (p, t1, t2, q) in enumerate(zip(s.parent.to_numpy(), s.t.to_numpy(), s.next_t.to_numpy(), s.next_parent.to_numpy())):
        gk = groups.get(key.iat[p])
        if gk is None or not fam[p]:
            continue
        bb, nodes = gk
        sel = (bb > max(t1, birth[p])) & (bb <= t2) & (nodes != p)
        nv[i] = sel.any()
        upg[i] = q != p and key.iat[q] == key.iat[p] and birth[q] > birth[p]
    s["new_version"], s["upgrade"] = nv, upg
    s["y3"] = np.where(s.R == 1, 0, np.where(s.upgrade == 1, 1, 2))  # stay / upgrade / switch
    # outcome of the child made with p, as seen at the next decision
    s["child_kids"] = np.log1p(kids_before(m, n, s.child.to_numpy(), s.next_t.to_numpy()))
    s["child_likes"] = np.log1p(n.likes.to_numpy()[s.child.to_numpy()])
    s["half"] = pd.to_datetime(s.t, unit="s").dt.to_period("Q").astype(str).str[:4] + \
        np.where(pd.to_datetime(s.t, unit="s").dt.quarter <= 2, "H1", "H2")
    s = s.join(idx, on="choice_id")
    return s.dropna(subset=["I_pop"])


def logit_cl(df, y, X, cluster):
    import statsmodels.api as sm
    Z = sm.add_constant(df[X].astype(float), has_constant="add")
    r = sm.Logit(df[y].to_numpy(float), Z).fit(disp=0, maxiter=200, cov_type="cluster",
                                                 cov_kwds={"groups": pd.factorize(df[cluster])[0]})
    return r


def part_binary():
    import statsmodels.api as sm
    from statsmodels.discrete.conditional_models import ConditionalLogit
    n, m = node_info()
    idx = chosen_indices()
    out, summ = [], []
    IV = ["I_pop", "I_qual", "I_match", "I_inertia"]
    CTRL = ["lgap", "same_task", "new_version", "lexp"]
    for w in (24, 6, 72):
        s = binary_sample(w, n, m, idx)
        s = s.loc[:, ~s.columns.duplicated()]
        fx = pd.get_dummies(s[["modality", "half"]].astype(str), drop_first=True, dtype=float)
        s = pd.concat([s, fx], axis=1)
        FX = list(fx.columns)
        IVu = [v for v in IV if s[v].std() > 0]
        specs = [("logit", "R", s, IVu + CTRL + FX)]
        if w == 24:
            specs += [("logit, same task next", "R", s[s.same_task == 1], IVu + ["lgap", "new_version", "lexp"] + FX),
                      ("logit, R within 180 days", "R180", s, IVu + CTRL + FX),
                      ("logit + child outcome (satisfaction)", "R", s, IVu + CTRL + ["child_kids", "child_likes"] + FX),
                      ("logit, demand signal only", "R", s[s.signal == 1], IVu + CTRL + FX)]
        for name, y, df, X in specs:
            X = [x for x in X if df[x].std() > 0]
            r = logit_cl(df, y, X, "author")
            for v in IVu + ["child_kids", "child_likes", "same_task", "new_version", "lgap", "lexp"]:
                if v in r.params:
                    out.append(dict(window_h=w, model=name, var=v, coef=r.params[v], se=r.bse[v], z=r.tvalues[v],
                                    n=int(r.nobs), mean_y=df[y].mean()))
            log(f"{w}h {name}: n {int(r.nobs)}")
        if w != 24:
            continue
        # author FE (conditional logit): authors with variation in R
        v = s.groupby("author").R.transform(lambda x: x.nunique() > 1)
        sf = s[v]
        Xc = [x for x in IVu + CTRL if sf[x].std() > 0]
        rc = ConditionalLogit(sf.R.to_numpy(float), sf[Xc].astype(float).to_numpy(), groups=sf.author.to_numpy()).fit(disp=0)
        for i, x in enumerate(Xc):
            out.append(dict(window_h=w, model="conditional logit, author FE", var=x, coef=rc.params[i], se=rc.bse[i],
                            z=rc.params[i] / rc.bse[i], n=len(sf), mean_y=sf.R.mean()))
        log(f"author FE: n {len(sf)} authors {sf.author.nunique()}")
        # stay / upgrade / switch
        Xm = [x for x in IVu + CTRL if s[x].std() > 0]
        mn = sm.MNLogit(s.y3.to_numpy(), sm.add_constant(s[Xm].astype(float))).fit(
            disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": pd.factorize(s.author)[0]})
        for k, lab in ((0, "upgrade vs stay"), (1, "switch vs stay")):
            for x in IVu:
                out.append(dict(window_h=w, model=f"MNL: {lab}", var=x, coef=mn.params.iloc[:, k][x],
                                se=mn.bse.iloc[:, k][x], z=mn.tvalues.iloc[:, k][x], n=int(mn.nobs), mean_y=np.nan))
        summ.append(dict(window_h=w, n=len(s), stay=(s.y3 == 0).mean(), upgrade=(s.y3 == 1).mean(),
                         switch=(s.y3 == 2).mean(), R180=s.R180.mean(), same_task=s.same_task.mean(),
                         new_version=s.new_version.mean()))
        s[["choice_id", "author", "R", "R180", "y3"] + IV + CTRL + ["child_kids", "signal"]].to_parquet(
            cdir(SPEC) / "v2_est" / "retention_sample.parquet")
    write(pd.DataFrame(out), "v2_retention_binary", "Stay or switch after a first author-parent decision (2.3); "
          "group indices of the parent at the first decision; SE clustered by author",
          "I_g = utility index of group g of the parent at its first adoption (v2 plan-A model, no FE). Positive "
          "I_match: better-matched parents are kept more often (satisfaction). I_pop near 0 or negative: popularity "
          "helps only at the first choice.")
    write(pd.DataFrame(summ), "v2_retention_outcomes", "Outcome shares of the next decision (24h decisions)")


# ---------------------------------------------------------------- nested
def nested_rows():
    """Rows of every decision (H kept) with nest id and the old-nest interactions r_kj."""
    n, m = node_info()
    d, ch = decisions(24)
    C = pickle.load(open(cdir(SPEC) / "v2_est" / "clogit.pkl", "rb"))
    feats = [f for f in C["feats"] if f != "author_reuse"]
    parts = []
    for f in sorted(glob.glob(str(cdir(SPEC) / "feats" / "*.parquet"))):
        if f.endswith(".notext.parquet") or not pq.read_schema(f).names:
            continue
        parts.append(pd.read_parquet(f, columns=["choice_id", "cand_id", "chosen", "log_q", "fe", "author_reuse"] + feats))
    R = pd.concat(parts, ignore_index=True)
    R = R[R.choice_id.isin(d.choice_id)].reset_index(drop=True)
    R["nest"] = R.author_reuse.astype(np.int64)
    R.loc[R.nest == 1, "log_q"] = 0.0
    # r_kj for old-nest rows: last use of j by the author before t
    ci = d.set_index("choice_id")
    old = R.nest.to_numpy() == 1
    Q = pd.DataFrame({"row": np.flatnonzero(old), "author": ci.author.reindex(R.choice_id[old]).to_numpy(),
                      "parent": R.cand_id[old].to_numpy(), "t": ci.t.reindex(R.choice_id[old]).to_numpy()})
    allc = ch[["author", "parent", "t", "child"]].rename(columns={"t": "tu"}).sort_values("tu")
    for c in ("author", "parent"):
        Q[c], allc[c] = Q[c].astype(np.int64), allc[c].astype(np.int64)
    Q["t"], allc["tu"] = Q.t.astype(np.float64), allc.tu.astype(np.float64)
    Q, allc = Q.sort_values("t"), allc.sort_values("tu")
    J = pd.merge_asof(Q, allc, left_on="t", right_on="tu", by=["author", "parent"], allow_exact_matches=False,
                      direction="backward")
    ok = J.child.notna().to_numpy()
    kid = np.where(ok, J.child.fillna(0).to_numpy(np.int64), 0)
    kb = kids_before(m, n, kid, J.t.to_numpy())
    r_out = np.where(ok, np.log1p(kb), 0.0)
    r_days = np.where(ok, np.log1p((J.t - J.tu).fillna(0).to_numpy() / DAY), 0.0)
    # newer family version since the last use
    org, fam, birth, deg = n.org.to_numpy(), n.fam.to_numpy(), n.birth.to_numpy(), n.deg.to_numpy()
    key = org + "|" + fam
    big = np.flatnonzero(deg >= 20)
    rel = pd.DataFrame({"k": key[big], "b": birth[big]}).sort_values(["k", "b"])
    groups = {k: g.b.to_numpy() for k, g in rel.groupby("k")}
    nv = np.zeros(len(J))
    for i, (p, tu, t) in enumerate(zip(J.parent.to_numpy(), J.tu.fillna(0).to_numpy(), J.t.to_numpy())):
        bb = groups.get(key[p])
        if bb is not None and fam[p]:
            nv[i] = ((bb > max(tu, birth[p])) & (bb < t)).any()
    for name, v in (("old_outcome", r_out), ("old_ldays", r_days), ("old_newver", nv)):
        col = np.zeros(len(R), np.float32)
        col[J.row.to_numpy()] = v
        R[name] = col
    R["old"] = R.nest.astype(np.float32)
    return R, feats, d, n


def part_nested():
    import torch
    from hfpa.models.clogit import CLogit, to_buckets
    from hfpa.models.nested2 import Nested2
    R, feats, d, n = nested_rows()
    X = feats + ["old", "old_outcome", "old_ldays", "old_newver"]
    X = [x for x in X if R[x].std() > 0]
    log(f"nested rows {len(R):,}, decisions {R.choice_id.nunique():,}, old rows {(R.nest == 1).sum():,}")
    hold = d.set_index("choice_id").holdout.reindex(R.choice_id).to_numpy()
    tr = R[~hold].reset_index(drop=True)
    D = to_buckets(tr, X, extra_int={"nest": "nest"})
    author = d.set_index("choice_id").author.reindex(D.choice_ids).to_numpy()
    cl = CLogit(len(X)).fit(D, se=False)
    m = Nested2(len(X))
    m.init(cl["coef"], [0.95, 0.95])
    res = m.fit(D, cluster=author, ckpt=str(cdir(SPEC) / "v2_est" / "nested_p0.npy"))
    log(f"nested: lam {res['lam']}, ll {res['ll']:.1f} vs clogit {cl['ll']:.1f}")
    out = []
    for i, x in enumerate(res["names"]):
        out.append(dict(param=x, coef=res["coef"][i], se=res["se"][i], se_cl=res["se_cl"][i],
                        clogit_coef=cl["coef"][i] if i < len(X) else np.nan))
    T = pd.DataFrame(out)
    lam = res["lam"]
    V = res["cov_cl"]
    for k, nm in ((len(X), "lambda_new"), (len(X) + 1, "lambda_old")):
        s_ = lam[k - len(X)] * (1 - lam[k - len(X)]) * np.sqrt(V[k, k])
        T = pd.concat([T, pd.DataFrame([dict(param=nm, coef=lam[k - len(X)], se_cl=s_)])])
    i_old, i_deg = X.index("old"), X.index("log_deg")
    ratio = res["coef"][i_old] / res["coef"][i_deg]
    g = np.zeros(len(res["coef"]))
    g[i_old], g[i_deg] = 1 / res["coef"][i_deg], -res["coef"][i_old] / res["coef"][i_deg] ** 2
    T = pd.concat([T, pd.DataFrame([dict(param="delta / beta_log_deg (log-degree units)", coef=ratio,
                                          se_cl=float(np.sqrt(g @ V @ g))),
                                     dict(param="degree multiple equal to delta", coef=float(np.exp(ratio))),
                                     dict(param="ll nested", coef=res["ll"]), dict(param="ll clogit", coef=cl["ll"]),
                                     dict(param="n decisions", coef=D.n)])])
    write(T, "v2_nested", "Two-nest logit over every decision (old nest = parents used before, all included; new nest = "
          "PPS sample); shared beta for first and repeat decisions; SE clustered by author",
          "old = switching cost / inertia delta; old_outcome = log(1 + other authors' children of the author's last child "
          "on j, before t) (kappa, satisfaction); old_ldays = log days since the last use; old_newver = a newer family "
          "version released since the last use.")
    pnest = m.nest_prob(D)
    pd.DataFrame({"choice_id": D.choice_ids, "p_old": pnest}).to_parquet(cdir(SPEC) / "v2_est" / "nested_pold.parquet")
    pickle.dump(dict(res=res, X=X), open(cdir(SPEC) / "v2_est" / "nested.pkl", "wb"))


# ---------------------------------------------------------------- v2.1: by author type (E)
STRATA = {"all": None, "individuals (light + mid)": ["indiv_light", "indiv_mid"], "indiv_light": ["indiv_light"],
          "indiv_mid": ["indiv_mid"], "indiv_heavy": ["indiv_heavy"],
          "organisations": ["org_major", "org_redistributor", "org_other"]}


def part_bytype():
    """Stay / switch by author type (scripts/v2_authors.py), five indices (inertia split into self-supply =
    same_author and habit = author_family; same full model, so I_self + I_habit = I_inertia). Per stratum: pooled
    logit, author-FE conditional logit, stay / upgrade / switch MNL, satisfaction test. Welfare (retention): Delta
    Pr(stay) = beta_match * fit_gap * p(1 - p) with the author-FE beta of the stratum (classify.parquet fit gaps)."""
    import statsmodels.api as sm
    from statsmodels.discrete.conditional_models import ConditionalLogit
    n, m = node_info()
    idx = chosen_indices()
    C = pickle.load(open(cdir(SPEC) / "v2_est" / "clogit.pkl", "rb"))
    feats, b = C["feats"], C["nofe"]["coef"]
    X = pd.concat([pd.read_parquet(f, columns=["choice_id", "chosen", "same_author", "author_family"],
                                   filters=[("chosen", "==", 1)])
                   for f in sorted(glob.glob(str(cdir(SPEC) / "feats" / "*.parquet")))
                   if not f.endswith(".notext.parquet") and pq.read_schema(f).names]).set_index("choice_id")
    idx["I_self"] = X.same_author.reindex(idx.index).to_numpy(float) * b[feats.index("same_author")]
    idx["I_habit"] = (X.author_family.reindex(idx.index).to_numpy(float) * b[feats.index("author_family")]
                      if "author_family" in feats else 0.0)
    assert np.allclose(idx.I_self + idx.I_habit, idx.I_inertia, atol=1e-6)
    AT = pd.read_parquet(paths.INTERIM / "v2_author_types.parquet").set_index("author_code").atype
    s0 = binary_sample(24, n, m, idx)
    s0 = s0.loc[:, ~s0.columns.duplicated()]
    s0["atype"] = s0.author.map(AT)
    fx = pd.get_dummies(s0[["modality", "half"]].astype(str), drop_first=True, dtype=float)
    s0 = pd.concat([s0, fx], axis=1)
    FX = list(fx.columns)
    IV = ["I_pop", "I_qual", "I_match", "I_self", "I_habit"]
    CTRL = ["lgap", "same_task", "new_version", "lexp"]
    K = pd.read_parquet(cdir(SPEC) / "v2_est" / "classify.parquet").set_index("choice_id")
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet", columns=["choice_id", "author"]).set_index("choice_id")
    K["atype"] = ch.author.reindex(K.index).map(AT)
    out, summ, wel = [], [], []
    for name, types in STRATA.items():
        s = s0 if types is None else s0[s0.atype.isin(types)]
        if len(s) < 500:
            log(f"{name}: n {len(s)} too small, skipped")
            continue
        IVu = [v for v in IV if s[v].std() > 0]
        specs = [("logit", "R", s, IVu + CTRL + FX),
                 ("logit + child outcome (satisfaction)", "R", s, IVu + CTRL + ["child_kids", "child_likes"] + FX)]
        for mname, y, df, Xs in specs:
            Xs = [x for x in Xs if df[x].std() > 0]
            try:
                r = logit_cl(df, y, Xs, "author")
            except Exception as e:  # noqa: BLE001  (perfect separation in a small stratum)
                log(f"{name} {mname}: {e}")
                continue
            for v in IVu + ["child_kids", "child_likes"] + CTRL:
                if v in r.params:
                    out.append(dict(stratum=name, model=mname, var=v, coef=r.params[v], se=r.bse[v], z=r.tvalues[v],
                                    n=int(r.nobs), authors=df.author.nunique(), mean_y=df[y].mean()))
        v = s.groupby("author").R.transform(lambda x: x.nunique() > 1)
        sf = s[v]
        bfe = {}
        if len(sf) > 200:
            Xc = [x for x in IVu + CTRL if sf[x].std() > 0]
            rc = ConditionalLogit(sf.R.to_numpy(float), sf[Xc].astype(float).to_numpy(),
                                  groups=sf.author.to_numpy()).fit(disp=0)
            for i, x in enumerate(Xc):
                out.append(dict(stratum=name, model="conditional logit, author FE", var=x, coef=rc.params[i],
                                se=rc.bse[i], z=rc.params[i] / rc.bse[i], n=len(sf), authors=sf.author.nunique(),
                                mean_y=sf.R.mean()))
                bfe[x] = (rc.params[i], rc.bse[i])
        Xm = [x for x in IVu + CTRL if s[x].std() > 0]
        try:
            mn = sm.MNLogit(s.y3.to_numpy(), sm.add_constant(s[Xm].astype(float))).fit(
                disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(s.author)[0]})
            for k, lab in ((0, "upgrade vs stay"), (1, "switch vs stay")):
                for x in IVu:
                    out.append(dict(stratum=name, model=f"MNL: {lab}", var=x, coef=mn.params.iloc[:, k][x],
                                    se=mn.bse.iloc[:, k][x], z=mn.tvalues.iloc[:, k][x], n=int(mn.nobs),
                                    authors=s.author.nunique(), mean_y=np.nan))
        except Exception as e:  # noqa: BLE001
            log(f"{name} MNL: {e}")
        summ.append(dict(stratum=name, pairs=len(s), authors=s.author.nunique(), stay=(s.y3 == 0).mean(),
                         upgrade=(s.y3 == 1).mean(), switch=(s.y3 == 2).mean(), R180=s.R180.mean(),
                         fe_sample=len(sf), fe_authors=sf.author.nunique()))
        # retention welfare with the stratum's author-FE beta_match
        if "I_match" in bfe:
            Kk = K if types is None else K[K.atype.isin(types)]
            pbar = s.R.mean()
            bm, sem = bfe["I_match"]
            gap = Kk.fit_gap.to_numpy()
            mism = gap > 1e-9
            for unit, g in (("per mismatched choice", gap[mism]), ("per choice", gap)):
                v = bm * np.nanmean(g) * pbar * (1 - pbar)
                wel.append(dict(stratum=name, unit=unit, beta_match_fe=bm, se_beta=sem, mean_fit_gap=np.nanmean(g),
                                pbar=pbar, d_pr_stay=v, se=abs(sem * np.nanmean(g) * pbar * (1 - pbar)),
                                rel_to_pbar=v / pbar, n_choices=len(g)))
        log(f"{name}: pairs {len(s)}, FE sample {len(sf)}")
    write(pd.DataFrame(out), "v2s_retention_bytype", "E: stay or switch by author type (24h decisions; five indices of "
          "the parent at its first adoption; SE clustered by author)",
          "I_self = same_author part, I_habit = author_family part of the former inertia index. Strata: author types of "
          "scripts/v2_authors.py.")
    write(pd.DataFrame(summ), "v2s_retention_bytype_outcomes", "E: next-decision outcomes by author type")
    write(pd.DataFrame(wel), "v2s_retention_welfare", "E: retention gain from choosing the best-matched candidate, "
          "author-FE beta_match of each stratum (Delta Pr(stay) = beta * fit_gap * p(1-p); SE by the delta method "
          "on beta only)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["sessions", "binary", "nested", "bytype"])
    a = ap.parse_args()
    {"sessions": part_sessions, "binary": part_binary, "nested": part_nested, "bytype": part_bytype}[a.part]()
