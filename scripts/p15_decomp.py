"""P15: why was this parent chosen? Shares of popularity / general quality / demand match / inertia / local search
/ availability in the choice of a parent, and ratios between them (conditional logit as in Overgoor, Benson &
Ugander; descriptive, not causal).

Feature groups (candidate j, chooser i):
  pop      log_deg, lrec30, major_org                                social signal, same for every chooser
  base     has_deg, legacy, log_age                                  visibility / life cycle: ever used at all
                                                                     (1.8M candidates were never used), launch age
  qual     bench_z, bench_na                                         model-index benchmarks (P14), same for all
  match    size_dist, size_na, task_hist, lang_hist, ds_lang, ds_task demand match (P13), differs by chooser
  inertia  author_reuse, author_family, same_author, is_last          habit and templates
  local    lco2, ds_couse                                            friend-of-friend analogues: authors sharing a
                                                                     niche parent; models trained on the same data
  avail    lquant, lib_* (common formats)                            quantized versions, tool formats
Main sample: plan A (first choice of each author-parent pair), unweighted; parents the author used before are
removed from its choice sets (a first choice cannot be a reuse). Comparison: all choices, n^-0.5.
Fit on the first 90% of choices in time; holdout = last 10%.

parts:
  build    assemble rows + features, cache processed/choices/<spec>/p15_rows.parquet
  clogit   full model, with and without hub FE, both samples; drop-one-group LR tests; holdout
  shapley  2^6 fits; Shapley shares of the log-likelihood gain over uniform attachment (log q baseline) for the
           6 groups and for coarser partitions (availability counted with popularity or with fit)
  mrs      marginal rates of substitution vs log degree and vs recent heat (delta-method SEs, clustered)
  classify per-choice classification among the sampled hub candidates: fit-type / popularity-type / both / neither
  mixture  latent-class logit (EM, paper section 2.3 / 3) with one mode per mechanism: pop, qual (optionally with
           hub FE = node fitness), match (choices with a demand signal), unif, reuse, self, local (candidates
           reached by a two-hop or same-dataset path; the paper's FoF choice set), template (last parent).
           Availability features enter the pop mode or the qual mode (--avail pop|fit). Shares = mean
           responsibilities (= pi at the EM fixed point). Run --part mixsim first.
  mixsim   semi-synthetic check: real candidate sets and features of the plan-A sample, choices redrawn from
           the mixture with known modes (the fitted parameters, and a variant with more match choices); EM
           should recover the realised share of choices of each mode.
usage: python scripts/p15_decomp.py --part build|clogit|shapley|mrs|classify|mixsim|mixture [--avail pop|fit] [--fe]
"""
import argparse
import importlib.util
import itertools
from math import factorial

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.models.clogit import CLogit, lr_test, to_padded


def _load_mod(name, file):
    spec = importlib.util.spec_from_file_location(name, paths.REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P5 = _load_mod("p5m", "p5_models.py")
P9 = _load_mod("p9m", "p9_extensions.py")
SPEC = "main_strat_frac0.1"
LIB = [f"lib_{x}" for x in P5.LIB if x not in ("gguf", "mlx", "timm")]  # < 0.03% of rows: separation
GROUPS = {
    "pop": ["log_deg", "lrec30", "major_org"],
    "base": ["has_deg", "legacy", "log_age"],
    "qual": ["bench_z", "bench_na"],
    "match": ["size_dist", "size_na", "task_hist", "lang_hist", "ds_lang", "ds_task"],
    "inertia": ["author_reuse", "author_family", "same_author", "is_last"],
    "local": ["lco2", "ds_couse"],
    "avail": ["lquant"] + LIB,
}
ALL = sum(GROUPS.values(), [])
NAMES = {"pop": "popularity", "base": "visibility / life cycle", "qual": "general quality", "match": "demand match", "inertia": "inertia",
         "local": "local search", "avail": "availability"}
PARTITIONS = {  # coarser classifications built from the same 2^7 fits
    "seven groups": {g: [g] for g in GROUPS},
    "A: visibility + availability as popularity": {"popularity": ["pop", "base", "avail"], "fit": ["qual", "match"],
                                                   "inertia": ["inertia"], "local search": ["local"]},
    "B: availability as fit, visibility separate": {"popularity": ["pop"], "visibility": ["base"],
                                                    "fit": ["qual", "match", "avail"], "inertia": ["inertia"],
                                                    "local search": ["local"]},
    "C: visibility and availability separate": {"popularity": ["pop"], "visibility": ["base"], "availability": ["avail"],
                                                "fit": ["qual", "match"], "inertia": ["inertia"], "local search": ["local"]},
}


def build():
    rows, ch, final_deg, nodes = P5.load(SPEC)
    d = paths.PROCESSED / "choices" / SPEC
    M = pd.read_parquet(d / "match.parquet")
    assert np.array_equal(M.choice_id.to_numpy(), rows.choice_id.to_numpy())
    assert np.array_equal(M.cand_id.to_numpy(), rows.cand_id.to_numpy())
    for c in M.columns[2:]:
        rows[c] = M[c].to_numpy(np.float32)
    B = pd.read_parquet(paths.PROCESSED / "nodes" / "bench.parquet")
    bz = np.full(len(nodes), np.nan, np.float32)
    bz[B.node.to_numpy()] = B.bench_z.to_numpy()
    b = bz[rows.cand_id.to_numpy()]
    rows["bench_na"] = np.isnan(b).astype(np.float32)
    rows["bench_z"] = np.nan_to_num(b).astype(np.float32)
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    nn = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code"])
    lp = pd.Series(P9.last_parent(ch.reset_index(drop=True), nn, E), index=ch.index)
    rows["is_last"] = (lp.loc[rows.choice_id].to_numpy() == rows.cand_id.to_numpy()).astype(np.float32)
    rows["lco2"] = np.log1p(P9.twohop(rows, ch, final_deg)).astype(np.float32)
    keep = ["choice_id", "cand_id", "chosen", "log_q", "stratum", "fe", "hist_has", "ds_has"] + ALL
    rows[keep].to_parquet(d / "p15_rows.parquet", index=False)
    # plan A flag: first choice of the author-parent pair, from all main choices
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent", "t"])
    a = nodes.author_code.to_numpy()
    full = full.assign(a=a[full.child]).sort_values("t", kind="stable")
    fp = set(map(tuple, full.loc[~full.duplicated(["a", "parent"]), ["child", "parent"]].to_numpy()))
    ch["first"] = [(c, p) in fp for c, p in zip(ch.child, ch.parent)]
    ch[["first"]].to_parquet(d / "p15_first.parquet")
    print(rows[keep].describe().T.to_string())


def load():
    d = paths.PROCESSED / "choices" / SPEC
    rows = pd.read_parquet(d / "p15_rows.parquet")
    ch = pd.read_parquet(d / "choices.parquet").set_index("choice_id", drop=False)
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["author_code", "major_org"])
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent"])
    a = nodes.author_code.to_numpy()
    npair = pd.Series(1, index=pd.MultiIndex.from_arrays([a[full.child], full.parent])).groupby(level=[0, 1]).size()
    ch["author"] = a[ch.child]
    ch["n_pair"] = npair.reindex(pd.MultiIndex.from_arrays([ch.author, ch.parent])).to_numpy()
    ch["first"] = pd.read_parquet(d / "p15_first.parquet")["first"].reindex(ch.index).to_numpy()
    return rows, ch


def sample(rows, ch, which):
    """('planA', gamma 0) first author-parent choices or ('all', gamma 0.5)."""
    if which == "planA":
        ids = ch.index[ch["first"]]
        r = rows[rows.choice_id.isin(ids).to_numpy() & (rows.author_reuse.to_numpy() == 0)]
        return r[r.choice_id.isin(r.choice_id[r.chosen == 1])], 0.0
    return rows, 0.5


def fit(rows, ch, feats, gamma, fe=False, hold=None, cluster=True):
    tr, ho = P5.split(rows, ch)
    cid = np.unique(tr.choice_id.to_numpy())
    w = ch.n_pair.loc[cid].to_numpy(float) ** -gamma
    n_fe = int(tr.fe.max()) if fe else 0
    if feats:
        D = to_padded(tr, feats, fe_col="fe" if fe else None, weight=w)
        m = CLogit(len(feats), n_fe=n_fe, l2_fe=1e-4 if n_fe else 0.0)
        res = m.fit(D, cluster=ch.author.loc[cid].to_numpy() if cluster else None)
        if hold:
            res["holdout"] = m.evaluate(to_padded(ho, feats, fe_col="fe" if fe else None))
        res["model"] = m
    else:
        res = dict(ll=P5.m0_ll(tr, ch, gamma)[0], k=0)
    return res


def drop_constant(rows, feats):
    sd = rows[feats].std()
    return [f for f in feats if sd[f] > 0]


def part_clogit(rows, ch):
    out, lr = [], []
    for which in ("planA", "all"):
        r, g = sample(rows, ch, which)
        feats = drop_constant(r, ALL)
        ll0 = fit(r, ch, [], g)["ll"]
        for fe in (False, True):
            res = fit(r, ch, feats, g, fe=fe, hold=True)
            tag = f"{which}{' + hub FE' if fe else ''}"
            for i, n in enumerate(feats):
                grp = next(k for k, v in GROUPS.items() if n in v)
                out.append(dict(model=tag, group=grp, param=n, coef=res["coef"][i], se_cl=res["se_cl"][i],
                                z=res["coef"][i] / res["se_cl"][i]))
            lr.append(dict(model=tag, test="full vs uniform", ll=res["ll"], k=res["k"], ll0=ll0,
                           pseudo_r2=1 - res["ll"] / ll0, **{f"hold_{k}": v for k, v in res["holdout"].items()}))
            if fe:
                continue
            for gname, gf in GROUPS.items():
                sub = [f for f in feats if f not in gf]
                rs = fit(r, ch, sub, g, cluster=False)
                st, p = lr_test(rs["ll"], res["ll"], len(feats) - len(sub))
                lr.append(dict(model=tag, test=f"drop {gname}", ll=rs["ll"], k=rs["k"], lr_stat=st, p=p))
            print(which, "done", flush=True)
    P5.write(pd.DataFrame(out), "p15_clogit", "P15 full conditional logit by feature group (clustered SE)")
    P5.write(pd.DataFrame(lr), "p15_clogit_lr", "P15 fit, holdout and drop-one-group likelihood-ratio tests")


def shapley(v, players):
    n = len(players)
    phi = {}
    for p in players:
        others = [q for q in players if q != p]
        s = 0.0
        for k in range(n):
            for S in itertools.combinations(others, k):
                wgt = factorial(k) * factorial(n - k - 1) / factorial(n)
                s += wgt * (v(frozenset(S) | {p}) - v(frozenset(S)))
        phi[p] = s
    return phi


def part_shapley(rows, ch, fe=False):
    """fe=True: the general-quality group also carries the hub fixed effects (node fitness, paper Table 1), an
    upper bound for quality (they also absorb brand and any unmeasured popularity of the 500 hubs)."""
    out = []
    for which in (("planA",) if fe else ("planA", "all")):
        r, g = sample(rows, ch, which)
        feats = drop_constant(r, ALL)
        ll = {}
        for k in range(len(GROUPS) + 1):
            for S in itertools.combinations(GROUPS, k):
                fs = [f for f in feats if any(f in GROUPS[x] for x in S)]
                use_fe = fe and "qual" in S
                ll[frozenset(S)] = fit(r, ch, fs, g, fe=use_fe, cluster=False)["ll"] if (fs or use_fe) else fit(r, ch, [], g)["ll"]
                print(which, sorted(S), round(ll[frozenset(S)], 1), flush=True)
        base, total = ll[frozenset()], ll[frozenset(GROUPS)] - ll[frozenset()]
        for pname, part in PARTITIONS.items():
            v = lambda S, part=part: ll[frozenset(sum((part[x] for x in S), []))] - base
            phi = shapley(v, list(part))
            for k_, val in phi.items():
                out.append(dict(sample=which, partition=pname, group=NAMES.get(k_, k_), shapley_ll=val,
                                share=val / total, total_gain=total))
        tag = "_fe" if fe else ""
        pd.Series({"|".join(sorted(k)): v for k, v in ll.items()}).to_csv(paths.TABLES / f"p15_shapley_ll_{which}{tag}.csv")
    P5.write(pd.DataFrame(out), f"p15_shapley{'_fe' if fe else ''}",
             "P15 Shapley shares of the log-likelihood gain over uniform attachment" + (" (quality group with hub FE)" if fe else ""))


SD_FEATS = ["log_deg", "lrec30", "log_age", "bench_z", "size_dist", "lco2", "ds_couse", "lquant"]


def part_mrs(rows, ch):
    """theta_x * dx / theta_ref: how many log units of the reference (log degree, recent heat) one change in x is
    worth; multiplier = exp(.) on (1 + degree) or (1 + recent children). dx = 1 for binary, 1 SD for continuous
    (SD over candidate rows of chosen-or-sampled sets, unweighted)."""
    out = []
    for which in ("planA", "all"):
        r, g = sample(rows, ch, which)
        feats = drop_constant(r, ALL)
        res = fit(r, ch, feats, g)
        b, V = res["coef"][: len(feats)], res["cov_cl"]
        for ref in ("log_deg", "lrec30"):
            j = feats.index(ref)
            for i, f in enumerate(feats):
                if f == ref:
                    continue
                dx = float(r[f].std()) if f in SD_FEATS else 1.0
                ratio = b[i] * dx / b[j]
                grad = np.zeros(len(feats))
                grad[i], grad[j] = dx / b[j], -b[i] * dx / b[j] ** 2
                se = float(np.sqrt(grad @ V @ grad))
                grp = next(k for k, v in GROUPS.items() if f in v)
                out.append(dict(sample=which, ref=ref, group=grp, feature=f, dx=dx, ratio=ratio, se=se,
                                multiplier=np.exp(ratio), mult_lo=np.exp(ratio - 1.96 * se), mult_hi=np.exp(ratio + 1.96 * se)))
    P5.write(pd.DataFrame(out), "p15_mrs", "P15 marginal rates of substitution (ratio of utility per change; multiplier on 1+degree or 1+recent)")


def part_classify(rows, ch):
    """Among the sampled hub candidates of each choice (stratum = hub; the chosen parent is added if not a hub),
    is the chosen parent the best on demand match and / or on popularity? Indices = fitted utility of each group
    (full model, plan A). Only choices with a demand signal and some variation in the match index are classified.
    Also the fit gap: max match index among hub candidates minus the chosen one's."""
    r, g = sample(rows, ch, "planA")
    feats = drop_constant(r, ALL)
    res = fit(r, ch, feats, g, cluster=False)
    b = dict(zip(feats, res["coef"][: len(feats)]))
    idx = {}
    for name, grp in [("match", GROUPS["match"]), ("pop", GROUPS["pop"]), ("base", GROUPS["base"]), ("qual", GROUPS["qual"]),
                      ("inertia", GROUPS["inertia"])]:
        idx[name] = sum(b.get(f, 0) * r[f].to_numpy() for f in grp)
    D = pd.DataFrame({"choice_id": r.choice_id.to_numpy(), "chosen": r.chosen.to_numpy(),
                      "hub": (r.stratum.to_numpy() == P9.HUB), **idx})
    D = D[D.hub | (D.chosen == 1)]
    out = []
    for k, gdf in D.groupby("choice_id"):
        if len(gdf) < 2:
            continue
        c = gdf[gdf.chosen == 1].iloc[0]
        rec = dict(choice_id=k, n=len(gdf), match_var=gdf.match.max() - gdf.match.min() > 1e-9,
                   fit_best=c.match >= gdf.match.max() - 1e-9, pop_best=c["pop"] >= gdf["pop"].max() - 1e-9,
                   inertia_pos=c.inertia > 1e-9, fit_gap=gdf.match.max() - c.match, pop_gap=gdf["pop"].max() - c["pop"])
        out.append(rec)
    C = pd.DataFrame(out).set_index("choice_id")
    sig = rows.groupby("choice_id")[["hist_has", "ds_has"]].max().max(1).reindex(C.index) > 0
    C["signal"] = sig.to_numpy() & C.match_var.to_numpy()
    lab = np.select([~C.signal, C.fit_best & C.pop_best, C.fit_best, C.pop_best],
                    ["no demand signal", "both", "fit-type", "popularity-type"], "neither")
    C["type"] = lab
    C.to_parquet(paths.PROCESSED / "choices" / SPEC / "p15_classify.parquet")
    tab = C.groupby("type").agg(n=("n", "size"), mean_fit_gap=("fit_gap", "mean"), mean_pop_gap=("pop_gap", "mean"),
                                share_inertia=("inertia_pos", "mean")).reset_index()
    tab["share_all"] = tab.n / tab.n.sum()
    s = tab[tab["type"] != "no demand signal"]
    tab["share_with_signal"] = np.where(tab["type"] != "no demand signal", tab.n / s.n.sum(), np.nan)
    P5.write(tab, "p15_classify", "P15 per-choice classification among sampled hub candidates (plan A, full model)")


MODES3 = ["pop", "qual", "match", "unif", "reuse", "self", "local", "template"]


def mode_feats(avail):
    return {"pop": GROUPS["pop"] + GROUPS["base"] + (GROUPS["avail"] if avail == "pop" else []),
            "qual": GROUPS["qual"] + (GROUPS["avail"] if avail == "fit" else []),
            "match": GROUPS["match"], "reuse": P9.REUSE, "self": P9.SELF, "local": GROUPS["local"]}


def mode_rows(rows, draw=False):
    """Candidate rows of each mode. For estimation a choice enters a restricted mode only if its chosen
    alternative is in the mode's set; for drawing (draw=True) every non-empty set is available."""
    sig = rows.groupby("choice_id")[["hist_has", "ds_has"]].transform("max").max(1).to_numpy() > 0
    sub = (lambda m: rows[m]) if draw else (lambda m: P9.restrict(rows, m, min_size=1))
    return {"pop": rows, "qual": rows, "unif": rows, "match": rows[sig],
            "reuse": sub(rows.author_reuse.to_numpy() > 0), "self": sub(rows.same_author.to_numpy() > 0),
            "local": sub((rows.lco2.to_numpy() > 0) | (rows.ds_couse.to_numpy() > 0))}


def build_modes3(rows, cid, avail, fe):
    F, R = mode_feats(avail), mode_rows(rows)
    modes = []
    for name in MODES3:
        if name == "template":
            hit = rows.loc[rows.chosen == 1].set_index("choice_id").is_last.reindex(cid).fillna(0).to_numpy()
            t = P9.Mode("template", None, [], cid)
            t.logp = lambda n, hit=np.where(hit > 0, 0.0, -np.inf): hit
            modes.append(t)
            continue
        r = R[name]
        if len(r) == 0:  # e.g. no reuse in plan A (a first choice cannot be a reuse): mode never applies
            modes.append(P9.Mode(name, None, [], cid))
            continue
        feats = drop_constant(r, F[name]) if name != "unif" else []
        modes.append(P9.Mode(name, r, feats, cid, fe_col="fe" if (fe and name == "qual") else None))
    return modes


def run_em(rows, w, avail, fe, seeds=(0, 1)):
    cid = np.unique(rows.choice_id.to_numpy())
    best = None
    for seed in seeds:
        modes = build_modes3(rows, cid, avail, fe)
        for m in modes:
            m.mstep(np.ones(len(cid)))
        r = P9.em(modes, w, len(cid), seed=seed, n_iter=40)
        print("EM", seed, dict(zip(MODES3, r["pi"].round(3))), round(r["ll"], 1), r["iters"], flush=True)
        if best is None or r["ll"] > best[0]["ll"]:
            best = (r, modes, cid)
    return best


def mode_utility(mode, r):
    """Per-row utility u - log q of a fitted mode on its rows (for drawing choices)."""
    if mode.model is None:
        return -r.log_q.to_numpy(float)
    p = mode.model.p.detach().cpu().numpy()
    u = r[mode.feats].to_numpy(float) @ p[: len(mode.feats)] - r.log_q.to_numpy(float)
    if mode.n_fe:
        u = u + np.r_[0.0, p[len(mode.feats):]][r.fe.to_numpy()]
    return u


def simulate(rows, modes, pi, seed):
    """Redraw the chosen alternative of every choice: mode ~ pi renormalised over the modes available for the
    choice, then an alternative from that mode's candidate set (Gumbel-max on u - log q)."""
    rng = np.random.default_rng(seed)
    base = rows.copy()
    base["chosen"] = 0
    cid = np.unique(base.choice_id.to_numpy())
    R = mode_rows(rows, draw=True)
    R["template"] = rows[rows.is_last.to_numpy() > 0]
    draws = {}
    for m in modes:
        r = R[m.name]
        if m.name in ("template",):
            g = pd.Series(r.index, index=r.choice_id.to_numpy())
            draws[m.name] = g[~g.index.duplicated()]
            continue
        u = mode_utility(m, r) + rng.gumbel(size=len(r))
        s = pd.Series(u, index=r.index)
        draws[m.name] = s.groupby(r.choice_id.to_numpy()).idxmax()
    avail = np.stack([np.isin(cid, draws[m.name].index) for m in modes], 1)
    pr = avail * pi[None, :]
    pr /= pr.sum(1, keepdims=True)
    pick = (rng.random(len(cid))[:, None] < np.cumsum(pr, 1)).argmax(1)
    rowpos = np.array([draws[modes[k].name].at[c] for c, k in zip(cid, pick)])
    base.loc[rowpos, "chosen"] = 1
    return base, pd.Series(np.array(MODES3)[pick], index=cid)


def part_mixsim(rows, ch, avail="pop", fe=False):
    r, g = sample(rows, ch, "planA")
    tr, _ = P5.split(r, ch)
    tr = tr.reset_index(drop=True)
    w = np.ones(tr.choice_id.nunique())
    fit_r, modes, cid = run_em(tr, w, avail, fe, seeds=(0,))
    out = []
    for label, pi in [("fitted pi", fit_r["pi"]),
                      ("more match", np.r_[fit_r["pi"][:2] * 0.6, [0.3], fit_r["pi"][3:] * 0.6])]:
        pi = pi / pi.sum()
        for seed in (1, 2):
            sim, truth = simulate(tr, modes, pi, seed)
            est, _, _ = run_em(sim, w, avail, fe, seeds=(0,))
            share = est["gamma"].mean(0)
            for k, name in enumerate(MODES3):
                out.append(dict(dgp=label, seed=seed, mode=name, pi_dgp=pi[k], realised_share=float((truth == name).mean()),
                                est_share=float(share[k]), error=float(share[k] - (truth == name).mean())))
            print(pd.DataFrame(out).tail(len(MODES3)).to_string(), flush=True)
    P5.write(pd.DataFrame(out), f"p15_mixsim_{avail}{'_fe' if fe else ''}", "P15 semi-synthetic check of the latent-class decomposition")


def part_mixture(rows, ch, avail="pop", fe=False):
    out, resp = [], []
    for which in ("planA", "all"):
        r, g = sample(rows, ch, which)
        tr, ho = P5.split(r, ch)
        tr = tr.reset_index(drop=True)
        cid0 = np.unique(tr.choice_id.to_numpy())
        w = ch.n_pair.loc[cid0].to_numpy(float) ** -g
        res, modes, cid = run_em(tr, w, avail, fe)
        share = (res["gamma"] * w[:, None]).sum(0) / w.sum()
        row = dict(sample=which, avail=avail, hub_fe=fe, ll=res["ll"], iters=res["iters"],
                   **{f"share_{k}": v for k, v in zip(MODES3, share)})
        for m in modes:
            if m.model is not None:
                for f, c, s_ in zip(m.feats, m.res["coef"], m.res["se"]):
                    row[f"{m.name}:{f}"] = c
        fitshare = share[MODES3.index("qual")] + share[MODES3.index("match")]
        row["R_fit_over_pop"] = fitshare / share[MODES3.index("pop")]
        row["R_match_over_pop"] = share[MODES3.index("match")] / share[MODES3.index("pop")]
        out.append(row)
        G = pd.DataFrame(res["gamma"], columns=MODES3, index=cid)
        G["sample"] = which
        resp.append(G)
    tag = f"{avail}{'_fe' if fe else ''}"
    P5.write(pd.DataFrame(out), f"p15_mixture_{tag}", f"P15 latent-class decomposition (availability in {avail}; hub FE in qual: {fe})")
    pd.concat(resp).to_parquet(paths.PROCESSED / "choices" / SPEC / f"p15_mixture_resp_{tag}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["build", "clogit", "shapley", "mrs", "classify", "mixsim", "mixture"])
    ap.add_argument("--avail", default="pop", choices=["pop", "fit"])
    ap.add_argument("--fe", action="store_true")
    a = ap.parse_args()
    if a.part == "build":
        build()
    else:
        rows, ch = load()
        kw = dict(avail=a.avail, fe=a.fe) if a.part in ("mixsim", "mixture") else dict(fe=a.fe) if a.part == "shapley" else {}
        globals()[f"part_{a.part}"](rows, ch, **kw)
