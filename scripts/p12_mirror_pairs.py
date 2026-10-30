"""Mirror pairs: an original model and a same-name copy under another account that declares the original
as its base model (ungated mirrors, re-uploads used by tools). Weights are (almost) identical, so a pair
fixed effect removes quality; the effect of accumulated degree is identified from how the split of new
children between the two copies moves with their degree ratio over time.

Model (pair x month, months after the copy appears, both copies alive):
  children_copy ~ Binomial(children_copy + children_orig, logistic(delta_pair + a * [log(1+D_copy) - log(1+D_orig)]
                                                                     + b * [log(1+age_copy) - log(1+age_orig)]))
D = degree at the start of the month (all children so far). Under PA with exponent a, the copy's share of
new children rises with its degree ratio; a = 0 means the split does not respond to accumulated degree.
Outcomes: all children, and first author-parent choices without self-choices (habit-free).

Validation first: Polya-urn pairs with known a (0, 0.5, 1) through the same estimator.

Pair samples: name (same repo name + declared base, from the name screen), hash (weight files identical,
p12_hash_pairs.py), union. The name sample is also split by its hash check.

usage: python scripts/p12_mirror_pairs.py
Writes results/tables/p12_mirror_pairs.csv (pairs), p12_mirror_panel.csv, p12_mirror.md, figure p12_mirror_share.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from hfpa import paths

MAIN = ["finetune", "adapter", "unspecified"]
MON = 30 * 86400


def fit(panel):
    """Binomial logit with pair FE; SE clustered by pair."""
    p = panel[(panel.n > 0)].copy()
    X = pd.get_dummies(p.pair.astype(str), prefix="pair", drop_first=False, dtype=float)
    X["dlogdeg"] = p.dlogdeg.to_numpy()
    X["dlogage"] = p.dlogage.to_numpy()
    y = np.c_[p.k.to_numpy(), (p.n - p.k).to_numpy()]
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": p.pair.to_numpy()})
    return {k: (r.params[k], r.bse[k]) for k in ("dlogdeg", "dlogage")}, len(p), p.pair.nunique()


def simulate(alpha, n_pairs=44, months=(3, 10), seed=0):
    """Two identical-quality copies; the copy appears after the original has a head start. The pair effect
    (name / gating / tool default) is drawn so that the copy's first-month share is U(0.1, 0.9) whatever
    alpha is, as in the real pairs; panel lengths and demand roughly as in the real data."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        d_o, d_c = rng.integers(50, 2000), 0
        s0 = rng.uniform(0.1, 0.9)
        delta = np.log(s0 / (1 - s0)) - alpha * (np.log1p(d_c) - np.log1p(d_o))
        lam = rng.lognormal(4, 1)
        for m in range(rng.integers(*months)):
            n = rng.poisson(lam)
            u = delta + alpha * (np.log1p(d_c) - np.log1p(d_o)) + 0.0 * np.log1p(m)
            k = rng.binomial(n, 1 / (1 + np.exp(-u)))
            rows.append(dict(pair=i, month=m, n=n, k=k, dlogdeg=np.log1p(d_c) - np.log1p(d_o),
                             dlogage=np.log1p(m * 30 + 15) - np.log1p(m * 30 + 15 + 365)))
            d_c, d_o = d_c + k, d_o + n - k
    return pd.DataFrame(rows)


def build(sample="name"):
    n = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "birth", "author_code"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    idx = pd.Series(np.arange(len(n)), index=n.model_id)
    tw = pd.read_csv(paths.TABLES / "p12_name_twins.csv")
    tw = tw[~tw.orig_deleted & tw.original.isin(idx.index) & tw.twin.isin(idx.index)]
    a, b = idx[tw.original].to_numpy(), idx[tw.twin].to_numpy()
    birth = n.birth.to_numpy()
    o = np.where(birth[a] <= birth[b], a, b)  # original = earlier-born
    c = np.where(birth[a] <= birth[b], b, a)
    P = pd.DataFrame({"orig": o, "cpy": c}).drop_duplicates()
    declared = set(zip(E.child_idx, E.parent_idx))  # copy declares the original as base (any relation type)
    P = P[[(x, y) in declared for x, y in zip(P.cpy, P.orig)]].reset_index(drop=True)
    hp = paths.TABLES / "p12_hash_pairs.csv"
    if hp.exists():
        H = pd.read_csv(hp)
        hm = dict(zip(zip(H.parent_idx, H.child_idx), H.match))
        P["hash_match"] = [hm.get((x, y), "not_checked") for x, y in zip(P.orig, P.cpy)]
        Hi = H[H.match == "identical"]
        Q = pd.DataFrame({"orig": np.where(birth[Hi.parent_idx] <= birth[Hi.child_idx], Hi.parent_idx, Hi.child_idx),
                          "cpy": np.where(birth[Hi.parent_idx] <= birth[Hi.child_idx], Hi.child_idx, Hi.parent_idx)})
        Q["hash_match"] = "identical"
        if sample == "hash":
            P = Q
        elif sample == "union":
            P = pd.concat([P[P.hash_match != "different"], Q]).drop_duplicates(["orig", "cpy"])
        elif sample == "name_not_different":
            P = P[P.hash_match != "different"]
        P = P.reset_index(drop=True)

    m = E[E.rel_type.isin(MAIN)].copy()
    author = n.author_code.to_numpy()
    m["t"] = birth[m.child_idx]
    m["author"] = author[m.child_idx]
    m = m.sort_values("t", kind="stable")
    m["first_noself"] = ~m.duplicated(["author", "parent_idx"]) & (m.author.to_numpy() != author[m.parent_idx])
    tmax = int(birth.max())
    panels = {}
    for variant in ("all", "first_noself"):
        mm = m if variant == "all" else m[m.first_noself]
        g = {k: v.t.to_numpy() for k, v in mm.groupby("parent_idx")}
        alltimes = {k: v.t.to_numpy() for k, v in m.groupby("parent_idx")}  # degree always counts every child
        rows = []
        for i, (oo, cc) in enumerate(zip(P.orig, P.cpy)):
            t0 = int(birth[cc])
            to, tc = g.get(oo, np.empty(0)), g.get(cc, np.empty(0))
            ao, ac = alltimes.get(oo, np.empty(0)), alltimes.get(cc, np.empty(0))
            for k in range(int((tmax - t0) // MON)):
                s, e = t0 + k * MON, t0 + (k + 1) * MON
                no, nc = int(((to >= s) & (to < e)).sum()), int(((tc >= s) & (tc < e)).sum())
                do, dc = int((ao < s).sum()), int((ac < s).sum())
                rows.append(dict(pair=i, month=k, n=no + nc, k=nc, deg_orig=do, deg_copy=dc,
                                 dlogdeg=np.log1p(dc) - np.log1p(do),
                                 dlogage=np.log1p((s - t0) / 86400 + 15) - np.log1p((s - birth[oo]) / 86400 + 15)))
        panels[variant] = pd.DataFrame(rows)
    P["orig_id"], P["copy_id"] = n.model_id.to_numpy()[P.orig], n.model_id.to_numpy()[P.cpy]
    P["gap_days"] = (birth[P.cpy] - birth[P.orig]) / 86400
    pa = panels["all"].groupby("pair")
    P["orig_deg_at_copy"] = pa.deg_orig.first().reindex(P.index).to_numpy()
    P["children_orig_after"] = (pa.n.sum() - pa.k.sum()).reindex(P.index).to_numpy()
    P["children_copy"] = pa.k.sum().reindex(P.index).to_numpy()
    P["copy_share_after"] = P.children_copy / (P.children_copy + P.children_orig_after)
    return P, panels


if __name__ == "__main__":
    out = []
    for alpha in (0.0, 0.5, 1.0):
        for seed in range(20):
            est, nobs, npairs = fit(simulate(alpha, seed=seed))
            out.append(dict(data=f"simulated, true a = {alpha}", seed=seed, a=est["dlogdeg"][0], a_se=est["dlogdeg"][1],
                            n_obs=nobs, n_pairs=npairs))
    sim = pd.DataFrame(out).groupby("data").agg(a=("a", "mean"), a_sd=("a", "std"), a_se=("a_se", "mean")).reset_index()
    res = []
    for sample in ("name", "name_not_different", "hash", "union"):
        P_, panels_ = build(sample)
        if sample == "name":
            P, panels = P_, panels_
            P.to_csv(paths.TABLES / "p12_mirror_pairs.csv", index=False)
        for variant, pn in panels_.items():
            if sample == "name":
                pn.assign(variant=variant).to_csv(paths.TABLES / f"p12_mirror_panel_{variant}.csv", index=False)
            for label, sub in [("all pairs", pn), ("gap <= 30 days", pn[pn.pair.isin(P_.index[P_.gap_days <= 30])])]:
                est, nobs, npairs = fit(sub)
                res.append(dict(pairs=sample, outcome=variant, subsample=label, a=est["dlogdeg"][0], a_se=est["dlogdeg"][1],
                                b_age=est["dlogage"][0], b_age_se=est["dlogage"][1], n_pair_months=nobs, n_pairs=npairs))
                print(res[-1], flush=True)
    res = pd.DataFrame(res)
    res.to_csv(paths.TABLES / "p12_mirror.csv", index=False)
    cols = ["orig_id", "copy_id", "hash_match", "gap_days", "orig_deg_at_copy", "children_orig_after", "children_copy", "copy_share_after"]
    md = ["# P12 mirror pairs: identical weights, different name / degree", "",
          f"Name sample: {len(P)} pairs (copy declares the original as base model; same repo name; both in the snapshot). "
          "name_not_different: name pairs whose weight hashes are not different; hash: weight files identical "
          "(p12_hash_pairs.py); union: name_not_different + hash.", "",
          "## Estimator check on simulated pairs (mean of 20 seeds; sd of estimates in a_sd)", "", sim.to_markdown(index=False, floatfmt=".3f"), "",
          "## Real pairs: pair-FE binomial logit of the copy's share of new children on the log degree ratio", "",
          res.to_markdown(index=False, floatfmt=".3f"), "",
          "## Pairs", "", P.sort_values("children_copy", ascending=False)[cols].to_markdown(index=False, floatfmt=".2f")]
    (paths.TABLES / "p12_mirror.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:14]))

    # figure: copy share by month since copy appeared, pooled over pairs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hfpa.viz.descriptive import GRID, SERIES, _save
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (variant, pn) in enumerate(panels.items()):
        g = pn[pn.month < 18].groupby("month")[["k", "n"]].sum()
        ax.plot(g.index, g.k / g.n, color=SERIES[i], lw=2, marker="o", ms=4, label={"all": "all children", "first_noself": "first choice per author, no self"}[variant])
    ax.set(xlabel="months since the copy appeared", ylabel="copy's share of new children", ylim=(0, 1),
           title=f"Mirror pairs (n = {len(P)}): share of new children going to the copy")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "p12_mirror_share")
