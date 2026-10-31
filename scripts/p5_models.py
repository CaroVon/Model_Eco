"""Stages P5-P8, P10 (conditional logit; plan 6.6-6.9, 6.11) on the 10% choice sample.

Read every alpha here as a conditional association, not a causal lock-in effect: P0-P0f showed that
with latent quality and a launch life cycle no conditional-logit specification recovers the true PA
coefficient (hub FE included). The causal question is handled by the natural experiment
(scripts/p10_sd15_event.py) and the node-FE GMM attempts (p5_gmm*.py).

Common settings: data main_strat_frac0.1 (stratified sampling, log q correction), choice weights
n^-0.5 (n = choices of the same author-parent pair, user decision), SEs clustered by child author,
fit on the first 90% of choices in time, holdout = last 10%.

parts (run separately):  ladder  -> M0-M7 (P5)          p6  -> degree-bin curves under M3 / M5
                         p7      -> hub FE with K in {200, 500, 1000, 2000}
                         p8      -> by quarter, by modality, author-type interactions
                         p10     -> R3 uniform sampling, R4 start 2023, R6 no bulk accounts,
                                    R9 recent window 7 / 90, gamma 0 / 1
usage: python scripts/p5_models.py --part ladder
Writes results/tables/p5_<part>.csv (+ .md) and figures for p6 / p8.
"""
import argparse
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hfpa import paths
from hfpa.models.clogit import CLogit, to_padded
from hfpa.viz.descriptive import GRID, INK2, MUTED, SEQ, SERIES, SURFACE, _save

MAIN = ["finetune", "adapter", "unspecified"]
GAMMA = 0.5
LIC = ["apache", "mit", "llama", "gemma", "openrail", "noncommercial", "cc_by", "other"]  # vs none / rest
LIB = ["transformers", "peft", "diffusers", "sentence-transformers", "gguf", "mlx", "timm"]  # vs none / rest
STATIC = (["major_org", "has_arxiv", "log_params", "params_na"] + [f"lic_{x}" for x in LIC]
          + [f"lib_{x}" for x in LIB])
AGEBIN = ["age_lt30", "age_30_90", "age_90_365"]
LOCAL = ["lrec30", "same_author", "author_reuse", "author_family", "lfam_recent", "lquant"]
M = {
    "M1": ["has_deg"],
    "M2": ["has_deg", "log_deg"],
    "M3": ["has_deg", "log_deg", "log_age", "legacy"],
}
M["M4"] = M["M3"] + STATIC
M["M5"] = M["M3"] + AGEBIN  # + hub FE
M["M6"] = M["M5"] + [f"{s}_nh" for s in STATIC]  # + hub FE; static features only for non-hubs
M["M7"] = M["M6"] + LOCAL
FE = {"M5": "fe", "M6": "fe", "M7": "fe"}
RELEASES = [("Llama 2", "2023-07-18"), ("Llama 3", "2024-04-18"), ("FLUX.1", "2024-08-01"),
            ("Qwen2.5", "2024-09-19"), ("DeepSeek-R1", "2025-01-20")]


# ---------------------------------------------------------------- data
def load(spec="main_strat_frac0.1", filters=None):
    """filters: optional pyarrow row filter, e.g. [("chosen", "==", 1)] to read only the chosen rows."""
    d = paths.PROCESSED / "choices" / spec
    ch = pd.read_parquet(d / "choices.parquet")
    rows = pd.concat([pd.read_parquet(f, filters=filters) for f in sorted(glob.glob(str(d / "rows" / "*.parquet")))],
                     ignore_index=True)
    assert np.all(np.diff(rows.choice_id.to_numpy()) >= 0)
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["birth", "author_code", "legacy", "major_org", "has_arxiv", "params_b",
                                     "license_cat", "library_cat", "modality"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    c = rows.cand_id.to_numpy()
    f32 = lambda a: np.asarray(a, np.float32)
    rows["log_age"] = f32(np.log1p(rows.age_days))
    for name, lo, hi in [("age_lt30", 0, 30), ("age_30_90", 30, 90), ("age_90_365", 90, 365)]:
        rows[name] = f32((rows.age_days >= lo) & (rows.age_days < hi))
    rows["legacy"] = f32(nodes.legacy.to_numpy()[c])
    rows["major_org"] = f32(nodes.major_org.to_numpy()[c])
    rows["has_arxiv"] = f32(nodes.has_arxiv.to_numpy()[c])
    pb = nodes.params_b.to_numpy()[c]
    ok = np.isfinite(pb) & (pb > 0) & (pb < 2000)  # names like "...-1000000b" are not sizes
    rows["log_params"] = f32(np.where(ok, np.log(np.where(ok, pb, 1)), 0))
    rows["params_na"] = f32(~ok)
    lic, lib = nodes.license_cat.to_numpy()[c], nodes.library_cat.to_numpy()[c]
    for x in LIC:
        rows[f"lic_{x}"] = f32(lic == x)
    for x in LIB:
        rows[f"lib_{x}"] = f32(lib == x)
    nh = f32(rows.fe.to_numpy() == 0)
    for s in STATIC:
        rows[f"{s}_nh"] = rows[s].to_numpy() * nh
    for w in (7, 30, 90):
        rows[f"lrec{w}"] = f32(np.log1p(rows[f"recent{w}"]))
    rows["lfam_recent"] = f32(np.log1p(rows.fam_recent))
    rows["lquant"] = f32(np.log1p(rows.quant_deg))

    # choice-level info: weights, cluster, child author type, modality, time
    author = nodes.author_code.to_numpy()
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent"])
    npair = pd.Series(1, index=pd.MultiIndex.from_arrays([author[full.child], full.parent])).groupby(level=[0, 1]).size()
    ch["author"] = author[ch.child]
    ch["n_pair"] = npair.reindex(pd.MultiIndex.from_arrays([ch.author, ch.parent])).to_numpy()
    ch["child_major"] = nodes.major_org.to_numpy()[ch.child].astype(float)
    order = np.argsort(nodes.birth.to_numpy(), kind="stable")
    prior = pd.Series(np.arange(len(nodes))).groupby(author[order]).cumcount().to_numpy()
    exp = np.empty(len(nodes))
    exp[order] = prior  # models the author uploaded before this one
    ch["child_lexp"] = np.log1p(exp[ch.child])
    ch = ch.set_index("choice_id", drop=False)
    # final main degree for alternative hub sets (P7)
    m = E[E.rel_type.isin(MAIN)]
    final_deg = np.bincount(m.parent_idx, minlength=len(nodes))
    return rows, ch, final_deg, nodes


def fit(rows, ch, feats, fe_col=None, gamma=GAMMA, hold=None):
    """Fit on non-holdout choices; return result dict (+ holdout metrics if hold rows given)."""
    cid = np.unique(rows.choice_id.to_numpy())
    w = ch.n_pair.loc[cid].to_numpy(float) ** -gamma
    n_fe = int(rows[fe_col].max()) if fe_col else 0
    D = to_padded(rows, feats, fe_col=fe_col, weight=w)
    model = CLogit(len(feats), n_fe=n_fe, l2_fe=1e-4 if n_fe else 0.0)
    res = model.fit(D, cluster=ch.author.loc[cid].to_numpy())
    del D
    if hold is not None:
        Dh = to_padded(hold, feats, fe_col=fe_col)
        res["holdout"] = model.evaluate(Dh)
        del Dh
    torch.cuda.empty_cache()
    return res


def m0_ll(rows, ch, gamma=GAMMA):
    """Uniform attachment within the sampled set: u = -log q."""
    g = rows.groupby("choice_id")
    lse = g.log_q.apply(lambda s: np.log(np.exp(-s.to_numpy(np.float64)).sum()))
    chosen = rows[rows.chosen == 1].set_index("choice_id").log_q
    ll = -chosen - lse.loc[chosen.index]
    w = ch.n_pair.loc[ll.index].to_numpy(float) ** -gamma
    return float((w * ll).sum()), ll


def records(label, res, keep=None, **extra):
    out = []
    for i, name in enumerate(res["names"]):
        if name.startswith("fe") and name[2:].isdigit():
            continue
        if keep is not None and name not in keep:
            continue
        out.append(dict(model=label, param=name, coef=res["coef"][i], se=res["se"][i], se_cl=res["se_cl"][i],
                        z_cl=res["coef"][i] / res["se_cl"][i], ll=res["ll"], k=res["k"], n=res["n"], aic=res["aic"],
                        **{f"hold_{k}": v for k, v in res.get("holdout", {}).items()}, **extra))
    return out


def split(rows, ch):
    hold_ids = ch.index[ch.holdout]
    h = rows.choice_id.isin(hold_ids).to_numpy()
    return rows[~h], rows[h]


def write(df, name, title):
    df.to_csv(paths.TABLES / f"{name}.csv", index=False)
    (paths.TABLES / f"{name}.md").write_text(f"# {title}\n\n" + df.to_markdown(index=False, floatfmt=".4g") + "\n")
    print(df.to_string(), flush=True)


# ---------------------------------------------------------------- parts
def part_ladder(rows, ch, **_):
    tr, ho = split(rows, ch)
    out = []
    ll0, _ = m0_ll(tr, ch)
    hll0 = m0_ll(ho, ch, gamma=0)[1].mean()
    out.append(dict(model="M0", param="-", ll=ll0, k=0, n=tr.choice_id.nunique(), aic=-2 * ll0, hold_mean_ll=hll0))
    for name, feats in M.items():
        res = fit(tr, ch, feats, FE.get(name), hold=ho)
        out += records(name, res)
        print(name, f"ll={res['ll']:.1f}", res.get("holdout"), flush=True)
    df = pd.DataFrame(out)
    df["alpha_doubling_effect"] = np.where(df.param == "log_deg", 2 ** df.coef - 1, np.nan)
    write(df, "p5_ladder", "P5 model ladder (conditional logit, gamma = 0.5 weights, SE clustered by author)")


def deg_bins(rows):
    edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, np.inf]
    d = rows.deg.to_numpy()
    names, med = [], []
    for lo, hi in zip(edges[1:-1], edges[2:]):
        n = f"db_{lo}"
        rows[n] = ((d >= lo) & (d < hi)).astype(np.float32)
        names.append(n)
        sel = (d >= lo) & (d < hi)
        med.append(float(np.median(d[sel])) if sel.any() else np.nan)
    return names, med  # baseline: degree 1 (has_deg carries 0 vs 1)


def part_p6(rows, ch, **_):
    """Degree-bin coefficients relative to degree 1, as in Overgoor et al. (2020) fig. 5: degree 0 is
    shown as a separate point (= -has_deg), with exact linear PA (alpha = 1) as a dotted reference."""
    tr, _ = split(rows, ch)
    names, med = deg_bins(tr)
    out = []
    x0 = -1.0  # plotting position of degree 0
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, (base, fe) in enumerate([("M3", None), ("M5", "fe")]):
        feats = [f for f in M[base] if f != "log_deg"] + names
        res = fit(tr, ch, feats, fe)
        slope = fit(tr, ch, M[base], fe)
        a = slope["coef"][slope["names"].index("log_deg")]
        idx = [res["names"].index(n) for n in names]
        coef, se = res["coef"][idx], res["se_cl"][idx]
        ih = res["names"].index("has_deg")
        z0, z0se = -res["coef"][ih], res["se_cl"][ih]
        x = np.log(med)
        ok = np.isfinite(coef) & np.isfinite(x)
        col = SERIES[i]
        ax.fill_between(np.r_[0, x[ok]], np.r_[0, coef[ok] - 1.96 * se[ok]], np.r_[0, coef[ok] + 1.96 * se[ok]],
                        color=col, alpha=0.15, lw=0)
        ax.plot(np.r_[0, x[ok]], np.r_[0, coef[ok]], color=col, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5,
                label=f"{base}: degree bins")
        ax.plot([0, x[ok].max()], [0, a * x[ok].max()], color=col, lw=1, ls="--", label=f"{base}: log-degree fit, α = {a:.2f}")
        ax.errorbar([x0 + 0.08 * i], [z0], yerr=[1.96 * z0se], fmt="s", ms=6, color=col, mec=SURFACE, mew=1.5, capsize=3)
        out.append(dict(model=base, bin="deg_0", median_deg=0, coef=z0, se_cl=z0se, alpha_linear=a))
        for n, m_, c_, s_ in zip(names, med, coef, se):
            out.append(dict(model=base, bin=n, median_deg=m_, coef=c_, se_cl=s_, alpha_linear=a))
    xm = np.log(max(m_ for m_ in med if np.isfinite(m_)))
    ax.plot([0, xm], [0, xm], color=MUTED, lw=1, ls=":", label="exact linear PA (α = 1)")
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    ax.set_xticks([x0] + [np.log(t) for t in ticks], ["0"] + [str(t) for t in ticks])
    ax.axvline(x0 / 2, color=GRID, lw=1)
    ax.set(xlabel="degree at the time of the choice (log scale; 0 shown separately)", ylabel="utility relative to degree 1",
           title="P6: non-parametric degree effect (conditional association)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color=GRID, lw=0.6)
    _save(fig, "p6_degree_bins")
    write(pd.DataFrame(out), "p6_degree_bins", "P6 degree-bin coefficients (M3 and M5 bases; deg_0 = -has_deg)")


def part_p7(rows, ch, final_deg, **_):
    tr, ho = split(rows, ch)
    out = []
    order = np.argsort(-final_deg, kind="stable")
    for K in (200, 500, 1000, 2000):
        rank = np.zeros(len(final_deg), np.int64)
        top = order[:K][final_deg[order[:K]] > 0]
        rank[top] = np.arange(1, len(top) + 1)
        for r in (tr, ho):
            r["feK"] = rank[r.cand_id.to_numpy()].astype(np.int16)
        res = fit(tr, ch, M["M5"], "feK", hold=ho)
        out += records(f"M5 K={K}", res, keep=["has_deg", "log_deg"], K=K)
        print(K, out[-1], flush=True)
    df = pd.DataFrame(out)
    df["alpha_doubling_effect"] = np.where(df.param == "log_deg", 2 ** df.coef - 1, np.nan)
    write(df, "p7_K_sensitivity", "P7 hub-FE size K (M5); compare P0: alpha under FE moves with K even when truth is 0")


def part_p8(rows, ch, nodes, **_):
    out = []
    # time: quarter windows, M3 and M5
    q = pd.to_datetime(ch.t, unit="s").dt.to_period("Q").astype(str)
    rows["quarter"] = q.loc[rows.choice_id].to_numpy()
    for qq, sub in rows.groupby("quarter"):
        if sub.choice_id.nunique() < 300:
            continue
        for base in ("M3", "M5"):
            fe = FE.get(base)
            if fe:  # re-index FE to the hubs present in this window
                u = np.unique(sub[fe])
                sub = sub.assign(fe_q=np.where(sub[fe] > 0, np.searchsorted(u[u > 0], sub[fe]) + 1, 0).astype(np.int16))
            res = fit(sub, ch, M[base], "fe_q" if fe else None)
            out += records(base, res, keep=["log_deg"], cut="quarter", value=qq, n_choices=sub.choice_id.nunique())
    # modality
    modname = ch.modality_name
    rows["mod"] = modname.loc[rows.choice_id].to_numpy()
    for md, sub in rows.groupby("mod"):
        if sub.choice_id.nunique() < 300:
            continue
        for base in ("M3", "M5"):
            fe = FE.get(base)
            if fe:
                u = np.unique(sub[fe])
                sub = sub.assign(fe_q=np.where(sub[fe] > 0, np.searchsorted(u[u > 0], sub[fe]) + 1, 0).astype(np.int16))
            res = fit(sub, ch, M[base], "fe_q" if fe else None)
            out += records(base, res, keep=["log_deg"], cut="modality", value=md, n_choices=sub.choice_id.nunique())
    # author type: log_deg x child-author features (no sample split)
    cm = ch.child_major.loc[rows.choice_id].to_numpy(np.float32)
    ce = ch.child_lexp.loc[rows.choice_id].to_numpy(np.float32)
    rows["ld_x_major"] = rows.log_deg.to_numpy() * cm
    rows["ld_x_lexp"] = rows.log_deg.to_numpy() * (ce - np.float32(ch.child_lexp.mean()))
    for base in ("M3", "M5"):
        res = fit(rows, ch, M[base] + ["ld_x_major", "ld_x_lexp"], FE.get(base))
        out += records(base, res, keep=["log_deg", "ld_x_major", "ld_x_lexp"], cut="author_type", value="interaction")
    df = pd.DataFrame(out)
    write(df, "p8_heterogeneity", "P8 heterogeneity of alpha (conditional association)")
    # figure: alpha by quarter
    t = df[df.cut == "quarter"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, base in enumerate(("M3", "M5")):
        s = t[(t.model == base) & (t.param == "log_deg")].copy()
        x = pd.PeriodIndex(s.value, freq="Q").to_timestamp(how="start") + pd.Timedelta(days=45)
        ax.fill_between(x, s.coef - 1.96 * s.se_cl, s.coef + 1.96 * s.se_cl, color=SERIES[i], alpha=0.15, lw=0)
        ax.plot(x, s.coef, color=SERIES[i], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5,
                label=f"{base}" + (" (hub FE + age bins)" if base == "M5" else " (no FE)"))
    for name, d in RELEASES:
        ax.axvline(pd.Timestamp(d), color=MUTED, lw=0.8, ls=":")
        ax.annotate(name, (pd.Timestamp(d), ax.get_ylim()[1]), xytext=(2, -10), textcoords="offset points",
                    fontsize=7, color=INK2, rotation=90, va="top")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(ylabel="α (log degree coefficient)", title="P8: α by quarter with 95% CI (conditional association)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(color=GRID, lw=0.6)
    _save(fig, "p8_alpha_by_quarter")


def part_p10(rows, ch, nodes, **_):
    tr, ho = split(rows, ch)
    out = []
    base = fit(tr, ch, M["M3"])
    out += records("main M3", base, keep=["has_deg", "log_deg"], check="main")
    for g in (0.0, 1.0):
        out += records("M3", fit(tr, ch, M["M3"], gamma=g), keep=["has_deg", "log_deg"], check=f"gamma={g:g}")
    # R4 start 2023-01-01
    t23 = pd.Timestamp(paths.CFG["replay"]["robust_start"], tz="UTC").timestamp()
    keep = ch.index[ch.t >= t23]
    out += records("M3", fit(tr[tr.choice_id.isin(keep)], ch, M["M3"]), keep=["has_deg", "log_deg"], check="R4 start 2023")
    # R6 drop bulk accounts: authors with > 50 uploads on one UTC day
    day = nodes.birth.to_numpy() // 86400
    per_day = pd.DataFrame({"a": nodes.author_code.to_numpy(), "d": day}).value_counts()
    bulk = set(per_day[per_day > 50].index.get_level_values(0))
    keep = ch.index[~ch.author.isin(bulk)]
    out += records("M3", fit(tr[tr.choice_id.isin(keep)], ch, M["M3"]), keep=["has_deg", "log_deg"],
                   check=f"R6 drop {len(bulk)} bulk authors")
    # R9 recent window in M7
    for w in (7, 90):
        feats = [f if f != "lrec30" else f"lrec{w}" for f in M["M7"]]
        out += records("M7", fit(tr, ch, feats, "fe"), keep=["log_deg", f"lrec{w}"], check=f"R9 recent {w}d")
    out += records("M7", fit(tr, ch, M["M7"], "fe"), keep=["log_deg", "lrec30"], check="R9 recent 30d (main)")
    # R3 uniform negative sampling (s = 50): every 10th choice of main_unif
    d = paths.PROCESSED / "choices" / "main_unif"
    parts = []
    for f in sorted(glob.glob(str(d / "rows" / "*.parquet"))):
        x = pd.read_parquet(f, columns=["choice_id", "cand_id", "chosen", "log_q", "deg", "log_deg", "has_deg", "age_days"])
        parts.append(x[x.choice_id % 10 == 0])
    u = pd.concat(parts, ignore_index=True)
    u["log_age"] = np.log1p(u.age_days).astype(np.float32)
    u["legacy"] = nodes.legacy.to_numpy()[u.cand_id].astype(np.float32)
    chu = pd.read_parquet(d / "choices.parquet").set_index("choice_id", drop=False)
    author = nodes.author_code.to_numpy()
    chu["author"] = author[chu.child]
    chu["n_pair"] = chu.groupby(["author", "parent"]).choice_id.transform("size")
    u = u[u.choice_id.isin(chu.index[~chu.holdout])]
    out += records("M3", fit(u, chu, M["M3"]), keep=["has_deg", "log_deg"], check="R3 uniform s=50 (every 10th choice)")
    write(pd.DataFrame(out), "p10_robustness", "P10 robustness (conditional logit, alpha = log_deg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["ladder", "p6", "p7", "p8", "p10"])
    a = ap.parse_args()
    rows, ch, final_deg, nodes = load()
    print(f"rows {len(rows):,}; choices {ch.shape[0]:,}", flush=True)
    globals()[f"part_{a.part}"](rows=rows, ch=ch, final_deg=final_deg, nodes=nodes)
    print("done")
