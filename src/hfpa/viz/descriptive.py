"""P3 descriptive analysis (plan 6.4). Each figure has a CSV table view next to it in results/tables."""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hfpa import paths

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#86b6ef", "#5598e7", "#256abf", "#104281"]  # ordinal blue steps 250..650
MAIN = ["finetune", "adapter", "unspecified"]
DAY = 86400

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(paths.FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)


def _label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.plot([x], [y], "o", ms=4, color=color, mec=SURFACE, mew=1)


def degree_ccdf(E):
    import powerlaw
    fig, ax = plt.subplots(figsize=(6, 4))
    rows, tab = [], []
    for i, r in enumerate(["finetune", "adapter", "quantized", "merge"]):
        d = E[E.rel_type == r].parent.value_counts().to_numpy()
        x = np.sort(np.unique(d))
        ccdf = 1 - np.searchsorted(np.sort(d), x, "left") / len(d)
        ax.loglog(x, ccdf, color=SERIES[i], lw=1.5)
        _label_end(ax, x[-1], ccdf[-1], r, SERIES[i])
        tab += [dict(rel_type=r, degree=a, ccdf=b) for a, b in zip(x, ccdf)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = powerlaw.Fit(d, discrete=True, verbose=False)
            R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        rows.append(dict(rel_type=r, n_parents=len(d), max_deg=int(d.max()), pl_alpha=fit.power_law.alpha,
                         pl_xmin=fit.power_law.xmin, R_pl_vs_lognormal=R, p=p))
    ax.set(xlabel="children per parent (final snapshot)", ylabel="P(D ≥ d)",
           title="Degree CCDF by edge type (parents with ≥1 child)")
    _save(fig, "p3_degree_ccdf")
    pd.DataFrame(tab).to_csv(paths.TABLES / "p3_degree_ccdf.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(paths.TABLES / "p3_powerlaw_fits.csv", index=False)
    return out


def concentration(E, nodes, t0):
    """Monthly share of new main-spec edges going to the top-k root families, HHI and Gini."""
    m = E[E.rel_type.isin(MAIN)].copy()
    m["t"] = nodes.birth_ts.to_numpy()[m.child_idx.to_numpy()]
    m = m[m.t > t0]
    m["root"] = nodes.root.to_numpy()[m.parent_idx.to_numpy()]
    m["month"] = m.t.dt.tz_localize(None).dt.to_period("M")
    rows = []
    for mo, g in m.groupby("month"):
        c = np.sort(g.root.value_counts().to_numpy())[::-1]
        s = c / c.sum()
        cs = np.sort(c)
        gini = 1 - 2 * np.sum(np.cumsum(cs) / cs.sum()) / len(cs) + 1 / len(cs)
        rows.append(dict(month=mo.to_timestamp(), n_edges=int(c.sum()), n_families=len(c), top1=s[:1].sum(),
                         top10=s[:10].sum(), top100=s[:100].sum(), hhi=float((s ** 2).sum()), gini=gini))
    df = pd.DataFrame(rows)
    df.to_csv(paths.TABLES / "p3_concentration_monthly.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    for i, k in enumerate(["top1", "top10", "top100"]):
        ax.plot(df.month, df[k], color=SERIES[i])
        _label_end(ax, df.month.iloc[-1], df[k].iloc[-1], k.replace("top", "Top-"), SERIES[i])
    ax.set(ylim=(0, 1), ylabel="share of new fine-tunes", title="Share flowing to top-k root families")
    axes[1].plot(df.month, df.hhi, color=SERIES[0])
    axes[1].set(ylim=(0, None), ylabel="HHI", title="HHI over root families")
    axes[2].plot(df.month, df.gini, color=SERIES[0])
    axes[2].set(ylim=(0, 1), ylabel="Gini", title="Gini over root families")
    for a in axes:
        a.tick_params(axis="x", labelrotation=45)
    _save(fig, "p3_concentration")
    return df


def family_growth(nodes, k=20):
    fam = nodes.root.value_counts()
    top = fam.index[:k]
    sub = nodes[nodes.root.isin(top)]
    tab = []
    fig, axes = plt.subplots(4, 5, figsize=(13, 9), sharex=True)
    for ax, r in zip(axes.flat, top):
        t = np.sort(sub.loc[sub.root == r, "birth_ts"].dt.tz_localize(None).to_numpy())
        y = np.arange(1, len(t) + 1)
        ax.step(t, y, where="post", color=SERIES[0])
        name = nodes.model_id.iloc[r]
        ax.set_title(name.split("/")[-1][:28], fontsize=8, fontweight="bold")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        tab += [dict(root=name, month=m, size=int(v)) for m, v in
                pd.Series(y, index=pd.DatetimeIndex(t)).resample("ME").max().ffill().items()]
    fig.suptitle("Cumulative size of the 20 largest root families (all relation types)", x=0.01, ha="left",
                 fontweight="bold")
    _save(fig, "p3_family_growth")
    pd.DataFrame(tab).to_csv(paths.TABLES / "p3_family_growth.csv", index=False)


def head_start(E, nodes, t0, horizon=180, early=30):
    """Mean children gained in the 6 months after day 30, grouped by children received in the first 30 days."""
    m = E[E.rel_type.isin(MAIN)]
    birth = nodes.birth.to_numpy()
    snap = birth.max()
    elig = (~nodes.legacy.to_numpy()) & (birth <= snap - (early + horizon) * DAY)
    age = (birth[m.child_idx.to_numpy()] - birth[m.parent_idx.to_numpy()]) / DAY
    par = m.parent_idx.to_numpy()
    keep = elig[par]
    par, age = par[keep], age[keep]
    early_n = np.bincount(par[age < early], minlength=len(birth))
    bins = [(0, 0, "0"), (1, 4, "1–4"), (5, 19, "5–19"), (20, np.inf, "20+")]
    days = np.arange(early, early + horizon + 1, 5)
    tab = []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (lo, hi, lab) in enumerate(bins):
        grp = elig & (early_n >= lo) & (early_n <= hi)
        n = int(grp.sum())
        sel = grp[par]
        a = age[sel]
        mean = [((a >= early) & (a < d)).sum() / n for d in days]
        ax.plot(days - early, mean, color=SEQ[i])
        _label_end(ax, days[-1] - early, mean[-1], f"{lab} in first 30d (n={n:,})", SEQ[i])
        tab += [dict(group=lab, n_parents=n, days_after_day30=d - early, mean_new_children=v) for d, v in zip(days, mean)]
    ax.set(yscale="symlog", xlabel="days after day 30", ylabel="mean new children (cumulative)",
           title="Head start: early children vs later growth", xlim=(0, horizon * 1.6))
    _save(fig, "p3_head_start")
    pd.DataFrame(tab).to_csv(paths.TABLES / "p3_head_start.csv", index=False)


def monthly_choices(E, nodes, t0):
    m = E[E.rel_type.isin(MAIN)]
    t = nodes.birth_ts.to_numpy()[m.child_idx.to_numpy()]
    s = pd.Series(1, index=pd.DatetimeIndex(t).tz_localize(None))
    s = s[s.index > t0.tz_localize(None)].resample("MS").sum()
    s.rename("n_choices").to_csv(paths.TABLES / "p3_monthly_choices.csv")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(s.index, s.values, width=24, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.set(ylabel="choices (main spec)", title="Choices per month (finetune + adapter + unspecified, after T0)")
    ax.grid(axis="x", visible=False)
    _save(fig, "p3_monthly_choices")
    return s
