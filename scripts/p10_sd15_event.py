"""Natural experiment: deletion of runwayml/stable-diffusion-v1-5 (late Aug 2024).

The repo with the most image-generation children disappeared; community mirrors with identical weights
(stable-diffusion-v1-5/stable-diffusion-v1-5, benjamin-paine/stable-diffusion-v1-5) appeared on
2024-08-30. Model quality is held fixed across original and mirrors, so the split of new SD-1.5 children
between them shows how much of "who gets chosen" is attachment to a name rather than to the weights.

Outputs: monthly children by base (all and first author-parent choice, self-choices removed), share of
SD-1.5 children still naming the deleted repo, SD-1.5 lineage share of image-generation choices
before / after, and whether post-deletion users of the deleted name had used it before.
Writes results/tables/p10_sd15_*.csv, p10_sd15.md and figures p10_sd15_monthly.png, p10_sd15_share.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hfpa import paths
import matplotlib.dates  # noqa: E402
import matplotlib.ticker  # noqa: E402

from hfpa.viz.descriptive import GRID, INK2, MUTED, SERIES, SURFACE, _save

MAIN = ["finetune", "adapter", "unspecified"]
EVENT = pd.Timestamp("2024-08-30")
ORIG = "runwayml/stable-diffusion-v1-5"
MIRRORS = ["stable-diffusion-v1-5/stable-diffusion-v1-5", "benjamin-paine/stable-diffusion-v1-5"]
OTHERS = {"SD 1.4": "CompVis/stable-diffusion-v1-4", "SD 2.1": "stabilityai/stable-diffusion-2-1",
          "SDXL": "stabilityai/stable-diffusion-xl-base-1.0"}

if __name__ == "__main__":
    n = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "birth_ts", "author_code", "modality"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    idx = pd.Series(np.arange(len(n)), index=n.model_id)
    m = E[E.rel_type.isin(MAIN)].copy()
    m["t"] = n.birth_ts.dt.tz_localize(None).to_numpy()[m.child_idx]
    m["author"] = n.author_code.to_numpy()[m.child_idx]
    m["self"] = m.author.to_numpy() == n.author_code.to_numpy()[m.parent_idx]
    m = m.sort_values(["t", "child_idx"], kind="stable")
    m["first"] = ~m.duplicated(["author", "parent_idx"])
    m["month"] = m.t.dt.to_period("M")
    base = {"runwayml (deleted)": [ORIG], "community mirrors": MIRRORS, **{k: [v] for k, v in OTHERS.items()}}
    pid = {k: idx[v].to_numpy() for k, v in base.items()}
    win = (m.month >= pd.Period("2024-03", "M")) & (m.month <= pd.Period("2025-07", "M"))

    tabs = {}
    for variant, sel in [("all", np.ones(len(m), bool)), ("first_noself", (m["first"] & ~m["self"]).to_numpy())]:
        cols = {k: m[sel & m.parent_idx.isin(v).to_numpy() & win].groupby("month").size() for k, v in pid.items()}
        img = m[sel & win.to_numpy() & (n.modality.to_numpy()[m.parent_idx] == "image_gen")].groupby("month").size()
        t = pd.DataFrame(cols).fillna(0).astype(int)
        t["image_gen total"] = img
        t["SD1.5 lineage"] = t["runwayml (deleted)"] + t["community mirrors"]
        t["share naming deleted repo"] = t["runwayml (deleted)"] / t["SD1.5 lineage"].replace(0, np.nan)
        t["SD1.5 share of image_gen"] = t["SD1.5 lineage"] / t["image_gen total"]
        t["SDXL share of image_gen"] = t["SDXL"] / t["image_gen total"]
        tabs[variant] = t
        t.to_csv(paths.TABLES / f"p10_sd15_monthly_{variant}.csv")

    # before / after (6 months each side; the event month 2024-08 is excluded from "before")
    t = tabs["all"]
    pre, post = t.loc["2024-02":"2024-07"], t.loc["2024-09":"2025-02"]
    later = t.loc["2025-03":"2025-07"]
    rows = []
    for name, s in [("SD1.5 lineage", "SD1.5 lineage"), ("SDXL", "SDXL"), ("SD 1.4", "SD 1.4"), ("image_gen total", "image_gen total")]:
        rows.append(dict(series=name, pre_mean=pre[s].mean(), post_mean=post[s].mean(), later_mean=later[s].mean(),
                         post_vs_pre=post[s].mean() / pre[s].mean() - 1))
    for s in ("SD1.5 share of image_gen", "SDXL share of image_gen"):
        rows.append(dict(series=s, pre_mean=pre[s].mean(), post_mean=post[s].mean(), later_mean=later[s].mean(),
                         post_vs_pre=post[s].mean() - pre[s].mean()))
    rows.append(dict(series="share naming deleted repo", pre_mean=1.0, post_mean=post["share naming deleted repo"].mean(),
                     later_mean=later["share naming deleted repo"].mean(), post_vs_pre=np.nan))
    ba = pd.DataFrame(rows)
    ba.to_csv(paths.TABLES / "p10_sd15_before_after.csv", index=False)

    # who keeps naming the deleted repo? prior users (habit) vs newcomers (templates / copied cards)
    o, mi = pid["runwayml (deleted)"], pid["community mirrors"]
    before = set(m.author[(m.t < EVENT) & m.parent_idx.isin(o)])
    after = m[(m.t >= EVENT) & m.parent_idx.isin(np.r_[o, mi])]
    who = after.assign(target=np.where(after.parent_idx.isin(o), "deleted repo", "mirror"),
                       prior_user=after.author.isin(before)).groupby(["target", "prior_user"]).size().unstack(fill_value=0)
    who["prior_user_share"] = who.get(True, 0) / who[[c for c in who.columns if isinstance(c, (bool, np.bool_))]].sum(1)
    who.to_csv(paths.TABLES / "p10_sd15_who.csv")

    md = ["# Natural experiment: deletion of runwayml/stable-diffusion-v1-5 (event 2024-08-30)", "",
          "## Before (2024-02..07) vs after (2024-09..2025-02) vs later (2025-03..07), all children", "",
          ba.to_markdown(index=False, floatfmt=".3f"), "",
          "## Who names the deleted repo after the event (authors who used it before the event = prior users)", "",
          who.to_markdown(floatfmt=".3f"), "",
          "## Monthly children, first author-parent choice without self-choices", "",
          tabs["first_noself"].to_markdown(floatfmt=".2f")]
    (paths.TABLES / "p10_sd15.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # figures
    t = tabs["all"]
    x = t.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, k in enumerate(["runwayml (deleted)", "community mirrors", "SD 1.4", "SDXL"]):
        ax.plot(x, t[k], color=SERIES[i], lw=2, marker="o", ms=4, mec=SURFACE, mew=1, label=k)
    ax.axvline(EVENT, color=MUTED, lw=1, ls=":")
    ax.annotate("runwayml repo deleted,\nmirrors created", (EVENT, ax.get_ylim()[1]), xytext=(4, -4),
                textcoords="offset points", fontsize=8, color=INK2, va="top")
    ax.set(ylabel="new child models per month", title="Children per month by base model (finetune + adapter + unspecified)",
           xlim=(x[0] - pd.Timedelta(days=15), x[-1] + pd.Timedelta(days=15)))
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.9))
    ax.grid(color=GRID, lw=0.6)
    _save(fig, "p10_sd15_monthly")

    post_t = tabs["first_noself"].loc["2024-09":]  # habit-free variant: one count per author-parent pair
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = post_t.index.to_timestamp()
    ax.bar(xs, post_t["share naming deleted repo"], width=24, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    for xi, v, nn in zip(xs, post_t["share naming deleted repo"], post_t["SD1.5 lineage"]):
        ax.annotate(f"{v:.0%}\n(n={nn})", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=INK2)
    ax.set(ylim=(0, 1.15), ylabel="share of SD-1.5 children",
           title="After deletion: share of SD-1.5 children still naming the deleted repo\n(first choice of each author, self-choices removed)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    _save(fig, "p10_sd15_share")
