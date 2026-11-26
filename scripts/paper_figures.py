"""Figures for Section 5 of the paper drafts (<draft>/figures/*.pdf), from results/tables/v2/*.csv.
    python scripts/paper_figures.py                 # all drafts in paper_tables.DRAFTS
    python scripts/paper_figures.py --draft "paper/TODO-Foundation Model Race"
fig_shares   five-group Shapley shares, in-sample (95% CI) and holdout          (Section 5.1)
fig_popsplit popularity split, seven-player game                                 (Section 5.3)
fig_types    shares by author type, stacked                                       (Section 5.5)
Colours: categorical slots 1-4 of the dataviz reference palette, fixed order; gray for "all other groups".
"""
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hfpa import paths

sys.path.insert(0, str(paths.REPO / "scripts"))
from paper_tables import DRAFTS  # noqa: E402

T = paths.TABLES / "v2"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRAY, INK, MUTED = "#8a8984", "#1f1f1d", "#5c5b57"
W = 5.4  # inches, about the acmsmall text width
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.grid": True, "axes.grid.axis": "x",
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True})


def save(fig, name, outs):
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
        fig.savefig(o / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_shares(outs):
    S = pd.read_csv(T / "v2s_P5_shapley.csv")
    S = S[(S.variant == "no FE") & (S.partition == "groups")]
    order = ["popularity", "demand match", "self-supply", "habit", "general quality"]
    lab = {"popularity": "Popularity", "demand match": "Demand match", "self-supply": "Self-supply",
           "habit": "Habit", "general quality": "General quality"}
    I = S[S.measure == "in-sample"].set_index("group").loc[order]
    H = S[S.measure == "holdout"].set_index("group").share.loc[order]
    fig, ax = plt.subplots(figsize=(W, 1.9))
    y = np.arange(len(order))
    ax.barh(y - 0.19, I.share * 100, height=0.36, color=C[0], label="In-sample (95% bootstrap CI)")
    ax.errorbar(I.share * 100, y - 0.19, xerr=[(I.share - I.lo) * 100, (I.hi - I.share) * 100], fmt="none",
                ecolor=INK, lw=0.8, capsize=1.5)
    ax.barh(y + 0.19, H * 100, height=0.36, color=C[1], label="Holdout")
    for i, g in enumerate(order):
        a, b = I.share.loc[g] * 100, H.loc[g] * 100
        txt = f"{a:.1f} / {b:.1f}" if a >= 0.05 else "<0.01 / <0.01"
        ax.text(max(a, b) + 2.2, i, txt, va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab[g] for g in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_xlabel("Share of the log-likelihood gain over uniform attachment (%)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_shares", outs)


def fig_popsplit(outs):
    Q = pd.read_csv(T / "v2x_popsplit_full.csv").set_index("group")
    order = ["momentum", "cumulative", "extensive", "visibility", "local", "lifecycle", "nonpop"]
    lab = {"extensive": "Used at all (has_deg)", "cumulative": "Cumulative adoptions (log_deg)",
           "momentum": "Momentum, 30 days (lrec30)", "visibility": "Visibility (org., re-distribution, ...)",
           "local": "Local (co-users, dataset co-use)", "lifecycle": "Lifecycle (age, legacy)",
           "nonpop": "All non-popularity groups"}
    Q = Q.loc[order]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    y = np.arange(len(order))
    ax.barh(y, Q.share * 100, height=0.6, color=[C[0]] * 6 + [GRAY], label="In-sample (95% bootstrap CI)")
    ax.errorbar(Q.share * 100, y, xerr=[(Q.share - Q.lo) * 100, (Q.hi - Q.share) * 100], fmt="none", ecolor=INK,
                lw=0.8, capsize=1.5)
    ax.scatter(Q.hold_share * 100, y, marker="D", s=16, color=C[1], edgecolor="white", linewidth=0.6, zorder=3,
               label="Holdout")
    for i, g in enumerate(order):
        ax.text(max(Q.share.loc[g], Q.hold_share.loc[g]) * 100 + 0.7, i, f"{100 * Q.share.loc[g]:.1f}",
                va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab[g] for g in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 29)
    ax.set_xlabel("Share of the total log-likelihood gain (%)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_popsplit", outs)


def fig_types(outs):
    R = pd.read_csv(T / "v2s_robustness.csv").set_index("config")
    types = ["indiv_light", "indiv_mid", "indiv_heavy", "org_other", "org_redistributor", "org_major"]
    lab = {"indiv_light": "Individuals, <10 models", "indiv_mid": "Individuals, 10-99", "indiv_heavy": "Individuals, >=100",
           "org_other": "Other organizations", "org_redistributor": "Re-distribution accounts",
           "org_major": "Major organizations"}
    X = R.loc[[f"P5_{t}" for t in types]]
    share = X.n_train / X.n_train.sum()
    parts = [("popularity", "Popularity"), ("fit (quality + match)", "Fit (quality + match)"),
             ("self-supply", "Self-supply"), ("habit", "Habit")]
    fig, ax = plt.subplots(figsize=(W, 2.1))
    y = np.arange(len(types))
    left = np.zeros(len(types))
    for (col, name), c in zip(parts, C):
        v = X[col].to_numpy() * 100
        ax.barh(y, v, left=left, color=c, height=0.62, label=name, edgecolor="white", linewidth=1)
        left += v
    for i, (n, s) in enumerate(zip(X.n_train, share)):
        ss = f"{100 * s:.0f}%" if s >= 0.005 else f"{100 * s:.1f}%"
        ax.text(101.5, i, f"n = {int(n):,} ({ss})", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([lab[t] for t in types])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Share of the log-likelihood gain (%), first adoptions")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.25))
    ax.spines["bottom"].set_bounds(0, 100)
    save(fig, "fig_types", outs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", nargs="*", default=DRAFTS)
    outs = [paths.REPO / d / "figures" for d in ap.parse_args().draft]
    fig_shares(outs)
    fig_popsplit(outs)
    fig_types(outs)
    print("figures in", [str(o) for o in outs])
