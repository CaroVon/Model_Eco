"""LaTeX tables for the paper drafts (<draft>/tables/*.tex), generated from results/tables/v2/*.csv so every number
in the paper traces to a result file. Re-run after any re-estimation:
    python scripts/paper_tables.py                 # EC draft (ec_adoption/) and ICML draft (paper/)
    python scripts/paper_tables.py --draft ec_adoption
Also writes <draft>/tables/numbers.tex: \\newcommand macros for numbers quoted in the text.
"""
import argparse
import numpy as np
import pandas as pd

from hfpa import paths

T = paths.TABLES / "v2"
DRAFTS = ["ec_adoption", "paper"]
GROUP_TEX = {"popularity": "Popularity", "general quality": "General quality", "demand match": "Demand match",
             "inertia": "Inertia"}
ORDER = list(GROUP_TEX)
MACROS = {}


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def macro(name, value):
    MACROS[name] = value


def put(name, body):
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.tex").write_text(body)



def boot(spec):
    B = pd.read_csv(T / f"v2_{spec}_shapley_boot.csv")
    return B[B.partition == "four groups"].set_index("group")


def point(spec, measure, variant="no FE"):
    S = pd.read_csv(T / f"v2_{spec}_shapley.csv")
    S = S[(S.variant == variant) & (S.measure == measure) & (S.partition == "four groups")]
    return S.set_index("group").share


def tab_shapley():
    B, H = boot("pps"), point("pps", "holdout")
    F = pd.read_csv(T / "v2_pps_clogit_fit.csv").set_index("model")
    rows = []
    for g in ORDER:
        b = B.loc[g]
        rows.append(f"{GROUP_TEX[g]} & {pct(b.share, 1 if b.share > 1e-3 else 3)} "
                    f"& [{pct(b.lo, 1 if b.share > 1e-3 else 3)}, {pct(b.hi, 1 if b.share > 1e-3 else 3)}] "
                    f"& {pct(H.loc[g], 1 if H.loc[g] > 1e-3 else 3)} \\\\")
        macro(f"sh{g.split()[-1].capitalize()}", pct(b.share))
        macro(f"sh{g.split()[-1].capitalize()}Lo", pct(b.lo))
        macro(f"sh{g.split()[-1].capitalize()}Hi", pct(b.hi))
        macro(f"shHold{g.split()[-1].capitalize()}", pct(H.loc[g]))
    full = F.loc["full"]
    macro("holdGain", f"{full.hold_gain:.2f}")
    macro("holdTopOne", pct(full.hold_top1))
    macro("holdTopTen", pct(full.hold_top10))
    macro("bootRepsShapley", str(int(B.reps.iloc[0])))
    lr = F.drop(index=["full", "full + FE"])
    body = ("\\begin{table}[t]\n\\centering\n\\caption{Shapley decomposition of the log-likelihood gain over uniform "
            "attachment (main specification: $(d{+}1)$ PPS candidate sets, first adoptions). In-sample shares with "
            f"95\\% multiplier-bootstrap intervals ({int(B.reps.iloc[0])} replicates, authors resampled); holdout = "
            "mean per-choice log-likelihood on the last 10\\% of choices in time.}\n\\label{tab:shapley}\n"
            "\\small\n\\begin{tabular}{lrcr}\n\\toprule\nGroup & Share (\\%) & 95\\% CI & Holdout (\\%) \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\midrule\n"
            f"\\multicolumn{{4}}{{l}}{{Holdout gain over uniform: {full.hold_gain:.2f} nats/choice; "
            f"top-1 / top-10 accuracy {pct(full.hold_top1)}\\% / {pct(full.hold_top10)}\\%}} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    put("tab_shapley", body)
    for m, r in lr.iterrows():
        macro("lr" + m.replace("drop ", "").capitalize(), f"{r.lr_stat:,.0f}")


def tab_q():
    specs = [("pps", "$(d{+}1)$ PPS"), ("strat", "Stratified"), ("unif", "Uniform")]
    C = {s: pd.read_csv(T / f"v2_{s}_clogit_boot.csv").set_index("param") for s, _ in specs}
    F = {s: pd.read_csv(T / f"v2_{s}_clogit_fit.csv").set_index("model").loc["full"] for s, _ in specs}
    B = {s: boot(s) for s, _ in specs}
    lines = []
    for g in ORDER:
        cells = []
        for s, _ in specs:
            b = B[s].loc[g]
            d = 1 if b.share > 1e-3 else 3
            cells.append(f"{pct(b.share, d)} [{pct(b.lo, d)}, {pct(b.hi, d)}]")
        lines.append(f"{GROUP_TEX[g]} share (\\%) & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for p, lab in (("log_deg", "$\\log d$"), ("has_deg", "$\\mathbf{1}[d>0]$"), ("major_org", "major org."),
                   ("lang_hist", "language history"), ("task_hist", "task history"), ("txt_sim", "card-text sim.")):
        lines.append(f"{lab} & " + " & ".join(f"{C[s].loc[p].coef:.2f} ({C[s].loc[p].se_boot:.2f})" for s, _ in specs)
                     + " \\\\")
    lines.append("\\midrule")
    lines.append("Holdout top-1 acc. (\\%) & " + " & ".join(pct(F[s].hold_top1) for s, _ in specs) + " \\\\")
    lines.append("Bootstrap replicates (shares / coef.) & " + " & ".join(
        f"{int(B[s].reps.iloc[0])} / {int(C[s].reps.iloc[0])}" for s, _ in specs) + " \\\\")
    body = ("\\begin{table*}[t]\n\\centering\n\\caption{The decomposition under three candidate-set designs. "
            "Same choices, features and estimator; only the inclusion probabilities $\\pi$ differ. Coefficients with "
            "bootstrap SEs. Under uniform sampling almost every sampled competitor is a never-used model, so "
            "top-1 accuracy is high but the data contain few comparisons among popular models.}\n\\label{tab:q}\n"
            "\\small\n\\begin{tabular}{lccc}\n\\toprule\n & " + " & ".join(l for _, l in specs) + " \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    put("tab_q", body)
    macro("logdegUnif", f"{C['unif'].loc['log_deg'].coef:.3f}")
    macro("logdegPps", f"{C['pps'].loc['log_deg'].coef:.3f}")
    macro("topOneUnif", pct(F["unif"].hold_top1))


def tab_bounds():
    Bd = pd.read_csv(T / "v2_bounds.csv").set_index("bound")
    J = pd.read_csv(T / "v2_placebo_joint.csv").set_index("group")
    P = pd.read_csv(T / "v2_placebo.csv")
    Q = pd.read_csv(T / "v2_quality_ext.csv")
    q = Q[(Q.source == "v2 average") & (Q.controls == "redistributor+lparams")].iloc[0]
    rows = [
        ("No FE; quality = benchmark z-score", Bd.loc["lower (no FE; quality = bench_z only)"], None),
        ("Hub FE counted as quality (in-sample)", Bd.loc["FE all quality (in-sample; old-style upper bound)"], None),
        ("Hub FE counted as quality (holdout)", Bd.loc["FE all quality (holdout; old-style upper bound)"], None),
        ("Externally explained part of FE (in-sample)", Bd.loc["external quality (v2 average, controls), in-sample"], None),
        ("Externally explained part of FE (holdout)", Bd.loc["external quality (v2 average, controls), holdout"], None),
    ]
    lines = [f"{lab} & {pct(r.fit)} & {pct(r.popularity)} \\\\" for lab, r, _ in rows]
    body = ("\\begin{table}[t]\n\\centering\n\\caption{How large could fit be? Fit = general quality + demand match. "
            "Hub FE: 500 hub-specific intercepts (top-500 by degree at the holdout cut). External quality explains "
            f"{pct(q.quality_part_share)}\\% of the FE variance among the {int(q.hubs)} hubs covered by the Open LLM "
            "Leaderboard v2 (noise-corrected, with redistributor and size controls).}\n\\label{tab:bounds}\n\\small\n"
            "\\begin{tabular}{lrr}\n\\toprule\nSpecification & Fit (\\%) & Popularity (\\%) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    put("tab_bounds", body)
    macro("feInQual", pct(Bd.loc["FE all quality (in-sample; old-style upper bound)"].fit))
    macro("feHoldQual", pct(Bd.loc["FE all quality (holdout; old-style upper bound)"].fit))
    macro("feShare", pct(Bd.loc["FE all quality (in-sample; old-style upper bound)"].fe_share))
    macro("feShareHold", pct(Bd.loc["FE all quality (holdout; old-style upper bound)"].fe_share))
    macro("extQ", pct(q.quality_part_share))
    macro("extHubs", str(int(q.hubs)))
    macro("extFitIn", pct(Bd.loc["external quality (v2 average, controls), in-sample"].fit))
    macro("extFitHold", pct(Bd.loc["external quality (v2 average, controls), holdout"].fit))
    macro("jointOwn", pct(J.loc["demand match"].share))
    macro("jointDonor", pct(J.loc["placebo (donor) match"].share))
    pl = P[(P.demand == "placebo") & (P.group == "demand match")].share.iloc[0]
    macro("placeboAlone", pct(pl))
    F = pd.read_csv(T / "v2_pps_clogit_fit.csv").set_index("model")
    macro("holdGainFE", f"{F.loc['full + FE'].hold_gain:.3f}")
    macro("holdGainNoFE", f"{F.loc['full'].hold_gain:.3f}")


def tab_coef():
    C = pd.read_csv(T / "v2_pps_clogit_boot.csv")
    M = pd.read_csv(T / "v2_mrs.csv")
    M = M[M.ref == "log_deg"].set_index("feature")
    gname = {"pop": "Popularity", "qual": "Quality", "match": "Demand match", "inertia": "Inertia"}
    lines, last = [], None
    for _, r in C.iterrows():
        if r.group != last:
            if last is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{gname[r.group]}}}}} \\\\")
            last = r.group
        mult = f"{M.loc[r.param].multiplier:.3g}" if r.param in M.index else "--"
        name = r.param.replace("_", "\\_")
        lines.append(f"\\texttt{{{name}}} & {r.coef:.3f} & {r.se_cl:.3f} & [{r.lo:.3f}, {r.hi:.3f}] & {mult} \\\\")
    body = ("\\begin{table}[h]\n\\centering\n\\caption{Conditional-logit coefficients, main specification (no FE). "
            f"SE clustered by author; 95\\% multiplier-bootstrap interval ({int(C.reps.iloc[0])} replicates). Last "
            "column: the factor on $(1+d)$ that is worth the same utility as a one-SD (binary: 0$\\to$1) change of the "
            "feature.}\n\\label{tab:coef}\n\\scriptsize\n\\begin{tabular}{lrrcr}\n\\toprule\nFeature & Coef. & SE & "
            "95\\% CI & $(1{+}d)$ multiple \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    put("tab_coef", body)


def tab_validate():
    V = pd.read_csv(T / "v2_validate_10pct.csv").dropna()
    lines = [f"\\texttt{{{r.param.replace('_', chr(92) + '_')}}} & {r.coef_10pct:.3f} ({r.se_cl_10pct:.3f}) & "
             f"{r.coef_full:.3f} ({r.se_cl_full:.3f}) & {'yes' if r.in_10pct_ci else 'no'} & {r.se_ratio:.2f} \\\\"
             for _, r in V.iterrows()]
    body = ("\\begin{table}[h]\n\\centering\n\\caption{Full-data estimates (stratified design) against the earlier "
            "10\\% sample. Differences outside the 10\\% interval concern features redefined in the rebuild.}\n"
            "\\label{tab:validate}\n\\scriptsize\n\\begin{tabular}{lccrr}\n\\toprule\nFeature & 10\\% sample & Full & "
            "In CI & SE ratio \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    put("tab_validate", body)
    macro("valIn", str(int(V.in_10pct_ci.sum())))
    macro("valN", str(len(V)))
    macro("valSeRatio", f"{V.se_ratio.median():.2f}")


def data_numbers():
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code", "modality_code"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["rel_type"])
    ch = pd.read_parquet(paths.PROCESSED / "choices" / "v2_pps" / "decisions.parquet",
                         columns=["choice_id", "t", "parent", "author", "pair_first", "dfirst24", "holdout", "modality",
                                  "modality_name", "parent_deg"])
    pa = ch[ch.pair_first]
    mb = {m: np.sort(nodes.birth.to_numpy()[nodes.modality_code.to_numpy() == m]) for m in np.unique(nodes.modality_code)}
    ms = np.array([np.searchsorted(mb[m], t) for m, t in zip(pa.modality, pa.t)])
    c = pa.parent.value_counts()
    p = c.to_numpy() / c.sum()
    tr = pd.read_csv(T / "v2_pps_shapley_boot.csv")
    fit = pd.read_csv(T / "v2_pps_clogit_fit.csv").set_index("model").loc["full"]
    vals = dict(nModels=f"{len(nodes):,}", nAuthorsAll=f"{nodes.author_code.nunique():,}", nEdges=f"{len(E):,}",
                nChoices=f"{len(ch):,}", nDecisions=f"{int(ch.dfirst24.sum()):,}", nFirst=f"{len(pa):,}",
                nFirstAuthors=f"{pa.author.nunique():,}", nFirstParents=f"{len(c):,}",
                nTrain=f"{int(fit.n):,}", topTenReal=pct(p[:10].sum()), topOneReal=pct(p[0]),
                effReal=f"{np.exp(-(p * np.log(p)).sum()):,.0f}", marketMedian=f"{np.median(ms):,.0f}",
                shareText=pct((pa.modality_name == "text").mean(), 0),
                tStart=pd.to_datetime(ch.t.min(), unit="s").strftime("%B %Y"),
                tEnd=pd.to_datetime(ch.t.max(), unit="s").strftime("%B %Y"),
                tHold=pd.to_datetime(ch.t[ch.holdout].min(), unit="s").strftime("%B %-d, %Y"),
                shareParentZero=pct((pa.parent_deg == 0).mean(), 0),
                nAdapter=f"{int((E.rel_type == 'adapter').sum()):,}", nFinetune=f"{int((E.rel_type == 'finetune').sum()):,}")
    for k, v in vals.items():
        macro(k, v)
    _ = tr


def extra_numbers():
    R = pd.read_csv(T / "v2_networks_recovery.csv")
    r4 = R[R.partition == "four groups"].set_index(["network", "group"]).share
    r3 = R[R.partition == "pop / fit / inertia"].set_index(["network", "group"]).share
    macro("recSfPop", pct(r4.loc[("SF-SIR", "popularity")]))
    macro("recQfFit", pct(r3.loc[("QF", "fit (quality + match)")]))
    macro("recFullPop", pct(r4.loc[("FULL", "popularity")]))
    macro("recFullMatch", pct(r4.loc[("FULL", "demand match")]))
    macro("recFullInertia", pct(r4.loc[("FULL", "inertia")]))
    S = pd.read_csv(T / "v2_signal_samples.csv")
    S = S[S.partition == "four groups"].set_index(["sample", "group"])
    n_all = S.loc[("all choices", "popularity")].n
    n_sig = S.loc[("choices with a demand signal", "popularity")].n
    macro("sigShare", pct(n_sig / n_all, 0))
    macro("sigMatch", pct(S.loc[("choices with a demand signal", "demand match")].share))
    macro("sigMatchRw", pct(S.loc[("signal choices reweighted to all", "demand match")].share))
    macro("sigPop", pct(S.loc[("choices with a demand signal", "popularity")].share))
    I = pd.read_csv(T / "v2_identical_pairs.csv").set_index("quantity")
    macro("idN", f"{int(I.loc['choices with both copies in the set and one chosen'].value):,}")
    macro("idHigher", pct(I.loc["share of choices where the chosen copy has the higher degree"].value))
    macro("idRec", f"{I.loc['coef d_lrec30'].value:.2f}")
    macro("idRecSe", f"{I.loc['coef d_lrec30'].se:.2f}")
    macro("idDeg", f"{I.loc['coef d_log_deg'].value:.2f}")
    macro("idDegSe", f"{I.loc['coef d_log_deg'].se:.2f}")
    M = pd.read_csv(T / "v2_mrs.csv")
    M = M[M.ref == "log_deg"].set_index("feature")
    macro("mrsTask", f"{M.loc['task_hist'].multiplier:,.0f}")
    macro("mrsTaskLo", f"{M.loc['task_hist'].mult_lo:,.0f}")
    macro("mrsTaskHi", f"{M.loc['task_hist'].mult_hi:,.0f}")
    macro("mrsRec", f"{M.loc['lrec30'].multiplier:.1f}")
    macro("mrsOrg", f"{M.loc['major_org'].multiplier:.1f}")
    C = pd.read_csv(T / "v2_classify.csv")
    tie = C[C.type.str.startswith("(ties) mean number")].expected_n.iloc[0]
    macro("tieMean", f"{tie:.1f}")


def tab_samples():
    """v2.1: sample selection (U / D / P), author weights, top-1% authors, author types; five groups."""
    R = pd.read_csv(T / "v2s_robustness.csv").set_index("config")
    lab = {"P5": "First adoptions (main)", "P5top": "First adoptions, top 1\\% of authors dropped",
           "P5a": "First adoptions, authors weighted equally", "D5": "All 24h decisions",
           "D5a": "All 24h decisions, authors weighted equally", "D5top": "All 24h decisions, top 1\\% dropped",
           "U5": "All uploads (no session merge)"}
    tlab = {"indiv_light": "Individuals, $<$10 models", "indiv_mid": "Individuals, 10--99 models",
            "indiv_heavy": "Individuals, $\\ge$100 models", "org_other": "Other organizations",
            "org_redistributor": "Re-distribution accounts", "org_major": "Major organizations"}
    f = lambda r, g: (f"{pct(r[g])}" + (f" [{pct(r[g + ' lo'])}, {pct(r[g + ' hi'])}]"
                                         if g + " lo" in r and pd.notna(r[g + " lo"]) else ""))
    line = lambda name, r: (f"{name} & {int(r.n_train):,} & {int(r.authors):,} & {f(r, 'popularity')} & "
                            f"{f(r, 'fit (quality + match)')} & {pct(r['self-supply'])} & {pct(r['habit'])} & "
                            f"{r['ratio fit / popularity']:.3f} \\\\")
    rows = [line(lab[k], R.loc[k]) for k in lab if k in R.index]
    rows.append("\\midrule\n\\multicolumn{8}{l}{\\emph{First adoptions by author type}} \\\\")
    rows += [line(tlab[t], R.loc[f"P5_{t}"]) for t in tlab if f"P5_{t}" in R.index]
    body = ("\\begin{table}[t]\n\\centering\n\\caption{Shapley shares (\\%, in-sample, no FE) by sample, weights and "
            "author type; 95\\% multiplier-bootstrap intervals by author.}\n\\label{tab:samples}\n\\small\n"
            "\\begin{tabular}{lrrrrrrr}\n\\toprule\nSample & Choices & Authors & Popularity & Fit & Self-supply & "
            "Habit & Fit/Pop \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    put("tab_samples", body)
    m = R.loc["P5"]
    for key, g in (("Pop", "popularity"), ("Fit", "fit (quality + match)"), ("Self", "self-supply"), ("Habit", "habit")):
        macro(f"fiveSh{key}", pct(m[g]))
        if g + " lo" in m and pd.notna(m[g + " lo"]):
            macro(f"fiveSh{key}Lo", pct(m[g + " lo"]))
            macro(f"fiveSh{key}Hi", pct(m[g + " hi"]))
    macro("fiveRatio", f"{m['ratio fit / popularity']:.3f}")
    for k, nm in (("D5", "Dec"), ("P5a", "AuthW"), ("D5a", "DecAuthW"), ("P5top", "Top"), ("U5", "Up")):
        if k in R.index:
            macro(f"samp{nm}Pop", pct(R.loc[k, "popularity"]))
            macro(f"samp{nm}Fit", pct(R.loc[k, "fit (quality + match)"]))
            macro(f"samp{nm}Habit", pct(R.loc[k, "habit"]))
    for t, nm in (("indiv_light", "Light"), ("indiv_heavy", "Heavy"), ("org_major", "Major")):
        macro(f"type{nm}Pop", pct(R.loc[f"P5_{t}", "popularity"]))
        macro(f"type{nm}Fit", pct(R.loc[f"P5_{t}", "fit (quality + match)"]))
        macro(f"type{nm}Self", pct(R.loc[f"P5_{t}", "self-supply"]))
    H = pd.read_csv(T / "v2s_hetero.csv").set_index("sample")
    macro("heteroLR", f"{H.loc['P'].lr:,.0f}")
    macro("heteroDf", str(int(H.loc["P"].df)))
    Cc = pd.read_csv(T / "v2s_conc_P5.csv").set_index("model")
    macro("concPopEff", f"{Cc.loc['pop'].eff_parents:,.0f}")
    macro("concPopSelfEff", f"{Cc.loc['pop + self'].eff_parents:,.0f}")
    Cd = pd.read_csv(T / "v2s_conc_D5.csv").set_index("model")
    macro("concDecPopEff", f"{Cd.loc['pop'].eff_parents:,.0f}")
    macro("concDecPopHabitEff", f"{Cd.loc['pop + habit'].eff_parents:,.0f}")
    L = pd.read_csv(T / "v2s_ladder.csv").set_index("sample")
    macro("ladderTopUp", pct(L.iloc[0].top1pct_share))
    macro("ladderTopDec", pct(L.iloc[1].top1pct_share))
    macro("ladderTopFirst", pct(L.iloc[2].top1pct_share))
    W = pd.read_csv(T / "v2s_retention_welfare.csv")
    w = W[(W.stratum == "individuals (light + mid)") & (W.unit == "per mismatched choice")].iloc[0]
    macro("retIndivBeta", f"{w.beta_match_fe:.3f}")
    macro("retIndivGain", pct(w.d_pr_stay))
    macro("retIndivRel", pct(w.rel_to_pbar))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", nargs="*", default=DRAFTS)
    OUTS = [paths.REPO / d / "tables" for d in ap.parse_args().draft]
    tab_shapley()
    tab_q()
    tab_bounds()
    tab_coef()
    tab_validate()
    data_numbers()
    extra_numbers()
    tab_samples()
    put("numbers", "% generated by scripts/paper_tables.py -- do not edit\n" +
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n")
    print(f"{len(MACROS)} macros; tables in {[str(o) for o in OUTS]}")
