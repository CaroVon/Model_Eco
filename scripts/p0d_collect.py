"""Collect P0d runs: alpha by estimator x period x outcome variant; mean, sd over seeds, mean SE."""
import pandas as pd

from hfpa import paths

if __name__ == "__main__":
    df = pd.concat([pd.read_csv(f) for f in sorted((paths.TABLES / "p0d").glob("*_s*.csv"))])
    df.to_csv(paths.TABLES / "p0d_all.csv", index=False)
    order = ["pa", "fit", "mix", "fit_decay", "mix_decay", "fit_bulk", "mix_bulk"]
    md = ["# P0d: quasi-differenced GMM; cell = mean [sd over seeds] {mean clustered SE}", ""]
    for param in ("log_deg", "has_deg"):
        a = df[df.param == param].copy()
        a["label"] = a.method + " p=" + a.period.astype(str) + " y=" + a.y
        s = a.groupby(["label", "scenario"]).agg(m=("est", "mean"), sd=("est", "std"), se=("se", "mean")).reset_index()
        s["cell"] = (s.m.map("{:.3f}".format) + " [" + s.sd.map("{:.3f}".format) + "]"
                     + s.se.map(lambda v: "" if pd.isna(v) else f" {{{v:.3f}}}"))
        tab = s.pivot(index="label", columns="scenario", values="cell")
        tab = tab.reindex(columns=[c for c in order if c in tab.columns])
        truth = a.groupby("scenario").truth.first()
        md += [f"## {param}", "truth: " + ", ".join(f"{k}={truth[k]:g}" for k in tab.columns), "", tab.to_markdown(), ""]
    j = df[(df.method == "QD_GMM2") & (df.param == "log_deg")].groupby(["scenario", "period", "y"]).J.mean()
    md += ["## Hansen J (mean over seeds; df = %d)" % df.J_df.iloc[0], "", j.unstack(["period", "y"]).round(1).to_markdown(), ""]
    (paths.TABLES / "p0d_summary.md").write_text("\n".join(md))
    print("\n".join(md))
