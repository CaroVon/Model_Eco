"""Collect P0c runs: alpha (log_deg) by estimator x period x gamma, mean and sd over seeds."""
import pandas as pd

from hfpa import paths

if __name__ == "__main__":
    df = pd.concat([pd.read_csv(f) for f in sorted((paths.TABLES / "p0c").glob("*_s*.csv"))])
    df.to_csv(paths.TABLES / "p0c_all.csv", index=False)
    order = ["pa", "fit", "mix", "fit_decay", "mix_decay", "fit_bulk", "mix_bulk"]
    md = ["# P0c: node-FE panel estimators; cell = mean (sd) of the estimate over seeds", ""]
    for param in ("log_deg", "has_deg"):
        a = df[df.param == param].copy()
        a["label"] = a.method + " p=" + a.period.astype(str) + a.gamma.map(lambda g: "" if g == 0 else f" g={g:g}")
        s = a.groupby(["label", "scenario"]).est.agg(["mean", "std"]).reset_index()
        s["cell"] = s["mean"].map("{:.3f}".format) + " (" + s["std"].map("{:.3f}".format) + ")"
        tab = s.pivot(index="label", columns="scenario", values="cell")
        tab = tab.reindex(columns=[c for c in order if c in tab.columns])
        truth = a.groupby("scenario").truth.first()
        md += [f"## {param}", "truth: " + ", ".join(f"{k}={truth[k]:g}" for k in tab.columns), "", tab.to_markdown(), ""]
    (paths.TABLES / "p0c_summary.md").write_text("\n".join(md))
    print("\n".join(md))
