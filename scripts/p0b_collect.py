"""Collect P0b runs into one table: alpha estimate per scenario x estimator (mean over seeds)."""
import pandas as pd

from hfpa import paths

if __name__ == "__main__":
    df = pd.concat([pd.read_csv(f) for f in sorted((paths.TABLES / "p0b").glob("*_s*.csv"))])
    df.to_csv(paths.TABLES / "p0b_all.csv", index=False)
    a = df[df.param == "log_deg"].copy()
    a["label"] = a.method + a.gamma.map(lambda g: "" if g == 0 else f" g={g:g}")
    a["cell"] = a.est.map("{:.3f}".format) + " (" + a.z_cl.map("{:+.1f}".format) + ")"
    s = a.groupby(["label", "scenario"]).agg(est=("est", "mean"), z=("z_cl", "mean"), n_seeds=("seed", "size"))
    s = s.reset_index()
    s["cell"] = s.est.map("{:.3f}".format) + " (" + s.z.map("{:+.1f}".format) + ")"
    order = ["pa", "fit", "mix", "fit_decay", "mix_decay", "fit_bulk", "mix_bulk"]
    tab = s.pivot(index="label", columns="scenario", values="cell").reindex(columns=[c for c in order if c in s.scenario.unique()])
    tab.to_csv(paths.TABLES / "p0b_alpha.csv")
    truth = df[df.param == "log_deg"].groupby("scenario").truth.first()
    md = ["# P0b: alpha (log_deg) by estimator; cell = estimate (z vs truth, clustered SE)", "",
          "truth: " + ", ".join(f"{k}={truth[k]:g}" for k in tab.columns), "", tab.to_markdown()]
    (paths.TABLES / "p0b_alpha.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
