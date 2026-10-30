"""P5 ladder in the layout of Overgoor et al. (2020) tables 3-4: one column per model, coefficient with
clustered SE below, then log-likelihood, observations and holdout metrics. Reads results/tables/p5_ladder.csv."""
import numpy as np
import pandas as pd

from hfpa import paths

LABEL = {"has_deg": "Has degree", "log_deg": "log Degree (α)", "log_age": "log(1 + age)", "legacy": "Legacy (≤ T0)",
         "age_lt30": "Age < 30d", "age_30_90": "Age 30–90d", "age_90_365": "Age 90–365d", "lrec30": "log(1 + children, 30d)",
         "same_author": "Same author", "author_reuse": "Author used this parent", "author_family": "Author used this family",
         "lfam_recent": "log(1 + family births, 30d)", "lquant": "log(1 + quantized children)"}

if __name__ == "__main__":
    d = pd.read_csv(paths.TABLES / "p5_ladder.csv")
    models = [m for m in ["M1", "M2", "M3", "M4", "M5", "M6", "M7"] if m in set(d.model)]
    params = [p for p in LABEL if p in set(d.param)]
    lines = ["| | " + " | ".join(models) + " |", "|:--|" + "--:|" * len(models)]
    for p in params:
        c, s = [], []
        for m in models:
            r = d[(d.model == m) & (d.param == p)]
            if len(r):
                z = abs(r.coef.iloc[0] / r.se_cl.iloc[0])
                c.append(f"{r.coef.iloc[0]:.3f}{'*' if z > 2.576 else ''}")
                s.append(f"({r.se_cl.iloc[0]:.3f})")
            else:
                c.append(""), s.append("")
        lines += [f"| {LABEL[p]} | " + " | ".join(c) + " |", "| | " + " | ".join(s) + " |"]
    first = d.groupby("model").first()
    extras = {"Static features (M4, M6)": ["", "", "", "yes", "", "yes", "yes"],
              "Hub fixed effects (K = 500)": ["", "", "", "", "yes", "yes", "yes"]}
    for k, v in extras.items():
        lines.append(f"| {k} | " + " | ".join(v[: len(models)]) + " |")
    for k, col, f in [("Observations", "n", "{:,.0f}"), ("Log-likelihood (weighted)", "ll", "{:,.0f}"),
                      ("Holdout mean log-likelihood", "hold_mean_ll", "{:.3f}"), ("Holdout top-1 accuracy", "hold_top1", "{:.3f}"),
                      ("Holdout top-10 accuracy", "hold_top10", "{:.3f}")]:
        lines.append(f"| {k} | " + " | ".join(f.format(first.loc[m, col]) for m in models) + " |")
    m0 = d[d.model == "M0"]
    note = (f"Note: conditional logit with stratified negative sampling and log q correction; choice weights n^-0.5; "
            f"SE clustered by child author; * p < 0.01. Holdout = last 10% of choices in time. "
            f"M0 (uniform): holdout mean log-likelihood {m0.hold_mean_ll.iloc[0]:.3f}. Coefficients are conditional associations.")
    md = "# P5 model ladder (layout of Overgoor et al. 2020, tables 3-4)\n\n" + "\n".join(lines) + "\n\n" + note + "\n"
    (paths.TABLES / "p5_ladder_table.md").write_text(md)
    print(md)
