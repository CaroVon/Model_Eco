"""Build demand-match features (hfpa.features.match) for the 10% modelling sample and describe them.

Demand comes only from the child's declared datasets and the author's earlier parents (P13 inheritance check);
candidate side uses the candidate's own labels.

usage: python scripts/p13_match.py [--spec main_strat_frac0.1]
Writes processed/choices/{spec}/match.parquet (row-aligned with the concatenated rows, plus choice_id/cand_id)
and results/tables/p13_match.md / .csv (coverage; feature means for chosen vs other candidates).
"""
import argparse
import glob
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.features.match import COLS, build, node_attrs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="main_strat_frac0.1")
    args = ap.parse_args()
    t0 = time.time()
    d = paths.PROCESSED / "choices" / args.spec
    ch = pd.read_parquet(d / "choices.parquet").set_index("choice_id", drop=False)
    rows = pd.concat([pd.read_parquet(f, columns=["choice_id", "cand_id", "chosen", "author_reuse", "same_author"])
                      for f in sorted(glob.glob(str(d / "rows" / "*.parquet")))], ignore_index=True)
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["model_id", "pipeline_tag", "params_b", "license_cat", "major_org", "author_code"])
    meta = pd.read_parquet(paths.INTERIM / "meta.parquet", columns=["model_id", "languages", "datasets"]).set_index("model_id")
    meta = meta.reindex(nodes.model_id)
    A = node_attrs(nodes, meta.languages.tolist(), meta.datasets.tolist())
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent", "t"])
    author = nodes.author_code.to_numpy()
    print(f"loaded {len(rows):,} rows, {len(ch):,} choices in {time.time() - t0:.0f}s", flush=True)
    F = build(rows, ch, A, full, author)
    print(f"built in {time.time() - t0:.0f}s", flush=True)
    F.insert(0, "cand_id", rows.cand_id.to_numpy())
    F.insert(0, "choice_id", rows.choice_id.to_numpy())
    F.to_parquet(d / "match.parquet", index=False)

    # description: plan-A first author-parent choices vs all
    full_s = full.assign(a=author[full.child]).sort_values("t", kind="stable")
    first_pairs = set(zip(full_s.child[~full_s.duplicated(["a", "parent"])], full_s.parent[~full_s.duplicated(["a", "parent"])]))
    ch["first"] = [(c, p) in first_pairs for c, p in zip(ch.child, ch.parent)]
    F["chosen"] = rows.chosen.to_numpy()
    F["first"] = ch["first"].reindex(F.choice_id).to_numpy()
    out = []
    for sample, sub in [("all choices", F), ("first author-parent choices", F[F["first"]])]:
        per = sub.groupby("choice_id")[["hist_has", "ds_has"]].max()
        for c in COLS:
            out.append(dict(sample=sample, feature=c, choices_with_signal_hist=per.hist_has.mean(),
                            choices_with_signal_ds=per.ds_has.mean(),
                            mean_chosen=sub.loc[sub.chosen == 1, c].mean(), mean_other=sub.loc[sub.chosen == 0, c].mean()))
    res = pd.DataFrame(out)
    res.to_csv(paths.TABLES / "p13_match.csv", index=False)
    md = ["# P13 demand-match features (10% sample)", "",
          "Coverage = share of choices with author history / usable dataset profile. Means over candidate rows "
          "(chosen vs not chosen; negative-sampled rows, not weighted by log q).", "",
          res.to_markdown(index=False, floatfmt=".3f")]
    (paths.TABLES / "p13_match.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
