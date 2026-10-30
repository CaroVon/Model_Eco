"""Screen for natural experiments like the SD-1.5 deletion (P10): events that change a model's name or
accumulated degree while the weights stay available.

1. Deleted parents: parents named in base_model tags that are not in the snapshot (main relation types),
   with children counts and the time span over which children kept naming them.
2. Name twins: repos with the same repo name (part after "/", case-insensitive) as a large parent
   (>= 200 children) or a deleted parent, under another account: mirrors, re-uploads, ungated copies.
   For each twin: birth, children, and children of the original after the twin appeared.

usage: python scripts/p12_event_screen.py
Writes results/tables/p12_deleted_parents.csv, p12_name_twins.csv, p12_event_screen.md.
"""
import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.data.build import raw_edges

MAIN = ["finetune", "adapter", "unspecified"]

if __name__ == "__main__":
    meta = pd.read_parquet(paths.INTERIM / "meta.parquet", columns=["model_id", "createdAt", "tags", "downloads", "likes"])
    meta["t"] = pd.to_datetime(meta.createdAt, utc=True, errors="coerce")
    R = raw_edges(meta)
    R = R[R.rel_type.isin(MAIN) & (R.child != R.parent)].drop_duplicates(["child", "parent"])
    ids = set(meta.model_id)
    low = set(meta.model_id.str.lower())
    R["exists"] = R.parent.isin(ids) | R.parent.str.lower().isin(low)
    R["t"] = R.child.map(meta.set_index("model_id").t)
    R["child_author"] = R.child.str.split("/").str[0]

    # 1. deleted parents
    d = R[~R.exists & R.parent.str.contains("/")]
    dp = d.groupby("parent").agg(children=("child", "size"), authors=("child_author", "nunique"),
                                 first_child=("t", "min"), last_child=("t", "max")).sort_values("children", ascending=False)
    dp = dp[dp.children >= 30]
    dp.to_csv(paths.TABLES / "p12_deleted_parents.csv")

    # 2. name twins of large or deleted parents
    E = R[R.exists]
    cnt = E.groupby("parent").size()
    big = cnt[cnt >= 200]
    orig = pd.DataFrame({"original": list(big.index) + list(dp.index),
                         "orig_children": list(big.values) + list(dp.children.values),
                         "orig_deleted": [False] * len(big) + [True] * len(dp)})
    orig["name"] = orig.original.str.split("/").str[-1].str.lower()
    m = meta[["model_id", "t", "downloads", "likes"]].assign(name=meta.model_id.str.split("/").str[-1].str.lower())
    tw = orig.merge(m, on="name")
    tw = tw[tw.model_id.str.lower() != tw.original.str.lower()]
    tw["twin_children"] = tw.model_id.map(cnt).fillna(0).astype(int)
    tw = tw[tw.twin_children >= 20]
    ot = meta.set_index("model_id").t
    tw["orig_birth"] = tw.original.map(ot)
    by_parent = E.groupby("parent")
    tw["orig_children_after_twin"] = [int((by_parent.get_group(o).t >= t).sum()) if o in cnt.index else
                                      int((d[d.parent == o].t >= t).sum()) for o, t in zip(tw.original, tw.t)]
    tw["twin_share_after"] = tw.twin_children / (tw.twin_children + tw.orig_children_after_twin)
    tw = tw.rename(columns={"model_id": "twin", "t": "twin_birth"}).sort_values("twin_children", ascending=False)
    cols = ["original", "orig_deleted", "orig_children", "orig_birth", "twin", "twin_birth", "twin_children",
            "orig_children_after_twin", "twin_share_after", "downloads", "likes"]
    tw[cols].to_csv(paths.TABLES / "p12_name_twins.csv", index=False)

    md = ["# P12 event screen", "",
          f"Deleted parents (not in snapshot, >= 30 children, main relation types): {len(dp)}", "",
          dp.head(40).to_markdown(), "",
          f"Name twins with >= 20 children of large (>= 200 children) or deleted parents: {len(tw)} "
          f"(originals: {tw.original.nunique()})", "",
          tw[cols].head(60).to_markdown(index=False, floatfmt=".2f")]
    (paths.TABLES / "p12_event_screen.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:8]))
    print(tw[cols].head(30).to_string())
