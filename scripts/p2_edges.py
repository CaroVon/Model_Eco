"""Stage P2: edges.parquet, nodes.parquet, cleaning funnel (plan 6.3)."""
import pandas as pd

from hfpa import paths
from hfpa.data.build import build_nodes, clean_edges, raw_edges

if __name__ == "__main__":
    meta = pd.read_parquet(paths.INTERIM / "meta.parquet")
    E, funnel, info = clean_edges(meta, raw_edges(meta))
    nodes, E = build_nodes(meta, E)
    (paths.PROCESSED / "edges").mkdir(parents=True, exist_ok=True)
    (paths.PROCESSED / "nodes").mkdir(parents=True, exist_ok=True)
    E.to_parquet(paths.PROCESSED / "edges" / "edges.parquet", index=False)
    nodes.to_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", index=False)
    funnel.to_csv(paths.TABLES / "cleaning_funnel.csv", index=False)

    main = E[E.rel_type.isin(["finetune", "adapter", "unspecified"])]
    top = main.parent.value_counts().head(10)
    L = ["", "# P2: edges and nodes", "", "## Cleaning funnel", funnel.to_markdown(index=False), "",
         f"- dangling share before removal: {info['dangling_share']:.2%} (< 2% -> removed, plan 5.7); largest:",
         *[f"  - `{p}`: {c:,} child edges" for p, c in info["dangling_top"].items()],
         "", "## Edges after cleaning", E.rel_type.value_counts().to_markdown(), "",
         "## Top-10 parents (main spec: finetune+adapter+unspecified)", top.to_markdown(), "",
         "## Nodes", f"- nodes: {len(nodes):,}; legacy: {nodes.legacy.sum():,}; roots (no parent): {nodes.is_root.sum():,}",
         f"- modality: {nodes.modality.value_counts().to_dict()}",
         f"- modality source: {nodes.modality_src.value_counts().to_dict()}",
         f"- generation: {nodes.generation.value_counts().sort_index().head(8).to_dict()}",
         f"- major_org models: {nodes.major_org.sum():,}; params parsed: {nodes.params_b.notna().sum():,}"]
    with open(paths.TABLES / "data_audit.md", "a") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
