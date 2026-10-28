"""Stage P1b: data audit (checkpoint 1) -> results/tables/data_audit.md; lookup tables -> data/external."""
import pandas as pd

from hfpa import paths
from hfpa.data.build import T0, raw_edges
from hfpa.data.lookups import MAJOR_ORGS, MODALITY, TAG2MOD

if __name__ == "__main__":
    paths.ensure()
    meta = pd.read_parquet(paths.INTERIM / "meta.parquet")
    L = ["# Data audit (P1, checkpoint 1)", "",
         f"Source: `modelbiome/ai_ecosystem_withmodelcards`, file `ai_ecosystem_withmodelcards.csv` "
         f"(5,194,246,959 bytes); parsed without `modelCard`.", ""]
    L += ["## 1. Rows and ids", f"- rows: {len(meta):,}", f"- `model_id` unique: {meta.model_id.is_unique}",
          f"- `private` values: {meta.private.value_counts().to_dict()}",
          "- no `author` column in the dataset: author = `model_id` prefix before `/`",
          f"- rows with `exception_raised` set: {meta.exception_raised.notna().sum():,} "
          "(scrape issues with model cards; metadata still present)", ""]
    ca = meta.createdAt
    vc = ca.value_counts()
    L += ["## 2. createdAt", f"- raw format example: `{meta.createdAt_raw.iloc[0]}` (ISO 8601, UTC `Z`)",
          f"- unparseable: {ca.isna().sum()}", f"- min: {ca.min()}  max (snapshot date): {ca.max()}",
          "- most frequent timestamps:"] + [f"  - {t}: {c:,}" for t, c in vc.head(5).items()]
    L += [f"- backfilled block = createdAt <= {T0}: {(ca <= T0).sum():,} models "
          f"(the two timestamps 23:29:04 and 23:29:05 on 2022-03-02); these form the initial graph G0 (legacy=1).",
          f"- study window: choices with child createdAt in ({T0}, {ca.max()}]", ""]
    L += ["## 3. Parent tags (20 random models with a base_model tag)"]
    has = meta[meta.tags.map(lambda t: any(x.startswith("base_model:") for x in t))]
    for _, r in has.sample(20, random_state=paths.CFG["seed"]).iterrows():
        L.append(f"- `{r.model_id}`: " + ", ".join(f"`{x}`" for x in r.tags if x.startswith("base_model:")))
    E = raw_edges(meta)
    L += ["", "Tag format confirmed: HF writes both a bare `base_model:<id>` and a typed `base_model:<rel>:<id>` tag;"
          " the parser keeps the typed one. Relation counts from tags: "
          + ", ".join(f"{k}={v:,}" for k, v in E.rel_type.value_counts().items()) + "."]
    cols = []
    for r in ["finetune", "adapter", "quantized", "merge"]:
        s = meta[["model_id", f"{r}_parent"]].explode(f"{r}_parent").dropna()
        cols.append(pd.DataFrame({"child": s.model_id, "parent": s[f"{r}_parent"], "rel_type": r}))
    C = pd.concat(cols)
    j = C.merge(E, on=["child", "parent"], how="outer", indicator=True, suffixes=("_col", "_tag"))
    L += [f"- comparison with the dataset's own `*_parent` columns: {j._merge.value_counts().to_dict()}; "
          "the tag-only rows are untyped (`unspecified`) parents, treated as finetune in the main spec.", ""]
    L += ["## 4. Snapshot", f"- snapshot (max createdAt): {ca.max()}", ""]
    L += ["## pipeline_tag", f"- missing: {meta.pipeline_tag.isna().sum():,} ({meta.pipeline_tag.isna().mean():.1%})",
          f"- tags not in modality map: {sorted(set(meta.pipeline_tag.dropna()) - set(TAG2MOD))}"]
    (paths.TABLES / "data_audit.md").write_text("\n".join(L) + "\n")
    pt = meta.pipeline_tag.value_counts()
    pd.DataFrame([(t, m, int(pt.get(t, 0))) for m, ts in MODALITY.items() for t in ts],
                 columns=["pipeline_tag", "modality", "n_models"]).to_csv(paths.EXTERNAL / "modality_map.csv", index=False)
    pd.DataFrame({"org": sorted(set(MAJOR_ORGS))}).to_csv(paths.EXTERNAL / "major_orgs.csv", index=False)
    print("\n".join(L))
