"""Inheritance check: how often does a child's metadata field simply repeat its parent's?

A field can carry the developer's own demand (a signal for demand-match features) only if children do not
copy it from the chosen parent. For each field and each main-spec choice (finetune / adapter / unspecified,
after T0), compare child and parent values:
  child_has   share of choices where the child has a value
  both_has    share where both have a value
  identical   among both_has: child value == parent value (sets compared as sets)
  overlap     among both_has: child and parent sets intersect
  chance      the same 'identical' rate against a random parent of the same modality chosen in the same month
              (agreement expected without copying)
  excess      identical - chance
Computed on all choices and on first author-parent choices (plan A).
Fields: pipeline_tag, library_name, languages, datasets, licenses, params_b (name-parsed, equal within 5%).

usage: python scripts/p13_inherit.py
Writes results/tables/p13_inherit.csv and p13_inherit.md.
"""
import numpy as np
import pandas as pd

from hfpa import paths

FIELDS = ["pipeline_tag", "library_name", "languages", "datasets", "licenses", "params_b"]
SETS = {"languages", "datasets", "licenses"}


def as_key(v, field):
    if field in SETS:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        s = frozenset(str(x).lower() for x in v)
        return s or None
    if field == "params_b":
        return float(v) if np.isfinite(v) and 0 < v < 2000 else None
    return None if pd.isna(v) or v == "" else str(v)


def same(a, b, field):
    if field == "params_b":
        return abs(np.log(a) - np.log(b)) < 0.05
    return a == b


def stats(c, p, r, field):
    has_c = np.array([x is not None for x in c])
    both = has_c & np.array([x is not None for x in p])
    both_r = has_c & np.array([x is not None for x in r])
    ident = np.array([same(a, b, field) if ok else False for a, b, ok in zip(c, p, both)])
    chance = np.array([same(a, b, field) if ok else False for a, b, ok in zip(c, r, both_r)])
    if field in SETS:
        ov = np.array([bool(a & b) if ok else False for a, b, ok in zip(c, p, both)])
    else:
        ov = ident
    nb, nr = max(both.sum(), 1), max(both_r.sum(), 1)
    return dict(child_has=has_c.mean(), both_has=both.mean(), identical=ident.sum() / nb, overlap=ov.sum() / nb,
                chance=chance.sum() / nr, excess=ident.sum() / nb - chance.sum() / nr, n_both=int(both.sum()))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    meta = pd.read_parquet(paths.INTERIM / "meta.parquet", columns=["model_id", "languages", "datasets", "licenses"])
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["model_id", "pipeline_tag", "library_name", "params_b", "author_code"])
    nodes = nodes.join(meta.set_index("model_id"), on="model_id")
    ch = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet",
                         columns=["child", "parent", "t", "modality"])
    ch["author"] = nodes.author_code.to_numpy()[ch.child]
    ch = ch.sort_values("t", kind="stable").reset_index(drop=True)
    ch["first"] = ~ch.duplicated(["author", "parent"])
    # random comparison parent: another choice in the same modality and month
    ch["month"] = (ch.t // (30 * 86400)).astype(int)
    ch["rand_parent"] = ch.parent.to_numpy()
    for _, idx in ch.groupby(["modality", "month"]).indices.items():
        ch.loc[ch.index[idx], "rand_parent"] = ch.parent.to_numpy()[rng.choice(idx, len(idx))]
    out = []
    for sample, sub in [("all choices", ch), ("first author-parent choices", ch[ch["first"]])]:
        for f in FIELDS:
            col = nodes[f].to_numpy()
            c = [as_key(v, f) for v in col[sub.child]]
            p = [as_key(v, f) for v in col[sub.parent]]
            r = [as_key(v, f) for v in col[sub.rand_parent]]
            out.append(dict(sample=sample, field=f, **stats(c, p, r, f)))
            print(out[-1], flush=True)
    res = pd.DataFrame(out)
    res.to_csv(paths.TABLES / "p13_inherit.csv", index=False)
    md = ["# P13 inheritance check: child metadata vs chosen parent", "",
          "identical / overlap: among choices where both child and parent have the field. chance: identical rate "
          "against a random parent chosen in the same modality and month. excess = identical - chance. "
          "Fields with a high excess are largely copied from the parent and cannot serve as demand signals.", "",
          res.to_markdown(index=False, floatfmt=".3f")]
    (paths.TABLES / "p13_inherit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
