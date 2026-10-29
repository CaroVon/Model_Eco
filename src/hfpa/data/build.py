"""P2: edge table, cleaning funnel and node table (plan 5.1, 5.2, 5.7, 5.8, 6.3)."""
import re

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.data.lookups import LIB2MOD, MAJOR_ORGS, MODALITIES, TAG2MOD
from hfpa.data.parse_tags import parse_parents

T0 = pd.Timestamp(paths.CFG["replay"]["t0"] + " 23:29:05", tz="UTC")  # last backfilled timestamp (see audit)
REL_PRIORITY = ["finetune", "adapter", "unspecified", "merge", "quantized"]  # dedupe + primary-parent order
# deleted parents with many children are kept as placeholder nodes (plan 5.7); known release dates
PLACEHOLDER_MIN = 100
PLACEHOLDER_BIRTH = {"runwayml/stable-diffusion-v1-5": "2022-10-20"}
PARAM_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)(b|m)(?![a-z])", re.I)


def raw_edges(meta):
    rows = [(c, p, r) for c, t in zip(meta.model_id, meta.tags) for p, r in parse_parents(list(t))]
    return pd.DataFrame(rows, columns=["child", "parent", "rel_type"])


def clean_edges(meta, E):
    """Returns cleaned edges and the cleaning funnel (one row per step)."""
    funnel = [("parsed base_model tags", len(E), 0)]

    def step(name, keep):
        nonlocal E
        funnel.append((name, int(keep.sum()), int((~keep).sum())))
        E = E[keep].copy()

    # same (child, parent) with several relation types: keep one by priority
    E["prio"] = E.rel_type.map({r: i for i, r in enumerate(REL_PRIORITY)})
    E = E.sort_values(["child", "parent", "prio"])
    step("dedupe (child, parent) pairs", ~E.duplicated(["child", "parent"]).to_numpy())
    # id normalisation: exact, then case-insensitive
    ids = set(meta.model_id)
    low = pd.Series(meta.model_id.values, index=meta.model_id.str.lower())
    low = low[~low.index.duplicated(keep=False)]  # ambiguous lower-case ids are not used
    exact = E.parent.isin(ids)
    fixed = ~exact & E.parent.str.lower().isin(low.index)
    E.loc[fixed, "parent"] = low.loc[E.loc[fixed, "parent"].str.lower()].values
    E["match"] = np.where(exact, "exact", np.where(fixed, "lowercase", "none"))
    funnel.append(("  parent matched exactly", int(exact.sum()), 0))
    funnel.append(("  parent matched case-insensitively", int(fixed.sum()), 0))
    step("drop self loops", (E.child != E.parent).to_numpy())
    dangling = E.match == "none"
    dangling_share = float(dangling.mean())
    dangling_top = E[dangling].parent.value_counts().head(10)
    ph = placeholders(meta, E[dangling])
    keep_ph = dangling & E.parent.isin(ph.model_id)
    E.loc[keep_ph, "match"] = "placeholder"
    funnel.append((f"  dangling parents kept as placeholders (>= {PLACEHOLDER_MIN} children)", int(keep_ph.sum()), 0))
    step("drop dangling parents (not in snapshot)", ~(dangling & ~keep_ph).to_numpy())
    ca = pd.concat([meta[["model_id", "createdAt"]], ph[["model_id", "createdAt"]]]).set_index("model_id").createdAt
    tc, tp = E.child.map(ca), E.parent.map(ca)
    legacy_pair = (tc <= T0) & (tp <= T0)
    step("drop child created before parent", ~(tc < tp).to_numpy())
    tc, tp, legacy_pair = tc[E.index], tp[E.index], legacy_pair[E.index]
    step("drop equal timestamps (non-legacy)", ~((tc == tp) & ~legacy_pair).to_numpy())
    cyc = find_cycle_edges(E)
    step("drop edges on cycles", ~cyc)
    # merge children: each parent counts as 1/m of a choice (plan 5.2 spec D)
    m = E[E.rel_type == "merge"].groupby("child").size()
    E["weight"] = np.where(E.rel_type == "merge", 1.0 / E.child.map(m).fillna(1), 1.0)
    info = dict(dangling_share=dangling_share, dangling_top=dangling_top, placeholders=ph)
    return E.drop(columns=["prio"]).reset_index(drop=True), pd.DataFrame(funnel, columns=["step", "kept", "removed"]), info


def placeholders(meta, D):
    """Meta-like rows for dangling parents with >= PLACEHOLDER_MIN children. Birth: known release date,
    else 1 s before the first child; pipeline_tag / library: most common among the children."""
    n = D.parent.value_counts()
    keep = n[n >= PLACEHOLDER_MIN].index
    m = meta.set_index("model_id")
    rows = []
    for p in keep:
        kids = m.loc[D.child[D.parent == p].unique()]
        first = kids.createdAt.min()
        birth = pd.Timestamp(PLACEHOLDER_BIRTH[p], tz="UTC") if p in PLACEHOLDER_BIRTH else first - pd.Timedelta(seconds=1)
        mode = lambda s: s.mode().iloc[0] if s.notna().any() else None
        rows.append(dict(model_id=p, author=p.split("/")[0], createdAt=min(birth, first - pd.Timedelta(seconds=1)),
                         pipeline_tag=mode(kids.pipeline_tag), library_name=mode(kids.library_name),
                         licenses=[], arxiv_papers=[], tags=[], placeholder=True))
    empty = meta.iloc[:0].assign(placeholder=pd.Series(dtype=bool))
    if not rows:
        return empty
    return pd.DataFrame(rows, columns=empty.columns).astype({"createdAt": meta.createdAt.dtype})


def find_cycle_edges(E):
    """Edges lying on a directed cycle (only possible among equal-time legacy nodes)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    ids = pd.Index(pd.unique(pd.concat([E.child, E.parent])))
    c, p = ids.get_indexer(E.child), ids.get_indexer(E.parent)
    A = sp.csr_matrix((np.ones(len(c)), (c, p)), shape=(len(ids), len(ids)))
    _, lab = connected_components(A, directed=True, connection="strong")
    size = np.bincount(lab)
    return ((lab[c] == lab[p]) & (size[lab[c]] > 1))


def _license_cat(ls):
    s = " ".join(ls).lower()
    if not s:
        return "none"
    for key, cat in [("apache", "apache"), ("mit", "mit"), ("llama", "llama"), ("gemma", "gemma"),
                     ("openrail", "openrail"), ("-nc", "noncommercial"), ("cc-by", "cc_by"), ("bsd", "bsd"),
                     ("gpl", "gpl"), ("other", "other")]:
        if key in s:
            return cat
    return "other"


def _params_b(model_id):
    name = model_id.split("/", 1)[-1]
    vals = [float(v) * (1.0 if u.lower() == "b" else 1e-3) for v, u in PARAM_RE.findall(name)]
    return max(vals) if vals else np.nan


def build_nodes(meta, E):
    n = meta[["model_id", "author", "createdAt", "pipeline_tag", "library_name", "licenses", "arxiv_papers",
              "tags", "downloads", "likes", "trendingScore"]].copy()
    n["placeholder"] = meta["placeholder"].fillna(False).astype(bool).to_numpy() if "placeholder" in meta else False
    n["legacy"] = n.createdAt <= T0
    n["birth_ts"] = n.createdAt.where(~n.legacy, T0)
    n = n.sort_values(["birth_ts", "model_id"], kind="stable").reset_index(drop=True)
    n["birth"] = n.birth_ts.dt.as_unit("s").astype("int64")  # unit-safe: parquet may store us
    assert n.birth.between(1.5e9, 2.0e9).all(), "birth must be unix seconds"
    idx = pd.Series(np.arange(len(n)), index=n.model_id)
    c, p = idx[E.child].to_numpy(), idx[E.parent].to_numpy()

    # primary parent: relation priority, then earliest parent
    pe = pd.DataFrame({"c": c, "p": p, "prio": E.rel_type.map({r: i for i, r in enumerate(REL_PRIORITY)}).values})
    pe = pe.sort_values(["c", "prio", "p"]).drop_duplicates("c")
    prim = np.full(len(n), -1)
    prim[pe.c.to_numpy()] = pe.p.to_numpy()
    # root and generation by pointer doubling over primary-parent links (acyclic after cleaning)
    root = np.where(prim < 0, np.arange(len(n)), prim)
    gen = (prim >= 0).astype(np.int64)
    while True:
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        gen = gen + gen[root] * (nxt != root)
        root = nxt
    n["primary_parent"], n["root"], n["generation"] = prim, root, gen
    n["is_root"] = ~np.isin(np.arange(len(n)), c)

    # modality: own pipeline_tag -> primary parent's modality -> library hint -> unknown
    own = n.pipeline_tag.map(TAG2MOD)
    mod = own.to_numpy(object)
    lib = n.library_name.map(LIB2MOD).to_numpy(object)
    order = np.argsort(gen, kind="stable")  # parents precede children (also within legacy timestamp ties)
    for i in order[pd.isna(mod[order])]:
        pp = prim[i]
        if pp >= 0 and isinstance(mod[pp], str):
            mod[i] = mod[pp]
        elif isinstance(lib[i], str):
            mod[i] = lib[i]
        else:
            mod[i] = "unknown"
    n["modality_src"] = np.where(own.notna(), "own", np.where(np.isin(np.arange(len(n)), pe.c), "parent", "lib/unknown"))
    n["modality"] = mod
    n["modality_code"] = pd.Categorical(n.modality, categories=MODALITIES).codes

    # static features (plan 5.4, determined at birth)
    n["major_org"] = n.author.isin(MAJOR_ORGS)
    n["license_cat"] = n.licenses.map(_license_cat)
    top_lib = ["transformers", "peft", "diffusers", "sentence-transformers", "gguf", "mlx", "timm", "keras"]
    n["library_cat"] = n.library_name.where(n.library_name.isin(top_lib), np.where(n.library_name.isna(), "none", "other"))
    n["params_b"] = n.model_id.map(_params_b)
    n["has_arxiv"] = (n.arxiv_papers.map(len) > 0) | n.tags.map(lambda t: any(x.startswith("arxiv:") for x in t))
    n["author_code"] = pd.factorize(n.author)[0]
    n = n.drop(columns=["licenses", "arxiv_papers", "tags"])
    E = E.assign(child_idx=c, parent_idx=p)
    return n, E
