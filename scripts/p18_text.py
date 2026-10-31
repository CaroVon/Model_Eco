"""P18: model-card text as a demand signal (hfpa.features.text).

--extract  read model_id + modelCard from the raw CSV for the nodes needed (children of all main choices,
           candidates of the 10% sample, parents of all main choices); clean to paragraphs; write
           interim/cards_clean.parquet (node, paras)
--build    child documents: own paragraphs minus templates, i.e. paragraphs that appear in >= MAX_PARA_DF cards
           or in a parent-side card (candidate or parent) created no later than the child, the child itself
           excluded; candidate documents: own paragraphs minus the frequent ones. Name tokens (orgs, families of
           models with >= NAME_MIN_DEG children; every name token of the N_HUBS largest parents; size / variant words)
           and tokens with digits removed on both sides. TF-IDF cosine.
           Writes processed/choices/<spec>/text.parquet (aligned with p15_rows.parquet),
           processed/choices/main_strat/text_chosen.parquet (child-chosen parent, all main choices) and
           results/tables/p18_text.md (coverage, chosen vs other similarity).

usage: python scripts/p18_text.py --extract   (Slurm: reads the 5 GB CSV)
       python scripts/p18_text.py --build
"""
import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.data.load_meta import CSV
from hfpa.features.text import MAX_PARA_DF, MIN_TOKENS, clean, drop_templates, name_vocab, para_key, row_cosine, tfidf, tokens

SPEC = "main_strat_frac0.1"
NAME_MIN_DEG = 20
N_HUBS = 500  # all repo-name tokens of these are removed (deg.index is sorted by degree)
MAIN = ["finetune", "adapter", "unspecified"]


def needed():
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent"])
    cand = pd.read_parquet(paths.PROCESSED / "choices" / SPEC / "p15_rows.parquet", columns=["cand_id"]).cand_id.unique()
    child = full.child.unique()
    parent = np.union1d(full.parent.unique(), cand)
    return child, parent


def extract(chunksize=50_000):
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id"])
    idx = pd.Series(np.arange(len(nodes)), index=nodes.model_id)
    child, parent = needed()
    want = set(nodes.model_id.to_numpy()[np.union1d(child, parent)])
    out, n = [], 0
    for ch in pd.read_csv(CSV, usecols=["model_id", "modelCard"], chunksize=chunksize, dtype=str,
                          keep_default_na=False, na_values=[""]):
        n += len(ch)
        sub = ch[ch.model_id.isin(want)]
        for mid, card in zip(sub.model_id, sub.modelCard):
            out.append((int(idx[mid]), clean(card)))
        print(f"{n:,} rows read, {len(out):,} cards kept", flush=True)
    pd.DataFrame(out, columns=["node", "paras"]).drop_duplicates("node").to_parquet(paths.INTERIM / "cards_clean.parquet", index=False)


def build():
    t0 = time.time()
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "birth"])
    birth = nodes.birth.to_numpy()
    C = pd.read_parquet(paths.INTERIM / "cards_clean.parquet")
    paras = dict(zip(C.node.to_numpy(), C.paras))
    child, parent = needed()
    pside = np.isin(C.node.to_numpy(), parent)
    print(f"{len(C):,} cards; parent side {pside.sum():,}; {time.time() - t0:.0f}s", flush=True)

    # paragraph keys: overall document frequency; earliest two parent-side births (to exclude the child itself)
    keys = {v: [k for k in map(para_key, ps) if k is not None] for v, ps in paras.items()}
    df = Counter(k for ks in keys.values() for k in set(ks))
    frequent = {k for k, c in df.items() if c >= MAX_PARA_DF}
    first = {}  # key -> [(birth, node), (birth, node)] two earliest parent-side cards
    for v in C.node.to_numpy()[pside]:
        b = birth[v]
        for k in set(keys[v]):
            f = first.get(k)
            if f is None:
                first[k] = [(b, v)]
            elif len(f) == 1 or b < f[1][0]:
                f.append((b, v))
                f.sort()
                del f[2:]
    print(f"paragraph keys {len(df):,}, frequent {len(frequent):,}; {time.time() - t0:.0f}s", flush=True)

    def copied(k, v):
        f = first.get(k)
        if not f:
            return False
        other = [b for b, u in f if u != v]
        return bool(other) and other[0] <= birth[v]

    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["parent_idx", "rel_type"])
    deg = E[E.rel_type.isin(MAIN)].parent_idx.value_counts()
    names = name_vocab(nodes.model_id.to_numpy()[deg.index[deg >= NAME_MIN_DEG]],
                       full_ids=nodes.model_id.to_numpy()[deg.index[:N_HUBS]])
    print(f"name tokens {len(names):,}", flush=True)

    child_nodes = np.intersect1d(child, C.node.to_numpy())
    cand_nodes = C.node.to_numpy()[pside]
    def child_doc(v):
        return tokens([p for p, k in zip(paras[v], map(para_key, paras[v])) if k is None or (k not in frequent and not copied(k, v))], names)
    docs = [child_doc(v) for v in child_nodes] + [tokens(drop_templates(paras[v], frequent), names) for v in cand_nodes]
    ntok = np.array([len(d) for d in docs])
    print(f"documents built; {time.time() - t0:.0f}s", flush=True)
    X, vocab = tfidf(docs)
    del docs
    print(f"tf-idf {X.shape}, nnz {X.nnz:,}; {time.time() - t0:.0f}s", flush=True)
    crow = np.full(len(nodes), -1, np.int64)
    crow[child_nodes] = np.arange(len(child_nodes))
    prow = np.full(len(nodes), -1, np.int64)
    prow[cand_nodes] = len(child_nodes) + np.arange(len(cand_nodes))
    ok_doc = ntok >= MIN_TOKENS

    def sims(kid, cand):
        a, b = crow[kid], prow[cand]
        ok = (a >= 0) & (b >= 0)
        ok[ok] = ok_doc[a[ok]] & ok_doc[b[ok]]
        s = np.zeros(len(kid), np.float32)
        s[ok] = row_cosine(X, a[ok], b[ok])
        has_child = np.zeros(len(kid), bool)
        has_child[a >= 0] = ok_doc[a[a >= 0]]
        return s, ~ok, has_child

    d = paths.PROCESSED / "choices" / SPEC
    R = pd.read_parquet(d / "p15_rows.parquet", columns=["choice_id", "cand_id", "chosen"])
    ch = pd.read_parquet(d / "choices.parquet").set_index("choice_id")
    kid = ch.child.loc[R.choice_id].to_numpy()
    s, na, hc = sims(kid, R.cand_id.to_numpy())
    T = pd.DataFrame({"choice_id": R.choice_id.to_numpy(), "cand_id": R.cand_id.to_numpy(), "txt_sim": s,
                      "txt_na": na.astype(np.float32), "txt_has": hc.astype(np.float32)})
    T.to_parquet(d / "text.parquet", index=False)
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["choice_id", "child", "parent"])
    s2, na2, hc2 = sims(full.child.to_numpy(), full.parent.to_numpy())
    pd.DataFrame({"choice_id": full.choice_id.to_numpy(), "txt_sim": s2, "txt_na": na2.astype(np.float32),
                  "txt_has": hc2.astype(np.float32)}).to_parquet(paths.PROCESSED / "choices" / "main_strat" / "text_chosen.parquet", index=False)

    # description: coverage and chosen vs other similarity (plan-A first choices of the 10% sample)
    first = pd.read_parquet(d / "p15_first.parquet")["first"]
    T["first"] = first.reindex(T.choice_id).to_numpy()
    T["chosen"] = R.chosen.to_numpy()
    out = []
    for smp, sub in [("all choices", T), ("first author-parent choices", T[T["first"]])]:
        per = sub.groupby("choice_id").agg(has=("txt_has", "max"))
        ok = sub[sub.txt_na == 0]
        rank = ok.assign(r=ok.groupby("choice_id").txt_sim.rank(ascending=False, method="average"),
                         n=ok.groupby("choice_id").txt_sim.transform("size"))
        cr = rank[rank.chosen == 1]
        out.append(dict(sample=smp, choices=len(per), share_child_text=per.has.mean(),
                        share_rows_with_both_texts=(sub.txt_na == 0).mean(),
                        sim_chosen=ok.loc[ok.chosen == 1, "txt_sim"].mean(), sim_other=ok.loc[ok.chosen == 0, "txt_sim"].mean(),
                        chosen_pct_rank=(1 - (cr.r - 1) / cr.n.clip(lower=1)).mean()))
    res = pd.DataFrame(out)
    tok = pd.Series(ntok[: len(child_nodes)])
    md = ["# P18 model-card text similarity (child's own text vs candidate card)", "",
          f"Cards kept: {len(C):,}; child documents: {len(child_nodes):,} (median {tok.median():.0f} tokens after "
          f"removing templates and names; {(tok >= MIN_TOKENS).mean():.1%} with >= {MIN_TOKENS} tokens); "
          f"name tokens removed: {len(names):,}; vocabulary {len(vocab):,}.", "",
          "chosen_pct_rank: percentile of the chosen parent among the sampled candidates with text (1 = most similar; "
          "0.5 = random; unweighted by log q).", "", res.to_markdown(index=False, floatfmt=".3f")]
    (paths.TABLES / "p18_text.md").write_text("\n".join(md) + "\n")
    res.to_csv(paths.TABLES / "p18_text.csv", index=False)
    print("\n".join(md), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.build:
        build()
