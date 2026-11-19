"""v2 card-text demand signal (fix F of the v2 plan), for every node.

--extract  model_id + modelCard of ALL nodes from the raw CSV (the v2 candidate sets touch almost every node);
           cleaned paragraphs -> interim/v2_cards_clean.parquet (node, paras)
--build    paragraph rules (all cards are "parent side" now):
             exact     key in >= MAX_PARA_DF cards (frequent template)          -> dropped from every document
             near-dup  5-word-shingle MinHash (63 hashes, LSH 9 x 7, verified Jaccard >= 0.8): a key whose near-
                       duplicate cluster covers >= MAX_PARA_DF cards           -> dropped from every document
             copied    (child documents only) the key or a near-duplicate appears in another card born no later than
                       the child                                              -> dropped from the child document
           names: organisation and family names only (hfpa.features.text.name_vocab_v2; descriptive words kept),
           size / variant words and tokens with digits. TF-IDF over child + candidate documents.
           Writes interim/v2_text_tfidf.npz, interim/v2_text_rows.npz (crow, prow, ntok_child, ntok_cand per node) and
           results/tables/v2/text_build.md. Similarities are computed per candidate row in v2_features.py; the minimum
           token count is applied there (txt_ntok = min of both sides; main threshold MIN_TOKENS = 5).
usage: sbatch slurm/cpu.sbatch scripts/v2_text.py --extract ; sbatch slurm/cpu.sbatch scripts/v2_text.py --build
"""
import argparse
import os
import re
from pathlib import Path
import time
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from hfpa import paths
from hfpa.data.load_meta import CSV
from hfpa.features.text import (ID, MAX_PARA_DF, clean, lsh_near_dup, minhash, name_vocab_v2, para_key, shingle_hashes,
                                tfidf, tokens)

MAIN = ["finetune", "adapter", "unspecified"]
NAME_MIN_DEG = 20
OUT = paths.TABLES / "v2"
WORD = re.compile(r"[a-z]+")


def extract(chunksize=50_000):
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id"])
    idx = pd.Series(np.arange(len(nodes)), index=nodes.model_id)
    out, n = [], 0
    for ch in pd.read_csv(CSV, usecols=["model_id", "modelCard"], chunksize=chunksize, dtype=str,
                          keep_default_na=False, na_values=[""]):
        n += len(ch)
        sub = ch[ch.model_id.isin(idx.index) & ch.modelCard.notna()]
        for mid, card in zip(sub.model_id, sub.modelCard):
            ps = clean(card)
            if ps:
                out.append((int(idx[mid]), ps))
        print(f"{n:,} rows read, {len(out):,} cards kept", flush=True)
    pd.DataFrame(out, columns=["node", "paras"]).drop_duplicates("node").to_parquet(
        paths.INTERIM / "v2_cards_clean.parquet", index=False)


def build():
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "birth"])
    birth = nodes.birth.to_numpy().astype(np.float64)
    C = pd.read_parquet(paths.INTERIM / "v2_cards_clean.parquet")
    sub = int(os.environ.get("V2_TEXT_SUB", 0))  # development: random subset of cards, outputs to V2_TEXT_OUT
    if sub:
        C = C.sample(sub, random_state=0)
    cnode = C.node.to_numpy()
    paras = [list(p) for p in C.paras]
    log(f"{len(C):,} cards")
    # ---- paragraph keys: df, earliest two cards, one representative text
    keys = [[para_key(p) for p in ps] for ps in paras]
    kid, df, first, rep = {}, [], [], []
    for v, ps, ks in zip(cnode, paras, keys):
        b = birth[v]
        for p, k in zip(ps, ks):
            if k is None:
                continue
            i = kid.get(k)
            if i is None:
                kid[k] = len(df)
                df.append(0)
                first.append([(b, v)])
                rep.append(p)
                i = kid[k]
            f = first[i]
            if all(u != v for _, u in f):
                if len(f) < 2 or b < f[1][0]:
                    f.append((b, v))
                    f.sort()
                    del f[2:]
        for k in set(k for k in ks if k is not None):
            df[kid[k]] += 1
    df = np.array(df)
    K = len(df)
    log(f"unique paragraph keys {K:,}; frequent (df >= {MAX_PARA_DF}) {(df >= MAX_PARA_DF).sum():,}")
    # ---- near duplicates among non-frequent keys
    cand = np.flatnonzero(df < MAX_PARA_DF)
    vocab = {}
    ids, off = [], [0]
    for i in cand:
        w = [vocab.setdefault(x, len(vocab)) for x in WORD.findall(ID.sub(" ", rep[i].lower()))]
        ids.extend(w)
        off.append(len(ids))
    del rep
    h, own = shingle_hashes(np.array(ids, np.int64), np.array(off, np.int64))
    del ids
    log(f"shingles {len(h):,} over {len(cand):,} keys")
    sig = minhash(h, own, len(cand))
    del h, own
    log("minhash done")
    fb = np.array([first[i][0][0] for i in cand])
    fnode = np.array([first[i][0][1] for i in cand])
    fo, fn, cdf = lsh_near_dup(sig, fb, fnode, df[cand])
    del sig
    near_first = np.full(K, np.inf)
    near_node = np.full(K, -1, np.int64)
    cluster_df = df.copy()
    near_first[cand], near_node[cand], cluster_df[cand] = fo, fn, cdf
    log(f"near-dup: keys with an earlier near-duplicate elsewhere {np.isfinite(fo).sum():,}; "
        f"keys in near-dup template clusters (>= {MAX_PARA_DF} cards) {(cdf >= MAX_PARA_DF).sum():,}")
    banned = cluster_df >= MAX_PARA_DF

    def copied(i, v):
        f = [b for b, u in first[i] if u != v]
        if f and f[0] <= birth[v]:
            return True
        return near_first[i] <= birth[v] and near_node[i] != v

    # ---- documents: child documents (children of main choices) and candidate documents (every card)
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    me = E[E.rel_type.isin(MAIN)]
    child_set = set(me.child_idx.unique().tolist())
    deg = me.parent_idx.value_counts()
    names = name_vocab_v2(nodes.model_id.to_numpy()[deg.index[deg >= NAME_MIN_DEG]])
    log(f"name tokens {len(names):,}")
    child_docs, child_nodes, cand_docs = [], [], []
    stats = Counter()
    for v, ps, ks in zip(cnode, paras, keys):
        kk = [kid.get(k) if k is not None else None for k in ks]
        cand_docs.append(tokens([p for p, i in zip(ps, kk) if i is None or not banned[i]], names))
        if v in child_set:
            keep = []
            for p, i in zip(ps, kk):
                if i is None:
                    keep.append(p)
                elif banned[i]:
                    stats["child_template"] += 1
                elif copied(i, v):
                    stats["child_copied"] += 1
                else:
                    keep.append(p)
            stats["child_kept"] += len(keep)
            child_docs.append(tokens(keep, names))
            child_nodes.append(v)
    del paras, keys
    log(f"documents: child {len(child_docs):,}, candidate {len(cand_docs):,}; {dict(stats)}")
    ntok_c = np.array([len(d) for d in child_docs])
    ntok_p = np.array([len(d) for d in cand_docs])
    X, voc = tfidf(child_docs + cand_docs)
    log(f"tf-idf {X.shape}, nnz {X.nnz:,}")
    N = len(nodes)
    crow = np.full(N, -1, np.int64)
    crow[np.array(child_nodes)] = np.arange(len(child_nodes))
    prow = np.full(N, -1, np.int64)
    prow[cnode] = len(child_nodes) + np.arange(len(cnode))
    nt_child = np.zeros(N, np.int32)
    nt_child[np.array(child_nodes)] = ntok_c
    nt_cand = np.zeros(N, np.int32)
    nt_cand[cnode] = ntok_p
    dest = Path(os.environ.get("V2_TEXT_OUT", paths.INTERIM))
    sp.save_npz(dest / "v2_text_tfidf.npz", X)
    np.savez(dest / "v2_text_rows.npz", crow=crow, prow=prow, ntok_child=nt_child, ntok_cand=nt_cand)
    tc = pd.Series(ntok_c)
    md = ["# v2 card text build", "",
          f"cards {len(C):,}; unique paragraph keys {K:,}; frequent {int((df >= MAX_PARA_DF).sum()):,}; banned incl. "
          f"near-duplicate clusters {int(banned.sum()):,}; child paragraphs dropped as template {stats['child_template']:,}, "
          f"as copied from an earlier card {stats['child_copied']:,}, kept {stats['child_kept']:,}.",
          f"name tokens removed {len(names):,}; vocabulary {len(voc):,}.",
          "", "child documents by token count (after removal):", "",
          pd.DataFrame({"min_tokens": [1, 5, 10, 20, 50],
                        "share_of_child_cards": [(tc >= k).mean() for k in (1, 5, 10, 20, 50)]}).to_markdown(index=False)]
    (dest / "text_build.md" if sub else OUT / "text_build.md").write_text("\n".join(md) + "\n")
    log("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.build:
        build()
