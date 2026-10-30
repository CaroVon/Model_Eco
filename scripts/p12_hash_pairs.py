"""Identify identical-weight copies by file hashes (extends the name-based mirror pairs of p12_mirror_pairs.py).

Candidates: every parent with >= 100 main-type children, and each of its direct children (any relation type)
that itself has >= 5 main-type children. For each repo, the LFS sha256 of its weight files (*.safetensors,
*.bin, *.pt, *.ckpt, *.pth) are fetched from the Hub (cached in external/hf_weight_hashes.json; repos that are
gone or gated without metadata are recorded as missing).
A child is an identical copy of its parent if both have weight files and the child's set of weight hashes
equals the parent's, or (sharded differently) if the child's safetensors hashes all appear in the parent's.

usage: python scripts/p12_hash_pairs.py      (network; run on the login node)
Writes results/tables/p12_hash_pairs.csv (all candidates with match class).
"""
import json
import time

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

from hfpa import paths

MAIN = ["finetune", "adapter", "unspecified"]
EXT = (".safetensors", ".bin", ".pt", ".ckpt", ".pth")
CACHE = paths.EXTERNAL / "hf_weight_hashes.json"


def fetch(api, repo, tries=5):
    for k in range(tries):
        try:
            info = api.model_info(repo, files_metadata=True)
            return {s.rfilename: s.lfs.sha256 for s in (info.siblings or []) if s.lfs and s.rfilename.endswith(EXT)}
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "404" in msg or "Repository Not Found" in msg or "401" in msg or "403" in msg:
                return None
            time.sleep(10 * 2 ** k if "429" in msg else 2)
    return None


def classify(c, p):
    if c is None or p is None:
        return "missing"
    if not c or not p:
        return "no_weights"
    hc, hp = set(c.values()), set(p.values())
    if hc == hp:
        return "identical"
    sc = {v for f, v in c.items() if f.endswith(".safetensors")}
    if sc and sc <= hp:
        return "identical"
    return "shared" if hc & hp else "different"


if __name__ == "__main__":
    n = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "placeholder"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    cnt = E[E.rel_type.isin(MAIN)].parent_idx.value_counts()
    big = cnt.index[cnt >= 100]
    C = E[E.parent_idx.isin(big) & (E.child_idx.map(cnt).fillna(0) >= 5)].copy()
    ids = n.model_id.to_numpy()
    C["parent"], C["child"] = ids[C.parent_idx], ids[C.child_idx]
    C = C[~n.placeholder.to_numpy()[C.parent_idx]]
    # also every name-based mirror pair (p12_mirror_pairs.py), so both samples carry a hash check
    mp = paths.TABLES / "p12_mirror_pairs.csv"
    if mp.exists():
        M = pd.read_csv(mp).rename(columns={"orig": "parent_idx", "cpy": "child_idx", "orig_id": "parent", "copy_id": "child"})
        M["rel_type"] = "name_pair"
        M = M[~M.set_index(["parent_idx", "child_idx"]).index.isin(C.set_index(["parent_idx", "child_idx"]).index)]
        C = pd.concat([C, M[["child_idx", "parent_idx", "rel_type", "parent", "child"]]], ignore_index=True)
    repos = sorted(set(C.parent) | set(C.child))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    api = HfApi()
    todo = [r for r in repos if r not in cache]
    print(f"{len(C)} candidate pairs, {len(repos)} repos, {len(todo)} to fetch", flush=True)
    for i, r in enumerate(todo):
        cache[r] = fetch(api, r)
        time.sleep(0.3)
        if i % 50 == 0:
            CACHE.write_text(json.dumps(cache))
            print(i, r, None if cache[r] is None else len(cache[r]), flush=True)
    CACHE.write_text(json.dumps(cache))
    C["match"] = [classify(cache.get(c), cache.get(p)) for c, p in zip(C.child, C.parent)]
    C["child_children"] = C.child_idx.map(cnt).fillna(0).to_numpy()
    C["parent_children"] = C.parent_idx.map(cnt).fillna(0).to_numpy()
    C = C.sort_values("child_children", ascending=False)
    C[["parent", "child", "rel_type", "match", "parent_children", "child_children", "parent_idx", "child_idx"]].to_csv(
        paths.TABLES / "p12_hash_pairs.csv", index=False)
    print(C.match.value_counts().to_string())
    print(pd.crosstab(C.rel_type, C.match).to_string())
