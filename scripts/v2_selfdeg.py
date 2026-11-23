"""v2.1 stage 0 (A): popularity net of the deciding author's own earlier children.

For a choice k by author a at time t and a candidate j, the replay records deg_j(t) = number of finetune / adapter /
unspecified edges into j whose child is born strictly before t, and recent30_j(t) = those with child birth in
(t - 30d, t). On a repeat decision these counts include the author's own earlier children of j, so log_deg and lrec30
mechanically absorb habit. This script writes, for every feats row (same files, same row order):

  own_prior   own earlier children of j (child author = a, birth < t)
  own_30d     the same in (t - 30d, t)
  log_deg_ex  log(max(deg - own_prior, 1));  has_deg_ex = deg - own_prior > 0;  lrec30_ex = log1p(recent30 - own_30d)

Only rows with author_reuse = 1 can be non-zero (the author's parent history is exactly the set of parents of their
earlier children), so on the first-adoption sample (plan A) the *_ex columns equal the originals by construction.
Output: processed/choices/<spec>/feats_self/<month>.parquet;  checks -> results/tables/v2/v2_selfdeg_check.md
usage: sbatch slurm/cpu.sbatch scripts/v2_selfdeg.py [--spec v2_pps]
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.features.groups import MAIN
from hfpa.v2data import cdir, write

DAY = 86400
SHIFT = np.int64(2) ** 32
T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def edge_index():
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    E = E[E.rel_type.isin(MAIN)]
    birth = nodes.birth.to_numpy(np.int64)
    auth = nodes.author_code.to_numpy(np.int64)
    N = np.int64(len(nodes))
    key = auth[E.child_idx.to_numpy()] * N + E.parent_idx.to_numpy(np.int64)
    te = birth[E.child_idx.to_numpy()]
    ukey, kid = np.unique(key, return_inverse=True)
    comb = np.sort(kid.astype(np.int64) * SHIFT + te)
    return ukey, comb, N, auth, birth


def own_counts(ukey, comb, N, a, j, t):
    """number of (author a -> parent j) edges with child birth < t and in (t - 30d, t)."""
    key = a * N + j
    pos = np.searchsorted(ukey, key)
    pos = np.minimum(pos, len(ukey) - 1)
    hit = ukey[pos] == key
    base = pos.astype(np.int64) * SHIFT
    lo = np.searchsorted(comb, base, "left")
    below_t = np.searchsorted(comb, base + t, "left")          # birth < t
    below_30 = np.searchsorted(comb, base + t - 30 * DAY, "right")  # birth <= t - 30d
    prior = np.where(hit, below_t - lo, 0)
    d30 = np.where(hit, below_t - np.maximum(below_30, lo), 0)
    return prior.astype(np.int32), d30.astype(np.int32)


def main(spec, sub="feats"):
    ukey, comb, N, auth, birth = edge_index()
    log(f"edge index: {len(ukey):,} author-parent pairs, {len(comb):,} edges")
    ch = pd.read_parquet(cdir(spec) / "decisions.parquet", columns=["choice_id", "author", "t"]).set_index("choice_id")
    out = cdir(spec) / {"feats": "feats_self", "feats_up": "feats_self_up"}[sub]
    out.mkdir(exist_ok=True)
    chk = []
    for f in sorted(glob.glob(str(cdir(spec) / sub / "*.parquet"))):
        if f.endswith(".notext.parquet"):
            continue
        if not __import__("pyarrow.parquet").parquet.read_schema(f).names:
            continue
        x = pd.read_parquet(f, columns=["choice_id", "cand_id", "deg", "lrec30", "author_reuse"])
        if len(x) == 0:
            continue
        a = ch.author.reindex(x.choice_id).to_numpy(np.int64)
        t = ch.t.reindex(x.choice_id).to_numpy(np.int64)
        prior = np.zeros(len(x), np.int32)
        d30 = np.zeros(len(x), np.int32)
        r = np.flatnonzero(x.author_reuse.to_numpy() == 1)
        if len(r):
            prior[r], d30[r] = own_counts(ukey, comb, N, a[r], x.cand_id.to_numpy(np.int64)[r], t[r])
        deg = x.deg.to_numpy(np.int64)
        rec30 = np.rint(np.expm1(x.lrec30.to_numpy(np.float64))).astype(np.int64)
        dex, rex = deg - prior, rec30 - d30
        nonhist = np.flatnonzero(x.author_reuse.to_numpy() == 0)
        # sanity: on non-history rows the author has no earlier child of j
        p0, _ = own_counts(ukey, comb, N, a[nonhist[:20000]], x.cand_id.to_numpy(np.int64)[nonhist[:20000]],
                           t[nonhist[:20000]]) if len(nonhist) else (np.zeros(0), None)
        chk.append(dict(file=os.path.basename(f), rows=len(x), hist_rows=len(r), mean_own_prior_hist=prior[r].mean()
                        if len(r) else np.nan, neg_deg=int((dex < 0).sum()), neg_rec30=int((rex < 0).sum()),
                        nonhist_with_own=int((p0 > 0).sum())))
        pd.DataFrame({"choice_id": x.choice_id.to_numpy(), "cand_id": x.cand_id.to_numpy(), "own_prior": prior,
                      "own_30d": d30, "log_deg_ex": np.log(np.maximum(dex, 1)).astype(np.float32),
                      "has_deg_ex": (dex > 0).astype(np.float32),
                      "lrec30_ex": np.log1p(np.maximum(rex, 0)).astype(np.float32)}).to_parquet(
            out / os.path.basename(f), index=False)
        log(f"{chk[-1]}")
    C = pd.DataFrame(chk)
    tot = C[["rows", "hist_rows", "neg_deg", "neg_rec30", "nonhist_with_own"]].sum()
    C = pd.concat([C, pd.DataFrame([dict(file="TOTAL", **tot.to_dict())])], ignore_index=True)
    write(C, "v2_selfdeg_check" + ("_up" if sub == "feats_up" else ""), "A: own earlier children removed from degree / recent30 (per feats month)",
          "neg_deg / neg_rec30 must be 0 (own counts cannot exceed the replay counts); nonhist_with_own must be 0 "
          "(the author's history equals the parents of their earlier children).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v2_pps")
    ap.add_argument("--sub", default="feats", choices=["feats", "feats_up"])
    a = ap.parse_args()
    main(a.spec, a.sub)
