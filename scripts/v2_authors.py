"""v2.1 stage 0 (F): author types.

--fetch     for every author with at least one decision, ask the public Hub API whether the account is an
            organisation (GET /api/organizations/<name>/overview -> 200; otherwise user or deleted = not_org).
            Anonymous, read-only, only public account names are sent. Adaptive rate (starts ~4 req/s, backs off on
            429). Resumable: cache interim/v2_authors_api.parquet, appended every 500 accounts.
--classify  six types: org_major (nodes.major_org), org_redistributor (groups.redistributors), org_other (API org),
            indiv_heavy (>= 100 models), indiv_mid (10-99), indiv_light (< 10). Accounts the API no longer knows
            (deleted / renamed) count as not_org and are classified as individuals by upload count.
            Writes interim/v2_author_types.parquet (author_code -> atype) and table v2_author_types.
usage: sbatch --time=24:00:00 --cpus-per-task=2 --mem=16G slurm/cpu.sbatch scripts/v2_authors.py --fetch
"""
import argparse
import time

import numpy as np
import pandas as pd
import requests

from hfpa import paths
from hfpa.features.groups import redistributors
from hfpa.v2data import cdir, write

SPEC = "v2_pps"
CACHE = paths.INTERIM / "v2_authors_api.parquet"
TYPES = paths.INTERIM / "v2_author_types.parquet"
API = "https://huggingface.co/api"
T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def author_names():
    n = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "author_code"])
    n["name"] = n.model_id.str.split("/").str[0]
    return n.drop_duplicates("author_code").set_index("author_code")["name"]


def get(sess, url, state):
    while True:
        time.sleep(state["delay"])
        try:
            r = sess.get(url, timeout=30)
        except requests.RequestException:
            state["delay"] = min(state["delay"] * 2, 30)
            continue
        if r.status_code == 429:
            state["delay"] = min(state["delay"] * 2, 60)
            log(f"429 -> delay {state['delay']:.2f}s")
            time.sleep(30)
            continue
        if r.status_code >= 500:
            time.sleep(5)
            continue
        state["delay"] = max(0.25, state["delay"] * 0.98)
        return r.status_code


def fetch():
    names = author_names()
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet", columns=["author"])
    order = ch.author.value_counts().index  # most decisions first: early partial results cover most choices
    todo = names.reindex(order)
    done = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["author_code", "name", "api"])
    todo = todo[~todo.index.isin(done.author_code)]
    log(f"{len(todo):,} accounts to query ({len(done):,} cached)")
    sess = requests.Session()
    sess.headers["User-Agent"] = "research-metadata-client"
    state, buf = {"delay": 0.25}, []
    for i, (code, name) in enumerate(todo.items()):
        s = get(sess, f"{API}/organizations/{name}/overview", state)
        api = "org" if s == 200 else "not_org"  # users and deleted accounts: classified by upload volume
        buf.append(dict(author_code=int(code), name=name, api=api))
        if len(buf) >= 500 or i == len(todo) - 1:
            done = pd.concat([done, pd.DataFrame(buf)], ignore_index=True)
            done.to_parquet(CACHE, index=False)
            buf = []
            log(f"{len(done):,} cached; {done.api.value_counts().to_dict()}; delay {state['delay']:.2f}s")


def classify():
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "author_code", "major_org"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    nodes2 = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "birth", "author_code"])
    redis, _ = redistributors(nodes2, E)
    A = pd.read_parquet(CACHE).set_index("author_code")
    up = nodes.groupby("author_code").size()
    major = nodes.groupby("author_code").major_org.max()
    name = author_names()
    T = pd.DataFrame({"name": name.reindex(A.index), "api": A.api, "n_models": up.reindex(A.index),
                      "major": major.reindex(A.index).fillna(False).astype(bool)})
    T["redis"] = T.name.isin(redis)
    vol = np.select([T.n_models >= 100, T.n_models >= 10], ["indiv_heavy", "indiv_mid"], "indiv_light")
    T["atype"] = np.select([T.major, T.redis, T.api == "org"], ["org_major", "org_redistributor", "org_other"], vol)
    T.reset_index().to_parquet(TYPES, index=False)
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet", columns=["author", "pair_first", "dfirst24"])
    ch["atype"] = ch.author.map(T.atype)
    rows = []
    for at, g in T.groupby("atype"):
        c = ch[ch.atype == at]
        rows.append(dict(atype=at, authors=len(g), api_org=(g.api == "org").mean(), 
                         uploads=len(c), decisions=int(c.dfirst24.sum()), first=int(c.pair_first.sum())))
    R = pd.DataFrame(rows)
    for col in ("uploads", "decisions", "first"):
        R[f"share_{col}"] = R[col] / R[col].sum()
    chk = dict(major_api_org=float((T[T.major].api == "org").mean()), redis_api_org=float((T[T.redis].api == "org").mean()),
               heavy_proxy_api_org=float((T[T.n_models >= 100].api == "org").mean()),
               unclassified_choices=int(ch.atype.isna().sum()))
    write(R, "v2_author_types", "F: author types (Hub API account type + major-org list + re-distributor rule + upload count)",
          "; ".join(f"{k} = {v:.3f}" if isinstance(v, float) else f"{k} = {v}" for k, v in chk.items()))
    log(R.to_string())
    log(chk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--classify", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    if a.classify:
        classify()
