"""P24: what developers saw on the Hub at choice time -- likes, 30-day downloads and gating of hub models, read
from archived model pages (Internet Archive Wayback Machine). Robustness only: coverage is good for high-degree
parents and ~0 for the long tail (feasibility check 2026-11-24, plan in reports/).

--targets   target models: the TOP_CHOSEN parents chosen most often in the plan-A sample + the 500 FE hubs.
            Aliases (old names that now redirect, e.g. bert-base-uncased -> google-bert/bert-base-uncased;
            meta-llama/Meta-Llama-3.1-8B -> meta-llama/Llama-3.1-8B): candidate old names are checked against the
            Hub API (GET /api/models/<alias> must return the target id). -> interim/wb_targets.parquet
--cdx       one CDX query per name (target + aliases): status-200 captures up to the snapshot date, one per month
            (collapse=timestamp:6). -> interim/wb_cdx.parquet
--fetch     every listed capture (raw `id_` page), parsed by hfpa.data.wayback.parse_page.
            -> interim/wb_panel.parquet (model_id, ts, likes, downloads30, gated, ...)
--summary   coverage tables: results/tables/v2/p24_wb_coverage.md
Network: the Internet Archive refuses connections after bursts (seen after ~100 CDX queries at 1/s), so all
requests are sequential, >= DELAY seconds apart, with long back-off on refusals; every stage is resumable.
Only public model ids are sent; generic User-Agent.
usage: sbatch --time=12:00:00 --cpus-per-task=1 --mem=8G slurm/cpu.sbatch scripts/p24_wayback.py --targets --cdx --fetch
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import requests

from hfpa import paths
from hfpa.data.wayback import parse_page
from hfpa.v2data import cdir, write

SPEC = "v2_pps"
TOP_CHOSEN = 2000
SNAP = "20250712"
DELAY = 4.0
TARGETS = paths.INTERIM / "wb_targets.parquet"
CDX = paths.INTERIM / "wb_cdx.parquet"
PANEL = paths.INTERIM / "wb_panel.parquet"
UA = {"User-Agent": "research-metadata-client"}
T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def targets():
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "birth", "placeholder"])
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet", columns=["parent", "pair_first"])
    cnt = ch[ch.pair_first].parent.value_counts()
    top = cnt.index[:TOP_CHOSEN].to_numpy()
    hubs = set()
    import glob
    for f in sorted(glob.glob(str(cdir(SPEC) / "feats" / "*.parquet"))):
        if f.endswith(".notext.parquet"):
            continue
        x = pd.read_parquet(f, columns=["cand_id", "fe"])
        hubs.update(x.cand_id[x.fe > 0].unique().tolist())
    idx = np.union1d(top, np.array(sorted(hubs), dtype=np.int64))
    T = pd.DataFrame({"node": idx, "model_id": nodes.model_id.to_numpy()[idx],
                      "birth": nodes.birth.to_numpy()[idx], "placeholder": nodes.placeholder.to_numpy()[idx],
                      "n_first": cnt.reindex(idx).fillna(0).astype(int).to_numpy(), "fe_hub": np.isin(idx, list(hubs))})
    log(f"targets {len(T):,} (top {TOP_CHOSEN} chosen + {len(hubs)} FE hubs); share of plan-A first choices covered "
        f"{T.n_first.sum() / cnt.sum():.3f}")
    # aliases
    sess = requests.Session()
    sess.headers.update(UA)
    al = []
    for i, m in enumerate(T.model_id):
        org, _, repo = m.partition("/")
        cands = {repo} if repo else set()
        if org == "meta-llama" and not repo.startswith("Meta-"):
            cands.add(f"meta-llama/Meta-{repo}")
        found = []
        for a in sorted(cands):
            for k in range(5):
                try:
                    r = sess.get(f"https://huggingface.co/api/models/{a}", timeout=30)
                except requests.RequestException:
                    time.sleep(10)
                    continue
                if r.status_code == 429:
                    time.sleep(60)
                    continue
                break
            else:
                continue
            if r.status_code == 200 and r.json().get("id", "").lower() == m.lower() and a.lower() != m.lower():
                found.append(a)
            time.sleep(0.3)
        al.append(json.dumps(found))
        if i % 200 == 0:
            log(f"aliases {i:,}/{len(T):,}")
    T["aliases"] = al
    T.to_parquet(TARGETS, index=False)
    n_al = (T.aliases != "[]").sum()
    log(f"targets with an alias: {n_al}: {T[T.aliases != '[]'].head(20)[['model_id', 'aliases']].to_dict('records')}")


class IA:
    """Sequential, paced client for web.archive.org with back-off on refusals / 429 / 5xx."""

    def __init__(self, delay=DELAY):
        self.s = requests.Session()
        self.s.headers.update(UA)
        self.delay, self.last = delay, 0.0

    def get(self, url, params=None, timeout=120):
        wait = 60
        for attempt in range(12):
            dt = time.time() - self.last
            if dt < self.delay:
                time.sleep(self.delay - dt)
            self.last = time.time()
            try:
                r = self.s.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                log(f"connection problem ({type(e).__name__}); sleeping {wait}s")
                time.sleep(wait)
                wait = min(wait * 2, 1800)
                continue
            if r.status_code in (429, 503) or r.status_code >= 520:
                log(f"HTTP {r.status_code}; sleeping {wait}s")
                time.sleep(wait)
                wait = min(wait * 2, 1800)
                continue
            return r
        return None


def cdx():
    T = pd.read_parquet(TARGETS)
    done = pd.read_parquet(CDX) if CDX.exists() else pd.DataFrame(columns=["model_id", "name", "ts", "original", "ok"])
    seen = set(zip(done.model_id, done.name))
    todo = [(m, n) for m, a in zip(T.model_id, T.aliases) for n in [m] + json.loads(a) if (m, n) not in seen]
    log(f"CDX: {len(todo):,} names to query ({len(seen):,} done)")
    ia, buf = IA(), []
    for i, (m, n) in enumerate(todo):
        r = ia.get("https://web.archive.org/cdx/search/cdx",
                   params=dict(url=f"huggingface.co/{n}", fl="timestamp,original", filter="statuscode:200",
                               to=SNAP, collapse="timestamp:6"))
        if r is None or r.status_code != 200:
            log(f"CDX failed for {n}: {None if r is None else r.status_code}")
            continue  # not cached -> retried on the next run
        lines = [ln.split(" ", 1) for ln in r.text.split("\n") if ln.strip()]
        if lines:
            buf += [dict(model_id=m, name=n, ts=t, original=o, ok=True) for t, o in lines]
        else:
            buf.append(dict(model_id=m, name=n, ts="", original="", ok=False))
        if len(buf) >= 200 or i == len(todo) - 1:
            done = pd.concat([done, pd.DataFrame(buf)], ignore_index=True)
            done.to_parquet(CDX, index=False)
            buf = []
            log(f"CDX {i + 1:,}/{len(todo):,}; captures {int(done.ok.sum()):,}")
    if buf:
        pd.concat([done, pd.DataFrame(buf)], ignore_index=True).to_parquet(CDX, index=False)


def fetch():
    T = pd.read_parquet(TARGETS).set_index("model_id")
    C = pd.read_parquet(CDX)
    C = C[C.ok].drop_duplicates(["model_id", "ts"])
    cols = ["model_id", "name", "ts", "http", "found_id", "likes", "downloads30", "gated", "last_modified", "fmt"]
    done = pd.read_parquet(PANEL) if PANEL.exists() else pd.DataFrame(columns=cols)
    seen = set(zip(done.model_id, done.ts))
    # most-chosen models first: partial runs are already useful
    C["n_first"] = T.n_first.reindex(C.model_id).to_numpy()
    C = C[[(m, t) not in seen for m, t in zip(C.model_id, C.ts)]].sort_values(["n_first", "ts"], ascending=[False, True])
    log(f"fetch: {len(C):,} captures to fetch ({len(done):,} done)")
    ia, buf = IA(), []
    for i, (m, n, ts, orig) in enumerate(zip(C.model_id, C.name, C.ts, C.original)):
        r = ia.get(f"https://web.archive.org/web/{ts}id_/{orig}")
        if r is None:
            continue
        rec = dict(model_id=m, name=n, ts=ts, http=r.status_code)
        if r.status_code == 200:
            rec.update(parse_page(r.text, [m] + json.loads(T.aliases[m])))
        buf.append(rec)
        if len(buf) >= 200 or i == len(C) - 1:
            done = pd.concat([done, pd.DataFrame(buf, columns=cols)], ignore_index=True)
            done.to_parquet(PANEL, index=False)
            buf = []
            ok = done.likes.notna()
            log(f"fetch {i + 1:,}/{len(C):,}; parsed likes {ok.mean():.3f}, downloads {done.downloads30.notna().mean():.3f}")
    if buf:
        pd.concat([done, pd.DataFrame(buf, columns=cols)], ignore_index=True).to_parquet(PANEL, index=False)


def summary():
    T = pd.read_parquet(TARGETS)
    C = pd.read_parquet(CDX)
    P = pd.read_parquet(PANEL) if PANEL.exists() else pd.DataFrame(columns=["model_id", "likes"])
    T["bin"] = pd.cut(T.n_first, [-1, 9, 49, 199, 999, 10**9], labels=["0-9", "10-49", "50-199", "200-999", ">=1000"])
    cap = C[C.ok].groupby("model_id").ts.nunique()
    par = P[P.likes.notna()].groupby("model_id").ts.nunique()
    T["months_cdx"] = cap.reindex(T.model_id).fillna(0).to_numpy()
    T["months_parsed"] = par.reindex(T.model_id).fillna(0).to_numpy()
    R = T.groupby("bin", observed=True).agg(models=("model_id", "size"), first_choices=("n_first", "sum"),
                                            any_capture=("months_cdx", lambda x: (x > 0).mean()),
                                            median_months=("months_cdx", "median"),
                                            any_parsed=("months_parsed", lambda x: (x > 0).mean())).reset_index()
    fm = P.fmt.value_counts().to_dict() if len(P) else {}
    write(R, "p24_wb_coverage", "P24 Wayback coverage of target models (by plan-A first choices)",
          f"targets {len(T):,}; captures listed {int(C.ok.sum()):,}; pages fetched {len(P):,}; parse format {fm}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for p in ("targets", "cdx", "fetch", "summary"):
        ap.add_argument(f"--{p}", action="store_true")
    a = ap.parse_args()
    if a.targets:
        targets()
    if a.cdx:
        cdx()
    if a.fetch:
        fetch()
    if a.summary:
        summary()
