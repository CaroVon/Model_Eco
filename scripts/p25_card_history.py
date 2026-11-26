"""P25: the model card each target model showed at choice time (git history of the repo), for the card-text match
(txt_sim, P18 / v2_text) and model-index quality (bench_z, P14), which otherwise use the 2025-07 snapshot card.

Targets: interim/wb_targets.parquet (scripts/p24_wayback.py --targets).
--commits   GET /api/models/<id>/commits/main (all pages) -> interim/card_commits.parquet (model_id, sha, date).
            Gated repos answer 401 to anonymous requests (403 with a token whose account has not accepted the terms) and
            deleted repos 404: they keep the snapshot card
            (hist_src = snapshot) -- the share is reported.
--readme    for every month M of the choice window in which the model existed: the last commit dated before the
            first day of M (the version visible during the whole month); README.md at each distinct sha
            (huggingface.co/<id>/raw/<sha>/README.md) -> interim/card_hist.parquet (model_id, sha, http, text).
            interim/card_month.parquet maps (model_id, month) -> sha.
--prep      cleaned paragraphs of every distinct historical version (hfpa.features.text.clean), in the layout of
            interim/v2_cards_clean.parquet, for `scripts/v2_text.py --build` with V2_TEXT_EXTRA (extra candidate
            documents; same template / name rules and TF-IDF vocabulary; outputs to V2_TEXT_OUT).
            Also historical bench_z: P14's parse_card on each version, standardised with the snapshot means / sds.
            -> interim/card_versions.parquet (vid, model_id, sha, bench_z)
Anonymous, read-only; Hub API limit 500 requests / 5 min -> >= 0.7 s between requests, back-off on 429.
usage: sbatch --time=12:00:00 --cpus-per-task=1 --mem=16G slurm/cpu.sbatch scripts/p25_card_history.py --commits --readme --prep
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from hfpa import paths

TARGETS = paths.INTERIM / "wb_targets.parquet"
COMMITS = paths.INTERIM / "card_commits.parquet"
HIST = paths.INTERIM / "card_hist.parquet"
MONTH = paths.INTERIM / "card_month.parquet"
VERS = paths.INTERIM / "card_versions.parquet"
EXTRA = paths.INTERIM / "card_hist_clean.parquet"
WINDOW = ("2022-03", "2025-07")
DELAY = 0.7
UA = {"User-Agent": "research-metadata-client"}
T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


TOKEN_FILE = Path.home() / ".config" / "acmec" / "hf_token"  # read-only token, outside the repo; never logged


def token():
    t = os.environ.get("HF_TOKEN")
    if not t and TOKEN_FILE.exists():
        t = TOKEN_FILE.read_text().strip()
    return t or None


class Hub:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(UA)
        if token():  # gated repos: readable only if the token's account accepted the terms (else 403)
            self.s.headers["Authorization"] = f"Bearer {token()}"
        self.last = 0.0

    def get(self, url):
        wait = 30
        for _ in range(10):
            dt = time.time() - self.last
            if dt < DELAY:
                time.sleep(DELAY - dt)
            self.last = time.time()
            try:
                r = self.s.get(url, timeout=60)
            except requests.RequestException:
                time.sleep(wait)
                wait = min(wait * 2, 600)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                log(f"HTTP {r.status_code}; sleeping {wait}s")
                time.sleep(wait)
                wait = min(wait * 2, 600)
                continue
            return r
        return None


def wait_targets():
    while not TARGETS.exists():
        log("waiting for interim/wb_targets.parquet (p24_wayback.py --targets)")
        time.sleep(300)
    return pd.read_parquet(TARGETS)


def commits():
    T = wait_targets()
    T = T[~T.placeholder.astype(bool)]
    done = pd.read_parquet(COMMITS) if COMMITS.exists() else pd.DataFrame(columns=["model_id", "sha", "date", "http"])
    if token():  # retry repos that refused access (anonymous 401, or 403 before the terms were accepted)
        done = done[~done.http.isin([401, 403])]
    todo = [m for m in T.model_id if m not in set(done.model_id)]
    log(f"commits: {len(todo):,} repos ({done.model_id.nunique():,} done)")
    hub, buf = Hub(), []
    for i, m in enumerate(todo):
        recs, page, http = [], 0, None
        while True:
            r = hub.get(f"https://huggingface.co/api/models/{m}/commits/main?p={page}")
            if r is None:
                http = -1
                break
            http = r.status_code
            if http != 200:
                break
            d = r.json()
            if not d:
                break
            recs += [(c["id"], c["date"]) for c in d]
            if len(d) < 50:  # page size is 50 when there are more pages
                break
            page += 1
        if http == -1:
            continue  # retried on the next run
        buf += [dict(model_id=m, sha=s, date=dt, http=200) for s, dt in recs] if recs else \
            [dict(model_id=m, sha="", date="", http=http)]
        if len(buf) >= 500 or i == len(todo) - 1:
            done = pd.concat([done, pd.DataFrame(buf)], ignore_index=True)
            done.to_parquet(COMMITS, index=False)
            buf = []
            st = done.drop_duplicates("model_id").http.value_counts().to_dict()
            log(f"commits {i + 1:,}/{len(todo):,}; repos by status {st}")
    if buf:
        pd.concat([done, pd.DataFrame(buf)], ignore_index=True).to_parquet(COMMITS, index=False)


def month_map():
    T = pd.read_parquet(TARGETS).set_index("model_id")
    C = pd.read_parquet(COMMITS)
    C = C[C.http == 200].copy()
    C["t"] = (pd.to_datetime(C.date, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")  # any resolution
    months = pd.period_range(*WINDOW, freq="M")
    start = np.array([M.start_time.value // 10**9 for M in months], np.int64)
    end = np.array([(M + 1).start_time.value // 10**9 for M in months], np.int64)
    rows = []
    for m, g in C.groupby("model_id"):
        g = g.sort_values("t")
        ts = g.t.to_numpy()
        born = int(T.birth[m]) if pd.notna(T.birth[m]) else int(ts[0])
        k = np.searchsorted(ts, start, side="right") - 1
        for M, e, kk in zip(months, end, k):
            if e <= born:
                continue
            # model born during M (or first commit after the month start): its first version
            rows.append((m, str(M), g.sha.iloc[max(kk, 0)]))
    return pd.DataFrame(rows, columns=["model_id", "month", "sha"])


def readme():
    Mm = month_map()
    Mm.to_parquet(MONTH, index=False)
    need = Mm.drop_duplicates(["model_id", "sha"])
    done = pd.read_parquet(HIST) if HIST.exists() else pd.DataFrame(columns=["model_id", "sha", "http", "text"])
    seen = set(zip(done.model_id, done.sha))
    need = need[[(m, s) not in seen for m, s in zip(need.model_id, need.sha)]]
    log(f"readme: {len(need):,} versions to fetch ({len(done):,} done); month map {len(Mm):,} rows")
    hub, buf = Hub(), []
    for i, (m, s) in enumerate(zip(need.model_id, need.sha)):
        r = hub.get(f"https://huggingface.co/{m}/raw/{s}/README.md")
        if r is None:
            continue
        buf.append(dict(model_id=m, sha=s, http=r.status_code, text=r.text if r.status_code == 200 else ""))
        if len(buf) >= 500 or i == len(need) - 1:
            done = pd.concat([done, pd.DataFrame(buf)], ignore_index=True)
            done.to_parquet(HIST, index=False)
            buf = []
            log(f"readme {i + 1:,}/{len(need):,}; status {done.http.value_counts().to_dict()}")
    if buf:
        pd.concat([done, pd.DataFrame(buf)], ignore_index=True).to_parquet(HIST, index=False)


def prep():
    import importlib.util
    import sys
    from hfpa.features.text import clean
    sys.path.insert(0, str(paths.REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("p14", paths.REPO / "scripts" / "p14_quality.py")
    p14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p14)
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id"])
    idx = pd.Series(np.arange(len(nodes)), index=nodes.model_id)
    H = pd.read_parquet(HIST)
    H = H[H.http.isin([200, 404])].drop_duplicates(["model_id", "sha"]).reset_index(drop=True)
    H["vid"] = np.arange(len(H))
    H["node"] = idx.reindex(H.model_id).to_numpy()
    # cleaned paragraphs (empty README at that commit -> empty document)
    paras = [clean(t) if isinstance(t, str) and t else [] for t in H.text]
    pd.DataFrame({"vid": H.vid, "node": H.node, "paras": paras}).to_parquet(EXTRA, index=False)
    # historical bench_z with the snapshot standardisation (same benchmark keys as P14)
    mi = pd.read_parquet(paths.INTERIM / "model_index.parquet")
    mi = mi[mi.model_id.isin(idx.index)].copy()  # as in p14_quality.analyze
    mi["bench"] = mi.dataset.str.lower() + "|" + mi.metric.str.lower()
    mi = mi.groupby(["model_id", "bench"]).value.mean().reset_index()
    cnt = mi.groupby("bench").model_id.nunique()
    keep = [f"{d}|{m}" for d, m in p14.LEADERBOARD if cnt.get(f"{d}|{m}", 0) >= p14.MIN_MODELS]
    mu_sd = mi[mi.bench.isin(keep)].groupby("bench").value.agg(["mean", "std"])
    bz = []
    for t in H.text:
        res = p14.parse_card(t) if isinstance(t, str) and t else []
        z = {}
        for task, ds, cfg, met, v in res:
            b = f"{str(ds).lower()}|{str(met).lower()}"
            if b in mu_sd.index:
                z.setdefault(b, []).append((v - mu_sd.at[b, "mean"]) / mu_sd.at[b, "std"])
        bz.append(np.mean([np.mean(v) for v in z.values()]) if z else np.nan)
    H["bench_z"] = bz
    H[["vid", "model_id", "node", "sha", "http", "bench_z"]].to_parquet(VERS, index=False)
    log(f"versions {len(H):,} of {H.model_id.nunique():,} models; empty {sum(not p for p in paras):,}; "
        f"with bench_z {int(np.isfinite(H.bench_z).sum()):,}; benchmarks used {len(keep)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for p in ("commits", "readme", "prep"):
        ap.add_argument(f"--{p}", action="store_true")
    a = ap.parse_args()
    if a.commits:
        commits()
    if a.readme:
        readme()
    if a.prep:
        prep()
