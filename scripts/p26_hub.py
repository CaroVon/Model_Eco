"""P26: does the decomposition change when popularity includes what developers saw on the Hub at choice time
(likes, 30-day downloads, gating; P24 Wayback panel) and when the card text / benchmarks are the versions visible at
choice time (P25 git history) instead of the 2025-07 snapshot?

Hub market: plan-A first choices whose chosen parent is a P24 target (top 2,000 chosen + FE hubs: 88% of first
choices), with the choice set restricted to target candidates. Under the conditional logit (IIA) a choice set
restricted by a fixed rule on alternatives keeps the same coefficients; log q is unchanged. Every variant is fitted
on this same sample, so differences come from the features alone.

--build   processed/choices/v2_pps/p26_hub.parquet: plan-A rows (main features) + new columns
            wb_llikes, wb_ldl, wb_gated, wb_na, wb_dl_na   latest Wayback capture <= t and >= t - MAX_GAP days
                                                           (log1p; missing -> 0 + indicator; wb_dl_na = likes
                                                           seen but downloads missing)
            snap_llikes, snap_ldl                          2025-07 snapshot values (leak future information; contrast)
            txt_sim_h, txt_na_h, bench_z_h, bench_na_h     card version of the first day of the choice month
                                                           (fallback: snapshot values when the history is missing --
                                                           gated / deleted repos; hist_src records which)
            txt_sim_r                                       snapshot card in the rebuilt TF-IDF (check vs txt_sim)
--fit     five players (GROUPS5), train split, in-sample Shapley shares for variants
            base | wb (+ Wayback signals in popularity) | hist (time-of-choice card text / benchmarks) | wb_hist |
            snap (+ snapshot likes / downloads)
          and full-model coefficients (cluster-robust by author). Tables results/tables/v2/p26_*.md.
usage: sbatch --mem=128G slurm/cpu.sbatch scripts/p26_hub.py --build ; sbatch slurm/gpu.sbatch scripts/p26_hub.py --fit
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from hfpa import paths
from hfpa.features.groups import ALL, GROUPS5, MIN_TOKENS, NAMES, SIGNAL
from hfpa.features.text import row_cosine
from hfpa.v2data import cdir, drop_constant, load_rows, shapley_fits, shares, split, write

SPEC = "v2_pps"
MAX_GAP = 60 * 86400
DEV = bool(os.environ.get("V2_DEV_GLOB"))  # development: month subset (hfpa.v2data.load_rows), separate outputs
OUT = cdir(SPEC) / ("p26_hub_dev.parquet" if DEV else "p26_hub.parquet")
TAG = "p26dev" if DEV else "p26"
TEXT_DIR = paths.INTERIM / "p25_text"  # V2_TEXT_OUT of the rebuilt TF-IDF with historical versions
WB = ["wb_llikes", "wb_ldl", "wb_gated", "wb_na", "wb_dl_na"]
SNAPF = ["snap_llikes", "snap_ldl"]
T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def asof(model, t, P, col):
    """Latest panel value of `col` for (model, t): capture ts <= t and t - ts <= MAX_GAP. NaN otherwise."""
    P = P[P[col].notna()]
    codes = {m: i for i, m in enumerate(sorted(set(P.model_id) | set(model)))}
    pk = np.array([codes[m] for m in P.model_id], np.int64) * 10**10 + P.t.to_numpy(np.int64)
    o = np.argsort(pk)
    pk, pv, pt = pk[o], P[col].to_numpy(np.float64)[o], P.t.to_numpy(np.int64)[o]
    rk = np.array([codes[m] for m in model], np.int64) * 10**10 + t
    k = np.searchsorted(pk, rk, side="right") - 1
    ok = (k >= 0) & (pk[np.maximum(k, 0)] // 10**10 == rk // 10**10)
    ok &= (t - pt[np.maximum(k, 0)]) <= MAX_GAP
    return np.where(ok, pv[np.maximum(k, 0)], np.nan)


def build():
    T = pd.read_parquet(paths.INTERIM / "wb_targets.parquet")
    tgt = T.node.to_numpy()
    rows, ch = load_rows(SPEC, "planA", cols=ALL + SIGNAL, extra=["txt_ntok"])
    n0 = rows.choice_id.nunique()
    chosen = rows.loc[rows.chosen == 1, ["choice_id", "cand_id"]]
    keep_c = chosen.choice_id[np.isin(chosen.cand_id, tgt)]
    rows = rows[rows.choice_id.isin(keep_c) & np.isin(rows.cand_id, tgt)].reset_index(drop=True)
    size = rows.groupby("choice_id").cand_id.transform("size")
    rows = rows[size >= 2].reset_index(drop=True)
    log(f"hub market: {rows.choice_id.nunique():,} of {n0:,} plan-A choices; {len(rows):,} rows; "
        f"mean set size {len(rows) / rows.choice_id.nunique():.1f}")
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "likes", "downloads"])
    mid = nodes.model_id.to_numpy()[rows.cand_id.to_numpy()]
    t = ch.t.reindex(rows.choice_id).to_numpy(np.int64)
    # Wayback panel
    P = pd.read_parquet(paths.INTERIM / "wb_panel.parquet")
    Cx = pd.read_parquet(paths.INTERIM / "wb_cdx.parquet")
    n_cap = Cx[Cx.ok].drop_duplicates(["model_id", "ts"]).shape[0]
    log(f"Wayback: CDX names queried {Cx.drop_duplicates(['model_id', 'name']).shape[0]:,}; captures fetched "
        f"{len(P):,} of {n_cap:,} ({len(P) / max(n_cap, 1):.1%})")
    P = P[P.http == 200].copy()
    P["t"] = (pd.to_datetime(P.ts, format="%Y%m%d%H%M%S", utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
    P["gated_b"] = P.gated.map({"false": 0.0, "manual": 1.0, "auto": 1.0, "true": 1.0})
    for c in ("likes", "downloads30"):
        P[c] = pd.to_numeric(P[c], errors="coerce")
    lk, dl, gt = (asof(mid, t, P, c) for c in ("likes", "downloads30", "gated_b"))
    rows["wb_na"] = np.isnan(lk).astype(np.float32)
    rows["wb_llikes"] = np.log1p(np.nan_to_num(lk)).astype(np.float32)
    rows["wb_dl_na"] = (np.isnan(dl) & ~np.isnan(lk)).astype(np.float32)  # else collinear with wb_na
    rows["wb_ldl"] = np.log1p(np.nan_to_num(dl)).astype(np.float32)
    rows["wb_gated"] = np.nan_to_num(gt).astype(np.float32)
    rows["snap_llikes"] = np.log1p(nodes.likes.fillna(0).to_numpy()[rows.cand_id]).astype(np.float32)
    rows["snap_ldl"] = np.log1p(nodes.downloads.fillna(0).to_numpy()[rows.cand_id]).astype(np.float32)
    ch_ = rows.chosen == 1
    log(f"Wayback likes available: all rows {1 - rows.wb_na.mean():.3f}, chosen {1 - rows.wb_na[ch_].mean():.3f}; "
        f"downloads {1 - rows.wb_dl_na.mean():.3f}; gated share among available {rows.wb_gated[rows.wb_na == 0].mean():.3f}")
    # historical card text and benchmarks
    V = pd.read_parquet(paths.INTERIM / "card_versions.parquet")
    Mm = pd.read_parquet(paths.INTERIM / "card_month.parquet")
    month = pd.to_datetime(t, unit="s").to_period("M").astype(str).to_numpy()
    assert t.min() > 1.5e9 and t.max() < 1.8e9, "choice times must be unix seconds"
    key = pd.DataFrame({"model_id": mid, "month": month})
    sha = key.merge(Mm, on=["model_id", "month"], how="left").sha.to_numpy()
    vmap = V.set_index(["model_id", "sha"]).vid
    vid = pd.Series(list(zip(mid, sha))).map(lambda k: vmap.get(k, -1) if isinstance(k[1], str) else -1).to_numpy(np.int64)
    has_h = vid >= 0
    X = sp.load_npz(TEXT_DIR / "v2_text_tfidf.npz").tocsr()
    R = np.load(TEXT_DIR / "v2_text_rows.npz")
    erow = np.full(int(R["extra_vid"].max()) + 1 if len(R["extra_vid"]) else 1, -1, np.int64)
    entok = np.zeros_like(erow, dtype=np.int32)
    erow[R["extra_vid"]], entok[R["extra_vid"]] = R["extra_row"], R["extra_ntok"]
    kid = ch.child.reindex(rows.choice_id).to_numpy()
    ra = R["crow"][kid]

    def sim(rb, ntok_b):
        ntok = np.minimum(R["ntok_child"][kid], ntok_b)
        ntok[(ra < 0) | (rb < 0)] = 0
        s = np.zeros(len(rows), np.float32)
        ok = ntok >= 1
        s[ok] = row_cosine(X, ra[ok], rb[ok])
        na = ntok < MIN_TOKENS
        return np.where(na, 0, s).astype(np.float32), na.astype(np.float32)

    c = rows.cand_id.to_numpy()
    rows["txt_sim_r"], _ = sim(R["prow"][c], R["ntok_cand"][c])
    sh, nah = sim(np.where(has_h, erow[np.maximum(vid, 0)], -1), np.where(has_h, entok[np.maximum(vid, 0)], 0))
    rows["txt_sim_h"] = np.where(has_h, sh, rows.txt_sim).astype(np.float32)
    rows["txt_na_h"] = np.where(has_h, nah, rows.txt_na).astype(np.float32)
    bz = V.bench_z.to_numpy()[np.maximum(vid, 0)] if len(V) else np.full(len(rows), np.nan)
    rows["bench_na_h"] = np.where(has_h, np.isnan(bz), rows.bench_na).astype(np.float32)
    rows["bench_z_h"] = np.where(has_h, np.nan_to_num(bz), rows.bench_z).astype(np.float32)
    rows["hist_src"] = np.where(has_h, "history", "snapshot")
    chk = dict(rebuilt_vs_prod_corr=float(np.corrcoef(rows.txt_sim, rows.txt_sim_r)[0, 1]),
               history_rows=float(has_h.mean()), history_chosen=float(has_h[ch_].mean()),
               txt_changed=float((rows.txt_sim_h != rows.txt_sim)[has_h].mean()),
               bench_changed=float((rows.bench_z_h != rows.bench_z)[has_h].mean()))
    log(chk)
    rows.to_parquet(OUT, index=False)


def variants():
    g = {k: list(v) for k, v in GROUPS5.items()}
    swap = lambda lst, m: [m.get(x, x) for x in lst]
    H = {"txt_sim": "txt_sim_h", "txt_na": "txt_na_h", "bench_z": "bench_z_h", "bench_na": "bench_na_h"}
    out = {"base": g,
           "wb": {**g, "pop": g["pop"] + WB},
           "hist": {k: swap(v, H) for k, v in g.items()},
           "snap": {**g, "pop": g["pop"] + SNAPF}}
    out["wb_hist"] = {**out["hist"], "pop": out["hist"]["pop"] + WB}
    return out


def fit():
    import torch
    from hfpa.models.clogit import fit_subset, to_buckets
    rows = pd.read_parquet(OUT)
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet").set_index("choice_id", drop=False)
    tr, _ = split(rows, ch)
    log(f"train: {tr.choice_id.nunique():,} choices, {len(tr):,} rows; device {'cuda' if torch.cuda.is_available() else 'cpu'}")
    share_rows, coef_rows = [], []
    for name, G in variants().items():
        feats = drop_constant(tr, list(dict.fromkeys(sum(G.values(), []))))
        G = {k: [f for f in v if f in feats] for k, v in G.items()}
        D = to_buckets(tr, feats)
        fits = shapley_fits(D, feats, groups=G, log=None)
        sh, total = shares(fits, list(G))
        fit_share = sh["qual"]["share"] + sh["match"]["share"]
        share_rows.append(dict(variant=name, ll_gain=total, **{NAMES[k]: 100 * v["share"] for k, v in sh.items()},
                               fit_over_pop=fit_share / sh["pop"]["share"]))
        author = ch.author.reindex(D.choice_ids).to_numpy()
        _, r = fit_subset(D, feats, se=True, cluster=author)
        for f, b, s in zip(feats, r["coef"], r["se_cl"]):
            coef_rows.append(dict(variant=name, feature=f, coef=b, se=s, z=b / s if s > 0 else np.nan))
        log(f"{name}: {share_rows[-1]}")
    S = pd.DataFrame(share_rows)
    C = pd.DataFrame(coef_rows)
    write(S, f"{TAG}_hub_shares", "P26 hub market: Shapley shares (%) by feature variant (five players, train, in-sample)",
          "base = main features on the hub market; wb = + Wayback likes / downloads / gating (popularity group); "
          "hist = card text and benchmarks as of the choice month; snap = + 2025-07 snapshot likes / downloads (leaky).")
    keep = C.feature.isin(WB + SNAPF + ["log_deg", "lrec30", "txt_sim", "txt_sim_h", "bench_z", "bench_z_h"])
    write(C[keep], f"{TAG}_hub_coef", "P26 hub market: selected coefficients (cluster-robust SE by author)")
    C.to_csv(paths.TABLES / "v2" / f"{TAG}_hub_coef_all.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.fit:
        fit()
