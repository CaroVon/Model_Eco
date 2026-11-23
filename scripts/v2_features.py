"""v2 stage 2: decisions (sessions) and candidate features, one month per task.

--prep SPEC   decisions: consecutive uploads by one author on one parent with gaps <= W hours form one decision
              (W = 24 main; 6 / 72 sensitivity). Flags per choice: sess{W} (decision id), dfirst{W} (first upload of
              its decision), pair_first (first decision of the author-parent pair = plan A), last_parent (the author's
              most recent parent strictly before, any modality). Featurised choices: v2_pps every dfirst24 (first and
              repeat decisions, part 2), other specs pair_first only (q comparison).
              Shared indices -> interim/v2_idx/ (node attributes, time-indexed dataset profiles, co-use index, author
              histories, redistributor flags). Writes processed/choices/SPEC/decisions.parquet.
--month SPEC YYYY-MM   features of the featurised choices of the month -> processed/choices/SPEC/feats/YYYY-MM.parquet
              (V2_COMPLEMENT=1: the other choices, i.e. later uploads of a 24h session -> feats_up/, v2.1 sample U)
--check SPEC  row counts, exactly one chosen, feature summary -> results/tables/v2/features_SPEC.md
usage: sbatch slurm/cpu.sbatch scripts/v2_features.py --prep v2_pps
       sbatch --array=0-40 slurm/cpu.sbatch scripts/v2_features.py --month v2_pps   (task id -> month)
"""
import argparse
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from hfpa import paths
from hfpa.features.groups import ALL, LIB, MAIN, MIN_TOKENS, SIGNAL, redistributors
from hfpa.features.match import TimeProfiles, author_history, build_v2, couse_index_v2, node_attrs
from hfpa.features.text import row_cosine

IDX = paths.INTERIM / "v2_idx"
WINDOWS = (6, 24, 72)
OUT = paths.TABLES / "v2"


def cdir(spec):
    return paths.PROCESSED / "choices" / spec


def months(spec):
    return sorted(os.path.basename(f)[:-8] for f in glob.glob(str(cdir(spec) / "rows" / "*.parquet")))


# ---------------------------------------------------------------- prep
def sessions(ch, hours):
    """Decision id per choice (choice order) and first-upload flag."""
    o = np.lexsort((ch.t.to_numpy(), ch.parent.to_numpy(), ch.author.to_numpy()))
    a, p, t = ch.author.to_numpy()[o], ch.parent.to_numpy()[o], ch.t.to_numpy()[o]
    new = np.r_[True, (a[1:] != a[:-1]) | (p[1:] != p[:-1]) | (t[1:] - t[:-1] > hours * 3600)]
    sid = np.empty(len(ch), np.int64)
    sid[o] = np.cumsum(new) - 1
    first = np.empty(len(ch), bool)
    first[o] = new
    return sid, first


def last_parent(ch, nodes, E):
    m = E[E.rel_type.isin(MAIN)]
    author, birth = nodes.author_code.to_numpy(), nodes.birth.to_numpy()
    h = pd.DataFrame({"author": author[m.child_idx], "t": birth[m.child_idx], "lp": m.parent_idx.to_numpy()})
    h = h.sort_values("t", kind="stable")
    q = pd.DataFrame({"choice_id": ch.choice_id.to_numpy(), "author": ch.author.to_numpy(), "t": ch.t.to_numpy()})
    q = q.sort_values("t", kind="stable")
    r = pd.merge_asof(q, h, on="t", by="author", allow_exact_matches=False, direction="backward")
    return r.set_index("choice_id").lp.reindex(ch.choice_id).to_numpy()


def prep(spec):
    t0 = time.time()
    d = cdir(spec)
    ch = pd.read_parquet(d / "choices.parquet")
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["model_id", "birth", "author_code", "pipeline_tag", "params_b", "license_cat",
                                     "major_org"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    if "author" not in ch:
        ch["author"] = nodes.author_code.to_numpy()[ch.child]
    for w in WINDOWS:
        ch[f"sess{w}"], ch[f"dfirst{w}"] = sessions(ch, w)
    o = np.argsort(ch.t.to_numpy(), kind="stable")
    pf = np.zeros(len(ch), bool)
    pf[o] = ~pd.DataFrame({"a": ch.author.to_numpy()[o], "p": ch.parent.to_numpy()[o]}).duplicated().to_numpy()
    ch["pair_first"] = pf
    ch["last_parent"] = last_parent(ch, nodes, E)
    ch["featurise"] = ch.dfirst24 if spec == "v2_pps" else ch.pair_first
    ch.to_parquet(d / "decisions.parquet", index=False)
    print({"choices": len(ch), **{f"decisions_{w}h": int(ch[f"dfirst{w}"].sum()) for w in WINDOWS},
           "pairs": int(pf.sum()), "featurised": int(ch.featurise.sum())}, flush=True)
    if (IDX / "done").exists():
        return
    IDX.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(paths.INTERIM / "meta.parquet", columns=["model_id", "languages", "datasets"]).set_index("model_id")
    meta = meta.reindex(nodes.model_id)
    A = node_attrs(nodes, meta.languages.tolist(), meta.datasets.tolist())
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent", "t"])
    author = nodes.author_code.to_numpy()
    prof = TimeProfiles(A, nodes.birth.to_numpy())
    cidx = couse_index_v2(full, A, author)
    hist = author_history(full, author)
    red_orgs, red = redistributors(nodes, E)
    print(f"indices built in {time.time() - t0:.0f}s; datasets with profiles {len(prof.prof):,}; redistributors "
          f"{len(red_orgs)}", flush=True)
    for name, obj in [("A", A), ("prof", prof), ("cidx", cidx), ("hist", hist)]:
        with open(IDX / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f, protocol=5)
    np.save(IDX / "redistributor.npy", red)
    pd.Series(red_orgs).to_csv(IDX / "redistributor_orgs.csv", index=False)
    (IDX / "done").write_text("ok")


# ---------------------------------------------------------------- month
def twohop_month(rows, ch, month, nodes, E):
    """B.3: distinct other authors who share a niche parent with the chooser's author and used candidate j, from
    edges strictly before the month; niche = not among the top 500 parents by degree at the month start."""
    m = E[E.rel_type.isin(MAIN)]
    author, birth = nodes.author_code.to_numpy(), nodes.birth.to_numpy()
    ea, ep, et = author[m.child_idx.to_numpy()], m.parent_idx.to_numpy(), birth[m.child_idx.to_numpy()]
    t0 = int(pd.Period(month, "M").to_timestamp().tz_localize("UTC").timestamp())
    pre = et < t0
    NA, NN = int(author.max()) + 1, len(nodes)
    dpre = np.bincount(ep[pre], minlength=NN)
    hubs = np.argsort(-dpre, kind="stable")[:500]
    is_hub = np.zeros(NN, bool)
    is_hub[hubs] = True
    A = sp.csr_matrix((np.ones(pre.sum()), (ea[pre], ep[pre])), shape=(NA, NN))
    A.sum_duplicates()
    A.data[:] = 1.0
    niche = pre & ~is_hub[ep]
    Ant = sp.csr_matrix((np.ones(niche.sum()), (ep[niche], ea[niche])), shape=(NN, NA))
    Ant.sum_duplicates()
    rows_author = ch.author.loc[rows.choice_id].to_numpy()
    cand = rows.cand_id.to_numpy()
    co = np.zeros(len(rows), np.float32)
    order = np.argsort(rows_author, kind="stable")
    ra = rows_author[order]
    st = np.r_[0, np.flatnonzero(np.diff(ra)) + 1]
    en = np.r_[st[1:], len(ra)]
    for s, e in zip(st, en):
        a = ra[s]
        S = A[a].indices
        S = S[~is_hub[S]]
        if len(S) == 0:
            continue
        users = np.unique(Ant[S].indices)
        users = users[users != a]
        if len(users) == 0:
            continue
        used, cnt = np.unique(A[users].indices, return_counts=True)  # distinct co-users per node
        ii = order[s:e]
        pos = np.searchsorted(used, cand[ii])
        pos = np.minimum(pos, len(used) - 1)
        hit = used[pos] == cand[ii]
        co[ii[hit]] = cnt[pos[hit]]
    return co


def month(spec, mon):
    t0 = time.time()
    d = cdir(spec)
    ch = pd.read_parquet(d / "decisions.parquet").set_index("choice_id", drop=False)
    comp = bool(os.environ.get("V2_COMPLEMENT"))  # v2.1: the non-featurised uploads (later uploads of a session)
    keep = ch.index[~ch.featurise.to_numpy() if comp else ch.featurise.to_numpy()]
    rows = pd.read_parquet(d / "rows" / f"{mon}.parquet")
    rows = rows[rows.choice_id.isin(keep)].reset_index(drop=True)
    out = d / ("feats_up" if comp else "feats") / f"{mon}.parquet"
    out.parent.mkdir(exist_ok=True)
    if os.environ.get("V2_SKIP_TEXT"):
        out = out.with_suffix(".notext.parquet")
    if len(rows) == 0:
        pd.DataFrame().to_parquet(out)
        return
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["birth", "author_code", "legacy", "major_org", "library_cat", "params_b"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    ld = lambda n: pickle.load(open(IDX / f"{n}.pkl", "rb"))
    A, prof, cidx, hist = ld("A"), ld("prof"), ld("cidx"), ld("hist")
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent", "t"])
    author = nodes.author_code.to_numpy()
    print(f"{mon}: {len(rows):,} rows, {rows.choice_id.nunique():,} choices; loaded in {time.time() - t0:.0f}s", flush=True)
    c = rows.cand_id.to_numpy()
    f32 = lambda a: np.asarray(a, np.float32)
    F = pd.DataFrame({k: rows[k].to_numpy() for k in ("choice_id", "cand_id", "chosen", "log_q", "stratum", "fe")})
    F["log_deg"] = rows.log_deg.to_numpy(np.float32)
    F["has_deg"] = f32(rows.has_deg)
    F["deg"] = rows.deg.to_numpy()
    F["log_age"] = f32(np.log1p(np.maximum(rows.age_days.to_numpy(), 0)))
    F["legacy"] = f32(nodes.legacy.to_numpy()[c])
    F["major_org"] = f32(nodes.major_org.to_numpy()[c])
    lib = nodes.library_cat.to_numpy()[c]
    for x in LIB:
        F[f"lib_{x}"] = f32(lib == x)
    F["lrec30"] = f32(np.log1p(rows.recent30))
    F["lquant"] = f32(np.log1p(rows.quant_deg))
    F["redistributor"] = f32(np.load(IDX / "redistributor.npy")[c])
    for k in ("same_author", "author_reuse", "author_family"):
        F[k] = f32(rows[k])
    # quality (model-index benchmarks, P14)
    B = pd.read_parquet(paths.PROCESSED / "nodes" / "bench.parquet")
    bz = np.full(len(nodes), np.nan, np.float32)
    bz[B.node.to_numpy()] = B.bench_z.to_numpy()
    b = bz[c]
    F["bench_na"] = f32(np.isnan(b))
    F["bench_z"] = f32(np.nan_to_num(b))
    # inertia: most recent parent
    lp = ch.last_parent.loc[rows.choice_id].to_numpy()
    F["is_last"] = f32(lp == c)
    # demand match (A2-A4)
    M = build_v2(rows, ch, A, full, author, prof, cidx, hist=hist)
    for k in M.columns:
        F[k] = M[k].to_numpy(np.float32)
    print(f"  match done {time.time() - t0:.0f}s", flush=True)
    # card text (F)
    if os.environ.get("V2_SKIP_TEXT"):  # debugging only
        for k in ("txt_raw", "txt_sim", "txt_has"):
            F[k] = np.float32(0)
        F["txt_ntok"], F["txt_na"] = np.int32(0), np.float32(1)
    else:
        text_features(F, rows, ch, c)
    # local search (B.3)
    F["lco2"] = f32(np.log1p(twohop_month(rows, ch, mon, nodes, E)))
    print(f"  text + twohop done {time.time() - t0:.0f}s", flush=True)
    assert F.groupby("choice_id").chosen.sum().eq(1).all()
    missing = [k for k in ALL + SIGNAL if k not in F]
    assert not missing, missing
    F.to_parquet(out, index=False)
    print(f"{mon} written: {len(F):,} rows in {time.time() - t0:.0f}s", flush=True)


def text_features(F, rows, ch, c):
    X = sp.load_npz(paths.INTERIM / "v2_text_tfidf.npz").tocsr()
    T = np.load(paths.INTERIM / "v2_text_rows.npz")
    kid = ch.child.loc[rows.choice_id].to_numpy()
    ra, rb = T["crow"][kid], T["prow"][c]
    ntok = np.minimum(T["ntok_child"][kid], T["ntok_cand"][c])
    ntok[(ra < 0) | (rb < 0)] = 0
    ok = ntok >= 1
    s = np.zeros(len(rows), np.float32)
    s[ok] = row_cosine(X, ra[ok], rb[ok])
    f32 = lambda a: np.asarray(a, np.float32)
    F["txt_raw"] = s  # cosine whenever both sides have >= 1 token; the threshold is applied below / in sensitivity
    F["txt_ntok"] = ntok.astype(np.int32)
    na = ntok < MIN_TOKENS
    F["txt_na"] = f32(na)
    F["txt_sim"] = np.where(na, 0, s).astype(np.float32)
    F["txt_has"] = f32(T["ntok_child"][kid] >= MIN_TOKENS)


def donors(ch, seed=0):
    """Placebo donor for each plan-A choice: another plan-A choice of a different author in the same month x
    modality cell (P20 design). Returns choice_id -> donor choice_id (missing if none)."""
    rng = np.random.default_rng(seed)
    c = ch[ch.pair_first.to_numpy()]
    mon = pd.to_datetime(c.t, unit="s").dt.to_period("M").astype(str).to_numpy()
    out = {}
    for _, g in c.groupby([mon, c.modality.to_numpy()]):
        ids, au = g.choice_id.to_numpy(), g.author.to_numpy()
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for _ in range(10):
                j = int(rng.integers(len(ids)))
                if au[j] != au[i]:
                    out[ids[i]] = ids[j]
                    break
    return pd.Series(out)


def placebo_month(spec, mon):
    """Demand-match and text features with the donor's demand (child, author history, datasets, card); candidates,
    times and everything else stay the choice's own."""
    d = cdir(spec)
    ch = pd.read_parquet(d / "decisions.parquet").set_index("choice_id", drop=False)
    don = donors(ch)
    rows = pd.read_parquet(d / "feats" / f"{mon}.parquet", columns=["choice_id", "cand_id", "chosen", "author_reuse"])
    rows = rows[rows.choice_id.isin(don.index) & (rows.author_reuse == 0)].reset_index(drop=True)
    out = d / "feats_placebo" / f"{mon}.parquet"
    out.parent.mkdir(exist_ok=True)
    if len(rows) == 0:
        pd.DataFrame().to_parquet(out)
        return
    ld = lambda n: pickle.load(open(IDX / f"{n}.pkl", "rb"))
    A, prof, cidx, hist = ld("A"), ld("prof"), ld("cidx"), ld("hist")
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent", "t"])
    author = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["author_code"]).author_code.to_numpy()
    ids = np.unique(rows.choice_id)
    dd = don.loc[ids].to_numpy()
    ch2 = pd.DataFrame({"choice_id": ids, "child": ch.child.loc[dd].to_numpy(), "parent": ch.parent.loc[dd].to_numpy(),
                        "t": ch.t.loc[ids].to_numpy()}).set_index("choice_id", drop=False)
    M = build_v2(rows, ch2, A, full, author, prof, cidx, hist=hist)
    F = pd.DataFrame({"choice_id": rows.choice_id.to_numpy(), "cand_id": rows.cand_id.to_numpy()})
    for k in M.columns:
        F[k] = M[k].to_numpy(np.float32)
    text_features(F, rows, ch2, rows.cand_id.to_numpy())
    F["has_donor"] = np.float32(1)
    F.to_parquet(out, index=False)
    print(f"placebo {mon}: {len(F):,} rows", flush=True)


def addtext(spec, mon):
    """Add the card-text features to a month built with V2_SKIP_TEXT=1 (.notext.parquet) -> final feats file."""
    d = cdir(spec)
    src = d / "feats" / f"{mon}.notext.parquet"
    out = d / "feats" / f"{mon}.parquet"
    if not pq.read_schema(src).names:
        pd.DataFrame().to_parquet(out)
        return
    F = pd.read_parquet(src)
    ch = pd.read_parquet(d / "decisions.parquet").set_index("choice_id", drop=False)
    text_features(F, F, ch, F.cand_id.to_numpy())
    F.to_parquet(out, index=False)
    print(f"{spec} {mon}: text added, {len(F):,} rows", flush=True)


def check(spec):
    d = cdir(spec)
    fs = [f for f in sorted(glob.glob(str(d / "feats" / "*.parquet"))) if not f.endswith(".notext.parquet")]
    ch = pd.read_parquet(d / "decisions.parquet")
    n_rows, n_ch, desc = 0, 0, []
    for f in fs:
        x = pd.read_parquet(f)
        if len(x) == 0:
            continue
        n_rows += len(x)
        n_ch += x.choice_id.nunique()
        desc.append(x[ALL + SIGNAL].describe().T[["mean"]].rename(columns={"mean": os.path.basename(f)[:7]}))
    S = pd.concat(desc, axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# v2 features: {spec}", "", f"months {len(fs)}; rows {n_rows:,}; choices {n_ch:,} "
          f"(featurised in decisions.parquet: {int(ch.featurise.sum()):,})", "",
          "Feature means by month (first, middle, last months):", "",
          S.iloc[:, [0, len(fs) // 2, -1]].to_markdown(floatfmt=".3f")]
    (OUT / f"features_{spec}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep")
    ap.add_argument("--month", nargs="+")
    ap.add_argument("--check")
    ap.add_argument("--placebo", action="store_true", help="with --month: donor demand features")
    ap.add_argument("--addtext", action="store_true", help="with --month: add text to the .notext month files")
    a = ap.parse_args()
    if a.prep:
        prep(a.prep)
    if a.month:
        spec = a.month[0]
        mons = a.month[1:] or [months(spec)[int(os.environ["SLURM_ARRAY_TASK_ID"])]]
        for m in mons:
            (placebo_month if a.placebo else addtext if a.addtext else month)(spec, m)
    if a.check:
        check(a.check)
