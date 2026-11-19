"""Demand-match features between a choosing child model and each candidate parent.

The developer's demand is read only from sources the child does not copy from its parent (P13 inheritance
check): the child's declared training datasets and the author's earlier choices. The candidate side uses the
candidate's own labels. No feature reads the child's own languages, task, license or size.

Features per (choice, candidate) row:
  hist_has     author chose at least one other parent before (leave-one-out: the candidate itself excluded)
  size_dist    |log params(candidate) - median log params of the author's earlier parents| (0 if unknown)
  size_na      candidate size or author size history unknown
  task_hist    candidate pipeline_tag appears among the author's earlier parents (leave-one-out)
  lang_hist    candidate shares a language with the author's earlier parents (leave-one-out)
  ds_has       child declares a dataset with a usable profile (>= MIN_DS other models declare it)
  ds_lang      candidate supports a language that is common (share >= SHARE) among other models trained on
               the child's datasets
  ds_task      candidate pipeline_tag is common among other models trained on the child's datasets
  ds_couse     log(1 + number of earlier models by other authors trained on one of the child's datasets that
               chose this candidate)
  org_noncomm  child author is a major organization x candidate license is non-commercial
Dataset profiles exclude the child itself.
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

NBITS = 63
SHARE = 0.3
MIN_DS = 3
COLS = ["hist_has", "size_dist", "size_na", "task_hist", "lang_hist", "ds_has", "ds_lang", "ds_task", "ds_couse",
        "org_noncomm"]
_SHIFT = np.arange(NBITS, dtype=np.uint64)


def _empty(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or len(v) == 0


def lang_masks(langs, vocab=None):
    """Bit mask (uint64) of the NBITS most common languages for each entry of `langs` (iterables or None)."""
    if vocab is None:
        c = Counter()
        for x in langs:
            if not _empty(x):
                c.update(x)
        vocab = {k: i for i, (k, _) in enumerate(c.most_common(NBITS))}
    out = np.zeros(len(langs), np.uint64)
    for i, x in enumerate(langs):
        if not _empty(x):
            m = 0
            for l in x:
                b = vocab.get(l)
                if b is not None:
                    m |= 1 << b
            out[i] = m
    return out, vocab


def bits(masks):
    return ((np.asarray(masks, np.uint64)[:, None] >> _SHIFT) & np.uint64(1)).astype(bool)


def node_attrs(nodes, meta_langs, meta_datasets):
    """nodes: frame with pipeline_tag, params_b, license_cat, major_org (row = node index)."""
    lm, _ = lang_masks(list(meta_langs))
    task = pd.Categorical(nodes.pipeline_tag).codes.astype(np.int32)  # -1 = missing
    pb = nodes.params_b.to_numpy(float)
    ok = np.isfinite(pb) & (pb > 0) & (pb < 2000)
    logp = np.where(ok, np.log(np.where(ok, pb, 1)), np.nan)
    ds = [None if _empty(x) else [str(d) for d in x] for x in meta_datasets]
    return dict(lang=lm, task=task, logp=logp, noncomm=(nodes.license_cat.to_numpy() == "noncommercial"),
                org=nodes.major_org.to_numpy().astype(bool), ds=ds)


def dataset_profiles(A):
    """Per dataset: number of models declaring it, per-language counts, per-task counts."""
    n, lang, task = Counter(), {}, defaultdict(Counter)
    for i, D in enumerate(A["ds"]):
        if D is None:
            continue
        b = bits(A["lang"][i:i + 1])[0].astype(np.int32)
        for d in set(D):
            n[d] += 1
            lang[d] = lang.get(d, 0) + b
            if A["task"][i] >= 0:
                task[d][A["task"][i]] += 1
    return n, lang, task


def child_demand(c, A, prof):
    """Languages (mask) and tasks common among other models declaring the child's datasets."""
    D = A["ds"][c]
    if D is None:
        return False, 0, set()
    n, lang, task = prof
    own = bits(A["lang"][c:c + 1])[0].astype(np.int32)
    L, T, ok = np.zeros(NBITS, bool), set(), False
    for d in set(D):
        m = n[d] - 1
        if m < MIN_DS:
            continue
        ok = True
        L |= (lang[d] - own) / m >= SHARE
        tc = task[d].copy()
        if A["task"][c] >= 0:
            tc[A["task"][c]] -= 1
        T |= {t for t, k in tc.items() if k / m >= SHARE}
    mask = int(np.sum(L.astype(np.uint64) << _SHIFT)) if ok else 0
    return ok, mask, T


def couse_index(full, A, author):
    """(dataset, parent) -> (sorted times, authors) of earlier main choices whose child declares the dataset."""
    idx = defaultdict(list)
    for c, p, t in zip(full.child.to_numpy(), full.parent.to_numpy(), full.t.to_numpy()):
        D = A["ds"][c]
        if D is not None:
            for d in set(D):
                idx[(d, p)].append((t, author[c]))
    out = {}
    for k, v in idx.items():
        v.sort()
        out[k] = (np.array([x[0] for x in v]), np.array([x[1] for x in v]))
    return out


def author_history(full, author):
    """author -> (times, parents) of all main choices, time-sorted."""
    f = pd.DataFrame({"a": author[full.child.to_numpy()], "t": full.t.to_numpy(), "p": full.parent.to_numpy()})
    f = f.sort_values("t", kind="stable")
    return {a: (g.t.to_numpy(), g.p.to_numpy()) for a, g in f.groupby("a")}


def build(rows, ch, A, full, author):
    """rows: choice rows sorted by choice_id (cand_id); ch: sampled choices indexed by choice_id (child, t).
    full: all main choices (child, parent, t) for author histories and dataset co-use."""
    prof = dataset_profiles(A)
    hist = author_history(full, author)
    cidx = couse_index(full, A, author)
    out = {k: np.zeros(len(rows), np.float32) for k in COLS}
    cid = rows.choice_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(cid)) + 1]
    ends = np.r_[starts[1:], len(rows)]
    cand_all = rows.cand_id.to_numpy()
    for s, e in zip(starts, ends):
        k = cid[s]
        c, t = int(ch.child.at[k]), ch.t.at[k]
        a = author[c]
        J = cand_all[s:e]
        # author history, strictly earlier, distinct parents
        ht, hp = hist.get(a, (np.empty(0), np.empty(0, int)))
        H = np.unique(hp[ht < t])
        inH = np.isin(J, H)
        nH = len(H) - inH
        out["hist_has"][s:e] = nH > 0
        if len(H):
            hl = A["logp"][H]
            hok = np.isfinite(hl)
            med = np.median(hl[hok]) if hok.any() else np.nan
            lj = A["logp"][J]
            dist = np.abs(lj - med)
            for i in np.flatnonzero(inH & hok.any()):  # leave the candidate out of the median
                rest = hl[hok & (H != J[i])]
                dist[i] = abs(lj[i] - np.median(rest)) if len(rest) else np.nan
            na = ~np.isfinite(dist)
            out["size_dist"][s:e] = np.where(na, 0, dist)
            out["size_na"][s:e] = na
            tc = Counter(A["task"][H][A["task"][H] >= 0].tolist())
            tj = A["task"][J]
            eff = np.array([tc.get(x, 0) for x in tj]) - inH
            out["task_hist"][s:e] = (tj >= 0) & (eff > 0)
            C = bits(A["lang"][H]).sum(0)
            B = bits(A["lang"][J])
            effl = C[None, :] - B * inH[:, None]
            out["lang_hist"][s:e] = (B & (effl > 0)).any(1)
        else:
            out["size_na"][s:e] = 1
        # dataset-based demand
        ok, L, T = child_demand(c, A, prof)
        if ok:
            out["ds_has"][s:e] = 1
            out["ds_lang"][s:e] = (A["lang"][J] & np.uint64(L)) > 0
            out["ds_task"][s:e] = np.isin(A["task"][J], list(T)) & (A["task"][J] >= 0)
            cnt = np.zeros(len(J))
            for d in set(A["ds"][c]):
                for i, j in enumerate(J):
                    v = cidx.get((d, j))
                    if v is not None:
                        m = v[0] < t
                        cnt[i] += m.sum() - (m & (v[1] == a)).sum()
            out["ds_couse"][s:e] = np.log1p(cnt)
        out["org_noncomm"][s:e] = A["org"][c] & A["noncomm"][J]
    return pd.DataFrame(out, index=rows.index)


# ============================================================================ v2 (time-filtered, A2-A4)
COLS_V2 = ["hist_has", "size_dist", "size_na", "task_hist", "lang_hist", "ds_has", "ds_lang", "ds_task", "ds_couse"]


class TimeProfiles:
    """A2: dataset demand profiles as of time t (only models born strictly before t declare the dataset).
    Per dataset: sorted declaration births, cumulative language-bit counts [n+1, NBITS] and cumulative task counts
    [n+1, n_task]. Datasets declared by fewer than MIN_DS + 1 models are dropped (never usable)."""

    def __init__(self, A, birth):
        dec = defaultdict(list)
        for i, D in enumerate(A["ds"]):
            if D is not None:
                for d in set(D):
                    dec[d].append(i)
        self.n_task = int(A["task"].max()) + 2  # column 0 = missing task
        self.prof = {}
        for d, v in dec.items():
            if len(v) < MIN_DS + 1:
                continue
            v = np.array(v)
            v = v[np.argsort(birth[v], kind="stable")]
            L = bits(A["lang"][v]).astype(np.int32)
            T = np.zeros((len(v), self.n_task), np.int32)
            T[np.arange(len(v)), A["task"][v] + 1] = 1
            z = lambda M: np.vstack([np.zeros((1, M.shape[1]), np.int32), np.cumsum(M, 0, dtype=np.int32)])
            self.prof[d] = (birth[v], z(L), z(T))

    def at(self, d, t):
        """(m, lang counts, task counts) of dataset d over models born before t, or None."""
        p = self.prof.get(d)
        if p is None:
            return None
        k = int(np.searchsorted(p[0], t, "left"))
        return k, p[1][k], p[2][k]


def couse_index_v2(full, A, author):
    """(dataset, parent) -> (sorted child births, child authors, child ids) of main choices whose child declares
    the dataset (for ds_couse and for A4: removing the candidate's own children from the profile)."""
    idx = defaultdict(list)
    for c, p, t in zip(full.child.to_numpy(), full.parent.to_numpy(), full.t.to_numpy()):
        D = A["ds"][c]
        if D is not None:
            for d in set(D):
                idx[(d, p)].append((t, author[c], c))
    out = {}
    for k, v in idx.items():
        v.sort()
        out[k] = tuple(np.array([x[i] for x in v]) for i in range(3))
    return out


def _demand(m, lc, tc):
    """Language mask (bool NBITS) and task set common (share >= SHARE) among m declaring models."""
    if m < MIN_DS:
        return None
    L = lc / m >= SHARE
    T = {int(t) - 1 for t in np.flatnonzero(tc / m >= SHARE) if t > 0}
    return L, T


def _mask(L):
    return np.uint64(int(np.sum(L.astype(np.uint64) << _SHIFT)))


def build_v2(rows, ch, A, full, author, prof, cidx, hist=None, drop_parent_ds=True, drop_cand_children=True):
    """v2 demand-match features. rows sorted by choice_id; ch indexed by choice_id with child, parent, t.
    A2 profiles as of the choice time (prof: TimeProfiles); A3 the child's datasets that the chosen parent also
    declares are not demand (drop_parent_ds); A4 the candidate's own earlier children are removed from the
    profile used for that candidate (drop_cand_children). org_noncomm dropped. Same author-history features as v1."""
    hist = hist if hist is not None else author_history(full, author)
    out = {k: np.zeros(len(rows), np.float32) for k in COLS_V2}
    cid = rows.choice_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(cid)) + 1]
    ends = np.r_[starts[1:], len(rows)]
    cand_all = rows.cand_id.to_numpy()
    lang_bits = None
    for s, e in zip(starts, ends):
        k = cid[s]
        c, t, par = int(ch.child.at[k]), ch.t.at[k], int(ch.parent.at[k])
        a = author[c]
        J = cand_all[s:e]
        # ---- author history (as v1)
        ht, hp = hist.get(a, (np.empty(0), np.empty(0, int)))
        H = np.unique(hp[ht < t])
        inH = np.isin(J, H)
        nH = len(H) - inH
        out["hist_has"][s:e] = nH > 0
        if len(H):
            hl = A["logp"][H]
            hok = np.isfinite(hl)
            med = np.median(hl[hok]) if hok.any() else np.nan
            lj = A["logp"][J]
            dist = np.abs(lj - med)
            for i in np.flatnonzero(inH & hok.any()):
                rest = hl[hok & (H != J[i])]
                dist[i] = abs(lj[i] - np.median(rest)) if len(rest) else np.nan
            na = ~np.isfinite(dist)
            out["size_dist"][s:e] = np.where(na, 0, dist)
            out["size_na"][s:e] = na
            tc = Counter(A["task"][H][A["task"][H] >= 0].tolist())
            tj = A["task"][J]
            eff = np.array([tc.get(x, 0) for x in tj]) - inH
            out["task_hist"][s:e] = (tj >= 0) & (eff > 0)
            C = bits(A["lang"][H]).sum(0)
            B = bits(A["lang"][J])
            effl = C[None, :] - B * inH[:, None]
            out["lang_hist"][s:e] = (B & (effl > 0)).any(1)
        else:
            out["size_na"][s:e] = 1
        # ---- dataset demand
        Dc = A["ds"][c]
        if Dc is None:
            continue
        Ds = set(Dc)
        if drop_parent_ds and A["ds"][par] is not None:
            Ds -= set(A["ds"][par])
        base = {}
        for d in Ds:
            r = prof.at(d, t)
            if r is not None:
                dm = _demand(*r)
                if dm is not None:
                    base[d] = (r, dm)
        cnt = np.zeros(len(J))
        override = {}  # candidate position -> (L, T) after removing its own children (A4)
        for d in Ds:
            for i, j in enumerate(J):
                v = cidx.get((d, j))
                if v is None:
                    continue
                pre = v[0] < t
                if not pre.any():
                    continue
                cnt[i] += pre.sum() - (pre & (v[1] == a)).sum()
                if drop_cand_children and d in base:
                    kids = v[2][pre]
                    (m, lc, tcn), _ = base[d]
                    lc2 = lc - bits(A["lang"][kids]).sum(0)
                    tc2 = tcn.copy()
                    np.subtract.at(tc2, A["task"][kids] + 1, 1)
                    override.setdefault(i, {})[d] = _demand(m - len(kids), lc2, tc2)
        out["ds_couse"][s:e] = np.log1p(cnt)
        if not base:
            continue
        out["ds_has"][s:e] = 1
        L0 = np.zeros(NBITS, bool)
        T0 = set()
        for _, (Ld, Td) in base.values():
            L0 |= Ld
            T0 |= Td
        lm = np.full(len(J), _mask(L0), np.uint64)
        tsets = [T0] * len(J)
        for i, od in override.items():
            L, T = np.zeros(NBITS, bool), set()
            for d, (_, dm) in base.items():
                dm = od.get(d, dm) if d in od else dm
                if dm is not None:
                    L |= dm[0]
                    T |= dm[1]
            lm[i] = _mask(L)
            tsets[i] = T
        tj = A["task"][J]
        out["ds_lang"][s:e] = (A["lang"][J] & lm) > 0
        out["ds_task"][s:e] = np.array([x >= 0 and x in T for x, T in zip(tj, tsets)])
    return pd.DataFrame(out, index=rows.index)
