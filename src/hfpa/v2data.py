"""Shared loading and estimation helpers for the v2 scripts (scripts/v2_*.py).

Samples (rows from processed/choices/<spec>/feats/*.parquet, choices from decisions.parquet):
  planA      first decision of each author-parent pair; the author's earlier parents (author_reuse = 1, the
             certainty stratum H_k) are removed from the choice set (a first choice cannot be a reuse).
  decisions  every first upload of a 24h decision (v2_pps only): first and repeat decisions, H_k kept.
Train = choices before the train / holdout cut (decisions.holdout False); holdout = after.
"""
import glob
import os
import itertools
from math import factorial

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hfpa import paths
from hfpa.features.groups import ALL, GROUPS, SIGNAL
from hfpa.models.clogit import CLogit, fit_subset, to_buckets, uniform_ll

TABLES = paths.TABLES / "v2"
BASE_COLS = ["choice_id", "cand_id", "chosen", "log_q", "stratum", "fe"]


def cdir(spec):
    return paths.PROCESSED / "choices" / spec


def load_choices(spec):
    return pd.read_parquet(cdir(spec) / "decisions.parquet").set_index("choice_id", drop=False)


def load_rows(spec, sample="planA", cols=None, extra=()):
    ch = load_choices(spec)
    ids = ch.index[ch.pair_first.to_numpy()] if sample == "planA" else ch.index[ch.dfirst24.to_numpy()]
    cols = list(dict.fromkeys(BASE_COLS + (cols or ALL + SIGNAL) + ["author_reuse"] + list(extra)))
    parts = []
    dev = os.environ.get("V2_DEV_GLOB")  # development only: e.g. "2023-0[1-3].notext.parquet"
    for f in sorted(glob.glob(str(cdir(spec) / "feats" / (dev or "*.parquet")))):
        if f.endswith(".notext.parquet") and not dev:
            continue
        have = pq.read_schema(f).names
        if not have:
            continue
        x = pd.read_parquet(f, columns=[c for c in cols if c in have])
        x = x[x.choice_id.isin(ids)]
        if sample == "planA":
            x = x[x.author_reuse.to_numpy() == 0]
        parts.append(x)
    rows = pd.concat(parts, ignore_index=True)
    ok = rows.groupby("choice_id").chosen.transform("max").to_numpy() == 1
    rows = rows[ok].reset_index(drop=True)
    assert np.all(np.diff(rows.choice_id.to_numpy()) >= 0)
    return rows, ch


def split(rows, ch):
    h = ch.holdout.reindex(rows.choice_id).to_numpy()
    return rows[~h].reset_index(drop=True), rows[h].reset_index(drop=True)


def drop_constant(rows, feats):
    sd = rows[feats].std()
    return [f for f in feats if f in sd and sd[f] > 0]


def shapley(v, players):
    n = len(players)
    phi = {}
    for p in players:
        others = [q for q in players if q != p]
        s = 0.0
        for k in range(n):
            for S in itertools.combinations(others, k):
                wgt = factorial(k) * factorial(n - k - 1) / factorial(n)
                s += wgt * (v(frozenset(S) | {p}) - v(frozenset(S)))
        phi[p] = s
    return phi


def subsets(players):
    for k in range(len(players) + 1):
        yield from itertools.combinations(players, k)


def shapley_fits(D, feats, groups=GROUPS, fe_player=False, n_fe=0, Dh=None, warm=None, log=print):
    """Fit every subset of the groups (and the FE as a fifth player if fe_player); warm start from the fit of the
    subset without the last added group (or from `warm`: {subset: coef} of an earlier run).
    Returns {subset: dict(ll, coef, hold_ll)} with frozenset keys; hold_ll = holdout mean log-likelihood."""
    players = list(groups) + (["fe"] if fe_player else [])
    out = {}
    for S in subsets(players):
        key = frozenset(S)
        cols = [f for f in feats if any(f in groups[g] for g in S if g != "fe")]
        use_fe = "fe" in S
        init = None
        if warm is not None and key in warm:
            init = warm[key]
        elif S:
            prev = out[frozenset(S[:-1])]
            prev_cols = [f for f in feats if any(f in groups[g] for g in S[:-1] if g != "fe")]
            init = np.zeros(len(cols) + (n_fe if use_fe else 0))
            pc = dict(zip(prev_cols, prev["coef"][: len(prev_cols)]))
            init[: len(cols)] = [pc.get(c, 0.0) for c in cols]
            if use_fe and "fe" in S[:-1]:
                init[len(cols):] = prev["coef"][len(prev_cols):]
        m, r = fit_subset(D, cols, fe=use_fe, n_fe=n_fe, init=init)
        rec = dict(ll=r["ll"], coef=np.asarray(r["coef"]))
        if Dh is not None:
            if cols or use_fe:
                rec["hold_ll"] = float(m.ll_per_choice(Dh.subset_feats(cols)).mean())
            else:
                rec["hold_ll"] = float(uniform_ll(Dh).mean())
        out[key] = rec
        if log:
            log(f"  {sorted(S)}: ll {r['ll']:.1f}" + (f" hold {rec['hold_ll']:.4f}" if Dh is not None else ""))
    return out


def shares(fits, players, partition=None, key="ll"):
    """Shapley shares of the gain over the empty model. partition: {name: [players]} coarser players."""
    part = partition or {p: [p] for p in players}
    base = fits[frozenset()][key]
    total = fits[frozenset(players)][key] - base
    v = lambda S: fits[frozenset(sum((part[x] for x in S), []))][key] - base
    phi = shapley(v, list(part))
    return {k: dict(value=val, share=val / total) for k, val in phi.items()}, total


def write(df, name, title, note=""):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(f"# {title}\n\n" + (note + "\n\n" if note else "") +
                                        df.to_markdown(index=False, floatfmt=".4g") + "\n")
    print(df.to_string(), flush=True)
