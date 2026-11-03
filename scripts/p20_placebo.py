"""P20: falsification and out-of-time checks of the demand-match (fit) features (plan A).

--placebo  every plan-A choice gets the demand of a donor: another choice of a different author in the same
           month x modality cell. Author history, dataset profile (hfpa.features.match.build with the donor's
           child; times stay the choice's own) and card text (P18 TF-IDF with the donor's card) are rebuilt.
           Candidates, popularity and everything else stay the choice's own. Donor demand shares the month's
           fashions, so the placebo is not exactly zero; it should be far below the real one.
           Reports coefficients (real, placebo instead of real, both together) and the Shapley shares
           (p19_robust.shares) with the placebo features in place of the real ones.
--oot      out-of-time validation (paper section 3, tables 3-4): fit on plan-A choices before an origin, evaluate
           on the next window; mean log-likelihood gain over uniform attachment and top-1 / top-10 accuracy for
           nested models, and the out-of-sample gain from adding fit to everything else.
usage: sbatch slurm/gpu.sbatch scripts/p20_placebo.py --placebo | --oot
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from hfpa import paths
from hfpa.features.match import build, node_attrs
from hfpa.features.text import row_cosine
from hfpa.models.clogit import CLogit, to_padded
from p15_decomp import GROUPS, P5, drop_constant, load, sample
from p19_robust import _m0, shares

MATCH_COLS = ["size_dist", "size_na", "task_hist", "lang_hist", "ds_lang", "ds_task"]
TXT_COLS = ["txt_sim", "txt_na"]
ALL = sum(GROUPS.values(), [])


def donors(ch, cid, seed=0):
    """Donor choice for each plan-A choice: same month x modality, different author; NaN if none."""
    c = ch.loc[cid]
    cell = pd.to_datetime(c.t, unit="s").dt.strftime("%Y-%m") + "|" + c.modality_name.astype(str)
    rng = np.random.default_rng(seed)
    out = pd.Series(np.nan, index=cid)
    for _, ids in c.groupby(cell.to_numpy()).groups.items():
        ids = np.asarray(ids)
        if len(ids) < 2:
            continue
        auth = c.author.loc[ids].to_numpy()
        for _ in range(20):  # random permutations until no choice keeps its own author
            perm = rng.permutation(len(ids))
            if (auth[perm] != auth).all():
                break
        ok = auth[perm] != auth
        out.loc[ids[ok]] = ids[perm][ok]
    return out


def text_sims(kid, cand):
    X = sp.load_npz(paths.INTERIM / "text_tfidf.npz").tocsr()
    R = np.load(paths.INTERIM / "text_rows.npz")
    crow, prow, ok_doc = R["crow"], R["prow"], R["ok_doc"]
    a, b = crow[kid], prow[cand]
    ok = (a >= 0) & (b >= 0)
    ok[ok] = ok_doc[a[ok]] & ok_doc[b[ok]]
    s = np.zeros(len(kid), np.float32)
    s[ok] = row_cosine(X, a[ok], b[ok])
    return s, (~ok).astype(np.float32)


def coefs(tr, w, feats, tag):
    D = to_padded(tr, feats, weight=w)
    res = CLogit(len(feats)).fit(D)
    return [dict(model=tag, param=f, coef=res["coef"][i], se=res["se"][i]) for i, f in enumerate(feats)
            if f in MATCH_COLS + TXT_COLS or f.startswith("pl_")], res["ll"]


def part_placebo(rows, ch):
    t0 = time.time()
    r, _ = sample(rows, ch, "planA")
    tr, _ = P5.split(r, ch)
    tr = tr.reset_index(drop=True)
    cid = np.unique(tr.choice_id.to_numpy())
    dn = donors(ch, cid)
    keep = dn.notna().to_numpy()
    print(f"{keep.sum():,} of {len(cid):,} choices have a donor", flush=True)
    tr = tr[tr.choice_id.isin(cid[keep]).to_numpy()].reset_index(drop=True)
    cid, dn = cid[keep], dn[keep].astype(int)
    w = np.ones(len(cid))

    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["model_id", "pipeline_tag", "params_b", "license_cat", "major_org", "author_code"])
    meta = pd.read_parquet(paths.INTERIM / "meta.parquet", columns=["model_id", "languages", "datasets"]).set_index("model_id")
    meta = meta.reindex(nodes.model_id)
    A = node_attrs(nodes, meta.languages.tolist(), meta.datasets.tolist())
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child", "parent", "t"])
    chp = ch.loc[cid, ["child", "t"]].copy()
    chp["child"] = ch.child.loc[dn.to_numpy()].to_numpy()  # donor's child (its author and datasets), own time
    M = build(tr, chp, A, full, nodes.author_code.to_numpy())
    kid = chp.child.loc[tr.choice_id].to_numpy()
    s, na = text_sims(kid, tr.cand_id.to_numpy())
    print(f"placebo features in {time.time() - t0:.0f}s", flush=True)
    P = tr.copy()
    for c in MATCH_COLS:
        P[c] = M[c].to_numpy(np.float32)
    P["txt_sim"], P["txt_na"] = s, na
    for c in MATCH_COLS + TXT_COLS:
        tr[f"pl_{c}"] = P[c].to_numpy()
    # description: share of choices whose donor has a signal; mean feature among chosen, real vs placebo
    desc = []
    for c in MATCH_COLS + TXT_COLS:
        ch1 = tr.chosen.to_numpy() == 1
        desc.append(dict(feature=c, real_chosen=tr.loc[ch1, c].mean(), real_other=tr.loc[~ch1, c].mean(),
                         placebo_chosen=P.loc[ch1, c].mean(), placebo_other=P.loc[~ch1, c].mean()))
    P5.write(pd.DataFrame(desc), "p20_placebo_desc", "P20 match features among chosen vs other candidates, real vs donor demand")

    feats = drop_constant(tr, ALL)
    pl = [f"pl_{c}" for c in MATCH_COLS + TXT_COLS if tr[f"pl_{c}"].std() > 0]
    out = []
    for tag, fs, data in [("real", feats, tr), ("placebo instead of real", feats, P),
                          ("real + placebo", feats + pl, tr)]:
        o, ll = coefs(data, w, fs, tag)
        out += [dict(x, ll=ll) for x in o]
        print(tag, round(ll, 1), flush=True)
    P5.write(pd.DataFrame(out), "p20_placebo_coef", "P20 demand-match coefficients with real vs donor (placebo) demand (plan A, same choices)")
    sh = pd.concat([shares(tr, w).assign(demand="real"), shares(P, w).assign(demand="placebo")], ignore_index=True)
    P5.write(sh, "p20_placebo_shapley", "P20 Shapley shares with real vs donor (placebo) demand (plan A, same choices)")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


NESTED = {"popularity": ["pop"], "+ visibility": ["pop", "base"],
          "+ availability, inertia, local (no fit)": ["pop", "base", "avail", "inertia", "local"],
          "full (with fit)": list(GROUPS), "fit only": ["qual", "match"]}


def part_oot(rows, ch):
    r, _ = sample(rows, ch, "planA")
    r = r.reset_index(drop=True)
    cid = np.unique(r.choice_id.to_numpy())
    t = ch.t.loc[cid].to_numpy()
    out = []
    for q in (0.5, 0.6, 0.7, 0.8):
        lo, hi = np.quantile(t, q), np.quantile(t, q + 0.1)
        tr = r[r.choice_id.isin(cid[t < lo]).to_numpy()].reset_index(drop=True)
        te = r[r.choice_id.isin(cid[(t >= lo) & (t < hi)]).to_numpy()].reset_index(drop=True)
        feats_all = drop_constant(tr, ALL)
        base = _m0(te).mean()
        for name, grps in NESTED.items():
            fs = [f for f in feats_all if any(f in GROUPS[g] for g in grps)]
            m = CLogit(len(fs))
            m.fit(to_padded(tr, fs))
            ev = m.evaluate(to_padded(te, fs))
            out.append(dict(origin=pd.to_datetime(lo, unit="s").strftime("%Y-%m-%d"), n_train=tr.choice_id.nunique(),
                            n_test=te.choice_id.nunique(), model=name, mean_ll=ev["mean_ll"], gain_over_uniform=ev["mean_ll"] - base,
                            top1=ev["top1"], top10=ev["top10"]))
            print(out[-1], flush=True)
    D = pd.DataFrame(out)
    piv = D.pivot(index="origin", columns="model", values="gain_over_uniform")
    D2 = pd.DataFrame({"origin": piv.index, "oos_gain_from_fit": piv["full (with fit)"] - piv["+ availability, inertia, local (no fit)"],
                       "oos_share_of_fit_in_gain": (piv["full (with fit)"] - piv["+ availability, inertia, local (no fit)"]) / piv["full (with fit)"]})
    P5.write(D, "p20_oot", "P20 out-of-time validation: fit before origin, test on the next 10% of plan-A choices in time")
    P5.write(D2, "p20_oot_fit", "P20 out-of-sample log-likelihood gain from adding fit (quality + match) to all other groups")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--oot", action="store_true")
    a = ap.parse_args()
    rows, ch = load()
    if a.placebo:
        part_placebo(rows, ch)
    if a.oot:
        part_oot(rows, ch)
