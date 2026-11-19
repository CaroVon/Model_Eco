"""v2 stage 7: robustness (F, H, I.1, J, K of the v2 plan).

--text     F: txt_sim coefficient and the match Shapley share as the minimum token count of the card text varies
           (5 main, 10, 20, 50; recomputed from txt_raw / txt_ntok).
--mrs      H: marginal rates of substitution vs log degree and vs recent heat (continuous features incl. txt_sim at
           one SD; delta-method SEs from the clustered covariance).
--inherit  H: inheritance check with the random comparison parent drawn from OTHER choices of the cell (never the
           choice's own parent), plus Cohen's kappa = (identical - chance) / (1 - chance).
--pairs    I.1: identical-weight copies (P12 hash-identical pairs) both present in a plan-A candidate set with one of
           them chosen: two-alternative conditional logit of the choice on the popularity differences (log q offset).
           If the more popular copy wins although the weights are the same, the preference is for popularity itself.
--signal   K: Shapley shares on choices with a demand signal, on all choices, and on the signal choices reweighted to
           the composition of all choices (author experience bins x major organisation).
--cards    J: card versions of the 50 largest FE parents before their heavy use (HF API, read-only), txt_sim recomputed
           with the old cards for the plan-A rows of those parents.
usage: sbatch slurm/gpu.sbatch scripts/v2_robust.py --text --mrs --pairs --signal ; python scripts/v2_robust.py --inherit
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.features.groups import ALL, GROUPS, MAIN, NAMES, SIGNAL, group_of
from hfpa.models.clogit import CLogit, to_buckets
from hfpa.v2data import cdir, drop_constant, load_rows, shapley_fits, shares, split, write

SPEC = "v2_pps"
T0 = time.time()
SD_FEATS = ["log_deg", "lrec30", "log_age", "bench_z", "size_dist", "lco2", "ds_couse", "lquant", "txt_sim"]


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def train():
    rows, ch = load_rows(SPEC, "planA", cols=ALL + SIGNAL + ["txt_raw", "txt_ntok"])
    tr, _ = split(rows, ch)
    return tr, ch


def part_text(tr):
    out = []
    for k in (5, 10, 20, 50):
        R = tr.copy()
        na = R.txt_ntok.to_numpy() < k
        R["txt_na"] = na.astype(np.float32)
        R["txt_sim"] = np.where(na, 0, R.txt_raw).astype(np.float32)
        feats = drop_constant(R, ALL)
        D = to_buckets(R, feats)
        m = CLogit(len(feats))
        res = m.fit(D, cluster=None, se=True)
        i = feats.index("txt_sim")
        fits = shapley_fits(D, feats, log=None)
        sh, _ = shares(fits, list(GROUPS))
        has = R.groupby("choice_id").txt_ntok.max() >= k
        out.append(dict(min_tokens=k, txt_sim_coef=res["coef"][i], txt_sim_se=res["se"][i],
                        share_choices_with_text=has.mean(), match_share=sh["match"]["share"], pop_share=sh["pop"]["share"]))
        log(out[-1])
        del D
    write(pd.DataFrame(out), "v2_text_threshold", "F: card-text similarity by minimum token count (plan A, v2)")


def part_mrs(tr):
    C = pickle.load(open(cdir(SPEC) / "v2_est" / "clogit.pkl", "rb"))
    feats, b, V = C["feats"], C["nofe"]["coef"], C["nofe"]["cov_cl"]
    out = []
    for ref in ("log_deg", "lrec30"):
        j = feats.index(ref)
        for i, f in enumerate(feats):
            if f == ref:
                continue
            dx = float(tr[f].std()) if f in SD_FEATS else 1.0
            ratio = b[i] * dx / b[j]
            g = np.zeros(len(feats))
            g[i], g[j] = dx / b[j], -b[i] * dx / b[j] ** 2
            se = float(np.sqrt(g @ V @ g))
            out.append(dict(ref=ref, group=group_of(f), feature=f, dx=dx, ratio=ratio, se=se, multiplier=np.exp(ratio),
                            mult_lo=np.exp(ratio - 1.96 * se), mult_hi=np.exp(ratio + 1.96 * se)))
    write(pd.DataFrame(out), "v2_mrs", "Marginal rates of substitution (v2 plan-A model; multiplier on 1+degree or 1+recent)")


def part_inherit():
    import sys
    sys.path.insert(0, str(paths.REPO / "scripts"))
    from p13_inherit import FIELDS, as_key, same
    rng = np.random.default_rng(42)
    meta = pd.read_parquet(paths.INTERIM / "meta.parquet", columns=["model_id", "languages", "datasets", "licenses"])
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["model_id", "pipeline_tag", "library_name", "params_b", "author_code"])
    nodes = nodes.join(meta.set_index("model_id"), on="model_id")
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet", columns=["child", "parent", "t", "modality", "pair_first"])
    ch["month"] = (ch.t // (30 * 86400)).astype(int)
    par_all = ch.parent.to_numpy()
    rp = np.full(len(ch), -1)
    for _, idx in ch.groupby(["modality", "month"]).indices.items():
        for i in idx:
            for _ in range(20):  # another choice of the cell whose parent is not this choice's own parent
                j = idx[rng.integers(len(idx))]
                if j != i and par_all[j] != par_all[i]:
                    rp[i] = par_all[j]
                    break
    ch["rand_parent"] = rp
    out = []
    for sample, sub in [("all choices", ch), ("first author-parent choices", ch[ch.pair_first])]:
        sub = sub[sub.rand_parent >= 0]
        for f in FIELDS:
            col = nodes[f].to_numpy()
            c = [as_key(v, f) for v in col[sub.child]]
            p = [as_key(v, f) for v in col[sub.parent]]
            r = [as_key(v, f) for v in col[sub.rand_parent]]
            both = np.array([a is not None and b is not None for a, b in zip(c, p)])
            bothr = np.array([a is not None and b is not None for a, b in zip(c, r)])
            ident = np.mean([same(a, b, f) for a, b, ok in zip(c, p, both) if ok]) if both.any() else np.nan
            chance = np.mean([same(a, b, f) for a, b, ok in zip(c, r, bothr) if ok]) if bothr.any() else np.nan
            out.append(dict(sample=sample, field=f, both_has=both.mean(), identical=ident, chance=chance,
                            excess=ident - chance, cohen_kappa=(ident - chance) / (1 - chance) if chance < 1 else np.nan))
            log(out[-1])
    write(pd.DataFrame(out), "v2_inherit", "Inheritance check (H): random comparison parent from other choices of the same "
          "modality x month (never the choice's own parent); Cohen's kappa")


def part_pairs(tr):
    P = pd.read_csv(paths.TABLES / "p12_mirror_pairs.csv")
    H = pd.read_csv(paths.TABLES / "p12_hash_pairs.csv")
    pairs = set(map(tuple, P.loc[P.hash_match == "identical", ["orig", "cpy"]].to_numpy()))
    pairs |= set(map(tuple, H.loc[H["match"] == "identical", ["parent_idx", "child_idx"]].to_numpy()))
    other = {}
    for a, b in pairs:
        other.setdefault(a, set()).add(b)
        other.setdefault(b, set()).add(a)
    R = tr[tr.cand_id.isin(list(other))]
    rec = []
    for k, g in R.groupby("choice_id"):
        if g.chosen.sum() != 1:
            continue
        c = g[g.chosen == 1].iloc[0]
        mates = g[g.cand_id.isin(other.get(int(c.cand_id), set()))]
        if len(mates) == 0:
            continue
        o = mates.iloc[0]
        rec.append(dict(choice_id=k, d_log_deg=c.log_deg - o.log_deg, d_lrec30=c.lrec30 - o.lrec30,
                        d_redist=c.redistributor - o.redistributor, d_lq=c.log_q - o.log_q,
                        chosen_more_popular=float(c.log_deg > o.log_deg)))
    D = pd.DataFrame(rec)
    out = [dict(quantity="choices with both copies in the set and one chosen", value=len(D))]
    if len(D) >= 20:
        import statsmodels.api as sm
        # two-alternative logit: P(chosen) = sigma(beta . dx - dlq); symmetric design: chosen vs the other copy
        y = np.r_[np.ones(len(D)), np.zeros(len(D))]
        X = np.r_[D[["d_log_deg", "d_lrec30", "d_redist"]].to_numpy(), -D[["d_log_deg", "d_lrec30", "d_redist"]].to_numpy()]
        off = np.r_[-D.d_lq.to_numpy(), D.d_lq.to_numpy()]
        keep = [i for i in range(X.shape[1]) if X[:, i].std() > 0]
        r = sm.GLM(y, X[:, keep], family=sm.families.Binomial(), offset=off).fit()
        names = np.array(["d_log_deg", "d_lrec30", "d_redist"])[keep]
        for nm, c_, s_ in zip(names, r.params, r.bse):
            out.append(dict(quantity=f"coef {nm}", value=c_, se=s_))
        out.append(dict(quantity="share of choices where the chosen copy has the higher degree",
                        value=D.chosen_more_popular[D.d_log_deg != 0].mean()))
    write(pd.DataFrame(out), "v2_identical_pairs", "I.1: identical-weight copies in the same candidate set (popularity "
          "without a quality difference)")


def part_signal(tr, ch):
    feats = drop_constant(tr, ALL)
    sig = tr.groupby("choice_id")[SIGNAL].max().max(1) > 0
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["birth", "author_code", "major_org"])
    c = ch.loc[sig.index]
    order = np.argsort(nodes.birth.to_numpy(), kind="stable")
    prior = pd.Series(np.arange(len(nodes))).groupby(nodes.author_code.to_numpy()[order]).cumcount().to_numpy()
    exp = np.empty(len(nodes))
    exp[order] = prior
    cell = pd.Series(np.minimum(np.floor(np.log2(1 + exp[c.child.to_numpy()])), 8).astype(int).astype(str), index=c.index) + \
        "|" + nodes.major_org.to_numpy()[c.child.to_numpy()].astype(int).astype(str)
    allc = cell.value_counts(normalize=True)
    sigc = cell[sig].value_counts(normalize=True)
    wcell = (allc / sigc).reindex(cell[sig]).to_numpy()
    out = []
    for lab, ids, w in (("all choices", sig.index, None), ("choices with a demand signal", sig.index[sig], None),
                        ("signal choices reweighted to all", sig.index[sig], wcell)):
        R = tr[tr.choice_id.isin(ids)]
        D = to_buckets(R, feats, weight=w)
        fits = shapley_fits(D, feats, log=None)
        for pname, part in (("four groups", None), ("pop / fit / inertia", {"popularity": ["pop"], "fit": ["qual", "match"],
                                                                         "inertia": ["inertia"]})):
            sh, tot = shares(fits, list(GROUPS), part)
            for g, v in sh.items():
                out.append(dict(sample=lab, partition=pname, group=NAMES.get(g, g), share=v["share"], n=D.n))
        log(f"{lab} done")
        del D
    write(pd.DataFrame(out), "v2_signal_samples", "K: shares on all choices, on choices with a demand signal, and on the "
          "signal choices reweighted to the composition of all choices (author experience x major organisation)",
          "Fit (demand match) can only be detected where a demand signal exists; on choices without a signal the match "
          "share is 0 by construction.")


def part_cards(tr):
    """J: old card versions (before the parent's 50th child) of the 50 largest FE parents."""
    from huggingface_hub import HfApi, hf_hub_download
    from hfpa.features.text import clean, name_vocab_v2, tokens
    import scipy.sparse as sp
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "birth"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["child_idx", "parent_idx", "rel_type"])
    m = E[E.rel_type.isin(MAIN)]
    top = tr[tr.fe > 0].groupby("cand_id").chosen.sum().sort_values(ascending=False).index[:50]
    api = HfApi()
    birth = nodes.birth.to_numpy()
    out = []
    for j in top:
        mid = nodes.model_id.iat[j]
        kids = np.sort(birth[m.child_idx[m.parent_idx == j].to_numpy()])
        cut = int(kids[min(49, len(kids) - 1)]) if len(kids) else None
        try:
            commits = api.list_repo_commits(mid)
            old = [c for c in commits if cut is None or c.created_at.timestamp() <= cut]
            rev = old[0].commit_id if old else commits[-1].commit_id
            f = hf_hub_download(mid, "README.md", revision=rev)
            out.append(dict(node=j, model_id=mid, revision=rev, n_commits=len(commits), text=open(f).read()))
        except Exception as e:  # noqa: BLE001  (gated / deleted repos)
            out.append(dict(node=j, model_id=mid, error=str(e)[:200]))
        log(f"{mid}: {out[-1].get('revision', out[-1].get('error'))}")
    O = pd.DataFrame(out)
    O.to_parquet(cdir(SPEC) / "v2_est" / "old_cards.parquet", index=False)
    ok = O.dropna(subset=["revision"]) if "revision" in O else O.iloc[:0]
    if len(ok) == 0:
        write(O[["model_id", "error"]], "v2_old_cards", "J: old card versions could not be fetched (no network?)")
        return
    # txt_sim with the old candidate card, same vocabulary as the v2 TF-IDF is not stored -> re-fit on the small set
    X = sp.load_npz(paths.INTERIM / "v2_text_tfidf.npz").tocsr()
    T = np.load(paths.INTERIM / "v2_text_rows.npz")
    deg = m.parent_idx.value_counts()
    names = name_vocab_v2(nodes.model_id.to_numpy()[deg.index[deg >= 20]])
    rows = tr[tr.cand_id.isin(ok.node)]
    ch = pd.read_parquet(cdir(SPEC) / "decisions.parquet").set_index("choice_id")
    kid = ch.child.reindex(rows.choice_id).to_numpy()
    res = dict(rows=len(rows), parents=len(ok), new_mean_sim_chosen=rows.loc[rows.chosen == 1, "txt_sim"].mean(),
               new_mean_sim_other=rows.loc[rows.chosen == 0, "txt_sim"].mean())
    # old-card similarity: TF-IDF fitted on child docs of these rows plus the old cards (vocabulary of this subset)
    from hfpa.features.text import tfidf
    from hfpa.features.text import row_cosine as rc
    C = pd.read_parquet(paths.INTERIM / "v2_cards_clean.parquet").set_index("node")
    kids_u = np.unique(kid)
    docs = [tokens(list(C.paras.get(k, [])), names) for k in kids_u] + [tokens(clean(t), names) for t in ok.text]
    Xs, _ = tfidf(docs, min_df=2)
    kpos = {k: i for i, k in enumerate(kids_u)}
    ppos = {n_: len(kids_u) + i for i, n_ in enumerate(ok.node)}
    a = np.array([kpos[k] for k in kid])
    b = np.array([ppos[c] for c in rows.cand_id])
    s_old = rc(Xs, a, b)
    na = rows.txt_na.to_numpy() > 0
    res["old_mean_sim_chosen"] = float(s_old[(rows.chosen.to_numpy() == 1) & ~na].mean())
    res["old_mean_sim_other"] = float(s_old[(rows.chosen.to_numpy() == 0) & ~na].mean())
    res["corr_new_old"] = float(np.corrcoef(rows.txt_sim.to_numpy()[~na], s_old[~na])[0, 1])
    write(pd.DataFrame([res]).T.reset_index().rename(columns={"index": "quantity", 0: "value"}), "v2_old_cards",
          "J: similarity with the card version before the parent's 50th child (50 largest FE parents)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("text", "mrs", "inherit", "pairs", "signal", "cards"):
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    if a.inherit:
        part_inherit()
    if a.text or a.mrs or a.pairs or a.signal or a.cards:
        tr, ch = train()
        if a.mrs:
            part_mrs(tr)
        if a.pairs:
            part_pairs(tr)
        if a.text:
            part_text(tr)
        if a.signal:
            part_signal(tr, ch)
        if a.cards:
            part_cards(tr)
