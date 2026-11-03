"""P19: uncertainty and robustness of the P15 shares (plan A, Shapley over the log-likelihood gain).

--boot K    author-clustered bootstrap replicates K*B_PER .. K*B_PER+B_PER-1 (authors drawn with replacement,
            a choice weighted by its author's draw count; coefficients re-estimated in every replicate).
            Without --fe: the 2^7 fits of the seven groups (partitions seven groups and C).
            With --fe: also the 2^5 fits with fit (quality + match) and the hub FE, for partition C with fit
            as one player (upper bound), and the P17-adjusted upper bound (s_total fixed at its P17 value).
            Writes results/tables/p19/boot[_fe]_<K>.csv.
--spec      the robustness matrix: one Shapley (2^7 fits, no hub FE) per specification; writes p19/spec_<name>.csv
            (--only NAME runs one).
--collect   percentile CIs for the bootstrap, the per-choice classification bootstrapped by author (fixed
            coefficients), the spec table and a forest plot of the fit / popularity ratio.
usage: sbatch --array=0-19 slurm/gpu.sbatch scripts/p19_robust.py --boot   (task id from SLURM_ARRAY_TASK_ID)
       sbatch --array=0-39 slurm/gpu.sbatch scripts/p19_robust.py --boot --fe --per 1
       sbatch slurm/gpu.sbatch scripts/p19_robust.py --spec
       python scripts/p19_robust.py --collect
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.models.clogit import CLogit, to_padded
from p15_decomp import GROUPS, PARTITIONS, P5, P9, drop_constant, load, sample, shapley

OUT = paths.TABLES / "p19"
B_PER = 10
PART_C = "C: visibility and availability separate"
S_TOTAL = 0.424  # P17 non-quality share of the hub-FE variance (publisher premium + pair noise), post-P18
BULK = 100       # an author with >= BULK main choices is a bulk account


def ll_fit(tr, feats, w, fe=False):
    """Weighted log-likelihood of a conditional logit on training rows (uniform attachment if no features)."""
    if not feats and not fe:
        return float((w * _m0(tr)).sum())
    D = to_padded(tr, feats, fe_col="fe" if fe else None, weight=w)
    n_fe = int(tr.fe.max()) if fe else 0
    return CLogit(len(feats), n_fe=n_fe, l2_fe=1e-4 if n_fe else 0.0).fit(D)["ll"]


def _m0(tr):
    """Per-choice log-likelihood of uniform attachment within the sampled set (u = -log q), sorted by choice."""
    lq = tr.log_q.to_numpy(np.float64)
    cid = tr.choice_id.to_numpy()
    _, start = np.unique(cid, return_index=True)
    lse = np.log(np.add.reduceat(np.exp(-lq), start))
    return -lq[tr.chosen.to_numpy() == 1] - lse


def shares(tr, w, fe=False):
    """Shapley shares, partitions 'seven groups' and C; with fe also C with the hub FE in fit (upper bound)."""
    feats = drop_constant(tr, sum(GROUPS.values(), []))
    ll = {}
    for k in range(len(GROUPS) + 1):
        for S in itertools.combinations(GROUPS, k):
            ll[frozenset(S)] = ll_fit(tr, [f for f in feats if any(f in GROUPS[x] for x in S)], w)
    out = []
    base, total = ll[frozenset()], ll[frozenset(GROUPS)] - ll[frozenset()]
    for pname in ("seven groups", PART_C):
        part = PARTITIONS[pname]
        phi = shapley(lambda S, part=part: ll[frozenset(sum((part[x] for x in S), []))] - base, list(part))
        out += [dict(partition=pname, group=g, share=v / total) for g, v in phi.items()]
    if fe:  # fit = quality + match with the hub FE; the other players as in C
        part = PARTITIONS[PART_C]
        llf = {}
        for S, v in ll.items():
            names = [p for p in part if set(part[p]) <= S]
            if sum(len(part[p]) for p in names) != len(S):
                continue
            if "fit" in names:
                fs = [f for f in feats if any(f in GROUPS[x] for x in S)]
                v = ll_fit(tr, fs, w, fe=True)
            llf[frozenset(names)] = v
        tot = llf[frozenset(part)] - base
        phi = shapley(lambda S: llf[frozenset(S)] - base, list(part))
        out += [dict(partition="C + hub FE", group=g, share=v / tot) for g, v in phi.items()]
    D = pd.DataFrame(out)
    piv = D.set_index(["partition", "group"]).share
    extra = [dict(partition=PART_C, group="ratio fit / popularity", share=piv[PART_C, "fit"] / piv[PART_C, "popularity"])]
    if fe:
        f1, p1, f0 = piv["C + hub FE", "fit"], piv["C + hub FE", "popularity"], piv[PART_C, "fit"]
        shift = S_TOTAL * (f1 - f0)  # non-quality part of the FE share, to popularity (P17 linear split)
        extra += [dict(partition="C + hub FE", group="ratio fit / popularity", share=f1 / p1),
                  dict(partition="C + hub FE, P17-adjusted", group="fit", share=f1 - shift),
                  dict(partition="C + hub FE, P17-adjusted", group="ratio fit / popularity", share=(f1 - shift) / (p1 + shift))]
    return pd.concat([D, pd.DataFrame(extra)], ignore_index=True)


def prep(rows, ch, which="planA", gamma=None):
    r, g = sample(rows, ch, which)
    tr, _ = P5.split(r, ch)
    tr = tr.reset_index(drop=True)
    cid = np.unique(tr.choice_id.to_numpy())
    g = g if gamma is None else gamma
    return tr, cid, ch.n_pair.loc[cid].to_numpy(float) ** -g


def boot(rows, ch, task, fe, per=B_PER):
    tr, cid, _ = prep(rows, ch)
    auth = ch.author.loc[cid].to_numpy()
    ua, inv = np.unique(auth, return_inverse=True)
    out = []
    for b in range(task * per, (task + 1) * per):
        t0 = time.time()
        rng = np.random.default_rng(1000 + b)
        cnt = np.bincount(rng.integers(0, len(ua), len(ua)), minlength=len(ua))
        w = cnt[inv].astype(float)
        keep = w > 0
        sub = tr[tr.choice_id.isin(cid[keep]).to_numpy()].reset_index(drop=True)
        out.append(shares(sub, w[keep], fe=fe).assign(rep=b))
        print(f"rep {b} in {time.time() - t0:.0f}s", flush=True)
        pd.concat(out).to_csv(OUT / f"boot{'_fe' if fe else ''}_{task}.csv", index=False)


def specs(rows, ch):
    """name -> (training rows, weights). Each changes one thing relative to plan A."""
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["author_code", "major_org"])
    full = pd.read_parquet(paths.PROCESSED / "choices" / "main_strat" / "choices.parquet", columns=["child"])
    n_auth = pd.Series(nodes.author_code.to_numpy()[full.child]).value_counts()
    tr, cid, w = prep(rows, ch)
    year = pd.to_datetime(ch.t.loc[cid], unit="s").dt.year.to_numpy()
    mod = ch.modality_name.loc[cid].to_numpy()
    bulk = n_auth.reindex(ch.author.loc[cid]).fillna(0).to_numpy() >= BULK
    org = nodes.major_org.to_numpy()[ch.child.loc[cid]] > 0

    def keep(mask):
        return lambda: (tr[tr.choice_id.isin(cid[mask]).to_numpy()].reset_index(drop=True), w[mask])

    def halve():
        rng = np.random.default_rng(0)
        drop = (tr.stratum.to_numpy() != P9.HUB) & (tr.chosen.to_numpy() == 0) & (rng.random(len(tr)) < 0.5)
        t = tr[~drop].copy()
        t["log_q"] = np.where(t.stratum.to_numpy() != P9.HUB, t.log_q.to_numpy() + np.log(0.5), t.log_q.to_numpy()).astype(np.float32)
        return t.reset_index(drop=True), w

    def used_only():  # consideration set: hubs and models already used (no zero-degree candidates)
        z = tr.stratum.to_numpy() == 2
        bad = tr.choice_id[z & (tr.chosen.to_numpy() == 1)].unique()
        t = tr[~z & ~tr.choice_id.isin(bad).to_numpy()].reset_index(drop=True)
        return t, w[~np.isin(cid, bad)]

    def plan_b():
        t, c, wb = prep(rows, ch, "all")
        return t, wb

    return {"baseline": lambda: (tr, w), "negatives halved": halve, "used candidates only": used_only,
            "no bulk accounts": keep(~bulk), "all choices (n^-0.5)": plan_b,
            "2022-2023": keep(year <= 2023), "2024": keep(year == 2024), "2025": keep(year >= 2025),
            "text": keep(mod == "text"), "image generation": keep(mod == "image_gen"),
            "other modalities": keep((mod != "text") & (mod != "image_gen")),
            "major-org authors": keep(org), "other authors": keep(~org)}


def collect():
    from p15_decomp import SPEC
    B = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("boot_[0-9]*.csv"))], ignore_index=True)
    rows = []
    fe_files = sorted(OUT.glob("boot_fe_*.csv"))
    if fe_files:
        F = pd.concat([pd.read_csv(f) for f in fe_files], ignore_index=True)
        B = pd.concat([B, F[F.partition.str.startswith("C + hub FE")]], ignore_index=True)
    for (p, g), d in B.groupby(["partition", "group"]):
        rows.append(dict(partition=p, group=g, reps=len(d), boot_mean=d.share.mean(), ci95_lo=d.share.quantile(0.025),
                         ci95_hi=d.share.quantile(0.975), boot_sd=d.share.std()))
    P5.write(pd.DataFrame(rows), "p19_boot", "P19 author-clustered bootstrap of the Shapley shares (plan A; coefficients re-estimated)")

    C = pd.read_parquet(paths.PROCESSED / "choices" / SPEC / "p15_classify.parquet")
    ch = pd.read_parquet(paths.PROCESSED / "choices" / SPEC / "choices.parquet").set_index("choice_id")
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["author_code"])
    a = nodes.author_code.to_numpy()[ch.child.loc[C.index]]
    ua, inv = np.unique(a, return_inverse=True)
    types = sorted(C["type"].unique())
    T = np.stack([(C["type"].to_numpy() == t) for t in types], 1).astype(float)
    sig = (C["type"].to_numpy() != "no demand signal").astype(float)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(1000):
        w = np.bincount(rng.integers(0, len(ua), len(ua)), minlength=len(ua))[inv]
        draws.append(np.r_[(w[:, None] * T).sum(0) / w.sum(), (w[:, None] * T * sig[:, None]).sum(0) / (w * sig).sum()])
    draws = np.array(draws)
    est = np.r_[T.mean(0), (T * sig[:, None]).sum(0) / sig.sum()]
    lab = [(t, "all") for t in types] + [(t, "with signal") for t in types]
    cl = pd.DataFrame([dict(type=t, base=bs, share=e, ci95_lo=lo, ci95_hi=hi) for (t, bs), e, lo, hi in
                       zip(lab, est, np.quantile(draws, 0.025, 0), np.quantile(draws, 0.975, 0))])
    cl = cl[~((cl.base == "with signal") & (cl["type"] == "no demand signal"))]
    P5.write(cl, "p19_classify_boot", "P19 per-choice classification shares, author-clustered bootstrap (fixed coefficients)")

    sp = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("spec_*.csv"))], ignore_index=True)
    P5.write(sp, "p19_spec", "P19 robustness matrix: Shapley shares by specification (plan A, no hub FE)")
    forest(sp, pd.DataFrame(rows))


def forest(sp, bt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = sp[(sp.partition == PART_C) & (sp.group == "ratio fit / popularity")].set_index("spec")
    order = [s for s in SPEC_ORDER if s in r.index]
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(order) + 1.2))
    y = np.arange(len(order))[::-1]
    ax.scatter(r.share.loc[order], y, s=36, color="#2a6fdb", zorder=3)
    for yi, s in zip(y, order):
        ax.text(r.share.loc[s] + 0.004, yi, f"{r.share.loc[s]:.2f}  (n={int(r.n.loc[s]):,})", va="center", fontsize=8, color="#444")
    b = bt[(bt.partition == PART_C) & (bt.group == "ratio fit / popularity")]
    if len(b):
        ax.axvspan(b.ci95_lo.iloc[0], b.ci95_hi.iloc[0], color="#2a6fdb", alpha=0.12, lw=0, label="baseline 95% bootstrap CI")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_yticks(y, order, fontsize=9)
    ax.set_xlabel("fit / popularity (Shapley, partition C, lower bound)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#ddd", lw=0.6)
    fig.tight_layout()
    fig.savefig(paths.FIGURES / "p19_ratio_forest.png", dpi=160)


SPEC_ORDER = ["baseline", "negatives halved", "used candidates only", "no bulk accounts", "all choices (n^-0.5)",
              "2022-2023", "2024", "2025", "text", "image generation", "other modalities", "major-org authors", "other authors"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--fe", action="store_true")
    ap.add_argument("--per", type=int, default=B_PER, help="replicates per array task")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.collect:
        collect()
    else:
        rows, ch = load()
        if a.boot:
            boot(rows, ch, a.task, a.fe, a.per)
        if a.spec:
            for name, make in specs(rows, ch).items():
                if a.only and name != a.only:
                    continue
                t0 = time.time()
                tr, w = make()
                D = shares(tr, w).assign(spec=name, n=len(w))
                D.to_csv(OUT / f"spec_{SPEC_ORDER.index(name):02d}.csv", index=False)
                print(name, len(w), D[D.group == "ratio fit / popularity"].share.round(3).tolist(), f"{time.time() - t0:.0f}s", flush=True)
