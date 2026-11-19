"""Why adding match/inertia makes SIR networks more concentrated: decompose."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import v2_networks as N
from hfpa.features.groups import GROUPS
from hfpa.v2data import load_rows
t0 = time.time()
rows, ch = load_rows(N.SPEC, "planA")
rows = rows.sort_values(["choice_id"], kind="stable").reset_index(drop=True)
feats, b, bfe = N.model_coefs()
X = rows[feats].to_numpy(np.float64)
gi = {g: [i for i, f in enumerate(feats) if f in GROUPS[g]] for g in GROUPS}
I = {g: X[:, gi[g]] @ b[gi[g]] for g in GROUPS}
cid = rows.choice_id.to_numpy(); st = np.r_[0, np.flatnonzero(np.diff(cid)) + 1]; ln = np.diff(np.r_[st, len(cid)])
cand = rows.cand_id.to_numpy(); chosen = rows.chosen.to_numpy() == 1
cid_t = ch.t.loc[cid[st]].to_numpy()
rng = np.random.default_rng(0)
def perm_within(v):  # shuffle a vector within each choice set (keeps its spread, breaks its link to candidates)
    key = np.repeat(np.arange(len(st)), ln) + rng.random(len(v))
    o = np.argsort(key, kind="stable"); out = np.empty_like(v); out[np.sort(o)] = v[o]
    # o sorts by choice then random; positions within choice preserved by np.sort(o)? rebuild explicitly:
    out = np.empty_like(v); out[np.arange(len(v))] = v[o]
    return out
print(f"[{time.time()-t0:.0f}s] rows {len(rows):,}, choices {len(st):,}", flush=True)
deg_rank_key = rows.log_deg.to_numpy() if "log_deg" in rows else X[:, feats.index("log_deg")]
V = {
 "pop": I["pop"],
 "pop+inertia": I["pop"] + I["inertia"],
 "pop+match": I["pop"] + I["match"],
 "pop+match(shuffled)": I["pop"] + perm_within(I["match"]),
 "pop+qual+match+inertia (no FE)": I["pop"] + I["qual"] + I["match"] + I["inertia"],
 "pop+inertia+match(shuffled)": I["pop"] + I["inertia"] + perm_within(I["match"]),
}
res = []
real = cand[chosen]
m = N.metrics(real, cid_t); res.append(dict(model="REAL", ll=np.nan, mean_entropy=0.0, **m))
for name, u in V.items():
    pr, _ = N.sir_probs(rows, u)
    ll = float(np.log(pr[chosen]).sum())
    ent = -np.add.reduceat(pr * np.log(np.clip(pr, 1e-300, None)), st).mean()
    # expected-count concentration (no draw noise)
    ec = pd.Series(pr).groupby(cand).sum(); p = ec.to_numpy() / ec.sum()
    eff_exp = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    # mass on top-10 overall parents of the REAL network
    top_real = pd.Series(real).value_counts().index[:10]
    ms = []
    for s in range(3):
        pos = N.draw(rows, pr, st, np.random.default_rng(100 + s))
        ms.append(N.metrics(cand[pos], cid_t))
    mm = pd.DataFrame(ms).mean().to_dict()
    res.append(dict(model=name, ll=ll, mean_entropy=ent, eff_expected=eff_exp,
                    mass_on_real_top10=float(p[ec.index.isin(top_real)].sum()), **mm))
    print(f"[{time.time()-t0:.0f}s] {res[-1]}", flush=True)
R = pd.DataFrame(res)
# alignment of match with popularity inside choice sets
def wdemean(v):
    mu = np.add.reduceat(v, st) / ln; return v - np.repeat(mu, ln)
a, c = wdemean(I["match"]), wdemean(I["pop"])
corr = float((a * c).sum() / np.sqrt((a * a).sum() * (c * c).sum()))
# among each choice set: top candidate by match index -> its pop rank within the set
pop_rank = pd.Series(I["pop"]).groupby(np.repeat(np.arange(len(st)), ln)).rank(ascending=False, method="first").to_numpy()
best_match = st + np.array([np.argmax(I["match"][s:s+l]) for s, l in zip(st, ln)])
deg = np.exp(X[:, feats.index("log_deg")]) - 1
print("\nwithin-choice corr(match index, pop index) =", round(corr, 3))
print("pop rank (1 = most popular in D) of the best-matched candidate: median",
      np.median(pop_rank[best_match]), "share in top 10:", (pop_rank[best_match] <= 10).mean().round(3))
print("same for the REAL chosen: median", np.median(pop_rank[chosen]), "share in top 10:", (pop_rank[chosen] <= 10).mean().round(3))
print("set size median", np.median(ln))
pd.set_option("display.width", 250)
print(R[["model","ll","mean_entropy","eff_expected","mass_on_real_top10","top10","eff_parents","gini","n_parents"]].round(4).to_string())
R.to_csv("/tmp/claude-1000038/-home-caroyf/044faf89-843f-4965-9207-845055269347/scratchpad/conc_decomp.csv", index=False)
