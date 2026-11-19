import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import v2_networks as N
from hfpa.features.groups import GROUPS
from hfpa.v2data import load_rows
rows, ch = load_rows(N.SPEC, "planA")
rows = rows.sort_values(["choice_id"], kind="stable").reset_index(drop=True)
feats, b, bfe = N.model_coefs()
X = rows[feats].to_numpy(np.float64); fe = rows.fe.to_numpy(); phi = np.r_[0.0, bfe[len(feats):]]
gi = {g: [i for i, f in enumerate(feats) if f in GROUPS[g]] for g in GROUPS}
I = {g: X[:, gi[g]] @ bfe[gi[g]] for g in GROUPS}  # FE-model coefficients
cid = rows.choice_id.to_numpy(); st = np.r_[0, np.flatnonzero(np.diff(cid)) + 1]
cand = rows.cand_id.to_numpy(); chosen = rows.chosen.to_numpy() == 1; cid_t = ch.t.loc[cid[st]].to_numpy()
V = {"FULL (FE model)": I["pop"] + I["qual"] + I["match"] + I["inertia"] + phi[fe],
     "FULL without FE term": I["pop"] + I["qual"] + I["match"] + I["inertia"],
     "pop + FE": I["pop"] + phi[fe],
     "pop + match + FE": I["pop"] + I["match"] + phi[fe],
     "pop + inertia + FE": I["pop"] + I["inertia"] + phi[fe]}
out = []
for k, u in V.items():
    pr, _ = N.sir_probs(rows, u)
    m = pd.DataFrame([N.metrics(cand[N.draw(rows, pr, st, np.random.default_rng(100 + s))], cid_t) for s in range(3)]).mean()
    out.append(dict(model=k, ll=float(np.log(pr[chosen]).sum()), top10=m.top10, eff_parents=m.eff_parents, gini=m.gini, n_parents=m.n_parents))
print(pd.DataFrame(out).round(4).to_string())
# who are the singletons: degree of parents adopted once, real vs pop-only SIR
pr, _ = N.sir_probs(rows, X[:, gi["pop"]] @ b[gi["pop"]])
sim = cand[N.draw(rows, pr, st, np.random.default_rng(100))]; real = cand[chosen]
for nm, a in (("REAL", real), ("pop SIR", sim)):
    c = pd.Series(a).value_counts()
    print(nm, "parents", len(c), "adopted once", int((c == 1).sum()), "share of choices to once-parents", round((c == 1).sum() / len(a), 4),
          "share to parents with <=5", round(c[c <= 5].sum() / len(a), 4))
