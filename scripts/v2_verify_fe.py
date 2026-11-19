"""Check the FE rewrite of hfpa.models.clogit on the full data: refit the v2_pps plan-A model with the 500 FE
(same start: no-FE coefficients, FE = 0) and compare with the fit saved by v2_main --part clogit (old code).
Writes results/tables/v2/v2_verify_fe.md."""
import pickle
import time

import numpy as np
import pandas as pd

from hfpa.models.clogit import CLogit
from hfpa.v2data import cdir, write
from v2_main import N_FE, data

C = pickle.load(open(cdir("v2_pps") / "v2_est" / "clogit.pkl", "rb"))
D, Dh, feats, author, ch = data("v2_pps")
assert feats == C["feats"]
t = time.time()
m = CLogit(len(feats), n_fe=N_FE, l2_fe=1e-4)
m.init(np.r_[C["nofe"]["coef"], np.zeros(N_FE)])
r = m.fit(D, cluster=author)
dt = time.time() - t
old, new = C["fe"]["coef"], r["coef"]
F = len(feats)
se_old = C["fe"]["se"]
rows = [dict(quantity="fit time with the new code (s), incl. Hessian and clustered SE", value=dt),
        dict(quantity="max |coef new - old|, 26 features", value=np.abs(new[:F] - old[:F]).max()),
        dict(quantity="max |coef new - old| / SE, 26 features", value=(np.abs(new[:F] - old[:F]) / se_old[:F]).max()),
        dict(quantity="max |phi new - old|, 500 FE", value=np.abs(new[F:] - old[F:]).max()),
        dict(quantity="max |phi new - old| / SE, 500 FE", value=(np.abs(new[F:] - old[F:]) / se_old[F:]).max()),
        dict(quantity="max |SE new - old| / SE old, all", value=(np.abs(r["se"] - se_old) / se_old).max()),
        dict(quantity="max |clustered SE new - old| / old, 26 features",
             value=(np.abs(r["se_cl"][:F] - np.sqrt(np.diag(C["fe"]["cov_cl"]))) / np.sqrt(np.diag(C["fe"]["cov_cl"]))).max())]
write(pd.DataFrame(rows), "v2_verify_fe", "FE rewrite check: new code vs the saved fit (v2_pps plan A, 500 FE)")
