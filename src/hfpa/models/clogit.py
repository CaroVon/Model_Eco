"""Conditional logit with sampling correction (plan 3.2, 7.3, 7.4).

u_kj = theta' x_kj - log q_kj + phi_{fe(kj)}, with phi_0 = 0 (non-hubs share the baseline).
Fit by full-batch L-BFGS (gradients accumulated over choice batches); standard errors from
the analytic Hessian of the conditional logit, H = sum_k w_k (E_p[zz'] - E_p[z]E_p[z]').
With choice weights or repeated choices by one author, pass `cluster` to fit() for
cluster-robust (sandwich) SEs of the non-FE coefficients: cov = sum_g (H^-1 s_g)(H^-1 s_g)'.

Two data layouts:
  to_padded(rows, ...)   one dense array [C, S_max, F] (float64). Fine for the 10% sample.
  to_buckets(rows, ...)  choices grouped by candidate-set size (sizes within a bucket differ by <= tol), each bucket
                         padded to its own maximum; features stored as float32, resident on the GPU; utilities,
                         log-likelihood, gradients and Hessians computed / accumulated in float64. For the full data.
Both are accepted by CLogit.fit / evaluate / ll_per_choice. Choice weights live in the data; set_weights() swaps them
(multiplier bootstrap) without moving the data.
"""
import numpy as np
import torch
from scipy.stats import chi2

torch.set_default_dtype(torch.float64)
KEYS = ("X", "lq", "mask", "chosen", "fe", "w", "idx")


def _layout(rows):
    cid = rows["choice_id"].to_numpy()
    assert np.all(np.diff(cid) >= 0), "rows must be sorted by choice_id"
    ids, start, counts = np.unique(cid, return_index=True, return_counts=True)
    return ids, start, counts


def to_padded(rows, feat_cols, fe_col=None, use_log_q=True, weight=None, offset_col=None):
    """rows: candidate rows sorted by choice_id -> dict of padded arrays [C, S, ...].
    offset_col: fixed utility term with coefficient 1 (e.g. fixed effects estimated elsewhere)."""
    ids, start, counts = _layout(rows)
    C, S = len(counts), int(counts.max())
    g = np.repeat(np.arange(C), counts)
    pos = np.arange(len(g)) - start[g]
    X = np.zeros((C, S, len(feat_cols)))
    X[g, pos] = rows[feat_cols].to_numpy(np.float64)
    lq = np.zeros((C, S))
    if use_log_q:
        lq[g, pos] = rows["log_q"].to_numpy(np.float64)
    if offset_col is not None:
        lq[g, pos] -= rows[offset_col].to_numpy(np.float64)  # utility subtracts lq
    mask = np.zeros((C, S), bool)
    mask[g, pos] = True
    chosen = rows["chosen"].to_numpy() == 1
    assert np.array_equal(np.bincount(g[chosen], minlength=C), np.ones(C)), "need exactly one chosen per choice"
    fe = np.zeros((C, S), np.int64)
    if fe_col is not None:
        fe[g, pos] = rows[fe_col].to_numpy(np.int64)
    w = np.ones(C) if weight is None else np.asarray(weight, np.float64)
    return dict(X=X, lq=lq, mask=mask, chosen=pos[chosen], fe=fe, w=w, names=list(feat_cols), idx=np.arange(C),
                choice_ids=ids)


class Buckets:
    """Choices grouped by candidate-set size, resident on `device` (see module docstring).
    .batches: list of dicts X [c,s,F] float32, lq [c,s] float64, mask, chosen [c], fe [c,s], w [c], idx [c] (position
    of each choice in choice_ids order). .n choices, .names features, .choice_ids sorted choice ids."""

    def __init__(self, batches, names, choice_ids, device):
        self.batches, self.names, self.choice_ids, self.device = batches, names, choice_ids, device
        self.n = len(choice_ids)

    def set_weights(self, w):
        w = torch.as_tensor(np.asarray(w, np.float64), device=self.device)
        assert len(w) == self.n
        for b in self.batches:
            b["w"] = w[b["idx"]]

    def subset_feats(self, cols):
        """View with a subset of feature columns (indices into names); shares lq / mask / fe."""
        j = [self.names.index(c) for c in cols]
        jt = torch.as_tensor(j, device=self.device, dtype=torch.long)
        nb = [{**b, "X": b["X"].index_select(2, jt) if len(j) else b["X"][..., :0]} for b in self.batches]
        return Buckets(nb, list(cols), self.choice_ids, self.device)

    def rows_total(self):
        return sum(int(b["mask"].sum()) for b in self.batches)


def to_buckets(rows, feat_cols, fe_col=None, use_log_q=True, weight=None, offset_col=None, tol=0.1, max_cells=3_000_000,
               device=None, extra_int=None):
    """Bucketed layout of to_padded. Choices sorted by set size; a new bucket starts when the size exceeds the
    bucket's smallest size by more than `tol` or the bucket would exceed `max_cells` padded cells.
    extra_int: {key: column} integer row columns carried along as [c, s] tensors (e.g. nest ids)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ids, start, counts = _layout(rows)
    C = len(counts)
    X_all = rows[list(feat_cols)].to_numpy(np.float32) if len(feat_cols) else np.zeros((len(rows), 0), np.float32)
    lq_all = rows["log_q"].to_numpy(np.float64) if use_log_q else np.zeros(len(rows))
    if offset_col is not None:
        lq_all = lq_all - rows[offset_col].to_numpy(np.float64)
    fe_all = rows[fe_col].to_numpy(np.int64) if fe_col is not None else np.zeros(len(rows), np.int64)
    ch_all = rows["chosen"].to_numpy() == 1
    w_all = np.ones(C) if weight is None else np.asarray(weight, np.float64)
    ext = {k: rows[c].to_numpy(np.int64) for k, c in (extra_int or {}).items()}
    order = np.argsort(counts, kind="stable")
    batches, i = [], 0
    while i < C:
        s0 = counts[order[i]]
        j = i
        while j < C and counts[order[j]] <= max(s0 * (1 + tol), s0 + 1) and (j - i + 1) * counts[order[j]] <= max_cells:
            j += 1
        sel = order[i:j]
        S = int(counts[sel].max())
        cnt = counts[sel]
        g = np.repeat(np.arange(len(sel)), cnt)
        pos = np.arange(len(g)) - np.repeat(np.r_[0, np.cumsum(cnt)[:-1]], cnt)
        src = np.repeat(start[sel], cnt) + pos  # row positions
        X = np.zeros((len(sel), S, X_all.shape[1]), np.float32)
        X[g, pos] = X_all[src]
        lq = np.zeros((len(sel), S))
        lq[g, pos] = lq_all[src]
        mask = np.zeros((len(sel), S), bool)
        mask[g, pos] = True
        fe = np.zeros((len(sel), S), np.int64)
        fe[g, pos] = fe_all[src]
        chm = ch_all[src]
        assert np.array_equal(np.bincount(g[chm], minlength=len(sel)), np.ones(len(sel))), "need exactly one chosen"
        chosen = pos[chm]
        t = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)
        b = dict(X=t(X), lq=t(lq), mask=t(mask), chosen=t(chosen, torch.long), fe=t(fe, torch.long),
                 w=t(w_all[sel]), idx=t(sel, torch.long))
        for k, a in ext.items():
            E_ = np.zeros((len(sel), S), np.int64)
            E_[g, pos] = a[src]
            b[k] = t(E_, torch.long)
        batches.append(b)
        i = j
    return Buckets(batches, list(feat_cols), ids, device)


def _fe_index(b):
    """Flat positions of the cells with fe > 0 and their FE column (fe - 1); cached in the batch dict."""
    if "fe_nz" not in b:
        f = b["fe"].reshape(-1)
        nz = torch.nonzero(f > 0).squeeze(1)
        b["fe_nz"], b["fe_v"] = nz, f[nz] - 1
    return b["fe_nz"], b["fe_v"]


def _fe_mass(b, pr, K):
    """sum of pr over the cells of each FE id, per choice: [C, K]."""
    nz, v = _fe_index(b)
    C, S = pr.shape
    return torch.zeros(C * K, device=pr.device, dtype=pr.dtype).index_add_(0, (nz // S) * K + v,
                                                                         pr.reshape(-1)[nz]).view(C, K)


class CLogit:
    def __init__(self, n_feat, n_fe=0, l2_fe=0.0, device=None):
        self.F, self.n_fe, self.l2 = n_feat, n_fe, l2_fe
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.p = torch.zeros(n_feat + n_fe, device=self.device, requires_grad=True)

    def _batches(self, D, bs):
        if isinstance(D, Buckets):
            yield from D.batches
            return
        C = len(D["chosen"])
        for i in range(0, C, bs):
            sl = slice(i, i + bs)
            b = {k: torch.as_tensor(D[k][sl], device=self.device) for k in ("X", "lq", "mask", "chosen", "fe", "w")}
            b["idx"] = torch.arange(i, min(i + bs, C), device=self.device)
            yield b

    def utility(self, b, p=None):
        p = self.p if p is None else p
        X = b["X"]
        u = (X.double() if X.dtype != torch.float64 else X) @ p[: self.F] - b["lq"]
        if self.n_fe:  # only cells with fe > 0 (phi_0 = 0): no atomic contention on id 0 in the backward pass
            nz, v = _fe_index(b)
            u = u.reshape(-1).index_add(0, nz, p[self.F:][v]).view_as(u)
        return u.masked_fill(~b["mask"], -torch.inf)

    def _ll_batch(self, b, p=None):
        u = self.utility(b, p)
        ll = u.gather(1, b["chosen"][:, None]).squeeze(1) - torch.logsumexp(u, 1)
        return ll

    def init(self, coef):
        """Warm start: coef aligned with the parameter vector (non-FE first)."""
        with torch.no_grad():
            self.p.copy_(torch.as_tensor(np.asarray(coef, np.float64), device=self.device))

    def fit(self, D, bs=20000, max_iter=500, tol=1e-9, cluster=None, se=True):
        batches = D.batches if isinstance(D, Buckets) else list(self._batches(D, bs))
        opt = torch.optim.LBFGS([self.p], lr=1, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-12,
                                history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total = 0.5 * self.l2 * (self.p[self.F:] ** 2).sum()
            total.backward()
            tot = total.item()
            for b in batches:
                loss = -(b["w"] * self._ll_batch(b)).sum()
                loss.backward()
                tot += loss.item()
            return torch.tensor(tot)

        opt.step(closure)
        n_eval = opt.state[opt._params[0]].get("func_evals", None)
        names = (D.names if isinstance(D, Buckets) else D["names"]) + [f"fe{i}" for i in range(1, self.n_fe + 1)]
        n = D.n if isinstance(D, Buckets) else len(D["chosen"])
        k = self.F + self.n_fe
        with torch.no_grad():
            ll = sum((b["w"] * self._ll_batch(b)).sum().item() for b in batches)
        self.result = dict(names=names, coef=self.p.detach().cpu().numpy().copy(), ll=ll, k=k, n=n,
                           aic=2 * k - 2 * ll, bic=k * np.log(n) - 2 * ll, func_evals=n_eval)
        if not se:
            return self.result
        with torch.no_grad():
            H = sum(self._hessian(b) for b in batches)
        if self.n_fe:
            H[self.F:, self.F:] += self.l2 * torch.eye(self.n_fe, device=self.device)
        cov = torch.linalg.pinv(H)
        se_ = torch.sqrt(torch.clamp(torch.diagonal(cov), min=0))
        se_cl = np.full(k, np.nan)
        cov_cl = None
        if cluster is not None:
            cov_cl = self._cluster_se(batches, cov[: self.F], np.asarray(cluster))
            se_cl[: self.F] = np.sqrt(np.clip(np.diag(cov_cl), 0, None))
        self.result.update(se=se_.cpu().numpy(), se_cl=se_cl, cov_cl=cov_cl, cov=cov.cpu().numpy())
        return self.result

    @torch.no_grad()
    def _cluster_se(self, batches, A, cluster):
        """A = rows of H^-1 for the non-FE coefficients [F, k]; per-choice scores projected by A.
        cluster: one label per choice, in choice order (positions idx)."""
        _, cl = np.unique(cluster, return_inverse=True)
        clt = torch.as_tensor(cl, device=self.device)
        G = torch.zeros(int(cl.max()) + 1, self.F, device=self.device)
        for b in batches:
            pr = torch.softmax(self.utility(b), 1)
            X, w = b["X"].double(), b["w"]
            xc = X.gather(1, b["chosen"][:, None, None].expand(-1, 1, self.F)).squeeze(1)
            proj = (w[:, None] * (xc - (pr[..., None] * X).sum(1))) @ A[:, : self.F].T
            if self.n_fe:
                P = _fe_mass(b, pr, self.n_fe)
                fc = b["fe"].gather(1, b["chosen"][:, None]).squeeze(1)
                onehot = torch.zeros_like(P)
                hit = torch.nonzero(fc > 0).squeeze(1)
                onehot[hit, fc[hit] - 1] = 1.0
                proj = proj + (w[:, None] * (onehot - P)) @ A[:, self.F:].T
            G.index_add_(0, clt[b["idx"]], proj)
        g = G.shape[0]
        V = G.T @ G * (g / max(g - 1, 1))
        return V.cpu().numpy()  # cluster-robust covariance of the non-FE coefficients

    def _hessian(self, b):
        pr = torch.softmax(self.utility(b), 1) * b["w"][:, None]  # [C,S], weighted
        X = b["X"].double()
        m = (pr[..., None] * X).sum(1)  # E_p[x] * w
        wsum = b["w"].clamp(min=1e-300)
        Hxx = torch.einsum("cs,csf,csg->fg", pr, X, X) - torch.einsum("cf,cg,c->fg", m, m, 1 / wsum)
        if not self.n_fe:
            return Hxx
        K = self.n_fe
        nz, v = _fe_index(b)
        P = _fe_mass(b, pr, K)  # [C, K] mass per FE id (ids 1..K -> columns 0..K-1)
        A = torch.zeros(K, self.F, device=self.device).index_add_(0, v, pr.reshape(-1)[nz, None] * X.reshape(-1, self.F)[nz])
        Hxf = (A - (P / wsum[:, None]).T @ m).T
        Hff = torch.diag(P.sum(0)) - (P / wsum[:, None]).T @ P
        return torch.cat([torch.cat([Hxx, Hxf], 1), torch.cat([Hxf.T, Hff], 1)], 0)

    @torch.no_grad()
    def ll_per_choice(self, D, bs=20000):
        """Unweighted log-likelihood of each choice, in choice order."""
        n = D.n if isinstance(D, Buckets) else len(D["chosen"])
        out = np.empty(n)
        for b in self._batches(D, bs):
            out[b["idx"].cpu().numpy()] = self._ll_batch(b).cpu().numpy()
        return out

    @torch.no_grad()
    def evaluate(self, D, bs=20000, topk=(1, 10)):
        """Mean log-likelihood and top-k hit rate (rank of chosen among candidates, by u without -log q;
        tied utilities are broken at random)."""
        lls, hits = [], {k: [] for k in topk}
        for b in self._batches(D, bs):
            lls.append(self._ll_batch(b).cpu().numpy())
            u = self.utility({**b, "lq": torch.zeros_like(b["lq"])})
            uc = u.gather(1, b["chosen"][:, None])
            greater = (u > uc).sum(1).cpu().numpy()
            ties = ((u == uc) & b["mask"]).sum(1).cpu().numpy()  # includes the chosen alternative
            for k in topk:  # expected hit rate when ties are broken at random
                hits[k].append(np.clip((k - greater) / ties, 0, 1))
        return dict(mean_ll=float(np.concatenate(lls).mean()),
                    **{f"top{k}": float(np.concatenate(v).mean()) for k, v in hits.items()})


def uniform_ll(D):
    """Per-choice log-likelihood of uniform attachment within the sampled set (u = -log q), choice order."""
    n = D.n if isinstance(D, Buckets) else len(D["chosen"])
    out = np.empty(n)
    m = CLogit(0, device=D.device if isinstance(D, Buckets) else "cpu")
    with torch.no_grad():
        for b in m._batches(D, 20000):
            out[b["idx"].cpu().numpy()] = m._ll_batch({**b, "X": b["X"][..., :0]}).cpu().numpy()
    return out


def fit_subset(D, cols, fe=False, n_fe=0, init=None, l2_fe=1e-4, max_iter=500, se=False, cluster=None):
    """Fit a clogit on a feature subset of Buckets D (optionally with FE); returns (model, result).
    With no features and no FE: uniform attachment (no fit)."""
    sub = D.subset_feats(cols)
    k_fe = n_fe if fe else 0
    m = CLogit(len(cols), n_fe=k_fe, l2_fe=l2_fe if k_fe else 0.0, device=D.device)
    if not cols and not k_fe:
        with torch.no_grad():
            ll = sum((b["w"] * m._ll_batch({**b, "X": b["X"][..., :0]})).sum().item() for b in sub.batches)
        return m, dict(ll=ll, coef=np.zeros(0), k=0, n=D.n, names=[])
    if init is not None:
        m.init(init)
    return m, m.fit(sub, max_iter=max_iter, se=se, cluster=cluster)


def multiplier_weights(author, base_w, rng):
    """Bayesian / multiplier bootstrap: one xi_g ~ Exp(1) per author, w_k = base_w_k * xi_{a(k)}."""
    _, a = np.unique(np.asarray(author), return_inverse=True)
    xi = rng.exponential(1.0, a.max() + 1)
    return np.asarray(base_w, np.float64) * xi[a]


def lr_test(ll_small, ll_big, df_diff):
    stat = 2 * (ll_big - ll_small)
    return stat, chi2.sf(stat, df_diff)


def summary(res, keep=None):
    import pandas as pd
    df = pd.DataFrame({"coef": res["coef"], "se": res["se"], "se_cl": res.get("se_cl")}, index=res["names"])
    df["z"] = df.coef / df.se
    return df if keep is None else df.loc[keep]
