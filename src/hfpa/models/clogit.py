"""Conditional logit with sampling correction (plan 3.2, 7.3, 7.4).

u_kj = theta' x_kj - log q_kj + phi_{fe(kj)}, with phi_0 = 0 (non-hubs share the baseline).
Fit by full-batch L-BFGS (gradients accumulated over choice batches); standard errors from
the analytic Hessian of the conditional logit, H = sum_k w_k (E_p[zz'] - E_p[z]E_p[z]').
With choice weights or repeated choices by one author, pass `cluster` to fit() for
cluster-robust (sandwich) SEs of the non-FE coefficients: cov = sum_g (H^-1 s_g)(H^-1 s_g)'.
"""
import numpy as np
import torch
from scipy.stats import chi2

torch.set_default_dtype(torch.float64)


def to_padded(rows, feat_cols, fe_col=None, use_log_q=True, weight=None, offset_col=None):
    """rows: candidate rows sorted by choice_id -> dict of padded arrays [C, S, ...].
    offset_col: fixed utility term with coefficient 1 (e.g. fixed effects estimated elsewhere)."""
    cid = rows["choice_id"].to_numpy()
    assert np.all(np.diff(cid) >= 0), "rows must be sorted by choice_id"
    _, start, counts = np.unique(cid, return_index=True, return_counts=True)
    C, S = len(counts), int(counts.max())
    g = np.repeat(np.arange(C), counts)
    pos = np.arange(len(cid)) - start[g]
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
    return dict(X=X, lq=lq, mask=mask, chosen=pos[chosen], fe=fe, w=w, names=list(feat_cols))


class CLogit:
    def __init__(self, n_feat, n_fe=0, l2_fe=0.0, device=None):
        self.F, self.n_fe, self.l2 = n_feat, n_fe, l2_fe
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.p = torch.zeros(n_feat + n_fe, device=self.device, requires_grad=True)

    def _batches(self, D, bs):
        C = len(D["chosen"])
        for i in range(0, C, bs):
            sl = slice(i, i + bs)
            yield {k: torch.as_tensor(D[k][sl], device=self.device) for k in ("X", "lq", "mask", "chosen", "fe", "w")}

    def utility(self, b, p=None):
        p = self.p if p is None else p
        u = b["X"] @ p[: self.F] - b["lq"]
        if self.n_fe:
            phi = torch.cat([torch.zeros(1, device=self.device), p[self.F:]])
            u = u + phi[b["fe"]]
        return u.masked_fill(~b["mask"], -torch.inf)

    def _ll_batch(self, b, p=None):
        u = self.utility(b, p)
        ll = u.gather(1, b["chosen"][:, None]).squeeze(1) - torch.logsumexp(u, 1)
        return ll

    def fit(self, D, bs=20000, max_iter=500, tol=1e-9, cluster=None):
        batches = list(self._batches(D, bs))
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
        with torch.no_grad():
            ll = sum((b["w"] * self._ll_batch(b)).sum().item() for b in batches)
            H = sum(self._hessian(b) for b in batches)
        if self.n_fe:
            H[self.F:, self.F:] += self.l2 * torch.eye(self.n_fe, device=self.device)
        cov = torch.linalg.pinv(H)
        se = torch.sqrt(torch.clamp(torch.diagonal(cov), min=0))
        k, n = self.F + self.n_fe, len(D["chosen"])
        se_cl = np.full(k, np.nan)
        cov_cl = None
        if cluster is not None:
            cov_cl = self._cluster_se(batches, cov[: self.F], np.asarray(cluster), bs)
            se_cl[: self.F] = np.sqrt(np.clip(np.diag(cov_cl), 0, None))
        self.result = dict(names=D["names"] + [f"fe{i}" for i in range(1, self.n_fe + 1)],
                           coef=self.p.detach().cpu().numpy(), se=se.cpu().numpy(), se_cl=se_cl, cov_cl=cov_cl, ll=ll, k=k, n=n,
                           aic=2 * k - 2 * ll, bic=k * np.log(n) - 2 * ll)
        return self.result

    @torch.no_grad()
    def _cluster_se(self, batches, A, cluster, bs):
        """A = rows of H^-1 for the non-FE coefficients [F, k]; per-choice scores projected by A."""
        _, cl = np.unique(cluster, return_inverse=True)
        G = torch.zeros(int(cl.max()) + 1, self.F, device=self.device)
        for i, b in enumerate(batches):
            pr = torch.softmax(self.utility(b), 1)
            X, w = b["X"], b["w"]
            xc = X.gather(1, b["chosen"][:, None, None].expand(-1, 1, self.F)).squeeze(1)
            proj = (w[:, None] * (xc - (pr[..., None] * X).sum(1))) @ A[:, : self.F].T
            if self.n_fe:
                P = torch.zeros(X.shape[0], self.n_fe + 1, device=self.device).scatter_add_(1, b["fe"], pr)
                onehot = torch.zeros_like(P).scatter_(1, b["fe"].gather(1, b["chosen"][:, None]), 1.0)
                proj = proj + (w[:, None] * (onehot - P))[:, 1:] @ A[:, self.F:].T
            G.index_add_(0, torch.as_tensor(cl[i * bs: i * bs + len(w)], device=self.device), proj)
        g = G.shape[0]
        V = G.T @ G * (g / max(g - 1, 1))
        return V.cpu().numpy()  # cluster-robust covariance of the non-FE coefficients

    def _hessian(self, b):
        pr = torch.softmax(self.utility(b), 1) * b["w"][:, None]  # [C,S], weighted
        X = b["X"]
        m = (pr[..., None] * X).sum(1)  # E_p[x] * w
        wsum = b["w"]
        Hxx = torch.einsum("cs,csf,csg->fg", pr, X, X) - torch.einsum("cf,cg,c->fg", m, m, 1 / wsum)
        if not self.n_fe:
            return Hxx
        K = self.n_fe + 1
        P = torch.zeros(X.shape[0], K, device=self.device).scatter_add_(1, b["fe"], pr)  # mass per hub
        A = torch.zeros(K, self.F, device=self.device).index_add_(0, b["fe"].reshape(-1),
                                                                   (pr[..., None] * X).reshape(-1, self.F))
        Hxf = (A - (P / wsum[:, None]).T @ m)[1:].T
        Hff = (torch.diag(P.sum(0)) - (P / wsum[:, None]).T @ P)[1:, 1:]
        return torch.cat([torch.cat([Hxx, Hxf], 1), torch.cat([Hxf.T, Hff], 1)], 0)

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


def lr_test(ll_small, ll_big, df_diff):
    stat = 2 * (ll_big - ll_small)
    return stat, chi2.sf(stat, df_diff)


def summary(res, keep=None):
    import pandas as pd
    df = pd.DataFrame({"coef": res["coef"], "se": res["se"], "se_cl": res.get("se_cl")}, index=res["names"])
    df["z"] = df.coef / df.se
    return df if keep is None else df.loc[keep]
