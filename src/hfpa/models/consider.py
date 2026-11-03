"""Choice models beyond the conditional logit, on padded candidate arrays (hfpa.models.clogit.to_padded layout).

consider  attention / consideration (Manzini & Mariotti 2014; Goeree 2008): candidate k enters the developer's
          consideration set independently with probability phi_k = sigmoid(g0 + Z_k g); among the considered,
          logit on v_k = X_k b. With w_k = exp(v_k - log q_k) (the sampled set with the log q shift, as in the
          conditional logit), a = w_j, S = sum_{k != j} B_k w_k, B_k ~ Bernoulli(phi_k):
              P(j) = phi_j * a * E[1 / (a + S)],  E[1/(a+S)] = int_0^inf exp(-t a) prod_k (1 - phi_k + phi_k e^{-t w_k}) dt
          exact up to Gauss-Laguerre quadrature (48 nodes, t scaled by a + E S). phi = 1 gives the conditional
          logit. (Ignores the small chance of an empty consideration set.)
          Popularity may enter Z (attention), X (utility) or both.
nested    nested logit with one dissimilarity lambda = sigmoid(l) in (0, 1] over the nests (e.g. model family);
          inclusive values from the sampled candidates with the log q correction.
Both are fitted by full-batch L-BFGS; standard errors from the exact Hessian (autograd), not clustered.
"""
import numpy as np
import torch


def _t(a, dev, dtype=None):
    return torch.as_tensor(a, device=dev, dtype=dtype)


class Model:
    def __init__(self, kind, n_x, n_z=0, device=None):
        assert kind in ("consider", "nested")
        self.kind, self.nx, self.nz = kind, n_x, n_z
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        k = n_x + (1 + n_z if kind == "consider" else 1)
        self.p = torch.zeros(k, device=self.dev, dtype=torch.float64)
        if kind == "consider":
            self.p.data[n_x] = 2.0  # start near "everything considered" (the conditional logit)
        else:
            self.p.data[n_x] = 3.0  # lambda ~ 0.95
        self.p.requires_grad_(True)
        x, wq = np.polynomial.laguerre.laggauss(48)
        self._gl = (_t(x, self.dev, torch.float64), _t(wq, self.dev, torch.float64))

    def batches(self, D, bs=1000):
        keys = ["X", "lq", "mask", "chosen", "w"] + (["Z"] if self.kind == "consider" else ["nest"])
        C = len(D["chosen"])
        return [{k: _t(D[k][i:i + bs], self.dev) for k in keys} for i in range(0, C, bs)]

    def ll_rows(self, b, p=None):
        p = self.p if p is None else p
        X, lq, mask, j = b["X"].double(), b["lq"].double(), b["mask"], b["chosen"]
        v = X @ p[: self.nx] - lq
        if self.kind == "consider":
            Z = b["Z"].double()
            eta = p[self.nx] + Z @ p[self.nx + 1:]
            logphi = torch.nn.functional.logsigmoid(eta)
            phi = torch.sigmoid(eta)
            v = v.masked_fill(~mask, -torch.inf)
            m = v.max(1, keepdim=True).values.detach()  # scale for stability
            w = torch.exp(v - m)
            a = w.gather(1, j[:, None]).squeeze(1)
            others = mask.clone()
            others.scatter_(1, j[:, None], False)
            mu = (phi * w).masked_fill(~others, 0).sum(1)
            c = a + mu
            # E[1/(a+S)] = int_0^inf exp(-t a) prod_k (1 - phi_k + phi_k exp(-t w_k)) dt, t = s / c, Gauss-Laguerre in s
            s_, om = self._gl
            r = (w / c[:, None])[..., None] * s_  # [C, S, Q]
            lf = torch.log1p(phi[..., None] * torch.expm1(-r)).masked_fill(~others[..., None], 0).sum(1)  # [C, Q]
            lg = s_ * (mu / c)[:, None] + lf
            return logphi.gather(1, j[:, None]).squeeze(1) + torch.log(a) - torch.log(c) + torch.logsumexp(lg + torch.log(om), 1)
        lam = torch.sigmoid(p[self.nx]).clamp(min=1e-3)
        nest = b["nest"]
        v = v.masked_fill(~mask, 0.0)  # finite on padding (no inf * 0 in the lambda gradient)
        C, S = v.shape
        K = int(nest.max()) + 1
        # inclusive values per (choice, nest): log sum exp(v / lam)
        y = v / lam
        m = torch.where(mask, y, torch.full_like(y, -1e30)).max(1, keepdim=True).values.detach()
        e = torch.exp(torch.where(mask, y - m, torch.full_like(y, -1e4))) * mask
        tot = torch.zeros(C, K, device=self.dev, dtype=e.dtype).scatter_add_(1, nest, e)
        iv = torch.log(tot.clamp(min=1e-300)) + m  # [C, K]; empty nests ~ -690 + m
        present = tot > 0
        top = torch.logsumexp((lam * iv).masked_fill(~present, -torch.inf), 1)
        nj = nest.gather(1, j[:, None]).squeeze(1)
        ivj = iv.gather(1, nj[:, None]).squeeze(1)
        return y.gather(1, j[:, None]).squeeze(1) - ivj + lam * ivj - top

    def total(self, B, p=None):
        return sum((b["w"] * self.ll_rows(b, p)).sum() for b in B)

    def fit(self, D, max_iter=500):
        B = self.batches(D)
        opt = torch.optim.LBFGS([self.p], lr=1, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
                                history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            tot = 0.0
            for b in B:
                loss = -(b["w"] * self.ll_rows(b)).sum()
                loss.backward()
                tot += loss.item()
            return torch.tensor(tot)

        opt.step(closure)
        p0 = self.p.detach().clone()
        H = torch.zeros(len(p0), len(p0), device=self.dev, dtype=torch.float64)
        for b in B:
            H += torch.autograd.functional.hessian(lambda q: -(b["w"] * self.ll_rows(b, q)).sum(), p0)
        cov = torch.linalg.pinv(H)
        with torch.no_grad():
            ll = float(self.total(B))
        k = len(p0)
        self.result = dict(coef=p0.cpu().numpy(), se=np.sqrt(np.clip(torch.diagonal(cov).cpu().numpy(), 0, None)),
                           ll=ll, k=k, n=len(D["chosen"]), aic=2 * k - 2 * ll)
        return self.result

    @torch.no_grad()
    def evaluate(self, D):
        return float(torch.cat([self.ll_rows(b) for b in self.batches(D)]).mean())

    @torch.no_grad()
    def phi(self, D):
        """Consideration probability of every candidate row (consider only), [C, S] with NaN padding."""
        out = []
        for b in self.batches(D):
            eta = self.p[self.nx] + b["Z"].double() @ self.p[self.nx + 1:]
            out.append(torch.sigmoid(eta).masked_fill(~b["mask"], float("nan")).cpu().numpy())
        S = max(o.shape[1] for o in out)
        return np.concatenate([np.pad(o, ((0, 0), (0, S - o.shape[1])), constant_values=np.nan) for o in out])


def add_padded(D, rows, cols, name, dtype=np.float64):
    """Add another padded block (e.g. Z or nest ids) with the same layout as to_padded(rows, ...)."""
    cid = rows["choice_id"].to_numpy()
    _, start, counts = np.unique(cid, return_index=True, return_counts=True)
    g = np.repeat(np.arange(len(counts)), counts)
    pos = np.arange(len(cid)) - start[g]
    shape = D["mask"].shape + ((len(cols),) if isinstance(cols, list) else ())
    A = np.zeros(shape, dtype)
    A[g, pos] = rows[cols].to_numpy(dtype)
    D[name] = A
    return D
