"""Two-nest logit for repeated decisions (v2 part 2.4): nest 1 = parents the author used before (H_k, all included,
pi = 1), nest 0 = every other model (Poisson-sampled with inclusion probability pi).

  P(j) = P(B) P(j | B),  P(j | B) = exp(V_j / lam_B) / sum_{l in B} exp(V_l / lam_B)
  P(B) = exp(lam_B IV_B) / sum_B' exp(lam_B' IV_B'),  IV_B = log sum_{l in B} exp(V_l / lam_B)
With a sampled nest, inside-nest terms carry the usual correction and IV_B is the Horvitz-Thompson estimate:
  y_l = V_l / lam_B - log pi_l,  IV_B = log sum_{l in D cap B} exp(y_l).
lam_B = sigmoid(t_B) in (0, 1); lam = 1 for both nests is the conditional logit. A first decision (H_k empty) has
only nest 0, so P(j) = softmax(V / lam_0 - log pi): the first-choice logit is the special case and the two can be
estimated jointly with shared beta.
V = X beta, where X may include 1[j in H] and its interactions (switching cost delta, satisfaction kappa).
Data: hfpa.models.clogit.to_buckets(..., extra_int={"nest": col}). Fit by L-BFGS; SEs from the autograd Hessian,
cluster-robust by author when `cluster` is given.
"""
import numpy as np
import torch

from hfpa.models.clogit import Buckets


class Nested2:
    def __init__(self, n_x, device=None, lam_init=3.0):
        self.nx = n_x
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.p = torch.zeros(n_x + 2, device=self.dev, dtype=torch.float64)
        self.p.data[n_x:] = lam_init  # lambda ~ 0.95
        self.p.requires_grad_(True)

    def lams(self, p=None):
        p = self.p if p is None else p
        return torch.sigmoid(p[self.nx:]).clamp(min=1e-3)

    def ll_rows(self, b, p=None):
        p = self.p if p is None else p
        lam = self.lams(p)
        X, mask, j, nest = b["X"].double(), b["mask"], b["chosen"], b["nest"]
        v = X @ p[: self.nx]
        lamr = lam[nest]  # [c, s]
        y = v / lamr - b["lq"]
        NEG = -1e9  # finite fill: an empty nest gets IV ~ -1e9 (zero weight, finite gradients)
        ivs, pres = [], []
        for k in (0, 1):
            inn = mask & (nest == k)
            ivs.append(torch.logsumexp(y.masked_fill(~inn, NEG), 1))
            pres.append(inn.any(1))
        iv = torch.stack(ivs, 1)  # [c, 2]
        present = torch.stack(pres, 1)
        top = torch.logsumexp(torch.where(present, lam[None, :] * iv, torch.full_like(iv, NEG)), 1)
        nj = nest.gather(1, j[:, None]).squeeze(1)
        ivj = iv.gather(1, nj[:, None]).squeeze(1)
        return y.gather(1, j[:, None]).squeeze(1) - ivj + lam[nj] * ivj - top

    def init(self, beta=None, lam=None):
        with torch.no_grad():
            if beta is not None:
                self.p[: self.nx] = torch.as_tensor(np.asarray(beta, np.float64), device=self.dev)
            if lam is not None:
                lam = np.clip(np.asarray(lam, np.float64), 1e-3, 1 - 1e-6)
                self.p[self.nx:] = torch.as_tensor(np.log(lam / (1 - lam)), device=self.dev)

    def fit(self, D: Buckets, max_iter=500, cluster=None, se=True, chunk_cells=1_000_000, ckpt=None):
        B = D.batches
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

        import os
        if ckpt is not None and os.path.exists(ckpt):  # converged point estimate from an earlier run
            with torch.no_grad():
                self.p.copy_(torch.as_tensor(np.load(ckpt), device=self.dev))
        else:
            opt.step(closure)
            if ckpt is not None:
                np.save(ckpt, self.p.detach().cpu().numpy())
        p0 = self.p.detach().clone()
        with torch.no_grad():
            ll = float(sum((b["w"] * self.ll_rows(b, p0)).sum() for b in B))
        k = len(p0)
        self.result = dict(names=D.names + ["t_new", "t_old"], coef=p0.cpu().numpy(), ll=ll, k=k, n=D.n,
                           aic=2 * k - 2 * ll, lam=self.lams(p0).detach().cpu().numpy())
        if not se:
            return self.result
        H = torch.zeros(k, k, device=self.dev, dtype=torch.float64)
        for b in B:
            H += torch.autograd.functional.hessian(lambda q: -(b["w"] * self.ll_rows(b, q)).sum(), p0)
        cov = torch.linalg.pinv(H)
        self.result["se"] = np.sqrt(np.clip(torch.diagonal(cov).cpu().numpy(), 0, None))
        if cluster is not None:
            _, cl = np.unique(np.asarray(cluster), return_inverse=True)
            clt = torch.as_tensor(cl, device=self.dev)
            G = torch.zeros(int(cl.max()) + 1, k, device=self.dev, dtype=torch.float64)
            eye = torch.eye(k, device=self.dev, dtype=torch.float64)
            for b in B:
                C = b["X"].shape[0]
                step = max(1, int(chunk_cells // max(1, b["X"].shape[1])))
                for s in range(0, C, step):
                    bs = {kk: (vv[s:s + step] if torch.is_tensor(vv) and vv.dim() > 0 and vv.shape[0] == C else vv)
                          for kk, vv in b.items()}
                    f = lambda q: bs["w"] * self.ll_rows(bs, q)
                    # forward mode: one jvp per parameter, memory linear in the chunk size
                    Jm = torch.stack([torch.func.jvp(f, (p0,), (eye[i],))[1] for i in range(k)], 1)
                    G.index_add_(0, clt[bs["idx"]], Jm.detach())
            g = G.shape[0]
            V = cov @ (G.T @ G) @ cov * (g / max(g - 1, 1))
            self.result["se_cl"] = np.sqrt(np.clip(torch.diagonal(V).cpu().numpy(), 0, None))
            self.result["cov_cl"] = V.cpu().numpy()
        return self.result

    @torch.no_grad()
    def ll_per_choice(self, D: Buckets):
        out = np.empty(D.n)
        for b in D.batches:
            out[b["idx"].cpu().numpy()] = self.ll_rows(b).cpu().numpy()
        return out

    @torch.no_grad()
    def nest_prob(self, D: Buckets):
        """P(old nest) per choice (0 if the author has no history)."""
        out = np.zeros(D.n)
        lam = self.lams()
        for b in D.batches:
            v = b["X"].double() @ self.p[: self.nx]
            y = v / lam[b["nest"]] - b["lq"]
            inn = [b["mask"] & (b["nest"] == k) for k in (0, 1)]
            iv = torch.stack([torch.logsumexp(y.masked_fill(~m, -1e9), 1) for m in inn], 1)
            pres = torch.stack([m.any(1) for m in inn], 1)
            z = torch.where(pres, lam[None, :] * iv, torch.full_like(iv, -1e9))
            out[b["idx"].cpu().numpy()] = torch.softmax(z, 1)[:, 1].cpu().numpy()
        return out
