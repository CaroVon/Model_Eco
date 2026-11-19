"""Latent-class (mixture) choice model with availability renormalisation (v2 fix C).

Mode m is available for choice k if its candidate set is non-empty (a_km = 1), whether or not the chosen parent
is in it. The likelihood is renormalised over the available modes,
    P(j_k) = sum_m pi_m a_km P_m(j_k) / sum_m pi_m a_km,
and simulation draws the mode with the same probabilities pi_m a_km / sum_m' pi_m' a_km', so estimation and
simulation agree. E step: gamma_km ∝ pi_m a_km P_m(j_k) (the denominator cancels). M step for pi: maximise
sum_k w_k [sum_m gamma_km log pi_m - log sum_m pi_m a_km] over pi = softmax(eta) (L-BFGS); M step for the mode
parameters: weighted conditional logits (weights w_k gamma_km), warm-started.

Modes:
  LogitMode(name, D, pos, feats)  conditional logit on its own candidate set; D = Buckets of the choices whose chosen
                                  alternative is in the set; pos = their positions among all n choices. feats = []
                                  is uniform attachment over the set.
  PointMode(name, hit)            point mass (e.g. the author's last parent): log P = 0 if hit else -inf.
"""
import numpy as np
import torch

from hfpa.models.clogit import CLogit, uniform_ll


class LogitMode:
    def __init__(self, name, D, pos, n_fe=0, l2_fe=1e-4):
        self.name, self.D, self.pos = name, D, np.asarray(pos)
        self.feats = D.names if D is not None else []
        self.n_fe = n_fe
        self.model = None
        if D is not None and (self.feats or n_fe):
            self.model = CLogit(len(self.feats), n_fe=n_fe, l2_fe=l2_fe if n_fe else 0.0, device=D.device)
        self._u0 = uniform_ll(D) if (D is not None and self.model is None) else None

    def logp(self, n):
        out = np.full(n, -np.inf)
        if self.D is None:
            return out
        out[self.pos] = self._u0 if self.model is None else self.model.ll_per_choice(self.D)
        return out

    def mstep(self, w_all, max_iter=60):
        if self.model is None:
            return
        self.D.set_weights(w_all[self.pos])
        self.res = self.model.fit(self.D, max_iter=max_iter, se=False)

    def coef(self):
        return {} if self.model is None else dict(zip(self.feats, self.model.p.detach().cpu().numpy()[: len(self.feats)]))


class PointMode:
    def __init__(self, name, hit):
        self.name, self.hit = name, np.asarray(hit, bool)
        self.model = None

    def logp(self, n):
        return np.where(self.hit, 0.0, -np.inf)

    def mstep(self, w_all, max_iter=None):
        pass

    def coef(self):
        return {}


def update_pi(gamma, avail, w, pi0):
    """argmax_pi sum_k w_k [sum_m gamma_km log pi_m - log sum_m pi_m a_km]."""
    G = torch.as_tensor(gamma * w[:, None])
    A = torch.as_tensor(avail.astype(np.float64))
    W = torch.as_tensor(w)
    eta = torch.as_tensor(np.log(np.clip(pi0, 1e-12, None)), dtype=torch.float64).requires_grad_(True)
    opt = torch.optim.LBFGS([eta], lr=1, max_iter=200, tolerance_grad=1e-10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lp = torch.log_softmax(eta, 0)
        den = torch.logsumexp(lp[None, :] + torch.log(A.clamp(min=1e-300)), 1)
        loss = -((G * lp[None, :]).sum() - (W * den).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return torch.softmax(eta.detach(), 0).numpy()


def loglik(L, avail, pi, w):
    lse = np.logaddexp.reduce(L + np.log(pi)[None, :], 1)
    den = np.log((avail * pi[None, :]).sum(1))
    return float((w * (lse - den)).sum()), lse


def em(modes, avail, w, n_iter=40, tol=1e-6, pi0=None, seed=0, log=None):
    """modes: list of LogitMode / PointMode; avail [n, M] bool; w [n] choice weights. Returns dict(pi, ll, gamma,
    iters, hist, share = mean responsibility)."""
    n, M = avail.shape
    rng = np.random.default_rng(seed)
    pi = np.full(M, 1 / M) if pi0 is None else np.asarray(pi0, float)
    if seed and pi0 is None:
        pi = rng.dirichlet(np.ones(M))
    for m in modes:
        m.mstep(w)  # start: every mode fitted on all its choices
    hist = []
    for it in range(n_iter):
        L = np.stack([m.logp(n) for m in modes], 1)
        ll, lse = loglik(L, avail, pi, w)
        hist.append(ll)
        g = np.exp(L + np.log(pi)[None, :] - lse[:, None])
        g = np.nan_to_num(g)
        pi = update_pi(g, avail, w, pi)
        for k, m in enumerate(modes):
            m.mstep(w * g[:, k])
        if log:
            log(f"EM {it}: ll {ll:.2f} pi {np.round(pi, 4)}")
        if it > 2 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-1]):
            break
    L = np.stack([m.logp(n) for m in modes], 1)
    ll, lse = loglik(L, avail, pi, w)
    g = np.nan_to_num(np.exp(L + np.log(pi)[None, :] - lse[:, None]))
    return dict(pi=pi, ll=ll, gamma=g, iters=len(hist), hist=hist, share=(g * w[:, None]).sum(0) / w.sum())


def draw_modes(avail, pi, rng):
    """Mode of each choice under the renormalised mixture (simulation uses the same model as estimation)."""
    pr = avail * pi[None, :]
    pr = pr / pr.sum(1, keepdims=True)
    return (rng.random(len(pr))[:, None] < np.cumsum(pr, 1)).argmax(1)
