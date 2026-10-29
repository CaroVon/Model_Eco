"""Node x period count panel with node fixed effects (P0c).

With large choice sets the conditional logit P(j at choice k) = exp(u_jk) / Z_k is, per period t,
a Poisson count with mean  E[y_jt] = exp(phi_j) * exp(tau_t + theta' x_jt) * exposure_jt,
where tau_t absorbs the period's sum of 1/Z_k and phi_j is node quality. A fixed effect for every
node is then cheap. Linear age decay is absorbed by phi_j + tau_t.

x_jt is built from the degree at the start of period t (predetermined: depends on y_j,<t), so
  * poisson_fe:  Poisson FE MLE (phi_j profiled out). Incidental-parameter (Nickell-type) bias of
                 order 1/T because x is predetermined, not strictly exogenous.
  * jackknife:   split-panel jackknife of poisson_fe (Dhaene & Jochmans 2015): 2*full - mean(halves).
  * qd_gmm:      Chamberlain (1992) / Wooldridge (1997) quasi-differenced moments
                 E[ y_jt - y_j,t+1 * mu_jt / mu_j,t+1 | F_t-1 ] = 0, instruments x_jt and period
                 dummies; consistent with predetermined x and fixed T.
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_panel(birth, choice_t, choice_parent, period, deg_t=None, deg_parent=None, y_weight=None, t0=None,
                age_unit=1):
    """Dense panel over nodes that receive >= 1 choice (others contribute nothing to either estimator).

    birth: node birth times; choice_t / choice_parent: time and parent of each choice;
    deg_t / deg_parent: edges that build degree (default = the choices); y_weight: weight per choice.
    Returns dict of [J, T] arrays: y (weighted counts), d (degree at period start), logE (log exposure),
    active mask, plus node ids.
    """
    deg_t = choice_t if deg_t is None else deg_t
    deg_parent = choice_parent if deg_parent is None else deg_parent
    t0 = min(birth.min(), choice_t.min()) if t0 is None else t0
    T = int((max(choice_t.max(), deg_t.max()) - t0) // period) + 1
    nodes = np.unique(choice_parent)
    J = len(nodes)
    row = {n: i for i, n in enumerate(nodes)}
    pj = np.array([row[p] for p in choice_parent])
    pt = ((choice_t - t0) // period).astype(np.int64)
    y = np.zeros((J, T))
    np.add.at(y, (pj, pt), 1.0 if y_weight is None else y_weight)
    inset = np.isin(deg_parent, nodes)
    dj = np.array([row[p] for p in deg_parent[inset]])
    dt = ((deg_t[inset] - t0) // period).astype(np.int64)
    pre = dt < 0  # edges before t0 form the initial degree
    cnt = np.zeros((J, T))
    np.add.at(cnt, (dj[~pre], dt[~pre]), 1.0)
    d0 = np.bincount(dj[pre], minlength=J).astype(float)
    d = d0[:, None] + np.cumsum(cnt, 1) - cnt  # degree at the start of each period
    b = (birth[nodes] - t0) / period
    start = np.arange(T)[None, :]
    alive = np.clip(start + 1 - b[:, None], 0, 1)  # share of the period the node exists
    active = alive > 0
    logE = np.where(active, np.log(np.where(active, alive, 1)), 0.0)
    age = np.maximum(t0 + start * period - birth[nodes][:, None], 0) / age_unit  # age at period start
    return dict(y=y, d=d, logE=logE, active=active, nodes=nodes, T=T, age=age)


def features(P, names=("has_deg", "log_deg")):
    """Degree terms are predetermined; age terms are deterministic. With node and period effects only
    the non-linear part of the age profile is identified (linear age = birth FE + period FE)."""
    d, age = P["d"], P["age"]
    f = {"has_deg": lambda: (d > 0).astype(float), "log_deg": lambda: np.log(np.maximum(d, 1)),
         "log1p_deg": lambda: np.log1p(d), "log_age": lambda: np.log1p(age),
         "age_lt30": lambda: (age < 30).astype(float), "age_30_90": lambda: ((age >= 30) & (age < 90)).astype(float),
         "age_90_365": lambda: ((age >= 90) & (age < 365)).astype(float)}
    return np.stack([f[n]() for n in names], -1)


def _t(a):
    return torch.as_tensor(a, device=DEV)


def poisson_fe(P, names=("has_deg", "log_deg"), periods=None, max_iter=300):
    """Poisson FE MLE with node and period FE; phi_j profiled out. periods: optional slice of periods."""
    sl = slice(None) if periods is None else periods
    y, X, logE, act = (_t(P["y"][:, sl]), _t(features(P, names)[:, sl]), _t(P["logE"][:, sl]), _t(P["active"][:, sl]))
    keep = (y.sum(1) > 0)
    y, X, logE, act = y[keep], X[keep], logE[keep], act[keep]
    T = y.shape[1]
    th = torch.zeros(len(names), device=DEV, requires_grad=True)
    tau = torch.zeros(T - 1, device=DEV, requires_grad=True)
    n = y.sum(1)

    def nll():
        eta = torch.cat([torch.zeros(1, device=DEV), tau])[None, :] + X @ th + logE
        eta = eta.masked_fill(~act, -torch.inf)
        return -((y * eta.masked_fill(~act, 0)).sum() - (n * torch.logsumexp(eta, 1)).sum())

    opt = torch.optim.LBFGS([th, tau], lr=1, max_iter=max_iter, tolerance_grad=1e-10, tolerance_change=1e-14,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    opt.step(closure)
    return dict(zip(names, th.detach().cpu().numpy()))


def jackknife(P, names=("has_deg", "log_deg")):
    """Split-panel jackknife: halves by period; degree state still carries the full history."""
    full = poisson_fe(P, names)
    h = P["T"] // 2
    a, b = poisson_fe(P, names, slice(0, h)), poisson_fe(P, names, slice(h, None))
    return {k: 2 * full[k] - 0.5 * (a[k] + b[k]) for k in names}, full


def qd_gmm(P, names=("has_deg", "log_deg"), max_iter=200):
    """Quasi-differenced GMM, just identified: period dummies (solved in closed form for tau) + x_jt."""
    y, X, logE, act = _t(P["y"]), _t(features(P, names)), _t(P["logE"]), _t(P["active"])
    pair = act[:, :-1] & act[:, 1:]  # consecutive periods both alive
    y0, y1 = y[:, :-1], y[:, 1:]
    X0, X1 = X[:, :-1], X[:, 1:]
    dE = logE[:, :-1] - logE[:, 1:]
    th = torch.zeros(len(names), device=DEV, requires_grad=True)
    scale = (y0 * pair).sum()

    def moments(th):
        r = torch.exp((X0 - X1) @ th + dE) * pair  # mu_t / mu_t+1 without tau
        # period-dummy moments give exp(tau_t - tau_t+1) = sum_j y_jt / sum_j y_j,t+1 r_jt
        num, den = (y0 * pair).sum(0), (y1 * r).sum(0)
        ratio = torch.where(den > 0, num / den.clamp_min(1e-300), torch.zeros_like(den))
        g = (y0 - y1 * r * ratio[None, :]) * pair
        return (g[..., None] * X0).sum((0, 1)) / scale

    opt = torch.optim.LBFGS([th], lr=1, max_iter=max_iter, tolerance_grad=1e-14, tolerance_change=1e-16,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        m = moments(th)
        loss = (m ** 2).sum() * 1e4
        loss.backward()
        return loss

    opt.step(closure)
    m = moments(th).detach().cpu().numpy()
    return dict(zip(names, th.detach().cpu().numpy())), float(np.abs(m).max())


def _instruments(P, X):
    """F_t-1 measurable instruments for period t: x_t, lagged x and lagged count, each raw and divided
    by (1 + d_t) so that hubs with large counts do not dominate the moments."""
    d, y = P["d"], P["y"]
    lag = lambda a: np.concatenate([np.zeros_like(a[:, :1]), a[:, :-1]], 1)
    base = [X[..., i] for i in range(X.shape[-1])]
    base += [lag(X[..., i]) for i in range(X.shape[-1])] + [np.log1p(lag(y))]
    w = 1.0 / (1.0 + d)
    return np.stack(base + [b * w for b in base], -1)


def pd_codes(a):
    return np.unique(np.asarray(a), return_inverse=True)[1].astype(np.int64)


BOUND = 10.0  # |theta| bound for qd_gmm2; estimates near it are flagged as not converged


def START_GRID(k):
    """Starting values for (has_deg, log_deg, ...): 0 and a few plausible PA strengths."""
    grid = [np.zeros(k)]
    for a in (0.5, 1.0):
        s = np.zeros(k)
        s[min(1, k - 1)] = a
        grid.append(s)
    return grid


def qd_gmm2(P, names=("has_deg", "log_deg"), max_iter=300, node_se=True, group=None):
    """Over-identified two-step GMM with the quasi-differenced moments (period dummies profiled in
    closed form). Step 1 identity weight, step 2 inverse of the node-clustered moment covariance.
    group: optional int per panel node (e.g. modality); period effects are then group x period.
    Returns estimates, node-clustered SEs, Hansen J statistic and its degrees of freedom."""
    y, X, logE, act = _t(P["y"]), _t(features(P, names)), _t(P["logE"]), _t(P["active"])
    Z = _t(_instruments(P, features(P, names)))
    pair = act[:, :-1] & act[:, 1:]
    y0, y1, X0, X1, Z0 = y[:, :-1], y[:, 1:], X[:, :-1], X[:, 1:], Z[:, :-1]
    dE = logE[:, :-1] - logE[:, 1:]
    J = y.shape[0]
    scale = (y0 * pair).sum() / J
    gi = _t(np.zeros(J, np.int64) if group is None else pd_codes(group))
    G = int(gi.max()) + 1

    def node_moments(th):
        r = torch.exp(((X0 - X1) @ th + dE).clamp(-40, 40)) * pair
        num = torch.zeros(G, r.shape[1], device=DEV).index_add_(0, gi, y0 * pair)
        den = torch.zeros(G, r.shape[1], device=DEV).index_add_(0, gi, y1 * r)
        ratio = torch.where(den > 0, num / den.clamp_min(1e-300), torch.zeros_like(den))[gi]
        g = (y0 - y1 * r * ratio) * pair
        return (g[..., None] * Z0).sum(1) / scale  # [J, L]

    def objective(th, W):
        m = node_moments(th).mean(0)
        return m @ W @ m

    def solve(W, starts):
        """Multi-start L-BFGS on theta = BOUND * tanh(v / BOUND); keeps the lowest finite objective."""
        best, best_f = None, np.inf
        for th0 in starts:
            v = (BOUND * torch.atanh((th0 / BOUND).clamp(-0.99, 0.99))).clone().requires_grad_(True)
            opt = torch.optim.LBFGS([v], lr=1, max_iter=max_iter, tolerance_grad=1e-14, tolerance_change=1e-18,
                                    history_size=20, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = objective(BOUND * torch.tanh(v / BOUND), W)
                if torch.isfinite(loss):
                    loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                th = BOUND * torch.tanh(v / BOUND)
                f = float(objective(th, W))
            if np.isfinite(f) and f < best_f:
                best, best_f = th.detach(), f
        return best

    L, k = Z.shape[-1], len(names)
    starts = [torch.as_tensor(s, device=DEV, dtype=torch.float64) for s in START_GRID(k)]
    th1 = solve(torch.eye(L, device=DEV), starts)
    M = node_moments(th1)
    S = (M - M.mean(0)).T @ (M - M.mean(0)) / J
    W = torch.linalg.pinv(S)
    th2 = solve(W, [th1] + starts)
    M = node_moments(th2)
    m = M.mean(0)
    S = (M - m).T @ (M - m) / J
    Jstat = float(J * m @ torch.linalg.pinv(S) @ m)
    out = dict(zip(names, th2.cpu().numpy()))
    se = dict.fromkeys(names, np.nan)
    if node_se:
        G = torch.autograd.functional.jacobian(lambda t: node_moments(t).mean(0), th2)  # [L, k]
        V = torch.linalg.pinv(G.T @ torch.linalg.pinv(S) @ G) / J
        se = dict(zip(names, torch.sqrt(torch.diagonal(V).clamp_min(0)).cpu().numpy()))
    out["converged"] = bool(torch.all(th2.abs() < 0.95 * BOUND))
    return out, se, Jstat, L - len(names)
