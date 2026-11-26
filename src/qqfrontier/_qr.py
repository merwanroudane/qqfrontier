"""Linear quantile-regression solvers.

* ``rq_fit(..., method="fn")`` - Frisch-Newton interior point (Koenker and
  Portnoy, 1997), a NumPy port of ``quantreg::rq.fit.fnb`` (Mehrotra
  predictor-corrector on the bounded dual LP).  Vectorised over a batch of
  problems sharing the same size, which is how the rolling estimators call it.
* ``rq_fit(..., method="br")`` - exact simplex vertex solution of the primal LP
  through HiGHS (``scipy.optimize.linprog``), the analogue of ``quantreg``'s
  Barrodale-Roberts ``"br"``.
* ``rq_noncross`` - Bondell, Reich and Wang (2010) non-crossing estimation of a
  set of quantile curves (one joint LP, solved with HiGHS).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

__all__ = ["rq_fit", "rq_fit_batch", "rq_noncross", "check_loss"]


def check_loss(u: np.ndarray, tau: float) -> np.ndarray:
    """Koenker-Bassett check function rho_tau(u) = u (tau - 1{u<0})."""
    u = np.asarray(u, dtype=float)
    return u * (tau - (u < 0))


def _bound(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Largest step keeping x + f dx >= 0, row-wise min over the last axis."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dx < 0, -x / dx, 1e20)
    return r.min(axis=-1)


def rq_fit_batch(X: np.ndarray, Y: np.ndarray, taus: np.ndarray,
                 beta: float = 0.99995, eps: float = 1e-6,
                 max_it: int = 100) -> np.ndarray:
    """Frisch-Newton quantile regression for a batch of problems.

    Parameters
    ----------
    X : ndarray, shape (B, n, p) or (n, p)
        Design matrices (a 2-D array is shared by every problem).
    Y : ndarray, shape (B, n)
        Responses.
    taus : ndarray, shape (B,)
        Quantile level of each problem.

    Returns
    -------
    ndarray, shape (B, p)
        Coefficient vectors.

    Notes
    -----
    Solves the dual ``max_a y'a  s.t.  X'a = (1-tau) X'1,  0 <= a <= 1`` with the
    primal-dual log-barrier method of ``quantreg``'s ``rqfnb`` (same ``beta``,
    ``eps`` and starting point a = 1 - tau); the coefficients are minus the
    dual multipliers of the equality constraints.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    Bn, n = Y.shape
    X = np.asarray(X, dtype=float)
    if _fnb_numba is not None and Bn > 1:
        Xc = np.ascontiguousarray(np.broadcast_to(X, (Bn, n, X.shape[-1])) if X.ndim == 2 else X)
        tc = np.ascontiguousarray(np.broadcast_to(np.asarray(taus, dtype=float), (Bn,)))
        return _fnb_numba(Xc, np.ascontiguousarray(Y), tc, beta, eps, max_it)
    return _rq_fit_batch_numpy(X, Y, taus, beta, eps, max_it)


def _rq_fit_batch_numpy(X, Y, taus, beta=0.99995, eps=1e-6, max_it=100):
    """Pure-NumPy vectorised Frisch-Newton (fallback when numba is unavailable)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    Bn, n = Y.shape
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = np.broadcast_to(X, (Bn, n, X.shape[1]))
    p = X.shape[2]
    taus = np.broadcast_to(np.asarray(taus, dtype=float), (Bn,)).copy()
    A = np.swapaxes(X, 1, 2)                        # (B, p, n) constraint matrix
    c = -Y                                          # dual objective (minimise c'x)
    u = np.ones((Bn, n))
    x = (1.0 - taus)[:, None] * u                   # feasible start
    b = np.einsum("bpn,bn->bp", A, x)               # rhs = (1-tau) X'1
    s = u - x
    # y = least-squares solution of A' y = c
    AAt = np.einsum("bpn,bqn->bpq", A, A)
    y = np.linalg.solve(AAt, np.einsum("bpn,bn->bp", A, c)[..., None])[..., 0]
    r = c - np.einsum("bpn,bp->bn", A, y)
    r = r + 0.001 * (r == 0)
    z = np.where(r > 0, r, 0.0)
    w = z - r
    gap = (c * x).sum(1) - (y * b).sum(1) + (w * u).sum(1)
    active = gap > eps
    it = 0
    while active.any() and it < max_it:
        it += 1
        ia = np.where(active)[0]
        Aa, xa, sa, za, wa, ya, ca = A[ia], x[ia], s[ia], z[ia], w[ia], y[ia], c[ia]
        q = 1.0 / (za / xa + wa / sa)
        r = za - wa
        AQA = np.einsum("bpn,bn,bqn->bpq", Aa, q, Aa)
        rhs = np.einsum("bpn,bn->bp", Aa, q * r)
        dy = np.linalg.solve(AQA, rhs[..., None])[..., 0]
        dx = q * (np.einsum("bpn,bp->bn", Aa, dy) - r)
        ds = -dx
        dz = -za * (dx / xa + 1.0)
        dw = -wa * (ds / sa + 1.0)
        fp = np.minimum(beta * np.minimum(_bound(xa, dx), _bound(sa, ds)), 1.0)
        fd = np.minimum(beta * np.minimum(_bound(wa, dw), _bound(za, dz)), 1.0)
        corr = np.minimum(fp, fd) < 1.0
        if corr.any():
            ic = np.where(corr)[0]
            xc, sc, zc, wc = xa[ic], sa[ic], za[ic], wa[ic]
            dxc, dsc, dzc, dwc = dx[ic], ds[ic], dz[ic], dw[ic]
            fpc, fdc = fp[ic][:, None], fd[ic][:, None]
            mu = (zc * xc).sum(1) + (wc * sc).sum(1)
            g = ((zc + fdc * dzc) * (xc + fpc * dxc)).sum(1) + \
                ((wc + fdc * dwc) * (sc + fpc * dsc)).sum(1)
            mu = (mu * (g / mu) ** 3 / (2.0 * n))[:, None]
            dxdz = dxc * dzc
            dsdw = dsc * dwc
            xinv = 1.0 / xc
            sinv = 1.0 / sc
            xi = mu * (xinv - sinv)
            qc, rc, Ac = q[ic], r[ic], Aa[ic]
            rhs_c = rhs[ic] + np.einsum("bpn,bn->bp", Ac, qc * (dxdz - dsdw - xi))
            dyc = np.linalg.solve(AQA[ic], rhs_c[..., None])[..., 0]
            dxc = qc * (np.einsum("bpn,bp->bn", Ac, dyc) + xi - rc - dxdz + dsdw)
            dsc = -dxc
            dzc = mu * xinv - zc - xinv * zc * dxc - dxdz
            dwc = mu * sinv - wc - sinv * wc * dsc - dsdw
            dx[ic], ds[ic], dz[ic], dw[ic], dy[ic] = dxc, dsc, dzc, dwc, dyc
            fp[ic] = np.minimum(beta * np.minimum(_bound(xc, dxc), _bound(sc, dsc)), 1.0)
            fd[ic] = np.minimum(beta * np.minimum(_bound(wc, dwc), _bound(zc, dzc)), 1.0)
        fp = fp[:, None]
        fd = fd[:, None]
        x[ia] = xa + fp * dx
        s[ia] = sa + fp * ds
        y[ia] = ya + fd * dy
        w[ia] = wa + fd * dw
        z[ia] = za + fd * dz
        gap_a = (ca * x[ia]).sum(1) - (y[ia] * b[ia]).sum(1) + (w[ia] * u[ia]).sum(1)
        active[ia] = gap_a > eps
    return -y


def _fnb_one(X, y, tau, beta, eps, max_it):
    """Single Frisch-Newton problem with explicit loops (compiled by numba)."""
    n, p = X.shape
    x = np.empty(n); s = np.empty(n); z = np.empty(n); w = np.empty(n)
    c = -y
    for i in range(n):
        x[i] = 1.0 - tau
        s[i] = tau
    b = np.zeros(p)
    for i in range(n):
        for j in range(p):
            b[j] += X[i, j] * x[i]
    XtX = X.T @ X
    yd = np.linalg.solve(XtX, X.T @ c)
    r = c - X @ yd
    for i in range(n):
        if r[i] == 0.0:
            r[i] = 0.001
        z[i] = r[i] if r[i] > 0 else 0.0
        w[i] = z[i] - r[i]
    gap = 0.0
    for i in range(n):
        gap += c[i] * x[i] + w[i]
    for j in range(p):
        gap -= yd[j] * b[j]
    q = np.empty(n); dx = np.empty(n); ds = np.empty(n); dz = np.empty(n); dw = np.empty(n)
    it = 0
    while gap > eps and it < max_it:
        it += 1
        M = np.zeros((p, p)); rhs = np.zeros(p)
        for i in range(n):
            q[i] = 1.0 / (z[i] / x[i] + w[i] / s[i])
            r[i] = z[i] - w[i]
            for a in range(p):
                rhs[a] += X[i, a] * q[i] * r[i]
                for bb in range(a, p):
                    M[a, bb] += X[i, a] * q[i] * X[i, bb]
        for a in range(p):
            for bb in range(a):
                M[a, bb] = M[bb, a]
        dy = np.linalg.solve(M, rhs)
        fx = 1e20; fs = 1e20; fw = 1e20; fz = 1e20
        for i in range(n):
            ad = 0.0
            for a in range(p):
                ad += X[i, a] * dy[a]
            dx[i] = q[i] * (ad - r[i])
            ds[i] = -dx[i]
            dz[i] = -z[i] * (dx[i] / x[i] + 1.0)
            dw[i] = -w[i] * (ds[i] / s[i] + 1.0)
            if dx[i] < 0:
                fx = min(fx, -x[i] / dx[i])
            if ds[i] < 0:
                fs = min(fs, -s[i] / ds[i])
            if dw[i] < 0:
                fw = min(fw, -w[i] / dw[i])
            if dz[i] < 0:
                fz = min(fz, -z[i] / dz[i])
        fp = min(beta * min(fx, fs), 1.0)
        fd = min(beta * min(fw, fz), 1.0)
        if min(fp, fd) < 1.0:
            mu = 0.0; g = 0.0
            for i in range(n):
                mu += z[i] * x[i] + w[i] * s[i]
                g += (z[i] + fd * dz[i]) * (x[i] + fp * dx[i]) + (w[i] + fd * dw[i]) * (s[i] + fp * ds[i])
            mu = mu * (g / mu) ** 3 / (2.0 * n)
            rhs2 = rhs.copy()
            xi = np.empty(n); dxdz = np.empty(n); dsdw = np.empty(n)
            for i in range(n):
                dxdz[i] = dx[i] * dz[i]
                dsdw[i] = ds[i] * dw[i]
                xi[i] = mu * (1.0 / x[i] - 1.0 / s[i])
                t = q[i] * (dxdz[i] - dsdw[i] - xi[i])
                for a in range(p):
                    rhs2[a] += X[i, a] * t
            dy = np.linalg.solve(M, rhs2)
            fx = 1e20; fs = 1e20; fw = 1e20; fz = 1e20
            for i in range(n):
                ad = 0.0
                for a in range(p):
                    ad += X[i, a] * dy[a]
                dx[i] = q[i] * (ad + xi[i] - r[i] - dxdz[i] + dsdw[i])
                ds[i] = -dx[i]
                dz[i] = mu / x[i] - z[i] - z[i] * dx[i] / x[i] - dxdz[i]
                dw[i] = mu / s[i] - w[i] - w[i] * ds[i] / s[i] - dsdw[i]
                if dx[i] < 0:
                    fx = min(fx, -x[i] / dx[i])
                if ds[i] < 0:
                    fs = min(fs, -s[i] / ds[i])
                if dw[i] < 0:
                    fw = min(fw, -w[i] / dw[i])
                if dz[i] < 0:
                    fz = min(fz, -z[i] / dz[i])
            fp = min(beta * min(fx, fs), 1.0)
            fd = min(beta * min(fw, fz), 1.0)
        gap = 0.0
        for i in range(n):
            x[i] += fp * dx[i]
            s[i] += fp * ds[i]
            w[i] += fd * dw[i]
            z[i] += fd * dz[i]
            gap += c[i] * x[i] + w[i]
        for j in range(p):
            yd[j] += fd * dy[j]
            gap -= yd[j] * b[j]
    return -yd


def _make_fnb_numba():
    try:
        from numba import njit, prange
    except Exception:  # pragma: no cover
        return None
    one = njit(cache=True, fastmath=False)(_fnb_one)

    @njit(cache=True, parallel=True)
    def batch(X, Y, taus, beta, eps, max_it):
        Bn = Y.shape[0]
        out = np.empty((Bn, X.shape[2]))
        for i in prange(Bn):
            out[i] = one(X[i], Y[i], taus[i], beta, eps, max_it)
        return out
    return batch


_fnb_numba = _make_fnb_numba()


def _rq_br(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Exact LP (simplex vertex) solution through HiGHS."""
    n, p = X.shape
    c = np.concatenate([np.zeros(p), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A_eq = sparse.hstack([sparse.csr_matrix(X), sparse.identity(n), -sparse.identity(n)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs-ds")
    if res.status != 0:
        raise RuntimeError(f"quantile LP failed: {res.message}")
    return res.x[:p]


def rq_fit(X: np.ndarray, y: np.ndarray, tau: float = 0.5,
           weights: Optional[np.ndarray] = None, method: str = "fn") -> np.ndarray:
    """Linear quantile regression of ``y`` on ``X`` at level ``tau``.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix (include a column of ones for an intercept).
    y : ndarray (n,)
    tau : float
        Quantile level in (0, 1).
    weights : ndarray (n,), optional
        Non-negative observation weights (as ``quantreg::rq.wfit``: rows of X
        and y are multiplied by the weights).
    method : {"fn", "br"}
        Frisch-Newton interior point or exact simplex (HiGHS).

    Returns
    -------
    ndarray (p,)

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=200); y = 1 + 2 * x + rng.normal(size=200)
    >>> X = np.column_stack([np.ones(200), x])
    >>> np.round(rq_fit(X, y, 0.5), 1)
    array([1., 2.])
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if not 0 < tau < 1:
        raise ValueError("tau must lie strictly in (0, 1)")
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        X = X * w[:, None]
        y = y * w
    if method == "br":
        return _rq_br(X, y, tau)
    if method != "fn":
        raise ValueError("method must be 'fn' or 'br'")
    return rq_fit_batch(X, y[None, :], np.array([tau]))[0]


def _box_vertices(X: np.ndarray) -> np.ndarray:
    """Vertices of the box spanned by the regressors (first column = intercept)."""
    p = X.shape[1]
    rng = [(X[:, j].min(), X[:, j].max()) for j in range(1, p)]
    if p == 1:
        return np.ones((1, 1))
    grids = np.meshgrid(*[np.array(r) for r in rng], indexing="ij")
    V = np.column_stack([g.ravel() for g in grids])
    return np.column_stack([np.ones(len(V)), V])


def rq_noncross(X: np.ndarray, y: np.ndarray, taus: Sequence[float],
                method: str = "ipm", eps: float = 1e-8, max_it: int = 200) -> np.ndarray:
    """Non-crossing quantile curves of Bondell, Reich and Wang (2010).

    Minimises ``sum_k sum_i rho_{tau_k}(y_i - x_i' b_k)`` subject to
    ``v'(b_k - b_{k-1}) >= 0`` at every vertex ``v`` of the box spanned by the
    regressors (intercept excluded), which guarantees non-crossing over the
    whole box (Bondell et al., 2010, Sec. 2.2).

    Parameters
    ----------
    method : {"ipm", "simplex"}
        ``"ipm"`` (default): a structured primal-dual interior point method
        (Mehrotra predictor-corrector on the bounded dual, as Koenker and Ng's
        ``rq.fit.sfnc``) whose Newton system is only (pK x pK).  ``"simplex"``:
        the same LP solved by HiGHS (slow; used to validate the default).

    Returns
    -------
    ndarray (p, K)
        One column of coefficients per (sorted) quantile level.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    taus = np.sort(np.asarray(taus, dtype=float))
    if method == "simplex" or len(taus) == 1:
        return _rq_noncross_joint(X, y, taus)
    if method != "ipm":
        raise ValueError("method must be 'ipm' or 'simplex'")
    if np.ptp(X[:, 1:], axis=0).min(initial=1.0) <= 0:
        return _rq_noncross_joint(X, y, taus)                 # constant regressor: degenerate box
    try:
        b = _rq_noncross_ipm(X, y, taus, eps, max_it)
        if np.isfinite(b).all():
            return b
    except np.linalg.LinAlgError:
        pass
    return _rq_noncross_joint(X, y, taus)                     # robust fallback (exact simplex)


def _rq_noncross_ipm(X, y, taus, eps=1e-8, max_it=200):
    """Infeasible-start Mehrotra IPM for the Bondell LP in bounded-dual form.

    min c1'x + c2'x2  s.t.  A1 x + A2 x2 = b,  0 <= x <= 1,  x2 >= 0, with
    A1 = blockdiag(X'), c1 = -y (stacked), b = (1 - tau_k) X'1 (stacked),
    A2 = R' (non-crossing constraints R beta >= 0), c2 = 0.  The quantile
    coefficients are minus the equality multipliers.
    """
    n, p = X.shape
    K = len(taus)
    E = _box_vertices(X)
    nv = E.shape[0]
    m = nv * (K - 1)
    P = p * K
    # R (m x P): rows e'(beta_k - beta_{k-1}) >= 0
    R = np.zeros((m, P))
    for k in range(1, K):
        R[(k - 1) * nv:k * nv, k * p:(k + 1) * p] = E
        R[(k - 1) * nv:k * nv, (k - 1) * p:k * p] = -E
    XT = X.T                                                   # (p, n)
    c1 = -np.tile(y, (K, 1))                                   # (K, n)
    b = ((1.0 - taus)[:, None] * XT.sum(1)[None, :]).ravel()  # (P,)

    def A1x(v):                                                # (K, n) -> (P,)
        return (v @ X).ravel()

    def A1ty(yy):                                              # (P,) -> (K, n)
        return yy.reshape(K, p) @ XT

    x = np.tile(1.0 - taus[:, None], (1, n))
    s = 1.0 - x
    x2 = np.ones(m)
    yd = -np.tile(np.linalg.lstsq(X, y, rcond=None)[0], K)   # same LS fit for every block
    r = c1 - A1ty(yd)
    z = np.maximum(r, 0) + 1e-2
    w = np.maximum(-r, 0) + 1e-2
    z2 = np.ones(m)
    npair = 2 * n * K + m


    def solve_dirs(rp, rd1, rd2, rxz, rsw, rx2):
        q1 = 1.0 / (z / x + w / s)
        q2 = x2 / z2
        h1 = -rd1 + rxz / x - rsw / s
        h2 = -rd2 + rx2 / x2
        M = (R.T * q2) @ R                                     # A2 Q2 A2'
        blk = np.einsum("pi,ki,qi->kpq", XT, q1, XT)           # A1 Q1 A1' (block diagonal)
        for k in range(K):
            M[k * p:(k + 1) * p, k * p:(k + 1) * p] += blk[k]
        rhs = rp - A1x(q1 * h1) - R.T @ (q2 * h2)
        dy = np.linalg.solve(M, rhs)
        dx = q1 * (A1ty(dy) + h1)
        dx2 = q2 * (R @ dy + h2)
        dz = (rxz - z * dx) / x
        dw = (rsw + w * dx) / s
        dz2 = (rx2 - z2 * dx2) / x2
        return dy, dx, dx2, dz, dw, dz2

    def steps(dx, dx2, dz, dw, dz2):
        fp = min(_bound(x.ravel(), dx.ravel()), _bound(s.ravel(), -dx.ravel()),
                 _bound(x2, dx2) if m else 1e20)
        fd = min(_bound(z.ravel(), dz.ravel()), _bound(w.ravel(), dw.ravel()),
                 _bound(z2, dz2) if m else 1e20)
        return min(fp, 1.0), min(fd, 1.0)


    for _ in range(max_it):
        rp = b - A1x(x) - R.T @ x2
        rd1 = c1 - A1ty(yd) - z + w
        rd2 = -(R @ yd) - z2
        mu = ((x * z).sum() + (s * w).sum() + x2 @ z2) / npair
        pobj = (c1 * x).sum()
        dobj = b @ yd - w.sum()
        if (abs(pobj - dobj) / (1 + abs(pobj)) < eps and
                np.abs(rp).max() / (1 + np.abs(b).max()) < eps and
                np.abs(rd1).max() / (1 + np.abs(c1).max()) < eps):
            break
        # predictor (affine scaling)
        dy, dx, dx2, dz, dw, dz2 = solve_dirs(rp, rd1, rd2, -x * z, -s * w, -x2 * z2)
        fp, fd = steps(dx, dx2, dz, dw, dz2)
        mu_aff = (((x + fp * dx) * (z + fd * dz)).sum() + ((s - fp * dx) * (w + fd * dw)).sum()
                  + (x2 + fp * dx2) @ (z2 + fd * dz2)) / npair
        sigma = (mu_aff / mu) ** 3
        # corrector
        dy, dx, dx2, dz, dw, dz2 = solve_dirs(
            rp, rd1, rd2,
            sigma * mu - x * z - dx * dz,
            sigma * mu - s * w + dx * dw,
            sigma * mu - x2 * z2 - dx2 * dz2)
        fp, fd = steps(dx, dx2, dz, dw, dz2)
        fp, fd = 0.99995 * fp, 0.99995 * fd
        x = x + fp * dx
        s = 1.0 - x
        x2 = x2 + fp * dx2
        yd = yd + fd * dy
        z = z + fd * dz
        w = w + fd * dw
        z2 = z2 + fd * dz2
    return (-yd).reshape(K, p).T


def _rq_noncross_joint(X: np.ndarray, y: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Single constrained LP of Bondell et al. (2010), solved with HiGHS."""
    n, p = X.shape
    K = len(taus)
    if K == 1:
        return _rq_br(X, y, float(taus[0]))[:, None]
    E = _box_vertices(X)
    nv = E.shape[0]
    # variables: [b_1..b_K (pK), u+ (nK), u- (nK)]
    nb = p * K
    c = np.concatenate([np.zeros(nb), np.repeat(taus, n), np.repeat(1 - taus, n)])
    Xb = sparse.block_diag([sparse.csr_matrix(X)] * K)
    I = sparse.identity(n * K)
    A_eq = sparse.hstack([Xb, I, -I]).tocsr()
    b_eq = np.tile(y, K)
    rows = []
    for k in range(1, K):
        blk = sparse.lil_matrix((nv, nb))
        blk[:, k * p:(k + 1) * p] = -E                 # -(b_k - b_{k-1}) <= 0
        blk[:, (k - 1) * p:k * p] = E
        rows.append(blk.tocsr())
    A_ub = sparse.hstack([sparse.vstack(rows), sparse.csr_matrix((nv * (K - 1), 2 * n * K))])
    bounds = [(None, None)] * nb + [(0, None)] * (2 * n * K)
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(nv * (K - 1)), A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"non-crossing LP failed: {res.message}")
    return res.x[:nb].reshape(K, p).T
