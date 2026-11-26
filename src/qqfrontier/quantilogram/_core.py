"""Cross-quantilogram of Han, Linton, Oka and Whang (2016).

Step -> equation map (HLOW = Han et al., 2016, J. Econometrics 193, 251-270)
--------------------------------------------------------------------------
psi_a(u) = 1[u < 0] - a ................................................. Eq. (1)
q_i,t(tau_i): sample quantile (x_it = 1) or linear QR x_it' beta_i ........ Sec. 2.1
rho_tau(k) = sum psi psi / sqrt(sum psi^2 sum psi^2), t = k+1..T ......... Eq. (2)
Box-Ljung Q(p) = T(T+2) sum_k rho^2 / (T-k); sup over a quantile range ... Sec. 2.1
Stationary bootstrap on the (T-p) tuples, quantiles re-estimated .......... Sec. 3.1.1
  CI for rho: [rho + c1/sqrt(T), rho + c2/sqrt(T)];  band for H0 (no
  predictability): [c1/sqrt(T), c2/sqrt(T)] (HLOW, Figs. 1-3)
Q* = T(T+2) sum_k (rho* - rho)^2/(T-k)  -> critical value c*_Q ........... Theorem 2
Mean block length 1/gamma: Politis-White (2004) with the Patton, Politis
  and White (2009) correction, applied to each series, gamma averaged ...... Sec. 6.1, fn. 16
Self-normalised S(p) = T rho' V^-1 rho, V from recursive subsamples ...... Sec. 3.1.2, Thm 4
Partial cross-quantilogram rho_{tau|z} = -p_12/sqrt(p_11 p_22), P = R^-1 . Eq. (10)

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._qr import rq_fit_batch

__all__ = ["cross_quantilogram", "cross_quantilogram_test", "partial_cross_quantilogram",
           "block_length", "stationary_bootstrap_index", "self_normalized_cv"]


# ------------------------------------------------------------------ primitives
def _hits(y: np.ndarray, taus: np.ndarray, X: Optional[np.ndarray] = None, hit: str = "lt",
          method: str = "type7") -> np.ndarray:
    """Quantile-hit matrix psi_tau(y - q_tau), shape (n, len(taus)).

    Unconditional: sample quantile (``np.quantile`` linear = R type 7). Conditional:
    linear quantile regression on ``X`` (n x d, include a constant).
    """
    y = np.asarray(y, dtype=float)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    if X is None:
        q = np.quantile(y, taus)                                # (m,)
        u = y[:, None] - q[None, :]
    else:
        B = rq_fit_batch(X, np.tile(y, (len(taus), 1)), taus)   # (m, d)
        u = y[:, None] - X @ B.T
    ind = (u < 0) if hit == "lt" else (u <= 0)
    return ind.astype(float) - taus[None, :]


def _corr(H1: np.ndarray, H2: np.ndarray) -> np.ndarray:
    """Cross-correlation of hit matrices without demeaning (HLOW Eq. 2): (m1, m2)."""
    num = H1.T @ H2
    return num / np.sqrt(np.outer((H1 ** 2).sum(0), (H2 ** 2).sum(0)))


def cross_quantilogram(y1, y2, tau1: Union[float, Sequence[float]], tau2: Union[float, Sequence[float]],
                       k: Union[int, Sequence[int]] = 1, x1: Optional[np.ndarray] = None,
                       x2: Optional[np.ndarray] = None, hit: str = "lt") -> np.ndarray:
    """Sample cross-quantilogram rho_tau(k) of Han et al. (2016, Eq. 2).

    Measures whether ``y2`` being below its tau2-quantile at t-k predicts ``y1``
    being below its tau1-quantile at t (directional predictability from y2 to y1).

    Parameters
    ----------
    y1, y2 : array-like (T,)
        Predicted series and predictor.
    tau1, tau2 : float or sequence
    k : int or sequence of int
        Lag(s), k >= 0.
    x1, x2 : ndarray (T x d), optional
        Regressors for conditional quantiles (include a constant column);
        default: unconditional sample quantiles.
    hit : {"lt", "le"}
        ``1[y < q]`` as in the paper, or ``1[y <= q]`` (the R package convention).

    Returns
    -------
    ndarray (len(k), len(tau1), len(tau2)), squeezed for scalars.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> e = rng.standard_normal(500); y2 = rng.standard_normal(500)
    >>> y1 = np.r_[0, 0.8 * y2[:-1]] + e
    >>> round(float(cross_quantilogram(y1, y2, 0.1, 0.1, 1)), 2) > 0.1
    True
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    T = len(y1)
    t1 = np.atleast_1d(tau1)
    t2 = np.atleast_1d(tau2)
    ks = np.atleast_1d(k).astype(int)
    H1 = _hits(y1, t1, x1, hit)
    H2 = _hits(y2, t2, x2, hit)
    out = np.empty((len(ks), len(t1), len(t2)))
    for i, kk in enumerate(ks):
        if kk < 0:
            raise ValueError("k must be >= 0 (swap y1 and y2 for leads)")
        out[i] = _corr(H1[kk:], H2[:T - kk])
    return out.squeeze() if (np.isscalar(tau1) and np.isscalar(tau2) and np.isscalar(k)) else out


# ---------------------------------------------------------------- bootstrap
def _flat_top(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def block_length(x, c: float = 1.959964) -> float:
    """Optimal mean block length of the stationary bootstrap for one series.

    Politis and White (2004) automatic rule with the Patton, Politis and White
    (2009) correction: b = (2 G^2 / D_SB)^(1/3) n^(1/3), D_SB = 2 g(0)^2, using a
    flat-top lag window with bandwidth 2 m_hat (m_hat: first lag after which K_n
    autocorrelations are insignificant at c sqrt(log10(n)/n)).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    Kn = max(5, int(np.ceil(np.log10(n))))
    mmax = int(np.ceil(np.sqrt(n))) + Kn
    Bmax = int(np.ceil(min(3 * np.sqrt(n), n / 3)))
    xc = x - x.mean()
    acov = np.array([(xc[h:] * xc[:n - h]).sum() / n for h in range(mmax + 1)])
    rho = acov[1:] / acov[0]
    crit = c * np.sqrt(np.log10(n) / n)
    insig = np.abs(rho) < crit
    mhat = None
    for j in range(mmax - Kn + 1):
        if insig[j:j + Kn].all():
            mhat = j + 1
            break
    if mhat is None:
        sig = np.where(~insig)[0]
        mhat = int(sig.max() + 1) if sig.size else 1
    M = min(2 * mhat, mmax)
    kk = np.arange(-M, M + 1)
    R = acov[np.abs(kk)]
    lam = _flat_top(kk / M)
    G = (lam * np.abs(kk) * R).sum()
    g0 = (lam * R).sum()
    D_sb = 2.0 * g0 ** 2
    b = (2.0 * G ** 2 / D_sb) ** (1 / 3) * n ** (1 / 3)
    return float(min(b, Bmax))


def stationary_bootstrap_index(n: int, gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap indices (geometric blocks, circular wrap)."""
    start = rng.integers(0, n, size=n)
    new_block = rng.random(n) < gamma
    new_block[0] = True
    # index = start of current block + position inside the block
    block_id = np.cumsum(new_block) - 1
    starts = start[new_block]
    first_pos = np.flatnonzero(new_block)
    offset = np.arange(n) - first_pos[block_id]
    return (starts[block_id] + offset) % n


# ------------------------------------------------------------ self-normalised
@lru_cache(maxsize=64)
def self_normalized_cv(p: int, omega: float = 0.1, alpha: float = 0.05, reps: int = 20000,
                       ngrid: int = 1000, seed: int = 20160301) -> float:
    """Critical value of B(1)' V^-1 B(1), V = int_omega^1 (B(r)-rB(1))(B(r)-rB(1))' dr (HLOW Thm 4)."""
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    r = np.arange(1, ngrid + 1) / ngrid
    keep = r >= omega
    chunk = max(1, 2000 // max(p, 1))
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        W = np.cumsum(rng.standard_normal((m, ngrid, p)), axis=1) / np.sqrt(ngrid)
        B1 = W[:, -1, :]
        D = W[:, keep, :] - r[keep][None, :, None] * B1[:, None, :]
        V = np.einsum("mti,mtj->mij", D, D) / ngrid
        out[done:done + m] = np.einsum("mi,mi->m", B1, np.linalg.solve(V, B1[..., None])[..., 0])
        done += m
    return float(np.quantile(out, 1 - alpha))


# ---------------------------------------------------------------- full test
def cross_quantilogram_test(y1, y2, tau1: Sequence[float], tau2: Sequence[float],
                            max_lag: int = 20, x1=None, x2=None, n_boot: int = 1000,
                            block: Union[str, float] = "auto", alpha: float = 0.05,
                            sup_range: Optional[Sequence[float]] = None,
                            self_normalized: bool = False, omega: float = 0.1,
                            hit: str = "lt", seed: Optional[int] = None, names=("y1", "y2"),
                            dates=None):
    """Cross-quantilogram with stationary-bootstrap inference (Han et al., 2016).

    Parameters
    ----------
    y1, y2 : array-like or Series
        Predicted series and predictor (``y2`` at t-k predicts ``y1`` at t).
    tau1, tau2 : sequence of float
        Quantile grids of y1 and y2.
    max_lag : int
        Largest lag p (paper: up to 60 for daily data).
    x1, x2 : ndarray, optional
        Regressors for conditional quantiles (with a constant); default unconditional.
    n_boot : int
        Bootstrap replications B (paper: 1,000).
    block : "auto" or float
        Mean block length 1/gamma; "auto" applies Politis-White-Patton to each series
        and averages gamma (paper, Sec. 6.1).
    alpha : float
        Significance level of bands and critical values.
    sup_range : sequence of float, optional
        Quantiles tau (with tau1 = tau2 = tau, as the paper) over which the
        sup-Box-Ljung statistic is computed, e.g. ``np.arange(0.1, 0.31, 0.02)``.
    self_normalized : bool
        Also compute the self-normalised statistic S(p) for every (tau1, tau2, p).
    seed : int, optional

    Returns
    -------
    CQResult
    """
    from ._results import CQResult
    nm1 = getattr(y1, "name", None) or names[0]
    nm2 = getattr(y2, "name", None) or names[1]
    if dates is None and isinstance(y1, pd.Series) and isinstance(y1.index, pd.DatetimeIndex):
        dates = y1.index
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if y1.shape != y2.shape or y1.ndim != 1:
        raise ValueError("y1 and y2 must be 1-D arrays of equal length")
    if np.isnan(y1).any() or np.isnan(y2).any():
        raise ValueError("missing values in y1 / y2")
    T = len(y1)
    t1 = np.sort(np.atleast_1d(np.asarray(tau1, dtype=float)))
    t2 = np.sort(np.atleast_1d(np.asarray(tau2, dtype=float)))
    p = int(max_lag)
    if not 1 <= p < T // 4:
        raise ValueError("max_lag must be in [1, T/4)")
    rng = np.random.default_rng(seed)
    lags = np.arange(1, p + 1)
    rho = cross_quantilogram(y1, y2, t1, t2, lags, x1, x2, hit)          # (p, m1, m2)
    Tk = (T - lags)[:, None, None]
    Q = T * (T + 2) * np.cumsum(rho ** 2 / Tk, axis=0)                    # Box-Ljung, (p, m1, m2)
    # tuples t = p+1..T: (y1_t, x1_t, y2_{t-1}, x2_{t-1}, ..., y2_{t-p}, x2_{t-p})
    N = T - p
    rows = np.arange(p, T)
    if block == "auto":
        b1 = block_length(y1[rows])
        b2 = block_length(y2[rows - 1])
        gamma = min(1.0, float(np.mean([1.0 / b1, 1.0 / b2])))   # b < 1 -> iid resampling
        mean_block = 1.0 / gamma
    else:
        mean_block = max(1.0, float(block))
        gamma = 1.0 / mean_block
    sup_taus = None if sup_range is None else np.sort(np.asarray(sup_range, dtype=float))
    rho_b = np.empty((n_boot, p, len(t1), len(t2)))
    sup_rho_b = None
    if sup_taus is not None:
        sup_rho_hat = cross_quantilogram(y1, y2, sup_taus, sup_taus, lags, x1, x2, hit)
        idx = np.arange(len(sup_taus))
        sup_rho_hat = sup_rho_hat[:, idx, idx]                            # (p, ns) tau1 = tau2
        sup_rho_b = np.empty((n_boot, p, len(sup_taus)))
    for b in range(n_boot):
        ii = rows[stationary_bootstrap_index(N, gamma, rng)]
        H1 = _hits(y1[ii], t1, None if x1 is None else x1[ii], hit)
        Hs1 = _hits(y1[ii], sup_taus, None if x1 is None else x1[ii], hit) if sup_taus is not None else None
        for kk in lags:
            jj = ii - kk
            H2 = _hits(y2[jj], t2, None if x2 is None else x2[jj], hit)
            rho_b[b, kk - 1] = _corr(H1, H2)
            if sup_taus is not None:
                Hs2 = _hits(y2[jj], sup_taus, None if x2 is None else x2[jj], hit)
                num = (Hs1 * Hs2).sum(0)
                sup_rho_b[b, kk - 1] = num / np.sqrt((Hs1 ** 2).sum(0) * (Hs2 ** 2).sum(0))
    cent = rho_b - rho[None]                                              # rho* - rho_hat
    lo = np.quantile(cent, alpha / 2, axis=0)
    hi = np.quantile(cent, 1 - alpha / 2, axis=0)
    pval = (np.abs(cent) >= np.abs(rho)[None]).mean(axis=0)
    Qb = T * (T + 2) * np.cumsum(cent ** 2 / Tk[None], axis=1)            # (B, p, m1, m2)
    Q_cv = np.quantile(Qb, 1 - alpha, axis=0)
    Q_p = (Qb >= Q[None]).mean(axis=0)
    sup = None
    if sup_taus is not None:
        Tk1 = (T - lags)[:, None]
        Qs = T * (T + 2) * np.cumsum(sup_rho_hat ** 2 / Tk1, axis=0)       # (p, ns)
        Qsb = T * (T + 2) * np.cumsum((sup_rho_b - sup_rho_hat[None]) ** 2 / Tk1[None], axis=1)
        stat = Qs.max(axis=1)
        dist = Qsb.max(axis=2)                                            # (B, p)
        sup = pd.DataFrame({"lag": lags, "sup_Q": stat, "cv": np.quantile(dist, 1 - alpha, axis=0),
                            "p_value": (dist >= stat[None]).mean(axis=0),
                            "argmax_tau": sup_taus[Qs.argmax(axis=1)]})
        sup.attrs["range"] = (float(sup_taus[0]), float(sup_taus[-1]))
    sn = None
    if self_normalized:
        sn = _self_normalized(y1, y2, t1, t2, p, x1, x2, hit, omega, rho)
        sn_cv = np.array([self_normalized_cv(int(pp), round(omega, 6), alpha) for pp in lags])
        sn = {"S": sn, "cv": sn_cv, "omega": omega}
    return CQResult(rho=rho, lo=lo, hi=hi, pvalues=pval, Q=Q, Q_cv=Q_cv, Q_p=Q_p, tau1=t1, tau2=t2,
                    lags=lags, T=T, n_boot=n_boot, mean_block=mean_block, alpha=alpha,
                    names=(str(nm1), str(nm2)), conditional=(x1 is not None or x2 is not None),
                    sup=sup, self_norm=sn, kind="cross-quantilogram")


def _self_normalized(y1, y2, t1, t2, p, x1, x2, hit, omega, rho_full):
    """S(p) = T rho' V^-1 rho with V = T^-2 sum_s s^2 (rho_s - rho)(rho_s - rho)' (HLOW Sec. 3.1.2)."""
    T = len(y1)
    s0 = max(int(np.floor(T * omega)), p + 10)
    lags = np.arange(1, p + 1)
    m1, m2 = len(t1), len(t2)
    V = np.zeros((m1, m2, p, p))
    for s in range(s0, T + 1):
        r = cross_quantilogram(y1[:s], y2[:s], t1, t2, lags,
                               None if x1 is None else x1[:s], None if x2 is None else x2[:s], hit)
        d = (r - rho_full).transpose(1, 2, 0)                             # (m1, m2, p)
        V += s ** 2 * np.einsum("abi,abj->abij", d, d)
    V /= T ** 2
    S = np.full((p, m1, m2), np.nan)
    for pp in range(1, p + 1):
        rr = rho_full[:pp].transpose(1, 2, 0)                             # (m1, m2, pp)
        Vp = V[:, :, :pp, :pp]
        sol = np.linalg.solve(Vp, rr[..., None])[..., 0]
        S[pp - 1] = T * (rr * sol).sum(-1)
    return S


def partial_cross_quantilogram(y1, y2, controls, tau1: float, tau2: float, tau_z: Sequence[float],
                               k: int = 1, n_boot: int = 1000, block: Union[str, float] = "auto",
                               alpha: float = 0.05, hit: str = "lt", seed: Optional[int] = None) -> dict:
    """Partial cross-quantilogram rho_{tau|z} of Han et al. (2016, Eq. 10) with bootstrap inference.

    ``controls`` (T x c) are turned into quantile hits at ``tau_z`` and enter
    contemporaneously with y2_{t-k}; rho_{tau|z} = -p_12 / sqrt(p_11 p_22) with
    P the inverse of the hit correlation matrix.

    Returns
    -------
    dict with ``rho`` (cross-quantilogram), ``partial``, ``ci`` (centered
    bootstrap band for H0: partial = 0), ``p_value`` and ``mean_block``.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    Z = np.asarray(controls, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    tz = np.atleast_1d(tau_z)
    if len(tz) != Z.shape[1]:
        raise ValueError("tau_z needs one quantile per control column")
    T = len(y1)
    D = np.column_stack([y1[k:], y2[:T - k], Z[:T - k]])
    taus = np.r_[tau1, tau2, tz]

    def stat(Dm):
        H = np.column_stack([_hits(Dm[:, j], [taus[j]], None, hit)[:, 0] for j in range(Dm.shape[1])])
        R = H.T @ H
        P = np.linalg.inv(R)
        return R[0, 1] / np.sqrt(R[0, 0] * R[1, 1]), -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])

    rho, prt = stat(D)
    rng = np.random.default_rng(seed)
    n = len(D)
    if block == "auto":
        gamma = min(1.0, float(np.mean([1.0 / block_length(D[:, 0]), 1.0 / block_length(D[:, 1])])))
    else:
        gamma = 1.0 / max(1.0, float(block))
    bs = np.array([stat(D[stationary_bootstrap_index(n, gamma, rng)])[1] for _ in range(n_boot)])
    cent = bs - prt
    return {"rho": float(rho), "partial": float(prt),
            "band": (float(np.quantile(cent, alpha / 2)), float(np.quantile(cent, 1 - alpha / 2))),
            "ci": (float(prt + np.quantile(cent, alpha / 2)), float(prt + np.quantile(cent, 1 - alpha / 2))),
            "p_value": float((np.abs(cent) >= abs(prt)).mean()), "mean_block": 1.0 / gamma}
