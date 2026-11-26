"""Quantile Granger causality robust to instability (Mayer, Wied and Troster, 2025).

Step -> equation map (MWT = Mayer, Wied and Troster, J. Econometrics 249, 105992)
--------------------------------------------------------------------------------
Q_y(tau | x_i) = x_i' beta_i(tau), x_i = (z_i', w_i')', H0: gamma_i(tau) = 0 ... Eq. (1)
psi_tau(u) = 1{u <= 0} - tau;  S_n(lambda, tau, t) = n^-1/2 sum_{i<=lambda n} x_i psi .. Eq. (3)
restricted fit alpha_n(tau) = argmin sum rho_tau(y_i - w_i' alpha) ................ Eq. (4)
H_n = (X'X/n)^{-1/2} S_n, A^{-1/2} = Cholesky factor with A^-1 = (A^-1/2)' A^-1/2  Eq. (5)
LM1(lambda, tau) = ||R' [H_n(lambda) - lambda H_n(1)]||_inf  (CUSUM, H0,1)
LM2(tau) = ||R' H_n(1)||_inf  (LM, H0,2)
LM(tau) = h(tau) (sup_lambda LM1 + LM2), h(tau) = [tau(1-tau)]^-1/2 ......... Sec. 3.1
supLM = sup_tau (sup_lambda LM1 + LM2);  expLM = int exp(1/2 (...)) dtau ....... Eq. (8)
Asymptotic null under Assumption D: Brownian bridge / pillow functionals ...... Cor. 2-3
Bootstrap: x resampled, y* = alpha_n(U)' w with U ~ U(0,1) ................... Algorithm 1
Regimes: expCUSUM on [a,b], expLM_{a,b}, Sidak levels ........................ Algorithm 2, Eq. (9)
Benchmark supWald (Koenker and Machado, 1999) with pairs-bootstrap covariance .. Eq. (10)

Extensions (clearly labelled, not in MWT): several lag orders in one table
(``qgc_table``), causality in variance (``moment=2``: squared series, as in
Nishiyama et al., 2011 and Balcilar et al., 2017), a lag x quantile map of LM(tau)
and a time x quantile CUSUM map.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._qr import rq_fit_batch

__all__ = ["qgc_instability", "qgc_table", "simulate_null"]

DEFAULT_INTERVALS = {"[0.05, 0.95]": (0.05, 0.95), "[0.05, 0.20]": (0.05, 0.20),
                     "[0.40, 0.60]": (0.40, 0.60), "[0.80, 0.95]": (0.80, 0.95)}


# ------------------------------------------------------------- design matrix
def build_design(y, x, lags: int = 1, own_lags: Optional[int] = None, controls=None,
                 moment: int = 1):
    """y_t, z_t = (x_{t-1},...,x_{t-q}), w_t = (1, y_{t-1},...,y_{t-r}, controls_t)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if moment == 2:
        y = (y - y.mean()) ** 2
        x = (x - x.mean()) ** 2
    elif moment != 1:
        raise ValueError("moment must be 1 (quantiles) or 2 (variance)")
    q = int(lags)
    r = q if own_lags is None else int(own_lags)
    m = max(q, r)
    T = len(y)
    Y = y[m:]
    Z = np.column_stack([x[m - j:T - j] for j in range(1, q + 1)])
    W = [np.ones(T - m)] + [y[m - j:T - j] for j in range(1, r + 1)]
    if controls is not None:
        C = np.asarray(controls, dtype=float)
        C = C[:, None] if C.ndim == 1 else C
        W += [C[m:, j] for j in range(C.shape[1])]
    return Y, Z, np.column_stack(W), m


def _inv_sqrt_upper(A: np.ndarray) -> np.ndarray:
    """A^{-1/2} with A^{-1} = (A^{-1/2})' A^{-1/2} (upper-triangular, MWT notation)."""
    L = np.linalg.cholesky(np.linalg.inv(A))
    return L.T


# ------------------------------------------------------------ the statistics
def _paths(Y, Z, W, taus, alpha_coef=None, seg: Optional[Tuple[int, int]] = None):
    """LM1(lambda, tau) paths and LM2(tau) on the sample (or a segment [a, b)).

    Returns LM1 (n_seg, n_tau), LM2 (n_tau,), and the restricted coefficients.
    """
    if seg is not None:
        a, b = seg
        Y, Z, W = Y[a:b], Z[a:b], W[a:b]
    n, p = Z.shape
    X = np.column_stack([Z, W])
    U = _inv_sqrt_upper(X.T @ X / n)
    V = X @ U[:p].T                                            # rows: (U_p x_i)', (n, p)
    if alpha_coef is None:
        alpha_coef = rq_fit_batch(W, np.tile(Y, (len(taus), 1)), taus)   # (n_tau, k)
    res = Y[:, None] - W @ alpha_coef.T                         # (n, n_tau)
    psi = (res <= 0).astype(float) - taus[None, :]
    G = np.cumsum(V[:, None, :] * psi[:, :, None], axis=0) / np.sqrt(n)   # (n, n_tau, p)
    lam = (np.arange(1, n + 1) / n)[:, None, None]
    LM1 = np.abs(G - lam * G[-1][None]).max(axis=2)             # (n, n_tau)
    LM2 = np.abs(G[-1]).max(axis=1)                             # (n_tau,)
    return LM1, LM2, alpha_coef


def _integral(f: np.ndarray, taus: np.ndarray, axis: int = -1) -> np.ndarray:
    if len(taus) == 1:
        return np.take(f, 0, axis=axis)
    return np.trapezoid(f, taus, axis=axis) if hasattr(np, "trapezoid") else np.trapz(f, taus, axis=axis)


def _interval_stats(A: np.ndarray, taus: np.ndarray, lo: float, hi: float):
    """supLM and expLM over [lo, hi] from A(tau) = sup_lambda LM1 + LM2 (last axis = tau)."""
    sel = (taus >= lo - 1e-9) & (taus <= hi + 1e-9)
    As = A[..., sel]
    return As.max(axis=-1), _integral(np.exp(0.5 * As), taus[sel], axis=-1)


# ---------------------------------------------------------- asymptotic nulls
@lru_cache(maxsize=32)
def _sim_null(p: int, taus_key: Tuple[float, ...], m: int = 500, reps: int = 2000, seed: int = 2025):
    """Simulated null draws under Assumption D (MWT Cor. 2-3).

    Partial sums of p independent uniform-hit processes approximate S_p(lambda, tau);
    returns A(tau) draws (reps, n_tau), the scaled fixed-tau LM draws and expCUSUM draws.
    """
    taus = np.asarray(taus_key)
    rng = np.random.default_rng(seed)
    A = np.empty((reps, len(taus)))
    cus = np.empty(reps)
    lam = (np.arange(1, m + 1) / m)[:, None, None]
    for r in range(reps):
        Uu = rng.random((m, p))
        hits = (Uu[:, None, :] <= taus[None, :, None]).astype(float) - taus[None, :, None]   # (m, n_tau, p)
        G = np.cumsum(hits, axis=0) / np.sqrt(m)
        LM1 = np.abs(G - lam * G[-1][None]).max(axis=2)
        LM2 = np.abs(G[-1]).max(axis=1)
        A[r] = LM1.max(axis=0) + LM2
        cus[r] = _integral(np.exp(0.5 * LM1), taus, axis=1).max()
    return A, cus


def simulate_null(p: int, taus: Sequence[float], m: int = 500, reps: int = 2000, seed: int = 2025):
    """Public access to the simulated asymptotic null (A(tau) draws and expCUSUM draws)."""
    return _sim_null(int(p), tuple(np.round(np.asarray(taus, dtype=float), 10)), m, reps, seed)


# ------------------------------------------------------------- bootstrap (Alg. 1)
def _bootstrap_A(Y, Z, W, taus, n_boot, rng, u_grid=None):
    """Algorithm 1: A*(tau) draws (n_boot, n_tau) and fixed-tau LM* draws."""
    n = len(Y)
    if u_grid is None:
        u_grid = np.linspace(0.005, 0.995, 199)
    a_grid = rq_fit_batch(W, np.tile(Y, (len(u_grid), 1)), u_grid)       # (n_u, k)
    out = np.empty((n_boot, len(taus)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Wb, Zb = W[idx], Z[idx]
        Ub = rng.random(n)
        pos = np.clip(np.searchsorted(u_grid, Ub) - 1, 0, len(u_grid) - 2)
        f = np.clip((Ub - u_grid[pos]) / (u_grid[pos + 1] - u_grid[pos]), 0, 1)
        coef = a_grid[pos] * (1 - f)[:, None] + a_grid[pos + 1] * f[:, None]
        Yb = (Wb * coef).sum(1)
        LM1, LM2, _ = _paths(Yb, Zb, Wb, taus)
        out[b] = LM1.max(axis=0) + LM2
    return out


def _sup_wald(Y, Z, W, taus, n_boot, rng):
    """Koenker-Machado sup-Wald with pairs-bootstrap covariance of gamma(tau) (MWT Eq. 10)."""
    n, p = Z.shape
    X = np.column_stack([Z, W])
    g = rq_fit_batch(X, np.tile(Y, (len(taus), 1)), taus)[:, :p]           # (n_tau, p)
    gb = np.empty((n_boot, len(taus), p))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        gb[b] = rq_fit_batch(X[idx], np.tile(Y[idx], (len(taus), 1)), taus)[:, :p]
    W_tau = np.empty(len(taus))
    for j in range(len(taus)):
        V = np.cov(gb[:, j, :].T).reshape(p, p)
        W_tau[j] = g[j] @ np.linalg.solve(V, g[j])
    return W_tau, g


@lru_cache(maxsize=32)
def _sim_supwald(p: int, taus_key: Tuple[float, ...], m: int = 1000, reps: int = 4000, seed: int = 99):
    """sup_tau ||BB_p(tau)||^2 / (tau(1-tau)) draws (Bessel process limit, MWT Eq. 10)."""
    taus = np.asarray(taus_key)
    rng = np.random.default_rng(seed)
    grid = np.arange(1, m + 1) / m
    idx = np.clip(np.searchsorted(grid, taus), 0, m - 1)
    out = np.empty(reps)
    for r in range(reps):
        Wm = np.cumsum(rng.standard_normal((m, p)), axis=0) / np.sqrt(m)
        BB = Wm - grid[:, None] * Wm[-1][None]
        v = (BB[idx] ** 2).sum(1) / (taus * (1 - taus))
        out[r] = v.max()
    return out


# ----------------------------------------------------------------- main test
def qgc_instability(y, x, lags: int = 1, own_lags: Optional[int] = None, controls=None,
                    taus: Sequence[float] = np.round(np.arange(0.05, 0.951, 0.01), 2),
                    intervals: Optional[Dict[str, Tuple[float, float]]] = None,
                    n_boot: int = 499, method: str = "bootstrap", moment: int = 1,
                    sup_wald: bool = True, regimes: bool = True, alpha: float = 0.05,
                    seed: Optional[int] = None, names=("y", "x"), dates=None):
    """Quantile Granger causality test robust to instability (Mayer, Wied and Troster, 2025).

    Tests H0: the lagged cause has no effect on the conditional quantiles of ``y``
    (H0,2) AND that effect is constant over time (H0,1), against breaks and/or
    Granger causality, at each tau (LM(tau)) and uniformly over quantile ranges
    (supLM, expLM).

    Parameters
    ----------
    y, x : array-like or Series
        Effect and candidate cause.
    lags : int
        Number q of lags of ``x`` in z_i (MWT application: q in {1, 3, 6, 9}).
    own_lags : int, optional
        Lags of ``y`` in the controls w_i (default q).
    controls : array-like (T x c), optional
        Additional contemporaneous controls in w_i.
    taus : sequence
        Quantile grid (MWT: 0.05, 0.06, ..., 0.95).
    intervals : dict, optional
        Quantile ranges for supLM / expLM / supWald (default: MWT Table 3 ranges).
    n_boot : int
        Bootstrap replications (MWT: 499; Fig. 2: 1,999).
    method : {"bootstrap", "asymptotic"}
        Critical values from Algorithm 1 (valid without Assumption D) or from the
        simulated pivotal limit (valid under Assumption D).
    moment : {1, 2}
        1 = causality in quantiles; 2 = causality in variance (squared demeaned series;
        an extension in the spirit of Nishiyama et al., 2011 and Balcilar et al., 2017).
    sup_wald : bool
        Also compute the Koenker-Machado supWald benchmark with bootstrap covariance.
    regimes : bool
        Run Algorithm 2 (break dating and regime-wise causality, at most two breaks).
    alpha : float
        Initial significance level of Algorithm 2 (MWT Table 4: 0.10) and of the tests.

    Returns
    -------
    GCResult

    References
    ----------
    Mayer, A., Wied, D. and Troster, V. (2025). Quantile Granger causality in the presence
    of instability. Journal of Econometrics, 249, 105992. https://doi.org/10.1016/j.jeconom.2025.105992
    """
    from ._results import GCResult
    nm_y = getattr(y, "name", None) or names[0]
    nm_x = getattr(x, "name", None) or names[1]
    if dates is None and isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
        dates = y.index
    taus = np.asarray(taus, dtype=float)
    intervals = intervals or DEFAULT_INTERVALS
    rng = np.random.default_rng(seed)
    Y, Z, W, m = build_design(y, x, lags, own_lags, controls, moment)
    n, p = Z.shape
    LM1, LM2, acoef = _paths(Y, Z, W, taus)
    A = LM1.max(axis=0) + LM2                                    # (n_tau,)
    h = 1.0 / np.sqrt(taus * (1 - taus))
    LM_tau = h * A
    # null distributions
    simA, sim_cus = _sim_null(p, tuple(np.round(taus, 10)))
    sim_LM = simA / np.sqrt(taus * (1 - taus))[None]
    bootA = _bootstrap_A(Y, Z, W, taus, n_boot, rng) if method == "bootstrap" else None
    refA = bootA if bootA is not None else simA
    p_tau = ((refA / np.sqrt(taus * (1 - taus))[None]) >= LM_tau[None]).mean(axis=0)
    p_tau_asy = (sim_LM >= LM_tau[None]).mean(axis=0)
    rows = []
    W_tau = gW = simW = None
    if sup_wald:
        W_tau, gW = _sup_wald(Y, Z, W, taus, n_boot, rng)
    for lab, (lo, hi) in intervals.items():
        s, e = _interval_stats(A, taus, lo, hi)
        rs, re = _interval_stats(refA, taus, lo, hi)
        row = {"interval": lab, "supLM": float(s), "p_supLM": float((rs >= s).mean()),
               "expLM": float(e), "p_expLM": float((re >= e).mean())}
        if sup_wald:
            sel = (taus >= lo - 1e-9) & (taus <= hi + 1e-9)
            sw = float(W_tau[sel].max())
            simW = _sim_supwald(p, tuple(np.round(taus[sel], 10)))
            row.update({"supWald": sw, "p_supWald": float((simW >= sw).mean())})
        rows.append(row)
    tests = pd.DataFrame(rows).set_index("interval")
    # CUSUM curve over the full sample (MWT Fig. 2)
    cusum_path = _integral(np.exp(0.5 * LM1), taus, axis=1)
    cus_cv = float(np.quantile(sim_cus, (1 - alpha) ** 0.5))
    reg = None
    if regimes:
        reg = _algorithm2(Y, Z, W, taus, alpha, tests.iloc[0]["p_expLM"], p)
    dts = None
    if dates is not None:
        dts = pd.to_datetime(pd.Index(dates))[m:]
    return GCResult(LM_tau=LM_tau, p_tau=p_tau, p_tau_asy=p_tau_asy, A=A, LM1=LM1, LM2=LM2, taus=taus,
                    tests=tests, cusum=cusum_path, cusum_cv=cus_cv, regimes=reg, n=n, p=p, k=W.shape[1],
                    lags=int(lags), method=method, n_boot=n_boot, moment=moment, alpha=alpha,
                    names=(str(nm_y), str(nm_x)), dates=dts, wald_tau=W_tau, gamma_hat=gW,
                    alpha_coef=acoef)


# ------------------------------------------------------------- Algorithm 2
def _seg_stats(Y, Z, W, taus, a, b):
    LM1, LM2, _ = _paths(Y, Z, W, taus, seg=(a, b))
    A = LM1.max(axis=0) + LM2
    expLM = float(_integral(np.exp(0.5 * A), taus))
    cpath = _integral(np.exp(0.5 * LM1), taus, axis=1)
    return expLM, float(cpath.max()), int(a + np.argmax(cpath)), cpath


def _algorithm2(Y, Z, W, taus, alpha, p_full, p):
    """Regime identification with at most two breaks (MWT Algorithm 2, Sidak levels)."""
    n = len(Y)
    simA, sim_cus = _sim_null(p, tuple(np.round(taus, 10)))
    sim_exp = _integral(np.exp(0.5 * simA), taus, axis=1)

    def p_exp(v):
        return float((sim_exp >= v).mean())

    def p_cus(v, nmax=1):
        F = (sim_cus[:, None] < v).mean()
        return float(1 - F ** nmax)

    lev = lambda j: 1 - (1 - alpha) ** (1 / j)
    out = {"breaks": [], "segments": [], "steps": []}
    if p_full > alpha:
        out["steps"].append(f"Step 1: expLM p = {p_full:.3f} > {alpha}: no Granger causality detected; stop.")
        out["segments"] = [(0, n, False, p_full)]
        return out
    out["steps"].append(f"Step 1: expLM p = {p_full:.3f} <= {alpha}: reject H0.")
    e, c, l1, _ = _seg_stats(Y, Z, W, taus, 0, n)
    pc = p_cus(c)
    if pc > lev(2):
        out["steps"].append(f"Step 2: expCUSUM p = {pc:.3f} > {lev(2):.4f}: stable Granger causality on the whole sample.")
        out["segments"] = [(0, n, True, p_full)]
        return out
    l1 = int(np.clip(l1 + 1, 10, n - 10))
    out["breaks"].append(l1)
    out["steps"].append(f"Step 2: expCUSUM p = {pc:.3f} <= {lev(2):.4f}: break at obs. {l1}.")
    eL, _, _, _ = _seg_stats(Y, Z, W, taus, 0, l1)
    eR, cR, lR, _ = _seg_stats(Y, Z, W, taus, l1, n)
    pL, pR = p_exp(eL), p_exp(eR)
    a3 = lev(4)
    rejL, rejR = pL <= a3, pR <= a3
    out["steps"].append(f"Step 3: expLM[0,l1] p = {pL:.3f}, expLM[l1,1] p = {pR:.3f} (level {a3:.4f}).")
    if not rejL and not rejR:
        out["steps"].append("Inconclusive on [0, 1].")
        out["segments"] = [(0, l1, None, pL), (l1, n, None, pR)]
        return out
    if rejL != rejR:
        gc_side = (l1, n) if rejR else (0, l1)
        nogc = (0, l1) if rejR else (l1, n)
        _, cS, lS, _ = _seg_stats(Y, Z, W, taus, *gc_side)
        pcs = p_cus(cS)
        if pcs > lev(5):
            out["steps"].append(f"expCUSUM on GC segment p = {pcs:.3f} > {lev(5):.4f}: no further break.")
            segs = [(nogc[0], nogc[1], False, pL if nogc[0] == 0 else pR),
                    (gc_side[0], gc_side[1], True, pR if rejR else pL)]
        else:
            l2 = int(np.clip(lS + 1, gc_side[0] + 10, gc_side[1] - 10))
            out["breaks"].append(l2)
            e1, _, _, _ = _seg_stats(Y, Z, W, taus, gc_side[0], l2)
            e2, _, _, _ = _seg_stats(Y, Z, W, taus, l2, gc_side[1])
            p1, p2 = p_exp(e1), p_exp(e2)
            a7 = lev(7)
            out["steps"].append(f"Second break at obs. {l2}; expLM p = {p1:.3f}, {p2:.3f} (level {a7:.4f}).")
            segs = [(nogc[0], nogc[1], False, pL if nogc[0] == 0 else pR),
                    (gc_side[0], l2, p1 <= a7 if (p1 <= a7 or p2 <= a7) else None, p1),
                    (l2, gc_side[1], p2 <= a7 if (p1 <= a7 or p2 <= a7) else None, p2)]
        out["segments"] = sorted(segs)
        return out
    # both sides reject
    _, cA, lA, _ = _seg_stats(Y, Z, W, taus, 0, l1)
    _, cB, lB, _ = _seg_stats(Y, Z, W, taus, l1, n)
    cm = max(cA, cB)
    pm = p_cus(cm, 2)
    if pm > lev(5):
        out["steps"].append(f"max expCUSUM p = {pm:.3f} > {lev(5):.4f}: Granger causality on both segments.")
        out["segments"] = [(0, l1, True, pL), (l1, n, True, pR)]
        return out
    l2 = int(np.clip((lA if cA >= cB else lB) + 1, 10, n - 10))
    out["breaks"].append(l2)
    lo_, hi_ = sorted([l1, l2])
    pts = [0, lo_, hi_, n]
    segs = []
    a7 = lev(7)
    for s0, s1 in zip(pts[:-1], pts[1:]):
        ee, _, _, _ = _seg_stats(Y, Z, W, taus, s0, s1)
        pp = p_exp(ee)
        segs.append((s0, s1, pp <= a7, pp))
    out["steps"].append(f"Second break at obs. {l2}; segment p-values: " + ", ".join(f"{s[3]:.3f}" for s in segs))
    out["segments"] = segs
    return out


def qgc_table(y, x, lags: Sequence[int] = (1, 3, 6, 9), **kwargs) -> pd.DataFrame:
    """p-values (%) of supWald, expLM and supLM by lag order and quantile range (MWT Table 3 layout)."""
    kwargs.setdefault("regimes", False)
    frames = []
    for q in lags:
        r = qgc_instability(y, x, lags=q, **kwargs)
        t = r.tests[[c for c in ("p_supWald", "p_expLM", "p_supLM") if c in r.tests]] * 100
        t.columns = [c[2:] for c in t.columns]
        t = t.stack().to_frame(f"q = {q}").T
        frames.append(t)
    out = pd.concat(frames)
    out.columns.names = ["interval", "test"]
    return out.round(3)
