"""Quantile-on-quantile ARDL (QQARDL): Python port of the author's Stata command ``qqardl`` 1.1.0.

Step -> equation map (Destek, Radulescu, Ozkan and Balsalobre-Lorente, 2025, JEM 389, 126089)
--------------------------------------------------------------------------------------------
S1  quantile series QSER_tau(y), QSER_theta(x) of Li (2022/2025), as qfa::qser:
    QSER_tau(t) = q_tau + sum_k [c1(k) cos(2 pi k s/n) + c2(k) sin(2 pi k s/n)] + c1(n/2)(-1)^s,
    (c1, c2) from Q_tau(y_t) = c0 + c1 cos(2 pi k t/n) + c2 sin(2 pi k t/n) (Frisch-Newton)
S2  bounds-test regression, OLS ................................................... Eq. (1)
    D.Y = c + sum_{i=1..a} w_i D.Y(t-i) + sum_{j=1..b} v_j D.X(t-j) + g1 Y(t-1) + g2 X(t-1) + e
    Wald F of g1 = g2 = 0 (classical variance) ...................................... Eqs. (2)-(3)
    p-values: Kripfganz and Schneider (2020) response surfaces, case III, I(1) bound;
    cointegration if p < 0.10 (Destek et al., p. 4)
S3  error-correction form, OLS .................................................... Eq. (4)
    D.Y = a0 + d0 Y(t-1) + d1 D.X(t) + b1 X(t-1) + e   (ARDL(1,1); (p, q) generalised)
S4  ECT = d0, short run = d1, long run = -b1/d0 (delta method) or b1 (lrtype="beta")
S5  grid tau, theta in {0.05, 0.10, 0.20, ..., 0.90, 0.95} ............................ Eq. (5), fn. 1

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._qr import rq_fit_batch

__all__ = ["qser", "qqardl", "ardl_bounds_pvalue", "PSS_CASE3_I1"]

DEFAULT_QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

#: Pesaran, Shin and Smith (2001, Table CI(iii), case III) asymptotic bounds (I(0), I(1)).
PSS_CASE3 = {1: {0.10: (4.04, 4.78), 0.05: (4.94, 5.73), 0.01: (6.84, 7.84)},
             2: {0.10: (3.17, 4.14), 0.05: (3.79, 4.85), 0.01: (5.15, 6.36)},
             3: {0.10: (2.72, 3.77), 0.05: (3.23, 4.35), 0.01: (4.29, 5.61)},
             4: {0.10: (2.45, 3.52), 0.05: (2.86, 4.01), 0.01: (3.74, 5.06)}}
PSS_CASE3_I1 = {k: {a: v[1] for a, v in d.items()} for k, d in PSS_CASE3.items()}


# --------------------------------------------------------------------- S1: QSER
def _q7(y: np.ndarray, tau: float) -> float:
    """Sample quantile, R type 7."""
    return float(np.quantile(y, tau, method="linear"))


def qser(y, taus: Sequence[float]) -> np.ndarray:
    """Quantile series of Li (2022/2025), operation by operation as ``qfa::qser``.

    For every Fourier frequency k/n (k = 1..floor((n-1)/2)) and level tau, the
    trigonometric quantile regression on (1, cos, sin) is solved by Frisch-Newton;
    the Nyquist cosine term is added for even n.

    Returns
    -------
    ndarray (n, len(taus))

    Examples
    --------
    >>> import numpy as np
    >>> y = np.sin(np.arange(40) / 3) + np.random.default_rng(0).normal(0, .1, 40)
    >>> qser(y, [0.5]).shape
    (40, 1)
    """
    y = np.asarray(y, dtype=float)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    n = len(y)
    t = np.arange(1, n + 1)
    s = np.arange(n)
    out = np.tile(np.array([_q7(y, tt) for tt in taus]), (n, 1))
    half = (n - 1) // 2
    if half >= 1:
        ks = np.arange(1, half + 1)
        X = np.stack([np.column_stack([np.ones(n), np.cos(2 * np.pi * k / n * t), np.sin(2 * np.pi * k / n * t)])
                      for k in ks])                                         # (K, n, 3)
        Xb = np.repeat(X, len(taus), axis=0)
        tb = np.tile(taus, len(ks))
        B = rq_fit_batch(Xb, np.tile(y, (len(tb), 1)), tb).reshape(len(ks), len(taus), 3)
        C = np.cos(2 * np.pi * np.outer(s, ks) / n)                         # (n, K)
        S = np.sin(2 * np.pi * np.outer(s, ks) / n)
        out += C @ B[:, :, 1] + S @ B[:, :, 2]
    if n % 2 == 0:
        X = np.column_stack([np.ones(n), np.cos(np.pi * t)])
        B = rq_fit_batch(X, np.tile(y, (len(taus), 1)), taus)
        out += np.outer(np.cos(np.pi * s), B[:, 1])
    return out


# ---------------------------------------------------------------- S2-S4: OLS
def _ecm(yv, xv, p, q, vce):
    """ARDL(p, q) error-correction OLS (Eq. 4). Same return order as the Stata Mata code."""
    n = len(yv)
    t0 = max(p, q)                        # 0-based first usable index (Stata: max(p,q)+1)
    idx = np.arange(t0, n)
    N = len(idx)
    Y = yv[idx] - yv[idx - 1]
    cols = [np.ones(N), yv[idx - 1], xv[idx - 1]]
    for i in range(1, p):
        cols.append(yv[idx - i] - yv[idx - i - 1])
    for i in range(0, q):
        cols.append(xv[idx - i] - xv[idx - i - 1])
    X = np.column_stack(cols)
    k = X.shape[1]
    XXi = np.linalg.pinv(X.T @ X)
    b = XXi @ X.T @ Y
    e = Y - X @ b
    s2 = e @ e / (N - k)
    if vce == "robust":
        V = XXi @ ((X * e[:, None]).T @ (X * e[:, None])) @ XXi * (N / (N - k))
    else:
        V = s2 * XXi
    R = np.zeros((2, k))
    R[0, 1] = R[1, 2] = 1
    Rb = R @ b
    F = float(Rb @ np.linalg.solve(R @ (s2 * XXi) @ R.T, Rb) / 2)
    a1, b1 = b[1], b[2]
    lr = -b1 / a1
    g = np.array([b1 / a1 ** 2, -1 / a1])
    vlr = g @ V[1:3, 1:3] @ g
    sr_i = 2 + p                          # 0-based column of D.X(t)
    return dict(ect=a1, ect_se=np.sqrt(V[1, 1]), b1=b1, b1_se=np.sqrt(V[2, 2]), sr=b[sr_i],
                sr_se=np.sqrt(V[sr_i, sr_i]), F=F, t_bdm=a1 / np.sqrt(s2 * XXi[1, 1]), lr=lr,
                lr_se=np.sqrt(vlr), r2=1 - e @ e / ((Y - Y.mean()) @ (Y - Y.mean())), N=N, k=k)


def _btest(yv, xv, a, b):
    """Bounds-test regression of Destek et al. (2025), Eq. (1): returns (F, t on g1, N)."""
    n = len(yv)
    t0 = max(a, b) + 1                    # 0-based (Stata: max(a,b)+2)
    idx = np.arange(t0, n)
    N = len(idx)
    Y = yv[idx] - yv[idx - 1]
    cols = [np.ones(N), yv[idx - 1], xv[idx - 1]]
    for i in range(1, a + 1):
        cols.append(yv[idx - i] - yv[idx - i - 1])
    for i in range(1, b + 1):
        cols.append(xv[idx - i] - xv[idx - i - 1])
    X = np.column_stack(cols)
    k = X.shape[1]
    XXi = np.linalg.pinv(X.T @ X)
    bh = XXi @ X.T @ Y
    e = Y - X @ bh
    s2 = e @ e / (N - k)
    R = np.zeros((2, k))
    R[0, 1] = R[1, 2] = 1
    Rb = R @ bh
    F = float(Rb @ np.linalg.solve(R @ (s2 * XXi) @ R.T, Rb) / 2)
    return F, bh[1] / np.sqrt(s2 * XXi[1, 1]), N


# ------------------------------------------------------- bounds-test p-values
def _find_ks_file(path: Optional[str] = None) -> Optional[Path]:
    """Locate Kripfganz-Schneider ``ardl_surfreg_coefs.dta`` (shipped with Stata's ardl package)."""
    cands = []
    if path:
        cands.append(Path(path))
    env = os.environ.get("QQFRONTIER_ARDL_COEFS")
    if env:
        cands.append(Path(env))
    for root in ("C:/ado/plus/a", os.path.expanduser("~/ado/plus/a"), "/usr/local/ado/plus/a",
                 os.path.expanduser("~/Library/Application Support/Stata/ado/plus/a")):
        cands.append(Path(root) / "ardl_surfreg_coefs.dta")
    for c in cands:
        if c.is_file():
            return c
    return None


_KS_CACHE = {}


def ardl_bounds_pvalue(stat: float, n: int, k: int = 1, sr: int = 2, case: int = 3, which: str = "F",
                       order: int = 1, coefs_path: Optional[str] = None) -> float:
    """Approximate p-value of the PSS bounds test from Kripfganz and Schneider (2020).

    Finite-sample response surface (cubic in 1/n, sr) for the 99 quantiles of the null
    distribution, then MacKinnon (1996, Eq. 12) local regression of the inverse F (or t)
    tail on the predicted critical values at the 9 quantiles nearest to ``stat``
    (as Stata's ``ardlbounds``).  ``order`` = 1 gives the I(1) bound, 0 the I(0) bound.

    Requires the coefficient file ``ardl_surfreg_coefs.dta`` of the Stata ``ardl`` package
    (searched in the Stata ado path, or given by ``coefs_path`` / the environment variable
    ``QQFRONTIER_ARDL_COEFS``); returns NaN if it cannot be found.

    References
    ----------
    Kripfganz, S. and Schneider, D. C. (2020). Response surface regressions for critical value
    bounds and approximate p-values in equilibrium correction models. Oxford Bulletin of
    Economics and Statistics, 82(6), 1456-1481. https://doi.org/10.1111/obes.12377
    """
    f = _find_ks_file(coefs_path)
    if f is None:
        return float("nan")
    key = str(f)
    if key not in _KS_CACHE:
        _KS_CACHE[key] = pd.read_stata(f)
    D = _KS_CACHE[key]
    D = D[(D["stat"] == which) & (D["c"] == case) & (D["I"] == order)].reset_index(drop=True)
    cv = (D["theta_0_2_0"] + D["theta_0_2_1"] * sr) / n ** 2 + (D["theta_0_3_0"] + D["theta_0_3_1"] * sr) / n ** 3
    for j in range(5):
        cv = cv + D[f"theta_{j}_0_0"] / (k + 1) ** j
        cv = cv + D[f"theta_{j}_1_0"] / ((k + 1) ** j * n) + D[f"theta_{j}_1_1"] * sr / ((k + 1) ** j * n)
    cv = cv.to_numpy(dtype=float)
    df1 = 1 + k + (case in (2, 4))
    df2 = n - (1 + k + sr + (case >= 2) + (case >= 4))
    if stat > cv.max():
        return 0.0 if which == "F" else 1.0
    if stat < cv.min():
        return 1.0 if which == "F" else 0.0
    pos = int(np.argmin(np.abs(cv - stat)))
    sel = np.arange(max(0, pos - 4), min(len(cv), pos + 5))
    p = D["p"].to_numpy(dtype=float)[sel] / 10000
    if which == "F":
        inv = stats.f.isf(p, df1, df2)
    else:
        inv = -stats.t.isf(p, df2)
    Xr = np.column_stack([np.ones(len(sel)), cv[sel], cv[sel] ** 2])
    beta = np.linalg.lstsq(Xr, inv, rcond=None)[0]
    crit = beta[0] + beta[1] * stat + beta[2] * stat ** 2
    return float(stats.f.sf(crit, df1, df2)) if which == "F" else float(1 - stats.t.sf(crit, df2))


# ------------------------------------------------------------------- main
def qqardl(y, x, tau: Sequence[float] = DEFAULT_QUANTILES, theta: Optional[Sequence[float]] = None,
           lags: Tuple[int, int] = (1, 1), btest: str = "eq1", btlags: Tuple[int, int] = (1, 1),
           vce: str = "ols", lrtype: str = "multiplier", lrall: bool = False,
           coint_level: float = 0.10, coefs_path: Optional[str] = None, names=("y", "x"), index=None):
    """Quantile-on-quantile ARDL of Destek et al. (2025) (port of the Stata command ``qqardl``).

    Parameters
    ----------
    y, x : array-like or Series
        Dependent variable and regressor, a contiguous time span without gaps (n >= 30).
    tau, theta : sequence of float
        Quantile levels of y (columns) and x (rows); default {0.05, 0.10, 0.20, ..., 0.90, 0.95}.
    lags : (p, q)
        ARDL order of the error-correction form, Eq. (4) (default (1, 1)).
    btest : {"eq1", "ecm"}
        Bounds test from Eq. (1) with ``btlags`` = (a, b) (default) or from Eq. (4).
    vce : {"ols", "robust"}
        Variance of the reported coefficients (the F test always uses the classical variance).
    lrtype : {"multiplier", "beta"}
        Long run as -b1/d0 (delta-method s.e.) or b1 itself.
    lrall : bool
        Report the long run for every pair; by default only for cointegrated pairs (p < 0.10).
    coefs_path : str, optional
        Path of the Kripfganz-Schneider coefficient file (else searched in the Stata ado path;
        without it the decision uses the PSS (2001) asymptotic 10% I(1) bound).

    Returns
    -------
    QQARDLResult

    References
    ----------
    Destek, M. A., Radulescu, M., Ozkan, O. and Balsalobre-Lorente, D. (2025). The hidden cost of
    renewable energy: A quantile view for environmental management on critical metals.
    Journal of Environmental Management, 389, 126089. https://doi.org/10.1016/j.jenvman.2025.126089
    """
    from ._results import QQARDLResult
    nm_y = getattr(y, "name", None) or names[0]
    nm_x = getattr(x, "name", None) or names[1]
    if index is None and isinstance(y, pd.Series):
        index = y.index
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    if yv.shape != xv.shape or yv.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of equal length")
    if np.isnan(yv).any() or np.isnan(xv).any():
        raise ValueError("y / x contain missing values; QSER needs a contiguous span without gaps")
    n = len(yv)
    if n < 30:
        raise ValueError(f"at least 30 observations are required (found {n})")
    if btest not in ("eq1", "ecm"):
        raise ValueError("btest must be 'eq1' (Destek et al. 2025, Eq. 1) or 'ecm' (Eq. 4)")
    if vce not in ("ols", "robust"):
        raise ValueError("vce must be 'ols' or 'robust'")
    if lrtype not in ("multiplier", "beta"):
        raise ValueError("lrtype must be 'multiplier' or 'beta'")
    taus = np.asarray(tau, dtype=float)
    ths = taus if theta is None else np.asarray(theta, dtype=float)
    p, q = int(lags[0]), int(lags[1])
    ba, bb = int(btlags[0]), int(btlags[1])
    QY = qser(yv, taus)
    QX = qser(xv, ths)
    shape = (len(ths), len(taus))
    out = {k: np.full(shape, np.nan) for k in ("F", "FT", "ECT", "ECTSE", "LR", "LRSE", "SR", "SRSE", "R2")}
    N = NB = None
    for i in range(len(ths)):
        for j in range(len(taus)):
            r = _ecm(QY[:, j], QX[:, i], p, q, vce)
            out["ECT"][i, j], out["ECTSE"][i, j] = r["ect"], r["ect_se"]
            out["SR"][i, j], out["SRSE"][i, j] = r["sr"], r["sr_se"]
            if lrtype == "beta":
                out["LR"][i, j], out["LRSE"][i, j] = r["b1"], r["b1_se"]
            else:
                out["LR"][i, j], out["LRSE"][i, j] = r["lr"], r["lr_se"]
            out["R2"][i, j] = r["r2"]
            N = r["N"]
            if btest == "ecm":
                out["F"][i, j], out["FT"][i, j], NB = r["F"], r["t_bdm"], N
            else:
                out["F"][i, j], out["FT"][i, j], NB = _btest(QY[:, j], QX[:, i], ba, bb)
    sr_n = ba + bb if btest == "eq1" else (p - 1) + q
    df = N - (3 + (p - 1) + q)
    tp = lambda est, se: 2 * stats.t.sf(np.abs(est / se), df)
    FP = np.full(shape, np.nan)
    have_ks = _find_ks_file(coefs_path) is not None
    if have_ks:
        for i in range(shape[0]):
            for j in range(shape[1]):
                FP[i, j] = ardl_bounds_pvalue(out["F"][i, j], NB, 1, sr_n, 3, "F", 1, coefs_path)
        coint = FP < coint_level
        pv_source = "Kripfganz-Schneider (2020) I(1) p-values"
    else:
        cv = PSS_CASE3_I1[1][0.10 if coint_level >= 0.10 else 0.05 if coint_level >= 0.05 else 0.01]
        coint = out["F"] > cv
        pv_source = f"PSS (2001) asymptotic I(1) bound, case III (F > {cv})"
    LRD = out["LR"].copy()
    if not lrall:
        LRD[~coint] = np.nan
    return QQARDLResult(taus=taus, thetas=ths, F=out["F"], F_p=FP, t_bdm=out["FT"], ect=out["ECT"],
                        ect_se=out["ECTSE"], ect_p=tp(out["ECT"], out["ECTSE"]), lr=out["LR"],
                        lr_se=out["LRSE"], lr_p=tp(out["LR"], out["LRSE"]), lr_display=LRD,
                        sr=out["SR"], sr_se=out["SRSE"], sr_p=tp(out["SR"], out["SRSE"]), r2=out["R2"],
                        coint=coint, N=N, NB=NB, df=df, qser_y=QY, qser_x=QX, y=yv, x=xv,
                        names=(str(nm_y), str(nm_x)), lags=(p, q), btest=btest, btlags=(ba, bb), vce=vce,
                        lrtype=lrtype, lrall=lrall, coint_level=coint_level, pv_source=pv_source,
                        index=None if index is None else pd.Index(index))
