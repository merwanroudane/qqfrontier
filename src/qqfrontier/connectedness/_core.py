"""Quantile-on-quantile connectedness estimators (port of the R package qqconnect).

* ``qqc_gs``     - Gabauer and Stenfors (2024): rolling QVAR, equation j at its own tau_j.
* ``qqc_kernel`` - Bonaccolto, Caporin and Gupta (2026): kernel-weighted local
                   Q-on-Q-QVAR with Bondell et al. (2010) conditional probability levels.
* ``qqc_tvp``    - Candelo-Viafara and Orrego-Reyes (2026): TVP-QQVAR from dynamic
                   quantile linear models and a forgetting-factor covariance.

All three share the GFEVD and the Diebold-Yilmaz measures in ``_fevd.py``.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import itertools
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._qr import _rq_br, rq_fit_batch, rq_noncross
from ._fevd import conn_summary, fevd_batch
from ._results import QQConnResult

__all__ = ["qqc_gs", "qqc_kernel", "qqc_tvp", "qqconnect", "cond_pit", "dqlm"]


# --------------------------------------------------------------------------- helpers
def _as_data(x):
    """Numeric matrix, dates and names from a DataFrame / ndarray (Date column detected)."""
    dates = None
    if isinstance(x, pd.DataFrame):
        df = x.copy()
        df = df.loc[:, [c for c in df.columns if df[c].dtype != bool]]
        dcols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)
                 or str(c).lower() == "date"]
        if dcols:
            dates = pd.to_datetime(df[dcols[0]]).dt.strftime("%Y-%m-%d").tolist()
            df = df.drop(columns=dcols)
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.strftime("%Y-%m-%d").tolist()
        names = [str(c) for c in df.columns]
        X = df.to_numpy(dtype=float)
    else:
        X = np.asarray(x, dtype=float)
        if X.ndim != 2:
            raise ValueError("'x' must be a 2-D array (T x k)")
        names = [f"y{i + 1}" for i in range(X.shape[1])]
    if X.shape[1] < 2:
        raise ValueError("'x' must contain at least two series (columns)")
    if np.isnan(X).any():
        raise ValueError("'x' contains missing values; remove or impute them first")
    if dates is None:
        dates = [str(i + 1) for i in range(X.shape[0])]
    return X, list(dates), names


def _tau_configs(taus, k: int) -> np.ndarray:
    """Quantile configurations (rows) x variables (cols); a vector gives the full grid."""
    t = np.asarray(taus, dtype=float)
    if t.ndim == 2:
        if t.shape[1] != k:
            raise ValueError("'taus' given as a matrix must have one column per variable")
        cfg = t
    else:
        u = np.unique(t)
        # first variable varies fastest, as R's expand.grid
        cfg = np.array([c[::-1] for c in itertools.product(*[u] * k)])
        if len(cfg) > 5000:
            warnings.warn(f"The quantile grid has {len(cfg)} configurations; this may be slow.")
    if (cfg <= 0).any() or (cfg >= 1).any():
        raise ValueError("All quantile levels must lie strictly in (0, 1)")
    return cfg


def _var_design(x: np.ndarray, nlag: int):
    """Y (rows t = nlag+1..T) and X = [1, y_{t-1}', ..., y_{t-p}'] (as stats::embed)."""
    T, k = x.shape
    Y = x[nlag:]
    lags = [x[nlag - l:T - l] for l in range(1, nlag + 1)]
    X = np.column_stack([np.ones(T - nlag)] + lags)
    return Y, X


def _resid_cov(R: np.ndarray, cov_type: str) -> np.ndarray:
    """Uncentered second moments R'R/n or the sample covariance (divisor n-1), batched."""
    if cov_type == "uncentered":
        return np.einsum("...ti,...tj->...ij", R, R) / R.shape[-2]
    Rc = R - R.mean(axis=-2, keepdims=True)
    return np.einsum("...ti,...tj->...ij", Rc, Rc) / (R.shape[-2] - 1)


def _parallel(fun, items, n_jobs: int, progress: bool = False, desc: str = ""):
    """Map ``fun`` over ``items`` serially or with joblib (optional dependency)."""
    items = list(items)
    if n_jobs == 1 or len(items) < 2:
        out = []
        for i, it in enumerate(items):
            out.append(fun(it))
            if progress and (i + 1) % max(1, len(items) // 20) == 0:
                print(f"\r{desc} {100 * (i + 1) / len(items):5.1f}%", end="", flush=True)
        if progress:
            print()
        return out
    try:
        from joblib import Parallel, delayed
    except ImportError:  # pragma: no cover
        warnings.warn("joblib not installed: running serially (pip install qqfrontier[fast])")
        return _parallel(fun, items, 1, progress, desc)
    return Parallel(n_jobs=n_jobs, verbose=5 if progress else 0)(delayed(fun)(it) for it in items)


def _check_int(v, name, lo, hi):
    if not (isinstance(v, (int, np.integer)) or float(v).is_integer()) or v < lo or v > hi:
        raise ValueError(f"'{name}' must be an integer in [{lo}, {hi}]")
    return int(v)


# ------------------------------------------------------------------ Gabauer-Stenfors
def qqc_gs(x, taus: Union[Sequence[float], np.ndarray] = np.arange(0.05, 1.0, 0.10),
           nlag: int = 1, window: Optional[int] = 200, horizon: int = 20,
           cov_type: str = "uncentered", corrected: bool = True, generalized: bool = True,
           rq_method: str = "fn", keep_fevd: bool = False, chunk: int = 64,
           progress: bool = False) -> QQConnResult:
    """Rolling quantile-on-quantile connectedness of Gabauer and Stenfors (2024).

    In every rolling window a QVAR(p) is estimated equation by equation, equation
    ``j`` at its own quantile level ``tau_j``; for each configuration
    ``(tau_1, ..., tau_k)`` the slope matrices and the residual second-moment
    matrix give a GFEVD and the Diebold-Yilmaz measures.

    Parameters
    ----------
    x : DataFrame or ndarray (T x k)
        Series in columns; a ``date`` (datetime) column or a DatetimeIndex is used for dates.
    taus : array-like or ndarray (C x k)
        Quantile levels (the full k-dimensional grid is formed) or explicit configurations.
    nlag : int
        QVAR lag order (GS: 1, by BIC).
    window : int or None
        Rolling-window length (GS: 200 days); None uses the full sample once.
    horizon : int
        Forecast horizon H of the GFEVD (GS: 20).
    cov_type : {"uncentered", "centered"}
        Residual matrix: uncentered second moments (GS and ``ConnectednessApproach``)
        or the centered covariance (Bonaccolto et al., 2026).
    corrected : bool
        Adjusted TCI, k/(k-1) x mean(TO) (Chatziantoniou et al., 2021).
    generalized : bool
        Generalized (True) or Cholesky FEVD.
    rq_method : {"fn", "br"}
        Frisch-Newton (as ``quantreg``'s ``"fn"``, the qqconnect default) or exact simplex.
    keep_fevd : bool
        Store every FEVD table (needed for connectedness tables).

    Returns
    -------
    QQConnResult

    References
    ----------
    Gabauer, D. and Stenfors, A. (2024). Quantile-on-quantile connectedness measures:
    Evidence from the US treasury yield curve. Finance Research Letters, 60, 104852.
    https://doi.org/10.1016/j.frl.2023.104852

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> x = rng.standard_t(5, size=(120, 2))
    >>> res = qqc_gs(x, taus=[0.1, 0.5, 0.9], window=80, horizon=10)
    >>> res.TCI.shape
    (41, 9)
    """
    X0, dates, names = _as_data(x)
    T, k = X0.shape
    nlag = _check_int(nlag, "nlag", 1, T // 3)
    horizon = _check_int(horizon, "horizon", 1, 10 ** 6)
    window = T if window is None else _check_int(window, "window", nlag + k * nlag + 3, T)
    if cov_type not in ("uncentered", "centered"):
        raise ValueError("cov_type must be 'uncentered' or 'centered'")
    cfg = _tau_configs(taus, k)
    nc = len(cfg)
    Y, Xd = _var_design(X0, nlag)
    p = Xd.shape[1]
    nrw = window - nlag
    nW = T - window + 1
    ut = [np.unique(cfg[:, j]) for j in range(k)]
    idx = np.column_stack([np.searchsorted(ut[j], cfg[:, j]) for j in range(k)])
    # problem list per window: (equation j, quantile index q)
    jobs = [(j, q) for j in range(k) for q in range(len(ut[j]))]
    jtau = np.array([ut[j][q] for j, q in jobs])
    COEF = [np.full((nW, p, len(ut[j])), np.nan) for j in range(k)]
    TCI = np.full((nW, nc), np.nan)
    TO = np.full((nW, k, nc), np.nan)
    FROM = TO.copy()
    NET = TO.copy()
    FE = np.full((k, k, nc, nW), np.nan) if keep_fevd else None
    from numpy.lib.stride_tricks import sliding_window_view
    Xwin = sliding_window_view(Xd, (nrw, p))[:, 0]                 # (nW, nrw, p)
    Ywin = sliding_window_view(Y, (nrw, k))[:, 0]                  # (nW, nrw, k)
    for w0 in range(0, nW, chunk):
        ws = np.arange(w0, min(nW, w0 + chunk))
        nb = len(ws)
        Xb = np.repeat(Xwin[ws], len(jobs), axis=0)                # (nb*J, nrw, p)
        Yb = Ywin[ws][:, :, [j for j, _ in jobs]].transpose(0, 2, 1).reshape(nb * len(jobs), nrw)
        tb = np.tile(jtau, nb)
        if rq_method == "fn":
            Bc = rq_fit_batch(Xb, Yb, tb)
        elif rq_method == "br":
            Bc = np.array([_rq_br(Xb[i], Yb[i], tb[i]) for i in range(len(tb))])
        else:
            raise ValueError("rq_method must be 'fn' or 'br'")
        Bc = Bc.reshape(nb, len(jobs), p)
        Rc = Yb.reshape(nb, len(jobs), nrw) - np.einsum("wjnp,wjp->wjn",
                                                         Xb.reshape(nb, len(jobs), nrw, p), Bc)
        pos = {jq: i for i, jq in enumerate(jobs)}
        for j in range(k):
            for q in range(len(ut[j])):
                COEF[j][ws, :, q] = Bc[:, pos[(j, q)], :]
        sel = np.array([[pos[(j, idx[c, j])] for j in range(k)] for c in range(nc)])  # (nc, k)
        Bsys = Bc[:, sel, 1:]                                       # (nb, nc, k, p-1)
        Rsys = Rc[:, sel, :].transpose(0, 1, 3, 2)                  # (nb, nc, nrw, k)
        Ssys = _resid_cov(Rsys, cov_type)
        F = fevd_batch(Bsys.reshape(-1, k, p - 1), Ssys.reshape(-1, k, k), horizon, generalized)
        s = conn_summary(F, corrected)
        TCI[ws] = s["TCI"].reshape(nb, nc)
        TO[ws] = s["TO"].reshape(k, nb, nc).transpose(1, 0, 2)
        FROM[ws] = s["FROM"].reshape(k, nb, nc).transpose(1, 0, 2)
        NET[ws] = s["NET"].reshape(k, nb, nc).transpose(1, 0, 2)
        if keep_fevd:
            FE[:, :, :, ws] = F.reshape(nb, nc, k, k).transpose(2, 3, 1, 0)
        if progress:
            print(f"\rGabauer-Stenfors QVAR: {100 * ws[-1] / max(nW - 1, 1):5.1f}%", end="", flush=True)
    if progress:
        print()
    config = dict(nlag=nlag, window=window, horizon=horizon, cov_type=cov_type,
                  corrected=corrected, generalized=generalized, rq_method=rq_method)
    coef = {names[j]: COEF[j] for j in range(k)}
    return QQConnResult(TCI=TCI, TO=TO, FROM=FROM, NET=NET, taus=cfg, dates=dates[window - 1:],
                        names=names, method="Gabauer-Stenfors rolling QVAR", config=config,
                        fevd=FE, coef=coef, coef_taus=ut)


# --------------------------------------------------------- conditional PIT (BCG step 1)
def _interp_pit(v: float, q: np.ndarray, g: np.ndarray) -> float:
    if v <= q[0]:
        return float(g[0])
    if v >= q[-1]:
        return float(g[-1])
    if np.any(np.diff(q) < 0):
        o = np.argsort(q, kind="stable")
        q, g = q[o], g[o]
    # stats::approx(ties = mean)
    uq, inv = np.unique(q, return_inverse=True)
    ug = np.bincount(inv, weights=g) / np.bincount(inv)
    return float(np.interp(v, uq, ug))


def cond_pit(y, window: int = 100, grid: Sequence[float] = np.arange(0.01, 1.0, 0.01),
             noncross: str = "bondell", insample: bool = False, n_jobs: int = 1,
             progress: bool = False) -> np.ndarray:
    """Conditional probability levels of Bonaccolto, Caporin and Gupta (2026), step 1.

    For each date t, quantile regressions of y_s on (1, y_{s-1}) over the
    ``window`` dates ending at t-1 (``insample=False``) are fitted on ``grid``;
    the probability level of y_t is read off the predicted conditional
    quantile curve by linear interpolation (clamped to the grid end points).

    Parameters
    ----------
    noncross : {"bondell", "rearrange", "none"}
        Bondell et al. (2010) joint non-crossing fit, Chernozhukov et al. (2010)
        rearrangement (sorting), or independent fits.

    Returns
    -------
    ndarray (T,) with NaN for the first dates without a full window.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    grid = np.unique(np.asarray(grid, dtype=float))
    window = _check_int(window, "window", 10, T - 2)
    if noncross not in ("bondell", "rearrange", "none"):
        raise ValueError("noncross must be 'bondell', 'rearrange' or 'none'")
    t0 = window if insample else window + 1                     # 0-based first date
    if t0 >= T:
        raise ValueError("Series too short for the auxiliary window")
    out = np.full(T, np.nan)
    ts = np.arange(t0, T)

    def one(t):
        last = t if insample else t - 1
        s = np.arange(last - window + 1, last + 1)
        Xa = np.column_stack([np.ones(window), y[s - 1]])
        ya = y[s]
        if noncross == "bondell":
            b = rq_noncross(Xa, ya, grid)
        else:
            b = rq_fit_batch(Xa, np.tile(ya, (len(grid), 1)), grid).T
        q = np.array([1.0, y[t - 1]]) @ b
        if noncross == "rearrange":
            q = np.sort(q)
        return _interp_pit(y[t], q, grid)

    if noncross == "bondell":
        vals = _parallel(one, ts, n_jobs, progress, "Bondell PIT")
    else:
        vals = [one(t) for t in ts]
    out[ts] = vals
    return out


# ----------------------------------------------------------- Bonaccolto-Caporin-Gupta
_KERNELS = {
    "gaussian": lambda u: np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi),
    "epanechnikov": lambda u: np.maximum(0.0, 0.75 * (1 - u ** 2)),
    "uniform": lambda u: 0.5 * (np.abs(u) <= 1),
}


def qqc_kernel(x, taus=np.arange(0.05, 1.0, 0.10), nlag: int = 1, window: Optional[int] = 200,
               horizon: int = 20, bandwidth: float = 0.15, kernel: str = "gaussian",
               cov_type: str = "centered", pit: Optional[np.ndarray] = None,
               presample: Optional[int] = None, aux_window: int = 100,
               aux_grid=np.arange(0.01, 1.0, 0.01), noncross: str = "bondell",
               aux_insample: bool = False, corrected: bool = True, generalized: bool = True,
               rq_method: str = "fn", keep_fevd: bool = False, n_jobs: int = 1,
               chunk: int = 16, progress: bool = False) -> QQConnResult:
    """Kernel-weighted local Q-on-Q-QVAR connectedness of Bonaccolto, Caporin and Gupta (2026).

    Step 1 (:func:`cond_pit`) maps every observation to its conditional probability
    level ``u_it`` from rolling auxiliary quantile regressions with Bondell et al.
    (2010) non-crossing.  Step 2 estimates equation ``j`` at ``tau_j`` by a
    weighted quantile regression with weights
    ``prod_{i != j} K((u_it - tau_i) / h)``, so the equation is local to the
    target quantiles of the other series; the residual matrix is centered.

    Parameters
    ----------
    bandwidth : float
        Kernel bandwidth h on the probability scale (BCG: 0.15).
    kernel : {"gaussian", "epanechnikov", "uniform"}
    pit : ndarray (T x k), optional
        Pre-computed probability levels (skips step 1; ``presample`` then defaults to 0).
    presample : int, optional
        Leading observations used only by step 1 (BCG: 100).
    aux_window : int
        Window of the auxiliary regressions (BCG: 100).
    rq_method : {"fn", "br"}
        ``"fn"`` (vectorised Frisch-Newton, fast) or ``"br"`` (exact simplex, the
        qqconnect default; slow in Python).  Both minimise the same weighted
        check loss; they can differ only where the LP optimum is not unique.
    n_jobs : int
        Parallel workers for step 1 (requires joblib).

    References
    ----------
    Bonaccolto, G., Caporin, M. and Gupta, R. (2026). Rethinking quantile-on-quantile
    connectedness. Finance Research Letters, 114, 111095.

    Bondell, H. D., Reich, B. J. and Wang, H. (2010). Noncrossing quantile regression
    curve estimation. Biometrika, 97(4), 825-838. https://doi.org/10.1093/biomet/asq048
    """
    X0, dates, names = _as_data(x)
    T, k = X0.shape
    nlag = _check_int(nlag, "nlag", 1, T // 3)
    if not bandwidth > 0:
        raise ValueError("'bandwidth' must be positive")
    if kernel not in _KERNELS:
        raise ValueError("kernel must be 'gaussian', 'epanechnikov' or 'uniform'")
    cfg = _tau_configs(taus, k)
    nc = len(cfg)
    if pit is None:
        if progress:
            print(f"Step 1/2: conditional probability levels ({noncross})")
        pit = np.column_stack([cond_pit(X0[:, j], aux_window, aux_grid, noncross,
                                        aux_insample, n_jobs, progress) for j in range(k)])
        if presample is None:
            presample = aux_window
    else:
        pit = np.asarray(pit, dtype=float)
        if pit.shape != X0.shape:
            raise ValueError("'pit' must have the same dimensions as 'x'")
        if presample is None:
            presample = 0
    presample = _check_int(presample, "presample", 0, T - 10)
    Xm = X0[presample:]
    Pm = pit[presample:]
    Tm = len(Xm)
    window = Tm if window is None else _check_int(window, "window", nlag + k * nlag + 3, Tm)
    Y, Xd = _var_design(Xm, nlag)
    Pd = Pm[nlag:]
    if np.isnan(Pd).any():
        raise ValueError("Probability levels are missing inside the estimation sample; "
                         "increase 'presample' (at least aux_window + 1 - nlag)")
    kf = _KERNELS[kernel]
    Wt = np.ones((len(Pd), k, nc))
    for c in range(nc):
        for j in range(k):
            for i in range(k):
                if i != j:
                    Wt[:, j, c] *= kf((Pd[:, i] - cfg[c, i]) / bandwidth)
    p = Xd.shape[1]
    nrw = window - nlag
    nW = Tm - window + 1
    TCI = np.full((nW, nc), np.nan)
    TO = np.full((nW, k, nc), np.nan)
    FROM, NET = TO.copy(), TO.copy()
    FE = np.full((k, k, nc, nW), np.nan) if keep_fevd else None
    from numpy.lib.stride_tricks import sliding_window_view
    Xwin = sliding_window_view(Xd, (nrw, p))[:, 0]
    Ywin = sliding_window_view(Y, (nrw, k))[:, 0]
    Wwin = sliding_window_view(Wt, (nrw, k, nc))[:, 0, 0]          # (nW, nrw, k, nc)
    taus_jc = cfg.T                                                 # (k, nc)
    if progress:
        print(f"Step 2/2: kernel-weighted QVAR over {nW} windows")
    for w0 in range(0, nW, chunk):
        ws = np.arange(w0, min(nW, w0 + chunk))
        nb = len(ws)
        Wb = Wwin[ws]                                               # (nb, nrw, k, nc)
        Xb = Xwin[ws][:, None, None] * Wb.transpose(0, 2, 3, 1)[..., None]   # (nb,k,nc,nrw,p)
        Yb = (Ywin[ws].transpose(0, 2, 1)[:, :, None, :] * Wb.transpose(0, 2, 3, 1))  # (nb,k,nc,nrw)
        tb = np.broadcast_to(taus_jc[None], (nb, k, nc)).ravel()
        keep = (Wb > 0).sum(axis=1)                                 # (nb, k, nc)
        if rq_method == "fn":
            Bc = rq_fit_batch(Xb.reshape(-1, nrw, p), Yb.reshape(-1, nrw), tb)
        elif rq_method == "br":
            Xf, Yf = Xb.reshape(-1, nrw, p), Yb.reshape(-1, nrw)
            Bc = np.array([_rq_br(Xf[i], Yf[i], tb[i]) for i in range(len(tb))])
        else:
            raise ValueError("rq_method must be 'fn' or 'br'")
        Bc = Bc.reshape(nb, k, nc, p)
        Bc[keep <= p + 1] = np.nan
        Rr = Ywin[ws].transpose(0, 2, 1)[:, :, None, :] - np.einsum("wnp,wjcp->wjcn", Xwin[ws], Bc)
        Bsys = Bc[..., 1:].transpose(0, 2, 1, 3)                    # (nb, nc, k, p-1)
        Rsys = Rr.transpose(0, 2, 3, 1)                              # (nb, nc, nrw, k)
        Ssys = _resid_cov(Rsys, cov_type)
        F = fevd_batch(Bsys.reshape(-1, k, p - 1), Ssys.reshape(-1, k, k), horizon, generalized)
        s = conn_summary(F, corrected)
        TCI[ws] = s["TCI"].reshape(nb, nc)
        TO[ws] = s["TO"].reshape(k, nb, nc).transpose(1, 0, 2)
        FROM[ws] = s["FROM"].reshape(k, nb, nc).transpose(1, 0, 2)
        NET[ws] = s["NET"].reshape(k, nb, nc).transpose(1, 0, 2)
        if keep_fevd:
            FE[:, :, :, ws] = F.reshape(nb, nc, k, k).transpose(2, 3, 1, 0)
        if progress:
            print(f"\r  windows {100 * ws[-1] / max(nW - 1, 1):5.1f}%", end="", flush=True)
    if progress:
        print()
    config = dict(nlag=nlag, window=window, horizon=horizon, bandwidth=bandwidth, kernel=kernel,
                  cov_type=cov_type, presample=presample, aux_window=aux_window,
                  noncross=noncross, aux_insample=aux_insample, corrected=corrected,
                  generalized=generalized, rq_method=rq_method)
    return QQConnResult(TCI=TCI, TO=TO, FROM=FROM, NET=NET, taus=cfg,
                        dates=dates[presample:][window - 1:], names=names,
                        method="Bonaccolto-Caporin-Gupta kernel Q-on-Q-QVAR", config=config,
                        fevd=FE, pit=pd.DataFrame(pit, index=dates, columns=names))


# ----------------------------------------------------------------- dynamic quantile LM
def _kf_discount_py(y, X, V, delta, m0, C0):
    n, p = X.shape
    m = np.zeros((n, p)); a = np.zeros((n, p)); Cs = np.zeros((n, p, p))
    mt = m0.copy(); Ct = C0.copy()
    for t in range(n):
        Rt = Ct / delta
        at = mt
        Ft = X[t]
        RF = Rt @ Ft
        Q = Ft @ RF + V[t]
        A = RF / Q
        mt = at + A * (y[t] - Ft @ at)
        Ct = Rt - np.outer(A, A) * Q
        Ct = (Ct + Ct.T) / 2
        a[t] = at; m[t] = mt; Cs[t] = Ct
    s = m.copy(); S = Cs.copy()
    q = np.zeros(n)
    q[n - 1] = X[n - 1] @ S[n - 1] @ X[n - 1]
    for t in range(n - 2, -1, -1):
        s[t] = m[t] + delta * (s[t + 1] - m[t])
        S[t] = Cs[t] + delta ** 2 * (S[t + 1] - Cs[t] / delta)
        q[t] = X[t] @ S[t] @ X[t]
    return m, a, s, np.maximum(q, 0)


try:  # optional acceleration
    from numba import njit as _njit
    from numba.core.errors import NumbaPerformanceWarning as _NPW
    warnings.simplefilter("ignore", category=_NPW)
    _kf_discount = _njit(cache=True)(_kf_discount_py)
except Exception:  # pragma: no cover
    _kf_discount = _kf_discount_py


def dqlm(y, X, tau: float = 0.5, delta: float = 0.99, m0=None, C0=None,
         maxit: int = 200, tol: float = 1e-6) -> dict:
    """Dynamic quantile linear model with random-walk coefficients (variational Bayes / EM).

    The asymmetric-Laplace DQLM used by Candelo-Viafara and Orrego-Reyes (2026)
    (``exdqlm::exdqlmISVB`` with ``dqlm.ind = TRUE``): a discount Kalman filter
    and smoother inside a location-scale-mixture EM for sigma.

    Returns
    -------
    dict with ``filtered`` (n x p), ``predictive`` (n x p, one-step-ahead),
    ``smoothed`` (n x p), ``sigma`` and ``iterations``.
    """
    y = np.ascontiguousarray(y, dtype=float)
    X = np.ascontiguousarray(X, dtype=float)
    n, p = X.shape
    if len(y) != n:
        raise ValueError("'X' and 'y' have incompatible dimensions")
    if not 0 < tau < 1:
        raise ValueError("tau must lie in (0, 1)")
    m0 = np.zeros(p) if m0 is None else np.asarray(m0, dtype=float)
    C0 = 10.0 * np.eye(p) if C0 is None else np.asarray(C0, dtype=float)
    cc = (1 - 2 * tau) / (tau * (1 - tau))
    bb = 2 / (tau * (1 - tau))
    psi = 2 + cc ** 2 / bb
    mad = 1.4826 * np.median(np.abs(y - np.median(y)))
    sigma = max(mad * tau * (1 - tau), 1e-8)
    E1v = np.ones(n)
    it = 0
    for it in range(1, maxit + 1):
        yt = y - sigma * cc / E1v
        Vt = sigma ** 2 * bb / E1v
        m, a, s, q = _kf_discount(yt, X, Vt, delta, m0, C0)
        r = y - (X * s).sum(1)
        Er2 = r ** 2 + q
        chi = np.maximum(Er2 / (sigma ** 2 * bb), 1e-12)
        E1v = np.sqrt(psi / chi)
        A = (Er2 * E1v).sum() / bb
        B = (cc * r).sum() / bb
        sig_new = (-B + np.sqrt(B ** 2 + 4 * n * A)) / (2 * n)
        conv = abs(sig_new - sigma) / sigma < tol
        sigma = sig_new
        if conv and it > 2:
            break
    yt = y - sigma * cc / E1v
    Vt = sigma ** 2 * bb / E1v
    m, a, s, q = _kf_discount(yt, X, Vt, delta, m0, C0)
    return {"filtered": m, "predictive": a, "smoothed": s, "sigma": sigma, "iterations": it}


# ------------------------------------------------------------------- TVP-QQVAR
def qqc_tvp(x, taus=np.arange(0.05, 1.0, 0.10), nlag: int = 1, horizon: int = 30,
            delta: float = 0.99, kappa: float = 0.99, coef_type: str = "filtered",
            resid_type: str = "predictive", init: int = 30, burnin: int = 0,
            corrected: bool = True, generalized: bool = True, m0=None, C0=None,
            keep_fevd: bool = False, n_jobs: int = 1, progress: bool = False) -> QQConnResult:
    """TVP quantile-on-quantile VAR connectedness of Candelo-Viafara and Orrego-Reyes (2026).

    Each equation and quantile level is a dynamic quantile linear model
    (:func:`dqlm`, discount factor ``delta``); the innovation matrix follows
    ``H_t = kappa H_{t-1} + (1 - kappa) u_t u_t'`` started at the second moment of
    the first ``init`` residuals.  No rolling window is needed.

    Parameters
    ----------
    horizon : int
        GFEVD horizon (paper: 30).
    delta, kappa : float
        Coefficient discount and covariance forgetting factors (paper: 0.99, 0.99).
    coef_type : {"filtered", "predictive"}
        Coefficients entering B_t.
    resid_type : {"predictive", "filtered"}
        Coefficients used for the residuals of the covariance recursion
        (one-step-ahead conditions on F_{t-1}, as in the paper).

    References
    ----------
    Candelo-Viafara, J. M. and Orrego-Reyes, J. E. (2026). Dynamic connectedness across
    heterogeneous market states: A TVP quantile-on-quantile VAR approach. SSRN 7434976.
    https://doi.org/10.2139/ssrn.7434976
    """
    X0, dates, names = _as_data(x)
    T, k = X0.shape
    nlag = _check_int(nlag, "nlag", 1, T // 3)
    if not 0 <= kappa <= 1:
        raise ValueError("kappa must lie in [0, 1]")
    cfg = _tau_configs(taus, k)
    nc = len(cfg)
    Y, Xd = _var_design(X0, nlag)
    n, p = Xd.shape
    init = _check_int(init, "init", 2, n)
    burnin = _check_int(burnin, "burnin", 0, n - 1)
    jobs = [(j, t) for j in range(k) for t in np.unique(cfg[:, j])]

    def fit(job):
        j, tt = job
        f = dqlm(Y[:, j], Xd, tau=tt, delta=delta, m0=m0, C0=C0)
        th = f["filtered"] if coef_type == "filtered" else f["predictive"]
        thr = f["filtered"] if resid_type == "filtered" else f["predictive"]
        return th, Y[:, j] - (Xd * thr).sum(1)

    if progress:
        print(f"Step 1/2: {len(jobs)} dynamic quantile regressions")
    fits = _parallel(fit, jobs, n_jobs, progress, "DQLM")
    key = {(j, round(float(t), 12)): i for i, (j, t) in enumerate(jobs)}
    keep = np.arange(burnin, n)
    nk = len(keep)
    TCI = np.full((nk, nc), np.nan)
    TO = np.full((nk, k, nc), np.nan)
    FROM, NET = TO.copy(), TO.copy()
    FE = np.full((k, k, nk, nc), np.nan) if keep_fevd else None
    for c in range(nc):
        ids = [key[(j, round(float(cfg[c, j]), 12))] for j in range(k)]
        U = np.column_stack([fits[i][1] for i in ids])
        Bt = np.stack([fits[i][0][:, 1:] for i in ids], axis=1)    # (n, k, kp)
        H = U[:init].T @ U[:init] / init
        St = np.empty((n, k, k))
        for t in range(n):
            H = kappa * H + (1 - kappa) * np.outer(U[t], U[t])
            St[t] = H
        F = fevd_batch(Bt[keep], St[keep], horizon, generalized)
        s = conn_summary(F, corrected)
        TCI[:, c] = s["TCI"]
        TO[:, :, c] = s["TO"].T
        FROM[:, :, c] = s["FROM"].T
        NET[:, :, c] = s["NET"].T
        if keep_fevd:
            FE[:, :, :, c] = F.transpose(1, 2, 0)
    if keep_fevd:
        FE = FE.transpose(0, 1, 3, 2)
    coef = {}
    for j in range(k):
        rr = [i for i, (jj, _) in enumerate(jobs) if jj == j]
        coef[names[j]] = np.stack([fits[i][0][keep] for i in rr], axis=2)
    config = dict(nlag=nlag, horizon=horizon, delta=delta, kappa=kappa, coef_type=coef_type,
                  resid_type=resid_type, init=init, burnin=burnin, corrected=corrected,
                  generalized=generalized)
    return QQConnResult(TCI=TCI, TO=TO, FROM=FROM, NET=NET, taus=cfg,
                        dates=list(np.array(dates[nlag:])[keep]), names=names,
                        method="Candelo-Viafara & Orrego-Reyes TVP-QQVAR", config=config,
                        fevd=FE, coef=coef,
                        coef_taus=[np.unique(cfg[:, j]) for j in range(k)])


def qqconnect(x, method: str = "gs", **kwargs) -> QQConnResult:
    """Dispatcher: ``method`` in {"gs", "kernel", "tvp"}; other arguments are passed on."""
    fun = {"gs": qqc_gs, "kernel": qqc_kernel, "tvp": qqc_tvp}.get(method)
    if fun is None:
        raise ValueError("method must be 'gs', 'kernel' or 'tvp'")
    return fun(x, **kwargs)
