"""Monte Carlo design of Candelo-Viafara and Orrego-Reyes (2026) (port of qqconnect).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .._qr import rq_fit_batch

__all__ = ["sim_qqvar", "mc_qqvar"]


def sim_qqvar(n: int = 1000, tau=(0.5, 0.5), sigma=(1.0, 1.0), rho: float = 0.3,
              burn: int = 100, seed=None) -> dict:
    """Simulate the bivariate TVP-QVAR(1) of the TVP-QQVAR paper's Monte Carlo.

    Coefficients follow smooth sine/cosine paths and a logistic regime shift at
    s = 0.55; the innovations are centred so that their tau-quantile is zero.

    Returns
    -------
    dict with ``x`` (n x 2), ``B`` (n x 4: b11, b12, b21, b22) and ``tau``.
    """
    rng = np.random.default_rng(seed)
    if len(tau) != 2 or min(tau) <= 0 or max(tau) >= 1:
        raise ValueError("'tau' must contain two levels in (0, 1)")
    s = np.arange(1, n + 1) / n
    g = 1 / (1 + np.exp(-(s - 0.55) / 0.025))
    B = np.column_stack([0.20 + 0.07 * np.sin(2 * np.pi * s) + 0.06 * (tau[0] - 0.5),
                         0.06 + 0.10 * g + 0.10 * (tau[0] - 0.5),
                         0.07 - 0.07 * g - 0.08 * (tau[1] - 0.5),
                         0.18 + 0.06 * np.cos(2 * np.pi * s) + 0.05 * (tau[1] - 0.5)])
    N = n + burn
    z1 = rng.standard_normal(N)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(N)
    e = np.column_stack([sigma[0] * (z1 - norm.ppf(tau[0])), sigma[1] * (z2 - norm.ppf(tau[1]))])
    Bf = np.vstack([np.repeat(B[[0]], burn, axis=0), B])
    x = np.zeros((N, 2))
    for t in range(1, N):
        A = Bf[t].reshape(2, 2)
        x[t] = A @ x[t - 1] + e[t]
    return {"x": x[burn:], "B": B, "tau": tuple(tau)}


def _recovery(est, true):
    e = est - true
    corr = np.nanmean([np.corrcoef(est[:, j], true[:, j])[0, 1] for j in range(4)])
    return np.array([e.mean(0).mean(), np.abs(e).mean(0).mean(),
                     np.sqrt((e ** 2).mean(0)).mean(), corr])


def mc_qqvar(reps: int = 100, n: int = 1000,
             taus: Sequence[Sequence[float]] = ((0.05, 0.95), (0.5, 0.5), (0.95, 0.05)),
             windows: Sequence[int] = (100, 200, 300), delta: float = 0.99,
             sigma=(1.0, 1.0), rho: float = 0.3, seed: Optional[int] = None) -> pd.DataFrame:
    """Coefficient-path recovery: TVP-QQVAR vs rolling QQVAR (TVP-QQVAR paper, Tables 2-3).

    Returns bias, MAE, RMSE and correlation with the true paths for both methods
    and the percentage MAE / RMSE improvement of the TVP filter.
    """
    from ._core import _var_design, dqlm
    rng = np.random.default_rng(seed)
    acc = np.zeros((len(taus), len(windows), 2, 4))
    for _ in range(reps):
        for a, tau in enumerate(taus):
            sim = sim_qqvar(n, tau, sigma, rho, seed=rng.integers(2 ** 31))
            Y, X = _var_design(sim["x"], 1)
            Btrue = sim["B"][1:]
            tvp = np.full((len(Y), 4), np.nan)
            for j in range(2):
                f = dqlm(Y[:, j], X, tau=tau[j], delta=delta)
                tvp[:, 2 * j:2 * j + 2] = f["filtered"][:, 1:3]
            for wi, wl in enumerate(windows):
                w = wl - 1
                rol = np.full((len(Y), 4), np.nan)
                from numpy.lib.stride_tricks import sliding_window_view
                Xw = sliding_window_view(X, (w, 3))[:, 0]
                for j in range(2):
                    Yw = sliding_window_view(Y[:, j], w)
                    b = rq_fit_batch(Xw, Yw, np.full(len(Yw), tau[j]))
                    rol[w - 1:, 2 * j:2 * j + 2] = b[:, 1:3]
                ev = np.arange(w - 1, len(Y))
                acc[a, wi, 0] += _recovery(tvp[ev], Btrue[ev]) / reps
                acc[a, wi, 1] += _recovery(rol[ev], Btrue[ev]) / reps
    rows = []
    for a, tau in enumerate(taus):
        for wi, wl in enumerate(windows):
            tv, ro = acc[a, wi, 0], acc[a, wi, 1]
            rows.append(dict(tau1=tau[0], tau2=tau[1], window=wl, bias_tvp=tv[0], mae_tvp=tv[1],
                             rmse_tvp=tv[2], corr_tvp=tv[3], bias_rolling=ro[0], mae_rolling=ro[1],
                             rmse_rolling=ro[2], corr_rolling=ro[3],
                             mae_improvement=100 * (1 - tv[1] / ro[1]),
                             rmse_improvement=100 * (1 - tv[2] / ro[2])))
    return pd.DataFrame(rows)
