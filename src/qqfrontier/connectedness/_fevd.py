"""Generalized FEVD and Diebold-Yilmaz connectedness measures.

Port of ``qqconnect``'s C++ core (``src/connectedness.cpp``).  The scaled GFEVD
of Koop, Pesaran and Potter (1996) and Pesaran and Shin (1998), horizons
h = 0, ..., H-1, reproduces the ``ConnectednessApproach`` convention:

    theta_ij = sigma_jj^{-1} sum_h (e_i' Psi_h Sigma e_j)^2,  rows normalised.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["gfevd", "connectedness_table", "fevd_batch", "conn_summary"]


def _wold_psi(B: np.ndarray, H: int) -> np.ndarray:
    """MA matrices Psi_0..Psi_{H-1} for a batch of VAR(p): B (N, k, kp) -> (N, H, k, k)."""
    N, k, kp = B.shape
    p = kp // k
    A = B.reshape(N, k, p, k).transpose(0, 2, 1, 3)       # A[:, l] = lag-(l+1) matrix
    Psi = np.zeros((N, H, k, k))
    Psi[:, 0] = np.eye(k)
    for h in range(1, H):
        acc = np.zeros((N, k, k))
        for l in range(1, min(h, p) + 1):
            acc += A[:, l - 1] @ Psi[:, h - l]
        Psi[:, h] = acc
    return Psi


def fevd_batch(B: np.ndarray, S: np.ndarray, horizon: int,
               generalized: bool = True) -> np.ndarray:
    """Row-normalised FEVD for many systems.

    Parameters
    ----------
    B : ndarray (N, k, k*p)
        Slope matrices [A_1, ..., A_p] (no intercept).
    S : ndarray (N, k, k)
        Innovation covariance matrices.
    horizon : int
        Forecast horizon H.
    generalized : bool
        Generalized (True) or Cholesky-orthogonalised (False) decomposition.

    Returns
    -------
    ndarray (N, k, k), rows sum to one; NaN for systems with missing inputs.
    """
    B = np.asarray(B, dtype=float)
    S = np.asarray(S, dtype=float)
    N, k, _ = B.shape
    out = np.full((N, k, k), np.nan)
    ok = ~(np.isnan(B).any(axis=(1, 2)) | np.isnan(S).any(axis=(1, 2)))
    if not ok.any():
        return out
    Bo, So = B[ok], S[ok]
    Psi = _wold_psi(Bo, int(horizon))                       # (n, H, k, k)
    if generalized:
        PS = Psi @ So[:, None]                              # (n, H, k, k)
        num = (PS ** 2).sum(axis=1)
        num = num / np.diagonal(So, axis1=1, axis2=2)[:, None, :]
    else:
        num = np.full((len(Bo), k, k), np.nan)
        for i in range(len(Bo)):
            try:
                P = np.linalg.cholesky(So[i])
            except np.linalg.LinAlgError:
                continue
            num[i] = ((Psi[i] @ P) ** 2).sum(axis=0)
    out[ok] = num / num.sum(axis=2, keepdims=True)
    return out


def conn_summary(F: np.ndarray, corrected: bool = True) -> Dict[str, np.ndarray]:
    """Diebold-Yilmaz TO / FROM / NET (k x N, percent) and TCI (N,) from FEVD tables.

    TCI is the adjusted index k/(k-1) * mean(TO) of Chatziantoniou, Gabauer and
    Stenfors (2021) when ``corrected`` is true.
    """
    F = np.asarray(F, dtype=float)
    k = F.shape[1]
    C = 100.0 * F
    own = np.diagonal(C, axis1=1, axis2=2)                  # (N, k)
    TO = C.sum(axis=1) - own
    FROM = C.sum(axis=2) - own
    NET = TO - FROM
    adj = k / (k - 1) if corrected else 1.0
    TCI = adj * TO.mean(axis=1)
    return {"TO": TO.T, "FROM": FROM.T, "NET": NET.T, "TCI": TCI}


def gfevd(B, Sigma, horizon: int = 10, generalized: bool = True,
          names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Generalized forecast error variance decomposition of a VAR(p).

    Element (i, j) is the share of the H-step forecast error variance of series
    i due to shocks in series j (rows sum to one).

    Parameters
    ----------
    B : array (k, k*p)
        VAR slope matrices [A_1, ..., A_p] (no intercept).
    Sigma : array (k, k)
        Innovation covariance.
    horizon : int
        Forecast horizon H (Koop, Pesaran and Potter, 1996; Pesaran and Shin, 1998).
    generalized : bool
        False gives the Cholesky (orthogonalised) decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> B = np.array([[0.5, 0.2], [0.1, 0.3]]); S = np.array([[1, 0.4], [0.4, 2]])
    >>> gfevd(B, S, horizon=10).round(4).values.tolist()
    [[0.7878, 0.2122], [0.0933, 0.9067]]
    """
    B = np.asarray(B, dtype=float)
    S = np.asarray(Sigma, dtype=float)
    k = S.shape[0]
    if B.shape[0] != k or B.shape[1] % k:
        raise ValueError("B must be k x (k*p)")
    if int(horizon) < 1:
        raise ValueError("horizon must be a positive integer")
    F = fevd_batch(B[None], S[None], int(horizon), generalized)[0]
    nm = list(names) if names is not None else [f"y{i + 1}" for i in range(k)]
    return pd.DataFrame(F, index=nm, columns=nm)


def connectedness_table(fevd, corrected: bool = True,
                        names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Diebold-Yilmaz (2012) connectedness table from a variance decomposition.

    Returns a dict with ``table`` (percent shares with FROM column and TO / NET
    rows, the TCI in the bottom-right cell), ``TO``, ``FROM``, ``NET``,
    ``NPDC`` (net pairwise directional connectedness) and ``TCI``.

    Examples
    --------
    >>> import numpy as np
    >>> B = np.array([[0.5, 0.2], [0.1, 0.3]]); S = np.array([[1, 0.4], [0.4, 2]])
    >>> round(connectedness_table(gfevd(B, S, 10))["TCI"], 3)
    30.551
    """
    if isinstance(fevd, pd.DataFrame):
        nm = list(fevd.columns) if names is None else list(names)
        F = fevd.values.astype(float)
    else:
        F = np.asarray(fevd, dtype=float)
        nm = list(names) if names is not None else [f"y{i + 1}" for i in range(F.shape[0])]
    s = conn_summary(F[None], corrected)
    CT = 100 * F
    TO, FROM, NET = s["TO"][:, 0], s["FROM"][:, 0], s["NET"][:, 0]
    tab = pd.DataFrame(CT, index=nm, columns=nm)
    tab["FROM"] = FROM
    tab.loc["TO"] = list(TO) + [TO.sum()]
    tab.loc["NET"] = list(NET) + [s["TCI"][0]]
    return {"table": tab, "TO": pd.Series(TO, nm), "FROM": pd.Series(FROM, nm),
            "NET": pd.Series(NET, nm),
            "NPDC": pd.DataFrame(CT - CT.T, index=nm, columns=nm),
            "TCI": float(s["TCI"][0])}
