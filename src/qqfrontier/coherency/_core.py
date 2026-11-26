"""Quantile coherency of Barunik and Kley (2019).

Step -> equation map (BK = Barunik and Kley, Econometrics Journal 22(2), 131-152)
--------------------------------------------------------------------------------
d_{n,R}(omega; tau) = sum_t 1{R_{n;t} <= n tau} e^{-i omega t}  (ranks, max-rank) . Eq. (2.4)
I_{n,R}^{j1 j2}(omega; tau1, tau2) = (2 pi n)^-1 d^{j1}(omega; tau1) d^{j2}(-omega; tau2) .. Eq. (2.4)
G_hat(omega) = (2 pi / n) sum_{s=1}^{n-1} W_n(omega - 2 pi s / n) I(2 pi s / n) ...... Eq. (2.6)
W_n(u) = sum_j b_n^-1 W(b_n^-1 (u + 2 pi j)), W Epanechnikov on [-pi, pi] ............. Ass. 4.2
G_tilde = G_hat / W^k, W^k = (2 pi / n) sum_{s != 0} W_n(omega_k - omega_s) ........... Supp. S5.1
R_hat = G^{12} / (G^{11} G^{22})^{1/2} ............................................. Eq. (2.8)
Pointwise CIs for Re and Im of R: delta-method covariance of L (Thm 4.1) with
  Cov(H_ab, H_cd) estimated by Eq. (S.10) .......................................... Supp. S5.2
Bandwidth default b_n = 0.5 n^(-1/4) (BK application, Sec. 5).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

__all__ = ["quantile_coherency", "kernel_weights"]

_KERN = {
    "epanechnikov": lambda x: np.where(np.abs(x) <= np.pi, 0.75 * (1 - (x / np.pi) ** 2) / np.pi, 0.0),
    "daniell": lambda x: np.where(np.abs(x) <= np.pi, 1 / (2 * np.pi), 0.0),
    "quartic": lambda x: np.where(np.abs(x) <= np.pi, (15 / 32) * (7 * (x / np.pi) ** 4 - 10 * (x / np.pi) ** 2 + 3) / np.pi, 0.0),
}


def kernel_weights(n: int, bandwidth: float, kernel: str = "epanechnikov") -> np.ndarray:
    """W_n(2 pi m / n), m = 0..n-1, for the periodised kernel of BK Assumption 4.2."""
    if kernel not in _KERN:
        raise ValueError(f"kernel must be one of {list(_KERN)}")
    W = _KERN[kernel]
    u = 2 * np.pi * np.arange(n) / n
    out = np.zeros(n)
    jmax = int(np.ceil(bandwidth / 2)) + 1
    for j in range(-jmax - 1, jmax + 2):
        out += W((u + 2 * np.pi * j) / bandwidth) / bandwidth
    return out


def _circ_conv(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """C(j) = sum_s w[(j - s) mod n] v(s) along the last axis."""
    return np.fft.ifft(np.fft.fft(w) * np.fft.fft(v, axis=-1), axis=-1)


def quantile_coherency(x, y, taus: Sequence[float] = np.round(np.arange(0.05, 0.951, 0.05), 2),
                       taus2: Optional[Sequence[float]] = None, bandwidth: Optional[float] = None,
                       kernel: str = "epanechnikov", ci: bool = True,
                       ci_freqs: Optional[Sequence[float]] = None, n_ci_freqs: int = 96,
                       alpha: float = 0.05, periods: Optional[Dict[str, float]] = None,
                       names=("x", "y")):
    """Quantile coherency R(omega; tau1, tau2) between two series (Barunik and Kley, 2019).

    Parameters
    ----------
    x, y : array-like or Series
        The two series (j1 = x, j2 = y). Rank-based, so invariant to monotone transforms.
    taus, taus2 : sequence of float
        Quantile levels tau1 of ``x`` and tau2 of ``y`` (default taus2 = taus).
    bandwidth : float, optional
        Kernel scale b_n (default 0.5 n^(-1/4), the BK application).
    kernel : {"epanechnikov", "daniell", "quartic"}
        Weight function W on [-pi, pi] (BK: Epanechnikov).
    ci : bool
        Compute pointwise (1 - alpha) confidence intervals (BK Supplement S5.2).
    ci_freqs : sequence of float, optional
        Frequencies in cycles per period (omega / 2 pi, in (0, 0.5]) at which the CIs are
        computed; default: ``n_ci_freqs`` log-spaced Fourier frequencies plus ``periods``.
    periods : dict, optional
        Named periods for tables and figures, e.g. ``{"W": 5, "M": 22, "Y": 250}`` (daily data).

    Returns
    -------
    CoherencyResult

    References
    ----------
    Barunik, J. and Kley, T. (2019). Quantile coherency: A general measure for dependence
    between cyclical economic variables. The Econometrics Journal, 22(2), 131-152.
    https://doi.org/10.1093/ectj/utz002
    """
    from ._results import CoherencyResult
    nm1 = getattr(x, "name", None) or names[0]
    nm2 = getattr(y, "name", None) or names[1]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("missing values in x / y")
    n = len(x)
    t1 = np.sort(np.atleast_1d(np.asarray(taus, dtype=float)))
    t2 = t1 if taus2 is None else np.sort(np.atleast_1d(np.asarray(taus2, dtype=float)))
    b = 0.5 * n ** (-0.25) if bandwidth is None else float(bandwidth)
    periods = periods if periods is not None else {"W": 5, "M": 22, "Y": 250}
    # Eq. (2.4): clipped rank indicators and their DFTs
    R1 = rankdata(x, method="max")
    R2 = rankdata(y, method="max")
    D1 = np.fft.fft((R1[None, :] <= n * t1[:, None]).astype(float), axis=1)   # (K1, n)
    D2 = np.fft.fft((R2[None, :] <= n * t2[:, None]).astype(float), axis=1)   # (K2, n)
    c = 1.0 / (2 * np.pi * n)
    I12 = c * D1[:, None, :] * np.conj(D2[None, :, :])                        # (K1, K2, n)
    I11 = c * np.abs(D1) ** 2                                                 # (K1, n) diagonal autos
    I22 = c * np.abs(D2) ** 2
    for I in (I12, I11, I22):
        I[..., 0] = 0.0                                                       # s = 0 excluded (Eq. 2.6)
    w = kernel_weights(n, b, kernel)
    Wk = (w.sum() - w)                                                        # (n / 2 pi) W^k
    G12 = _circ_conv(w, I12) / Wk                                             # G_tilde, (K1, K2, n)
    G11 = _circ_conv(w, I11).real / Wk
    G22 = _circ_conv(w, I22).real / Wk
    jpos = np.arange(1, n // 2 + 1)                                           # omega in (0, pi]
    freq = jpos / n                                                           # cycles per period
    Rc = G12[:, :, jpos] / np.sqrt(G11[:, None, jpos] * G22[None, :, jpos])   # (K1, K2, F)
    Rc = np.moveaxis(Rc, 2, 0)                                                # (F, K1, K2)
    ci_out = None
    if ci:
        if ci_freqs is None:
            grid = np.unique(np.round(np.geomspace(1, n // 2, n_ci_freqs)).astype(int))
            extra = [int(round(n / p)) for p in periods.values() if 1 <= round(n / p) <= n // 2]
            jj = np.unique(np.r_[grid, extra])
        else:
            jj = np.unique(np.clip(np.round(np.asarray(ci_freqs) * n).astype(int), 1, n // 2))
        sd_re, sd_im = _coherency_sd(G12, G11, G22, w, Wk, jj, n)
        z = norm.ppf(1 - alpha / 2)
        Rj = np.moveaxis(G12[:, :, jj] / np.sqrt(G11[:, None, jj] * G22[None, :, jj]), 2, 0)
        ci_out = {"freq": jj / n, "index": jj, "sd_re": sd_re, "sd_im": sd_im,
                  "re_low": Rj.real - z * sd_re, "re_high": Rj.real + z * sd_re,
                  "im_low": Rj.imag - z * sd_im, "im_high": Rj.imag + z * sd_im}
    return CoherencyResult(R=Rc, freq=freq, tau1=t1, tau2=t2, n=n, bandwidth=b, kernel=kernel,
                           ci=ci_out, alpha=alpha, periods=periods, names=(str(nm1), str(nm2)))


def _coherency_sd(G12, G11, G22, w, Wk, jj, n):
    """Standard deviations of Re and Im of R_hat at frequency indices ``jj`` (BK Supp. S5.2)."""
    K1, K2, _ = G12.shape
    s = np.arange(1, n)                                                       # s = 1..n-1
    sd_re = np.zeros((len(jj), K1, K2))
    sd_im = np.zeros((len(jj), K1, K2))
    # sparse support of the weights
    nz = np.flatnonzero(w > 0)
    half = int(max(np.minimum(nz, n - nz).max(), 1)) if nz.size else 1
    for f, j in enumerate(jj):
        ss = (j + np.arange(-half, half + 1)) % n
        ss = ss[ss != 0]
        w1 = w[(j - ss) % n] ** 2                                             # W(j-s)^2
        w2 = w[(j - ss) % n] * w[(j + ss) % n]                                # W(j-s) W(j+s)
        g11 = G11[:, ss]                                                      # (K1, S)
        g22 = G22[:, ss]
        g12 = G12[:, :, ss]                                                   # (K1, K2, S)
        g21c = np.conj(g12)                                                   # G_21(s) = conj(G_12(s))

        def H(Gac, Gbd, Gad, Gbc):
            # Eq. (S.10): sum_s W^2 G_ac(s) conj(G_bd(s)) + sum_s W W G_ad(s) conj(G_bc(s))
            return (w1 * Gac * np.conj(Gbd)).sum(-1) + (w2 * Gad * np.conj(Gbc)).sum(-1)

        A11 = g11[:, None, :]
        A22 = g22[None, :, :]
        H1111 = H(A11, A11, A11, A11)
        H2222 = H(A22, A22, A22, A22)
        H1122 = H(g12, g12, g12, g12)                     # a=1,b=1,c=2,d=2: G12 conj(G12) twice
        H1112 = H(A11, g12, g12, A11)                     # a=1,b=1,c=1,d=2
        H1212 = H(A11, A22, g12, g21c)                    # a=1,b=2,c=1,d=2
        H1221 = H(g12, g21c, A11, A22)                    # a=1,b=2,c=2,d=1
        H1222 = H(g12, A22, g12, A22)                     # a=1,b=2,c=2,d=2
        f12 = G12[:, :, j]
        f11 = G11[:, None, j]
        f22 = G22[None, :, j]
        Q = H1111 / f11 ** 2 + 2 * np.real(H1122 / (f11 * f22)) + H2222 / f22 ** 2
        A = H1212 - np.real(f12 * H1112 / f11) - np.real(f12 * np.conj(H1222) / f22) + np.abs(f12) ** 2 / 4 * Q
        A = A / (f11 * f22)
        B = H1221 - f12 * H1222 / f22 - f12 * np.conj(H1112) / f11 + f12 ** 2 / 4 * Q
        B = B / (f11 * f22)
        scale = 1.0 / (Wk[j]) ** 2                                            # (2 pi / n)^2 / (W^k)^2
        var_re = np.maximum(np.real(A + B) / 2 * scale, 1e-18)
        var_im = np.maximum(np.real(A - B) / 2 * scale, 1e-18)
        sd_re[f] = np.sqrt(np.real(var_re))
        sd_im[f] = np.sqrt(np.real(var_im))
    return sd_re, sd_im
