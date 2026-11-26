"""Results object for quantile coherency (Barunik and Kley, 2019).

Paper outputs reproduced: Re R(omega; tau1, tau2) against frequency for selected
quantile pairs with pointwise CIs and a top axis of named periods (BK Fig. 3 left
and centre), and Re R against tau1 = tau2 at fixed periods (BK Fig. 3 right).
Additions: the (tau1, tau2) heatmap / 3D surface / contour at any period (the
quantile-on-quantile view of coherency), a frequency x quantile 3D surface, the
imaginary part (lead-lag), journal tables with CI-based stars, LaTeX / Markdown export
and a plain-language interpretation.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._grid import QQGrid
from .._style import LINE_CYCLE, fmt_q, journal_style, latex_escape

__all__ = ["CoherencyResult"]


@dataclass
class CoherencyResult:
    """Output of :func:`quantile_coherency`.

    Attributes
    ----------
    R : complex ndarray (F, K1, K2)
        Quantile coherency at the positive Fourier frequencies ``freq`` (cycles per period).
    ci : dict or None
        ``freq``, ``re_low``/``re_high``, ``im_low``/``im_high`` (F_ci, K1, K2) and standard deviations.
    """

    R: np.ndarray
    freq: np.ndarray
    tau1: np.ndarray
    tau2: np.ndarray
    n: int
    bandwidth: float
    kernel: str
    ci: Optional[dict]
    alpha: float
    periods: Dict[str, float]
    names: Tuple[str, str]

    def _fi(self, freq: Optional[float] = None, period: Optional[float] = None) -> int:
        if period is not None:
            freq = 1.0 / float(self.periods.get(period, period)) if isinstance(period, str) else 1.0 / period
        return int(np.argmin(np.abs(self.freq - freq)))

    def _ti(self, arr, v):
        return int(np.argmin(np.abs(arr - v)))

    def _ci_at(self, f_idx: int):
        if self.ci is None:
            return None
        k = np.where(self.ci["index"] == f_idx + 1)[0]
        return int(k[0]) if len(k) else None

    def _note(self) -> str:
        return (f"Notes: rank-based quantile coherency (Barunik and Kley, 2019); n = {self.n}; {self.kernel} "
                f"kernel, b_n = {self.bandwidth:.4f}; {int(100 * (1 - self.alpha))}% pointwise CIs (Supp. S5.2); "
                f"stars: CI of Re R excludes zero.")

    # ------------------------------------------------------------- tables
    def grid(self, period=None, freq: Optional[float] = None, part: str = "real") -> QQGrid:
        """(tau1, tau2) surface of the real (or imaginary / modulus) part at one period or frequency."""
        if period is None and freq is None:
            period = list(self.periods)[0] if self.periods else None
            freq = None if period is not None else self.freq[len(self.freq) // 10]
        i = self._fi(freq, period)
        Z = {"real": self.R[i].real, "imag": self.R[i].imag, "abs": np.abs(self.R[i])}[part]
        P = None
        ci_i = self._ci_at(i)
        if ci_i is not None and part in ("real", "imag"):
            lo = self.ci["re_low" if part == "real" else "im_low"][ci_i]
            hi = self.ci["re_high" if part == "real" else "im_high"][ci_i]
            P = np.where((lo > 0) | (hi < 0), 0.01, 0.5)                       # CI excludes zero
        per = 1 / self.freq[i]
        lab = {"real": r"Re $\hat R$", "imag": r"Im $\hat R$", "abs": r"$|\hat R|$"}[part]
        return QQGrid(Z, self.tau1, self.tau2, row_label=f"{self.names[0]} quantile",
                      col_label=f"{self.names[1]} quantile", value_label=lab,
                      title=f"Quantile coherency ({part} part), period = {per:.1f}",
                      pvalues=P, note=self._note(), center=0.0, decimals=2)

    def table(self, pairs: Optional[Sequence[Tuple[float, float]]] = None,
              periods: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Re R with CI-based stars for quantile pairs (rows) and periods (columns)."""
        periods = periods or dict(self.periods, **{"Short (2)": 2})
        if pairs is None:
            qs = [q for q in (0.05, 0.10, 0.50, 0.90, 0.95) if np.min(np.abs(self.tau1 - q)) < 1e-8]
            pairs = [(q, q) for q in qs] + [(self.tau1[0], self.tau2[-1]), (self.tau1[-1], self.tau2[0])]
        data = {}
        for name, per in periods.items():
            i = self._fi(period=per)
            ci_i = self._ci_at(i)
            col = []
            for a, b in pairs:
                ia, ib = self._ti(self.tau1, a), self._ti(self.tau2, b)
                v = self.R[i, ia, ib].real
                s = ""
                if ci_i is not None:
                    lo, hi = self.ci["re_low"][ci_i, ia, ib], self.ci["re_high"][ci_i, ia, ib]
                    s = "*" if (lo > 0 or hi < 0) else ""
                col.append(f"{v:.3f}{s}")
            data[f"{name} ({per:g})" if not str(name).startswith("Short") else name] = col
        return pd.DataFrame(data, index=pd.Index([f"{fmt_q(a)}|{fmt_q(b)}" for a, b in pairs],
                                                 name="tau1|tau2"))

    def summary(self) -> str:
        W = 84
        tab = self.table()
        L = ["=" * W, "Quantile Coherency (Barunik and Kley, 2019)".center(W), "=" * W,
             f"Series j1 / j2     : {self.names[0]} / {self.names[1]}",
             f"Observations       : {self.n:<14} Kernel / bandwidth : {self.kernel} / {self.bandwidth:.4f}",
             f"Quantile levels    : {len(self.tau1)} x {len(self.tau2)}      Frequencies       : {len(self.freq)}",
             "-" * W, "Real part of quantile coherency by period (columns) and quantile pair (rows)", "-" * W,
             f"{'tau1|tau2':>12}" + "".join(f"{c:>14}" for c in tab.columns), "-" * W]
        for i, r in tab.iterrows():
            L.append(f"{i:>12}" + "".join(f"{v:>14}" for v in r.values))
        L += ["=" * W, "* the pointwise CI of Re R excludes zero." if self.ci is not None else "", self._note()]
        return "\n".join(L)

    def to_latex(self, caption: str = "Quantile coherency (real part)", label: str = "tab:coh") -> str:
        tab = self.table()
        L = [r"\begin{table}[htbp]\centering", r"\caption{" + caption + r"}\label{" + label + "}",
             r"\begin{tabular}{l" + "c" * tab.shape[1] + "}", r"\toprule",
             r"$\tau_1|\tau_2$ & " + " & ".join(latex_escape(c) for c in tab.columns) + r" \\", r"\midrule"]
        L += [f"{i} & " + " & ".join(v.replace("*", r"$^{*}$") for v in r.values) + r" \\" for i, r in tab.iterrows()]
        L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\begin{minipage}{0.95\linewidth}\footnotesize"
              r"\textit{Notes:} " + latex_escape(self._note()[7:]) + r"\end{minipage}", r"\end{table}"]
        return "\n".join(L)

    def to_frame(self) -> pd.DataFrame:
        F, A, B = np.meshgrid(self.freq, self.tau1, self.tau2, indexing="ij")
        return pd.DataFrame({"freq": F.ravel(), "period": 1 / F.ravel(), "tau1": A.ravel(), "tau2": B.ravel(),
                             "re": self.R.real.ravel(), "im": self.R.imag.ravel(), "abs": np.abs(self.R).ravel()})

    def interpret(self) -> str:
        txt = []
        tab_f = {}
        for name, per in self.periods.items():
            i = self._fi(period=per)
            diag = np.array([self.R[i, a, self._ti(self.tau2, t)].real for a, t in enumerate(self.tau1)])
            tab_f[name] = diag
        lo_q = self.tau1 <= 0.15
        hi_q = self.tau1 >= 0.85
        for name, d in tab_f.items():
            if lo_q.any() and hi_q.any():
                l, h = d[lo_q].mean(), d[hi_q].mean()
                txt.append(f"At the {name} period ({self.periods[name]:g}), coherency between the lower tails "
                           f"({l:.2f}) is {'stronger' if l > h else 'weaker'} than between the upper tails ({h:.2f})"
                           + (", i.e. downside co-movement dominates." if l > h else "."))
        i_s, i_l = self._fi(freq=0.5), self._fi(period=max(self.periods.values()))
        a0 = self._ti(self.tau1, 0.05)
        b0 = self._ti(self.tau2, 0.05)
        txt.append(f"For the 0.05|0.05 pair, Re R moves from {self.R[i_s, a0, b0].real:.2f} at the shortest "
                   f"cycle (2 periods) to {self.R[i_l, a0, b0].real:.2f} at the longest named period, so "
                   + ("long-run cycles are more strongly connected in the tails."
                      if self.R[i_l, a0, b0].real > self.R[i_s, a0, b0].real else
                      "short-run fluctuations carry most of the tail dependence."))
        b1 = self._ti(self.tau2, 0.95)
        v = self.R[i_l, a0, b1].real
        txt.append(f"The opposite-tail pair 0.05|0.95 has Re R = {v:.2f} at the longest period "
                   f"({'positive' if v > 0 else 'negative'}: a low {self.names[0]} tends to coincide with "
                   f"{'a low' if v > 0 else 'a high'} {self.names[1]} over those cycles).")
        return " ".join(txt)

    def __repr__(self):
        return (f"CoherencyResult({self.names[0]} & {self.names[1]}; n={self.n}; "
                f"{len(self.tau1)}x{len(self.tau2)} quantiles; {len(self.freq)} frequencies)")

    # --------------------------------------------------------------- plots
    def _top_axis(self, ax):
        ax2 = ax.secondary_xaxis("top")
        ticks = [1 / p for p in self.periods.values() if 1 / p <= 0.5]
        labs = [k for k, p in self.periods.items() if 1 / p <= 0.5]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labs)
        for t in ticks:
            ax.axvline(t, color="0.7", lw=0.6, ls=":")
        return ax2

    def plot_frequency(self, pairs: Optional[Sequence[Tuple[float, float]]] = None, part: str = "real",
                       ax=None, figsize=(6.6, 3.8), log_freq: bool = False):
        """Re (or Im) R against omega / 2 pi with pointwise CIs (BK Fig. 3, left / centre)."""
        import matplotlib.pyplot as plt
        if pairs is None:
            pairs = [(0.5, 0.5), (0.05, 0.05), (0.95, 0.95)]
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            for k, (a, b) in enumerate(pairs):
                ia, ib = self._ti(self.tau1, a), self._ti(self.tau2, b)
                c, ls = LINE_CYCLE[k % len(LINE_CYCLE)]
                v = self.R[:, ia, ib].real if part == "real" else self.R[:, ia, ib].imag
                ax.plot(self.freq, v, color=c, ls=ls, lw=1.2, label=f"{fmt_q(a)} | {fmt_q(b)}")
                if self.ci is not None:
                    key = "re" if part == "real" else "im"
                    ax.fill_between(self.ci["freq"], self.ci[f"{key}_low"][:, ia, ib], self.ci[f"{key}_high"][:, ia, ib],
                                    color=c, alpha=0.18, lw=0)
            ax.axhline(0, color="0.3", lw=0.7)
            if log_freq:
                ax.set_xscale("log")
            ax.set_xlim(self.freq[0], 0.5)
            ax.set_xlabel(r"$\omega / 2\pi$ (cycles per period)")
            ax.set_ylabel(r"Re $\hat R(\omega;\tau_1,\tau_2)$" if part == "real" else r"Im $\hat R(\omega;\tau_1,\tau_2)$")
            self._top_axis(ax)
            ax.legend(title=r"$\tau_1|\tau_2$", fontsize=8)
        return fig, ax

    def plot_quantile(self, periods: Optional[Dict[str, float]] = None, ax=None, figsize=(5.4, 3.8)):
        """Re R along tau1 = tau2 at fixed periods (BK Fig. 3, right)."""
        import matplotlib.pyplot as plt
        periods = periods or self.periods
        common = [t for t in self.tau1 if np.min(np.abs(self.tau2 - t)) < 1e-8]
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            for k, (name, per) in enumerate(periods.items()):
                i = self._fi(period=per)
                v = [self.R[i, self._ti(self.tau1, t), self._ti(self.tau2, t)].real for t in common]
                c, ls = LINE_CYCLE[k % len(LINE_CYCLE)]
                ax.plot(common, v, color=c, ls=ls, marker="o", ms=3, label=f"{name} ({per:g})")
                ci_i = self._ci_at(i)
                if ci_i is not None:
                    lo = [self.ci["re_low"][ci_i, self._ti(self.tau1, t), self._ti(self.tau2, t)] for t in common]
                    hi = [self.ci["re_high"][ci_i, self._ti(self.tau1, t), self._ti(self.tau2, t)] for t in common]
                    ax.fill_between(common, lo, hi, color=c, alpha=0.15, lw=0)
            ax.set_xlabel(r"$\tau_1 = \tau_2$")
            ax.set_ylabel(r"Re $\hat R$")
            ax.legend(fontsize=8)
        return fig, ax

    def plot_freq_surface(self, which: str = "diagonal", cmap: str = "Parula", elev=28, azim=-125,
                          max_freq: float = 0.5, figsize=(7.4, 5.4)):
        """3D surface of Re R over frequency x quantile (diagonal tau1 = tau2 or anti-diagonal)."""
        import matplotlib.pyplot as plt
        from .._colors import get_cmap
        common = np.array([t for t in self.tau1 if np.min(np.abs(self.tau2 - t)) < 1e-8])
        sel = self.freq <= max_freq
        step = max(1, sel.sum() // 200)
        fi = np.where(sel)[0][::step]
        Z = np.empty((len(fi), len(common)))
        for c_, t in enumerate(common):
            ib = self._ti(self.tau2, t if which == "diagonal" else 1 - t)
            Z[:, c_] = self.R[fi, self._ti(self.tau1, t), ib].real
        Fm, Qm = np.meshgrid(self.freq[fi], common, indexing="ij")
        with journal_style():
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            s = ax.plot_surface(Fm, Qm, Z, cmap=get_cmap(cmap), linewidth=0, antialiased=True)
            ax.set_xlabel(r"$\omega/2\pi$", labelpad=6)
            ax.set_ylabel(r"$\tau_1 = \tau_2$" if which == "diagonal" else r"$\tau_1 = 1-\tau_2$", labelpad=6)
            ax.set_zlabel(r"Re $\hat R$")
            ax.set_box_aspect(None, zoom=0.88)
            ax.view_init(elev=elev, azim=azim)
            fig.colorbar(s, ax=ax, shrink=0.6, pad=0.1, label=r"Re $\hat R$")
            ax.set_title("Quantile coherency across frequencies and quantiles")
        return fig, ax

    def plot(self, kind: str = "frequency", **kw):
        """Dispatcher: frequency, quantile, heatmap, surface, surface_plotly, contour, panel,
        freq_surface, imaginary."""
        period = kw.pop("period", None)
        f = {"frequency": lambda: self.plot_frequency(**kw),
             "imaginary": lambda: self.plot_frequency(part="imag", **kw),
             "quantile": lambda: self.plot_quantile(**kw),
             "heatmap": lambda: self.grid(period).heatmap(**kw),
             "surface": lambda: self.grid(period).surface3d(**kw),
             "surface_plotly": lambda: self.grid(period).plot_surface(**kw),
             "contour": lambda: self.grid(period).contour(**kw),
             "panel": lambda: self.grid(period).panel(**kw),
             "freq_surface": lambda: self.plot_freq_surface(**kw)}.get(kind)
        if f is None:
            raise ValueError(f"unknown plot kind '{kind}'")
        return f()
