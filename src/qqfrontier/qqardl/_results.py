"""Results object for the quantile-on-quantile ARDL (port of Stata ``qqardl``).

Tables reproduce the Stata output: bounds-test F, long run (cointegrated pairs only),
short run d1 and error-correction term d0, rows theta (x quantiles, high at the top),
columns tau (y quantiles), stars from p-values.  Figures: heatmaps / 3D surfaces /
contours of every coefficient (Parula, or a diverging map centred at zero), a
cointegration map, the observed series against its quantile series, and the long-run
elasticity along tau = theta with a 95% band.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._grid import QQGrid
from .._style import LINE_CYCLE, fmt_q, journal_style, latex_escape, stars

__all__ = ["QQARDLResult"]

_LAB = {"f": "Bounds test F-statistic, Eq. (1) (H0: no cointegration)",
        "lr": "Long-run coefficient",
        "sr": "Short-run coefficient d1 on D.x (Eq. 4)",
        "ect": "Error-correction term d0 on L.y (Eq. 4)",
        "r2": "R-squared of Eq. (4)"}


@dataclass
class QQARDLResult:
    taus: np.ndarray
    thetas: np.ndarray
    F: np.ndarray
    F_p: np.ndarray
    t_bdm: np.ndarray
    ect: np.ndarray
    ect_se: np.ndarray
    ect_p: np.ndarray
    lr: np.ndarray
    lr_se: np.ndarray
    lr_p: np.ndarray
    lr_display: np.ndarray
    sr: np.ndarray
    sr_se: np.ndarray
    sr_p: np.ndarray
    r2: np.ndarray
    coint: np.ndarray
    N: int
    NB: int
    df: int
    qser_y: np.ndarray
    qser_x: np.ndarray
    y: np.ndarray
    x: np.ndarray
    names: Tuple[str, str]
    lags: Tuple[int, int]
    btest: str
    btlags: Tuple[int, int]
    vce: str
    lrtype: str
    lrall: bool
    coint_level: float
    pv_source: str
    index: Optional[pd.Index] = None

    # ------------------------------------------------------------- access
    def _vals(self, stat: str):
        stat = stat.lower()
        if stat == "f":
            P = self.F_p if np.isfinite(self.F_p).any() else None
            return self.F, P, None
        if stat == "lr":
            return self.lr_display, self.lr_p, self.lr_se
        if stat == "lr_all":
            return self.lr, self.lr_p, self.lr_se
        if stat == "sr":
            return self.sr, self.sr_p, self.sr_se
        if stat == "ect":
            return self.ect, self.ect_p, self.ect_se
        if stat == "r2":
            return self.r2, None, None
        raise ValueError("stat must be 'F', 'lr', 'lr_all', 'sr', 'ect' or 'r2'")

    def grid(self, stat: str = "lr") -> QQGrid:
        """Coefficient (or F) surface over (theta of x, tau of y) as a :class:`QQGrid`."""
        V, P, SE = self._vals(stat)
        lab = {"f": "F-statistic", "lr": "Long-run coefficient", "lr_all": "Long-run coefficient",
               "sr": "Short-run coefficient", "ect": "ECT coefficient", "r2": "R-squared"}[stat.lower()]
        title = _LAB.get(stat.lower().replace("_all", ""), lab)
        if stat.lower().startswith("lr"):
            title += "  LR = -b1/d0" if self.lrtype == "multiplier" else "  b1 (Eq. 4)"
        return QQGrid(V, self.thetas, self.taus, row_label=f"{self.names[1]} quantile (theta)",
                      col_label=f"{self.names[0]} quantile (tau)", value_label=lab, title=title,
                      pvalues=P, note=self._note(), decimals=3,
                      center=None if stat.lower() in ("f", "r2") else 0.0)

    def _note(self) -> str:
        return (f"Notes: rows: quantiles (theta) of {self.names[1]}; columns: quantiles (tau) of {self.names[0]}. "
                f"Bounds test: {self.pv_source}; N = {self.NB}. Coefficients: t({self.df}) with "
                f"{'OLS' if self.vce == 'ols' else 'robust'} standard errors; LR s.e. by the delta method.")

    def to_frame(self) -> pd.DataFrame:
        Th, Ta = np.meshgrid(self.thetas, self.taus, indexing="ij")
        return pd.DataFrame({"theta": Th.ravel(), "tau": Ta.ravel(), "F": self.F.ravel(), "F_p": self.F_p.ravel(),
                             "cointegrated": self.coint.ravel(), "ect": self.ect.ravel(), "ect_se": self.ect_se.ravel(),
                             "ect_p": self.ect_p.ravel(), "lr": self.lr.ravel(), "lr_se": self.lr_se.ravel(),
                             "lr_p": self.lr_p.ravel(), "sr": self.sr.ravel(), "sr_se": self.sr_se.ravel(),
                             "sr_p": self.sr_p.ravel(), "r2": self.r2.ravel()})

    # ------------------------------------------------------------- tables
    def _table(self, stat: str) -> str:
        V, P, _ = self._vals(stat)
        w0, w = 10, 10
        W = w0 + w * len(self.taus)
        L = [_LAB[stat.lower().replace("_all", "")] + (("  LR = -b1/d0" if self.lrtype == "multiplier" else "  b1")
                                                      + ("" if self.lrall else "  [cointegrated pairs only]")
                                                      if stat.lower() == "lr" else ""),
             f"{'theta|tau':>{w0 - 1}} " + "".join(f"{fmt_q(t):>{w}}" for t in self.taus), "-" * W]
        for i in range(len(self.thetas) - 1, -1, -1):
            cells = []
            for j in range(len(self.taus)):
                v = V[i, j]
                if not np.isfinite(v):
                    cells.append(f"{'.':>{w}}")
                    continue
                s = stars(P[i, j]) if P is not None else ""
                cells.append(f"{f'{v:.3f}{s}':>{w}}")
            L.append(f"{fmt_q(self.thetas[i]):>{w0 - 1}} " + "".join(cells))
        return "\n".join(L)

    def summary(self) -> str:
        W = 10 + 10 * len(self.taus)
        span = ""
        if self.index is not None:
            span = f"   Sample: {self.index[0]}-{self.index[-1]}"
        hdr = ["-" * W, f"{'Quantile-on-Quantile ARDL (QQARDL)':<48}Number of obs   = {self.NB:>6}",
               f"{'Destek et al. (2025); QSER of Li (2022)':<48}ARDL order      = ({self.lags[0]},{self.lags[1]})",
               f"{'Dependent: ' + self.names[0] + '   Regressor: ' + self.names[1]:<48}Quantile pairs  = {self.F.size:>6}" + span,
               "-" * W, ""]
        body = [self._table(s) + "\n" for s in ("F", "lr", "sr", "ect")]
        foot = ["-" * W, f"Rows: quantiles (theta) of {self.names[1]}; columns: quantiles (tau) of {self.names[0]}.",
                f"*** p<0.01, ** p<0.05, * p<0.10. Bounds test: {self.pv_source}; N = {self.NB}.",
                f"Coefficients: t({self.df}) with {'OLS' if self.vce == 'ols' else 'robust'} standard errors; "
                f"LR SE by the delta method.",
                f"Cointegrated pairs (p<{self.coint_level:.2f}): {int(self.coint.sum())} of {self.coint.size}"]
        return "\n".join(hdr + body + foot)

    def to_latex(self, stat: str = "lr", caption: Optional[str] = None, label: str = "tab:qqardl") -> str:
        return self.grid(stat).to_latex(caption=caption or _LAB.get(stat.lower(), stat), label=label)

    def to_markdown(self, stat: str = "lr") -> str:
        return self.grid(stat).to_markdown()

    def interpret(self) -> str:
        c = self.coint
        txt = [f"Cointegration between the quantiles of {self.names[0]} and {self.names[1]} is found for "
               f"{int(c.sum())} of {c.size} quantile pairs ({100 * c.mean():.0f}%)."]
        lo = self.taus <= 0.2
        hi = self.taus >= 0.8
        txt.append(f"It is most frequent at low quantiles of {self.names[0]} ({100 * c[:, lo].mean():.0f}% of pairs "
                   f"with tau <= 0.2) versus high quantiles ({100 * c[:, hi].mean():.0f}% with tau >= 0.8).")
        L = self.lr_display
        if np.isfinite(L).any():
            sig = np.isfinite(L) & (self.lr_p < 0.05)
            txt.append(f"Among cointegrated pairs, the long-run coefficient ranges from {np.nanmin(L):.3f} to "
                       f"{np.nanmax(L):.3f} (median {np.nanmedian(L):.3f}); {int(sig.sum())} are significant at 5%.")
            i, j = np.unravel_index(np.nanargmax(np.where(sig, np.abs(L), -1)), L.shape) if sig.any() else (None, None)
            if i is not None:
                txt.append(f"The largest significant long-run effect ({L[i, j]:.3f}) arises when {self.names[1]} is at "
                           f"its {fmt_q(self.thetas[i])} quantile and {self.names[0]} at its {fmt_q(self.taus[j])} "
                           f"quantile: a 1% rise in {self.names[1]} is associated with a {L[i, j]:.2f}% change in "
                           f"{self.names[0]} in the long run (log-log specification).")
        neg = (self.ect < 0) & (self.ect_p < 0.05)
        txt.append(f"The error-correction term is negative and significant for {int(neg.sum())} of {neg.size} pairs "
                   f"(median {np.median(self.ect):.3f}: about {100 * abs(np.median(self.ect)):.0f}% of a deviation "
                   f"from equilibrium is corrected each period).")
        srs = self.sr_p < 0.05
        txt.append(f"Short-run effects are significant for {int(srs.sum())} pairs, with median {np.median(self.sr):.3f}.")
        return " ".join(txt)

    def __repr__(self):
        return (f"QQARDLResult({self.names[0]} on {self.names[1]}; N={self.NB}; {self.F.shape[0]}x{self.F.shape[1]} "
                f"pairs; cointegrated {int(self.coint.sum())})")

    # ------------------------------------------------------------- plots
    def plot_coint_map(self, ax=None, figsize=(6.4, 5.2)):
        """Bounds-test F heatmap with cointegrated pairs outlined (white frame)."""
        fig, ax = self.grid("F").heatmap(ax=ax, figsize=figsize, cmap="Parula")
        for i in range(len(self.thetas)):
            for j in range(len(self.taus)):
                if self.coint[i, j]:
                    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=1.6))
        ax.set_title(f"Bounds test F (frames: cointegrated, p < {self.coint_level:.2f})")
        return fig, ax

    def plot_qser(self, which: str = "y", taus: Sequence[float] = (0.05, 0.5, 0.95), ax=None, figsize=(6.8, 3.6)):
        """Observed series and its quantile series (Li, 2022) at selected levels."""
        import matplotlib.pyplot as plt
        v = self.y if which == "y" else self.x
        Q = self.qser_y if which == "y" else self.qser_x
        grid = self.taus if which == "y" else self.thetas
        t = self.index if self.index is not None else np.arange(1, len(v) + 1)
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            ax.plot(t, v, color="k", lw=1.5, label="Observed")
            for k, q in enumerate(taus):
                j = int(np.argmin(np.abs(grid - q)))
                c, ls = LINE_CYCLE[k % len(LINE_CYCLE)]
                ax.plot(t, Q[:, j], color=c, ls=ls, lw=1.0, label=f"QSER {fmt_q(grid[j])}")
            ax.set_ylabel(self.names[0] if which == "y" else self.names[1])
            ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            ax.set_title("Observed series and quantile series")
        return fig, ax

    def plot_diagonal(self, stat: str = "lr", ax=None, figsize=(6.0, 3.6)):
        """Coefficient along tau = theta with 95% band (e.g. the long-run elasticity by quantile)."""
        import matplotlib.pyplot as plt
        V, P, SE = self._vals("lr_all" if stat == "lr" else stat)
        common = [t for t in self.taus if np.min(np.abs(self.thetas - t)) < 1e-9]
        v = np.array([V[int(np.argmin(np.abs(self.thetas - t))), int(np.argmin(np.abs(self.taus - t)))] for t in common])
        s = np.array([SE[int(np.argmin(np.abs(self.thetas - t))), int(np.argmin(np.abs(self.taus - t)))] for t in common])
        from scipy.stats import t as tdist
        z = tdist.ppf(0.975, self.df)
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            ax.fill_between(common, v - z * s, v + z * s, color="#1F4E9E", alpha=0.18, lw=0, label="95% CI")
            ax.plot(common, v, color="#1F4E9E", marker="o", ms=4, label=stat.upper())
            ax.axhline(0, color="0.3", lw=0.8)
            ax.set_xlabel(r"Quantile ($\tau = \theta$)")
            ax.set_ylabel({"lr": "Long-run coefficient", "sr": "Short-run coefficient",
                           "ect": "ECT coefficient"}.get(stat, stat))
            ax.legend(fontsize=8)
        return fig, ax

    def plot(self, kind: str = "heatmap", stat: str = "lr", **kw):
        """Dispatcher: heatmap, surface, surface_plotly, contour, panel, slices, coint, qser, diagonal."""
        f = {"heatmap": lambda: self.grid(stat).heatmap(**kw),
             "surface": lambda: self.grid("lr_all" if stat == "lr" else stat).surface3d(**kw),
             "surface_plotly": lambda: self.grid("lr_all" if stat == "lr" else stat).plot_surface(**kw),
             "contour": lambda: self.grid("lr_all" if stat == "lr" else stat).contour(**kw),
             "panel": lambda: self.grid("lr_all" if stat == "lr" else stat).panel(**kw),
             "slices": lambda: self.grid(stat).slices(**kw),
             "coint": lambda: self.plot_coint_map(**kw),
             "qser": lambda: self.plot_qser(**kw),
             "diagonal": lambda: self.plot_diagonal(stat, **kw)}.get(kind)
        if f is None:
            raise ValueError(f"unknown plot kind '{kind}'")
        return f()
