"""Results object for the cross-quantilogram (Han, Linton, Oka and Whang, 2016).

Paper outputs reproduced: cross-quantilogram bars with the 95% bootstrap band for
no predictability (HLOW Figs. 2a/3a), Box-Ljung Q(p) against its bootstrap critical
value (Figs. 2b/3b) and the sup-Q statistic over a quantile range (Fig. 1).
Additions: a (tau1, tau2) heatmap / 3D surface / contour at any lag with stars, a
lag x quantile 3D surface, journal tables (rho with stars by lag, Q(p) with stars),
LaTeX / Markdown export and a plain-language interpretation.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .._grid import QQGrid
from .._style import LINE_CYCLE, fmt_q, journal_style, latex_escape, stars

__all__ = ["CQResult"]


@dataclass
class CQResult:
    """Output of :func:`cross_quantilogram_test`.

    Attributes
    ----------
    rho : ndarray (p, m1, m2)
        Cross-quantilogram by lag, tau1 and tau2.
    lo, hi : ndarray (p, m1, m2)
        Bootstrap band for H0: rho = 0 (centered percentiles of rho* - rho).
    pvalues : ndarray (p, m1, m2)
        Two-sided bootstrap p-values of rho.
    Q, Q_cv, Q_p : ndarray (p, m1, m2)
        Box-Ljung Q(p) up to each lag, its bootstrap critical value and p-value.
    """

    rho: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    pvalues: np.ndarray
    Q: np.ndarray
    Q_cv: np.ndarray
    Q_p: np.ndarray
    tau1: np.ndarray
    tau2: np.ndarray
    lags: np.ndarray
    T: int
    n_boot: int
    mean_block: float
    alpha: float
    names: tuple
    conditional: bool = False
    sup: Optional[pd.DataFrame] = None
    self_norm: Optional[dict] = None
    kind: str = "cross-quantilogram"

    # ------------------------------------------------------------- helpers
    def _i(self, arr, v):
        i = int(np.argmin(np.abs(arr - v)))
        if abs(arr[i] - v) > 1e-8:
            raise ValueError(f"quantile {v} not in the grid {list(arr)}")
        return i

    def _note(self) -> str:
        q = "conditional (quantile regression)" if self.conditional else "unconditional (sample)"
        return (f"Notes: {self.names[1]} at t-k predicting {self.names[0]} at t; {q} quantiles; "
                f"T = {self.T}; stationary bootstrap with {self.n_boot} replications, mean block "
                f"length {self.mean_block:.2f}; stars from two-sided bootstrap p-values.")

    # --------------------------------------------------------------- data
    def ci(self, k: int = 1) -> pd.DataFrame:
        """Rho with its 95% confidence interval [rho + c1, rho + c2] (HLOW Sec. 3.1.1) at lag k."""
        i = k - 1
        rows = []
        for a, t1 in enumerate(self.tau1):
            for b, t2 in enumerate(self.tau2):
                r = self.rho[i, a, b]
                rows.append((t1, t2, r, r + self.lo[i, a, b], r + self.hi[i, a, b],
                             self.lo[i, a, b], self.hi[i, a, b], self.pvalues[i, a, b]))
        return pd.DataFrame(rows, columns=["tau1", "tau2", "rho", "ci_low", "ci_high",
                                           "band0_low", "band0_high", "p_value"])

    def to_frame(self) -> pd.DataFrame:
        """Long table: lag, tau1, tau2, rho, band, p-value, Q(p), cv, p-value of Q."""
        L, A, B = np.meshgrid(self.lags, self.tau1, self.tau2, indexing="ij")
        return pd.DataFrame({"lag": L.ravel(), "tau1": A.ravel(), "tau2": B.ravel(),
                             "rho": self.rho.ravel(), "band_low": self.lo.ravel(),
                             "band_high": self.hi.ravel(), "p_value": self.pvalues.ravel(),
                             "Q_BL": self.Q.ravel(), "Q_cv": self.Q_cv.ravel(),
                             "Q_p_value": self.Q_p.ravel(),
                             "sig": [stars(p) for p in self.pvalues.ravel()]})

    def grid(self, k: int = 1) -> QQGrid:
        """Cross-quantilogram at lag k on the (tau1, tau2) grid, with bootstrap p-values."""
        i = k - 1
        return QQGrid(self.rho[i], self.tau1, self.tau2,
                      row_label=f"{self.names[0]} quantile (t)",
                      col_label=f"{self.names[1]} quantile (t-{k})",
                      value_label=r"$\hat\rho_\tau$" + f"({k})", title=f"Cross-quantilogram, lag {k}",
                      pvalues=self.pvalues[i], note=self._note(), center=0.0, decimals=3)

    def q_grid(self, p: Optional[int] = None) -> QQGrid:
        """Box-Ljung Q(p) on the (tau1, tau2) grid with bootstrap p-values."""
        p = p or int(self.lags[-1])
        return QQGrid(self.Q[p - 1], self.tau1, self.tau2,
                      row_label=f"{self.names[0]} quantile", col_label=f"{self.names[1]} quantile",
                      value_label=f"Q({p})", title=f"Box-Ljung portmanteau statistic Q({p})",
                      pvalues=self.Q_p[p - 1], note=self._note(), decimals=2)

    # ------------------------------------------------------------- tables
    def table(self, tau2: Optional[float] = None, lags: Optional[Sequence[int]] = None,
              stat: str = "rho") -> pd.DataFrame:
        """Rows = tau1, columns = lags, cells = rho (or Q) with stars, for one tau2."""
        b = 0 if tau2 is None else self._i(self.tau2, tau2)
        lags = list(lags) if lags is not None else [k for k in (1, 2, 3, 5, 10, 20, 30, 60) if k <= self.lags[-1]]
        V, P = (self.rho, self.pvalues) if stat == "rho" else (self.Q, self.Q_p)
        fmt = "{:.3f}" if stat == "rho" else "{:.2f}"
        data = {f"k={k}": [fmt.format(V[k - 1, a, b]) + stars(P[k - 1, a, b]) for a in range(len(self.tau1))]
                for k in lags}
        return pd.DataFrame(data, index=pd.Index([fmt_q(t) for t in self.tau1], name="tau1"))

    def summary(self, tau2: Optional[Sequence[float]] = None, lags: Optional[Sequence[int]] = None) -> str:
        """Journal tables: rho by lag with stars and Q(p) with stars, for each tau2 (+ sup / SN tests)."""
        tau2 = self.tau2 if tau2 is None else np.atleast_1d(tau2)
        out = []
        W = 86
        hdr = ["=" * W, "Cross-Quantilogram (Han, Linton, Oka and Whang, 2016)".center(W), "=" * W,
               f"Predicted (y1)   : {self.names[0]:<20} Predictor (y2)   : {self.names[1]}",
               f"Observations     : {self.T:<20} Bootstrap reps   : {self.n_boot}",
               f"Max lag          : {int(self.lags[-1]):<20} Mean block length: {self.mean_block:.2f}",
               f"Quantiles        : {'conditional (QR)' if self.conditional else 'unconditional':<20} "
               f"Significance     : {self.alpha}", "=" * W]
        out.append("\n".join(hdr))
        for t2 in tau2:
            for stat, title in (("rho", "Cross-quantilogram rho_tau(k)"), ("Q", "Box-Ljung Q(p), p = k")):
                tab = self.table(t2, lags, stat)
                lines = ["", f"{title}   [{self.names[1]} quantile tau2 = {fmt_q(t2)}]", "-" * W,
                         f"{'tau1':>8}" + "".join(f"{c:>13}" for c in tab.columns), "-" * W]
                for idx, row in tab.iterrows():
                    lines.append(f"{idx:>8}" + "".join(f"{v:>13}" for v in row.values))
                lines.append("-" * W)
                out.append("\n".join(lines))
        if self.sup is not None:
            s = self.sup
            r = s.attrs.get("range", ("", ""))
            lines = ["", f"Sup-Box-Ljung test over tau1 = tau2 in [{r[0]}, {r[1]}]", "-" * W,
                     f"{'p':>6}{'sup Q':>12}{'cv':>12}{'p-value':>12}{'argmax tau':>14}", "-" * W]
            for _, row in s.iterrows():
                lines.append(f"{int(row.lag):>6}{row.sup_Q:>12.3f}{row.cv:>12.3f}{row.p_value:>12.3f}"
                             f"{row.argmax_tau:>14.2f} {stars(row.p_value)}")
            out.append("\n".join(lines + ["-" * W]))
        if self.self_norm is not None:
            S, cv = self.self_norm["S"], self.self_norm["cv"]
            lines = ["", f"Self-normalised omnibus test S(p) (omega = {self.self_norm['omega']}); "
                         f"5% critical values: " + ", ".join(f"p={k}: {cv[k - 1]:.1f}" for k in (1, 2, 3, 5) if k <= len(cv))]
            out.append("\n".join(lines))
        out.append("=" * W + "\nSignificance: *** p<0.01, ** p<0.05, * p<0.10 (bootstrap).\n" + self._note())
        return "\n".join(out)

    def to_latex(self, tau2: Optional[float] = None, lags: Optional[Sequence[int]] = None,
                 caption: str = "Cross-quantilogram estimates", label: str = "tab:cq") -> str:
        """booktabs table: rows tau1, columns lags, rho with stars (and Q(p) panel)."""
        b_tau = self.tau2[0] if tau2 is None else tau2
        A = self.table(b_tau, lags, "rho")
        B = self.table(b_tau, lags, "Q")
        n = A.shape[1]

        def tex(s):
            core = s.rstrip("*")
            st = s[len(core):]
            return core + (r"\sym{" + st + "}" if st else "")
        L = [r"% Preamble: \newcommand{\sym}[1]{\ensuremath{^{#1}}}", r"\begin{table}[htbp]\centering",
             r"\caption{" + caption + r"}\label{" + label + "}", r"\begin{tabular}{l" + "c" * n + "}",
             r"\toprule", r"$\tau_1$ & " + " & ".join(c.replace("k=", "$k=") + "$" for c in A.columns) + r" \\",
             r"\midrule", r"\multicolumn{" + str(n + 1) + r"}{l}{\textit{Panel A. Cross-quantilogram } $\hat\rho_\tau(k)$} \\"]
        L += [f"{i} & " + " & ".join(tex(v) for v in r.values) + r" \\" for i, r in A.iterrows()]
        L += [r"\midrule", r"\multicolumn{" + str(n + 1) + r"}{l}{\textit{Panel B. Box-Ljung } $\hat Q^{(p)}_\tau$, $p=k$} \\"]
        L += [f"{i} & " + " & ".join(tex(v) for v in r.values) + r" \\" for i, r in B.iterrows()]
        L += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\begin{minipage}{0.95\linewidth}\footnotesize\textit{Notes:} " +
              latex_escape(f"{self.names[1]} quantile tau2 = {fmt_q(b_tau)}. ") +
              latex_escape(self._note()[7:]) + r" \sym{***} $p<0.01$, \sym{**} $p<0.05$, \sym{*} $p<0.10$."
              r"\end{minipage}", r"\end{table}"]
        return "\n".join(L)

    def to_markdown(self, tau2: Optional[float] = None, lags=None) -> str:
        A = self.table(tau2, lags, "rho")
        head = "| tau1 | " + " | ".join(A.columns) + " |"
        return "\n".join([head, "|---|" + "---|" * A.shape[1]] +
                         [f"| {i} | " + " | ".join(r.values) + " |" for i, r in A.iterrows()])

    def interpret(self, k: int = 1) -> str:
        """Plain-language reading of directional predictability."""
        i = k - 1
        R, P = self.rho[i], self.pvalues[i]
        sig = P < 0.05
        txt = [f"At lag {k}, {100 * sig.mean():.0f}% of the {R.size} quantile pairs show significant "
               f"directional predictability from {self.names[1]} to {self.names[0]} (5% level)."]
        if sig.any():
            a, b = np.unravel_index(np.nanargmax(np.where(sig, np.abs(R), -1)), R.shape)
            v = R[a, b]
            txt.append(f"The strongest effect is rho = {v:.3f} at (tau1 = {fmt_q(self.tau1[a])}, tau2 = "
                       f"{fmt_q(self.tau2[b])}): when {self.names[1]} is below its {fmt_q(self.tau2[b])} "
                       f"quantile, the probability that {self.names[0]} falls below its "
                       f"{fmt_q(self.tau1[a])} quantile {k} period(s) later is "
                       f"{'higher' if v > 0 else 'lower'} than unconditionally (equivalently, an "
                       f"exceedance of that quantile makes the event {'less' if v > 0 else 'more'} likely).")
            pos = (R > 0) & sig
            neg = (R < 0) & sig
            txt.append(f"Significant positive values: {pos.sum()}, negative: {neg.sum()}.")
        last = int(self.lags[-1])
        qsig = self.Q_p[last - 1] < 0.05
        txt.append(f"The Box-Ljung test up to lag {last} rejects 'no predictability' for "
                   f"{100 * qsig.mean():.0f}% of the quantile pairs.")
        mid = np.argmin(np.abs(self.tau1 - 0.5))
        if np.abs(self.tau1[mid] - 0.5) < 1e-8:
            m = (self.pvalues[:, mid, :] < 0.05).mean()
            txt.append(f"Around the median of {self.names[0]} the cross-quantilogram is significant in "
                       f"{100 * m:.0f}% of lag-quantile cells, compared with "
                       f"{100 * (self.pvalues < 0.05).mean():.0f}% overall"
                       + (", so predictability concentrates in the tails." if m < (self.pvalues < 0.05).mean() else "."))
        if self.sup is not None:
            rej = self.sup[self.sup.p_value < 0.05]
            txt.append(f"The sup-Q test over the quantile range rejects at 5% for {len(rej)} of "
                       f"{len(self.sup)} lag orders.")
        return " ".join(txt)

    def __repr__(self):
        return (f"CQResult({self.names[1]} -> {self.names[0]}; T={self.T}; lags=1..{int(self.lags[-1])}; "
                f"{len(self.tau1)}x{len(self.tau2)} quantiles; B={self.n_boot})")

    # ---------------------------------------------------------------- plots
    def plot_correlogram(self, tau2: Optional[float] = None, tau1: Optional[Sequence[float]] = None,
                         ncols: int = 3, figsize=None):
        """Cross-quantilogram bars over lags with the 95% band for no predictability (HLOW Fig. 2a)."""
        import matplotlib.pyplot as plt
        b = 0 if tau2 is None else self._i(self.tau2, tau2)
        tau1 = self.tau1 if tau1 is None else np.atleast_1d(tau1)
        n = len(tau1)
        nrows = int(np.ceil(n / ncols))
        with journal_style():
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (3.3 * ncols, 2.4 * nrows),
                                     sharex=True, squeeze=False)
            for idx, t1 in enumerate(tau1):
                ax = axes.flat[idx]
                a = self._i(self.tau1, t1)
                r = self.rho[:, a, b]
                sig = self.pvalues[:, a, b] < self.alpha
                ax.bar(self.lags, r, color=np.where(sig, "#1F4E9E", "#9DB4D8"), width=0.7)
                ax.plot(self.lags, self.lo[:, a, b], "r--", lw=0.9)
                ax.plot(self.lags, self.hi[:, a, b], "r--", lw=0.9)
                ax.axhline(0, color="0.3", lw=0.7)
                ax.set_title(rf"$\tau_1$ = {fmt_q(t1)}", fontsize=9)
                if idx % ncols == 0:
                    ax.set_ylabel(r"$\hat\rho_\tau(k)$")
                if idx >= n - ncols:
                    ax.set_xlabel("Lag k")
            for ax in list(axes.flat)[n:]:
                ax.set_visible(False)
            fig.suptitle(f"Cross-quantilogram: {self.names[1]} (" + r"$\tau_2$" + f" = {fmt_q(self.tau2[b])}) "
                         f"$\\rightarrow$ {self.names[0]}", y=1.01)
            fig.tight_layout()
        return fig, axes

    def plot_portmanteau(self, tau2: Optional[float] = None, tau1: Optional[Sequence[float]] = None,
                         ncols: int = 3, figsize=None):
        """Box-Ljung Q(p) by p with the bootstrap critical value (HLOW Fig. 2b)."""
        import matplotlib.pyplot as plt
        b = 0 if tau2 is None else self._i(self.tau2, tau2)
        tau1 = self.tau1 if tau1 is None else np.atleast_1d(tau1)
        n = len(tau1)
        nrows = int(np.ceil(n / ncols))
        with journal_style():
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (3.3 * ncols, 2.4 * nrows),
                                     sharex=True, squeeze=False)
            for idx, t1 in enumerate(tau1):
                ax = axes.flat[idx]
                a = self._i(self.tau1, t1)
                ax.bar(self.lags, self.Q[:, a, b], color="#2CA02C", width=0.7, label="Q(p)")
                ax.plot(self.lags, self.Q_cv[:, a, b], "r--", lw=1.0, label=f"{int(100 * (1 - self.alpha))}% c.v.")
                ax.set_title(rf"$\tau_1$ = {fmt_q(t1)}", fontsize=9)
                if idx == 0:
                    ax.legend(fontsize=7)
                if idx >= n - ncols:
                    ax.set_xlabel("p")
            for ax in list(axes.flat)[n:]:
                ax.set_visible(False)
            fig.suptitle(f"Box-Ljung test: {self.names[1]} (" + r"$\tau_2$" + f" = {fmt_q(self.tau2[b])}) "
                         f"$\\rightarrow$ {self.names[0]}", y=1.01)
            fig.tight_layout()
        return fig, axes

    def plot_sup(self, ax=None, figsize=(6.0, 3.4)):
        """Sup-Box-Ljung statistic by p with its bootstrap critical value (HLOW Fig. 1)."""
        import matplotlib.pyplot as plt
        if self.sup is None:
            raise ValueError("re-run with sup_range=... to compute the sup test")
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            ax.bar(self.sup.lag, self.sup.sup_Q, color="#1F4E9E", width=0.7, label="sup Q")
            ax.plot(self.sup.lag, self.sup.cv, "r--", label=f"{int(100 * (1 - self.alpha))}% c.v.")
            r = self.sup.attrs.get("range")
            ax.set_title(f"Sup-Box-Ljung test, " + r"$\tau \in$" + f" [{r[0]}, {r[1]}]")
            ax.set_xlabel("p")
            ax.legend()
        return fig, ax

    def plot_lag_surface(self, tau2: Optional[float] = None, cmap: str = "BlueRed", elev=28, azim=-130,
                         figsize=(7.2, 5.4)):
        """3D surface of rho over lag k x tau1 for one tau2 (significant cells in black)."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from .._colors import get_cmap
        b = 0 if tau2 is None else self._i(self.tau2, tau2)
        Z = self.rho[:, :, b]
        K, A = np.meshgrid(self.lags, self.tau1, indexing="ij")
        m = np.nanmax(np.abs(Z)) or 1
        with journal_style():
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            s = ax.plot_surface(K, A, Z, cmap=get_cmap(cmap), norm=mcolors.TwoSlopeNorm(0, -m, m),
                                linewidth=0.2, edgecolor="0.4", alpha=0.95)
            sig = self.pvalues[:, :, b] < 0.05
            ax.scatter(K[sig], A[sig], Z[sig], color="k", s=6, depthshade=False, label="p < 0.05")
            ax.set_xlabel("Lag k")
            ax.set_ylabel(r"$\tau_1$ (" + self.names[0] + ")")
            ax.set_zlabel(r"$\hat\rho_\tau(k)$")
            ax.set_box_aspect(None, zoom=0.88)
            ax.view_init(elev=elev, azim=azim)
            ax.legend(loc="upper left")
            fig.colorbar(s, ax=ax, shrink=0.6, pad=0.1)
            ax.set_title(f"Cross-quantilogram surface, {self.names[1]} " + r"$\tau_2$" + f" = {fmt_q(self.tau2[b])}")
        return fig, ax

    def plot(self, kind: str = "correlogram", **kw):
        """Dispatcher: correlogram, portmanteau, sup, heatmap, surface, surface_plotly, contour,
        panel, lag_surface, q_heatmap."""
        k = kw.pop("k", 1)
        f = {"correlogram": lambda: self.plot_correlogram(**kw),
             "portmanteau": lambda: self.plot_portmanteau(**kw),
             "sup": lambda: self.plot_sup(**kw),
             "heatmap": lambda: self.grid(k).heatmap(**kw),
             "surface": lambda: self.grid(k).surface3d(**kw),
             "surface_plotly": lambda: self.grid(k).plot_surface(**kw),
             "contour": lambda: self.grid(k).contour(**kw),
             "panel": lambda: self.grid(k).panel(**kw),
             "lag_surface": lambda: self.plot_lag_surface(**kw),
             "q_heatmap": lambda: self.q_grid(kw.pop("p", None)).heatmap(**kw)}.get(kind)
        if f is None:
            raise ValueError(f"unknown plot kind '{kind}'")
        return f()
