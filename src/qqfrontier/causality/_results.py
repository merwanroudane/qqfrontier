"""Results object for the instability-robust quantile Granger causality test.

Paper outputs reproduced (Mayer, Wied and Troster, 2025): p-values of supWald,
expLM and supLM over quantile ranges (Table 3), regimes of Granger causality with
dates (Table 4), and the CUSUM curve with break dates and critical value (Fig. 2).
Additions: LM(tau) curve with bootstrap critical values, time x quantile map of the
CUSUM detector LM1(lambda, tau), unrestricted gamma(tau) profile, journal tables,
LaTeX export and a plain-language interpretation.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .._grid import QQGrid
from .._style import fmt_q, journal_style, latex_escape, stars

__all__ = ["GCResult", "qgc_map"]


@dataclass
class GCResult:
    LM_tau: np.ndarray
    p_tau: np.ndarray
    p_tau_asy: np.ndarray
    A: np.ndarray
    LM1: np.ndarray
    LM2: np.ndarray
    taus: np.ndarray
    tests: pd.DataFrame
    cusum: np.ndarray
    cusum_cv: float
    regimes: Optional[dict]
    n: int
    p: int
    k: int
    lags: int
    method: str
    n_boot: int
    moment: int
    alpha: float
    names: tuple
    dates: Optional[pd.DatetimeIndex] = None
    wald_tau: Optional[np.ndarray] = None
    gamma_hat: Optional[np.ndarray] = None
    alpha_coef: Optional[np.ndarray] = None

    # --------------------------------------------------------------- helpers
    def _date(self, i: int) -> str:
        if self.dates is None:
            return str(i)
        i = int(np.clip(i, 0, len(self.dates) - 1))
        return self.dates[i].strftime("%Y-%m-%d")

    def _what(self) -> str:
        return "variance" if self.moment == 2 else "quantiles"

    def regime_table(self) -> pd.DataFrame:
        """Regimes of Granger causality (MWT Table 4 layout): start, end, decision, p-value."""
        if not self.regimes:
            return pd.DataFrame()
        rows = []
        for s0, s1, gc, pv in self.regimes["segments"]:
            rows.append({"start": self._date(s0), "end": self._date(s1 - 1),
                         "observations": s1 - s0,
                         "Granger causality": {True: "yes", False: "no", None: "inconclusive"}[gc],
                         "expLM p-value": pv})
        return pd.DataFrame(rows)

    def tau_table(self, taus: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)) -> pd.DataFrame:
        """LM(tau) at selected quantiles with bootstrap and asymptotic p-values."""
        rows = []
        for t in taus:
            i = int(np.argmin(np.abs(self.taus - t)))
            rows.append({"tau": self.taus[i], "LM(tau)": self.LM_tau[i],
                         "sup LM1 (instability)": self.LM1[:, i].max(),
                         "LM2 (causality)": self.LM2[i],
                         "p-value": self.p_tau[i], "p-value (asy)": self.p_tau_asy[i]})
        return pd.DataFrame(rows).set_index("tau")

    # ---------------------------------------------------------------- tables
    def summary(self) -> str:
        W = 92
        span = f"{self._date(0)} to {self._date(self.n - 1)}" if self.dates is not None else f"{self.n} obs."
        L = ["=" * W, "Quantile Granger Causality Robust to Instability".center(W),
             "(Mayer, Wied and Troster, 2025)".center(W), "=" * W,
             f"H0: {self.names[1]} does not Granger-cause {self.names[0]} in {self._what()}, with stable effect",
             f"{'Lags of cause (q)':<26}: {self.lags:<17}{'Observations':<20}: {self.n}",
             f"{'Regressors in w (incl. 1)':<26}: {self.k:<17}{'Sample':<20}: {span}",
             f"{'Critical values':<26}: {self.method + (f' (B={self.n_boot})' if self.method == 'bootstrap' else ''):<17}"
             f"{'Quantile grid':<20}: {fmt_q(self.taus[0])}-{fmt_q(self.taus[-1])} ({len(self.taus)} points)",
             "-" * W]
        has_w = "supWald" in self.tests
        head = f"{'Quantile range':<18}" + (f"{'supWald':>12}{'p-value':>10}" if has_w else "") + \
               f"{'expLM':>14}{'p-value':>10}{'supLM':>12}{'p-value':>10}"
        L += [head, "-" * W]
        for lab, r in self.tests.iterrows():
            line = f"{lab:<18}"
            if has_w:
                line += f"{r.supWald:>12.3f}{r.p_supWald:>7.3f}{stars(r.p_supWald):<3}"
            line += f"{r.expLM:>14.3f}{r.p_expLM:>7.3f}{stars(r.p_expLM):<3}{r.supLM:>12.3f}{r.p_supLM:>7.3f}{stars(r.p_supLM):<3}"
            L.append(line)
        L += ["-" * W, "LM(tau) at selected quantiles", "-" * W,
              f"{'tau':>8}{'LM(tau)':>12}{'sup LM1':>12}{'LM2':>12}{'p-value':>12}{'p (asy)':>12}", "-" * W]
        for t, r in self.tau_table().iterrows():
            L.append(f"{t:>8.2f}{r['LM(tau)']:>12.3f}{r['sup LM1 (instability)']:>12.3f}{r['LM2 (causality)']:>12.3f}"
                     f"{r['p-value']:>9.3f}{stars(r['p-value']):<3}{r['p-value (asy)']:>12.3f}")
        if self.regimes:
            L += ["-" * W, "Regimes of Granger causality (Algorithm 2, Sidak-corrected levels)", "-" * W]
            for _, r in self.regime_table().iterrows():
                L.append(f"  {r.start} -> {r.end}  ({r.observations:>5} obs.)   GC: {r['Granger causality']:<14}"
                         f"  expLM p = {r['expLM p-value']:.3f}")
            L += ["  " + s for s in self.regimes["steps"]]
        L += ["=" * W, "Significance: *** p<0.01, ** p<0.05, * p<0.10. supLM/expLM: MWT Eq. (8); supWald: Koenker and",
              "Machado (1999) with pairs-bootstrap covariance; LM(tau) = h(tau)(sup LM1 + LM2), Sec. 3.1."]
        return "\n".join(L)

    def to_latex(self, caption: str = "Quantile Granger causality robust to instability",
                 label: str = "tab:qgc") -> str:
        has_w = "supWald" in self.tests
        cols = (["supWald"] if has_w else []) + ["expLM", "supLM"]
        L = [r"% Preamble: \newcommand{\sym}[1]{\ensuremath{^{#1}}}", r"\begin{table}[htbp]\centering",
             r"\caption{" + caption + r"}\label{" + label + "}", r"\begin{tabular}{l" + "cc" * len(cols) + "}",
             r"\toprule", "Quantile range & " + " & ".join(r"\multicolumn{2}{c}{" + c + "}" for c in cols) + r" \\",
             "".join(r"\cmidrule(lr){" + f"{2 + 2 * i}-{3 + 2 * i}" + "}" for i in range(len(cols))),
             " & " + " & ".join(["Stat. & $p$-value"] * len(cols)) + r" \\", r"\midrule"]
        for lab, r in self.tests.iterrows():
            cells = []
            for c in cols:
                pv = r[f"p_{c}"]
                st = stars(pv)
                cells += [f"{r[c]:.3f}" + (r"\sym{" + st + "}" if st else ""), f"{pv:.3f}"]
            L.append(f"${lab}$ & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\begin{minipage}{0.95\linewidth}\footnotesize\textit{Notes:} " +
              latex_escape(f"H0: {self.names[1]} does not Granger-cause {self.names[0]} in {self._what()} "
                           f"with a stable effect; q = {self.lags} lags; n = {self.n}; ") +
              ("bootstrap critical values (Algorithm 1, B = " + str(self.n_boot) + ")" if self.method == "bootstrap"
               else "asymptotic critical values under Assumption D") +
              r". \sym{***} $p<0.01$, \sym{**} $p<0.05$, \sym{*} $p<0.10$.\end{minipage}", r"\end{table}"]
        return "\n".join(L)

    def interpret(self) -> str:
        t = self.tests
        full = t.iloc[0]
        txt = [f"Over the quantile range {t.index[0]}, the expLM test "
               f"{'rejects' if full.p_expLM < 0.05 else 'does not reject'} the joint null of no and stable "
               f"Granger causality from {self.names[1]} to {self.names[0]} in {self._what()} "
               f"(p = {full.p_expLM:.3f}); supLM gives p = {full.p_supLM:.3f}."]
        if "supWald" in t:
            txt.append(f"The conventional supWald test, which ignores instability, gives p = {full.p_supWald:.3f}"
                       + (", so allowing for breaks strengthens the evidence." if full.p_expLM < full.p_supWald else "."))
        rej = [lab for lab, r in t.iloc[1:].iterrows() if r.p_expLM < 0.05]
        if rej:
            txt.append("Causality is detected in the ranges " + ", ".join(rej) + ".")
        sig = self.taus[self.p_tau < 0.05]
        if sig.size:
            txt.append(f"Quantile by quantile, LM(tau) is significant at 5% for {sig.size} of {len(self.taus)} "
                       f"levels (from {fmt_q(sig.min())} to {fmt_q(sig.max())}).")
        i = int(np.argmax(self.LM1.max(axis=0) / (self.LM2 + 1e-12)))
        share = np.mean(self.LM1.max(axis=0) > self.LM2)
        txt.append(f"The instability component (sup LM1) dominates the causality component (LM2) at "
                   f"{100 * share:.0f}% of the quantiles, so " +
                   ("time variation is a main source of the rejection." if share > 0.5 else
                    "the rejection is driven mainly by a stable non-zero effect."))
        if self.regimes and self.regimes.get("breaks"):
            bk = ", ".join(self._date(b) for b in self.regimes["breaks"])
            gcs = [f"{self._date(s0)} to {self._date(s1 - 1)}" for s0, s1, gc, _ in self.regimes["segments"] if gc]
            txt.append(f"Algorithm 2 dates break(s) at {bk}; Granger causality holds on: " +
                       ("; ".join(gcs) if gcs else "no segment") + ".")
        return " ".join(txt)

    def __repr__(self):
        r = self.tests.iloc[0]
        return (f"GCResult({self.names[1]} -> {self.names[0]}, q={self.lags}, n={self.n}; "
                f"{self.tests.index[0]}: expLM p={r.p_expLM:.3f}, supLM p={r.p_supLM:.3f})")

    # ----------------------------------------------------------------- plots
    def _x(self):
        if self.dates is not None:
            return self.dates[:self.n]
        return np.arange(1, self.n + 1) / self.n

    def plot_cusum(self, ax=None, figsize=(7.2, 3.4)):
        """CUSUM curve int exp(LM1/2) dtau over time with break dates and critical value (MWT Fig. 2);
        Granger-causality regimes are shaded."""
        import matplotlib.pyplot as plt
        x = self._x()
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            ax.plot(x, self.cusum, color="k", lw=1.0, label="CUSUM curve")
            ax.axhline(self.cusum_cv, color="#D62728", ls="--", lw=0.9,
                       label=f"critical value (level {1 - (1 - self.alpha) ** 0.5:.3f})")
            if self.regimes:
                for b in self.regimes.get("breaks", []):
                    ax.axvline(x[min(b, self.n - 1)], color="#1F4E9E", ls=":", lw=1.0)
                for s0, s1, gc, _ in self.regimes.get("segments", []):
                    if gc:
                        ax.axvspan(x[s0], x[min(s1, self.n) - 1], color="#2CA02C", alpha=0.12, lw=0)
            ax.set_ylabel(r"$\int \exp(\frac{1}{2}LM_1(\lambda,\tau))\,d\tau$")
            ax.set_title(f"CUSUM detector: {self.names[1]} " + r"$\rightarrow$" + f" {self.names[0]} (q = {self.lags})")
            ax.legend(loc="upper left", fontsize=8)
        return fig, ax

    def plot_lm(self, ax=None, figsize=(6.8, 3.6)):
        """LM(tau) across quantiles; filled markers where the bootstrap p-value < 5%."""
        import matplotlib.pyplot as plt
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            sig = self.p_tau < 0.05
            ax.plot(self.taus, self.LM_tau, color="#1F4E9E", lw=1.2, label=r"LM($\tau$)")
            ax.scatter(self.taus[sig], self.LM_tau[sig], color="#D62728", s=14, zorder=3, label="p < 0.05")
            ax.plot(self.taus, self.LM1.max(axis=0) / np.sqrt(self.taus * (1 - self.taus)), color="#2CA02C",
                    ls="--", lw=1.0, label=r"instability part $h(\tau)\sup_\lambda LM_1$")
            ax.plot(self.taus, self.LM2 / np.sqrt(self.taus * (1 - self.taus)), color="#FF7F0E", ls="-.",
                    lw=1.0, label=r"causality part $h(\tau)LM_2$")
            ax.set_xlabel(r"Quantile $\tau$ of " + self.names[0])
            ax.set_ylabel("Statistic")
            ax.legend(fontsize=8)
            ax.set_title("Quantile-by-quantile LM test")
        return fig, ax

    def plot_cusum_map(self, cmap: str = "Parula", ax=None, figsize=(7.6, 3.6)):
        """Time x quantile map of the CUSUM detector LM1(lambda, tau) (where and when instability arises)."""
        import matplotlib.pyplot as plt
        from .._colors import get_cmap
        x = self._x()
        with journal_style({"axes.grid": False}):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            im = ax.pcolormesh(x, self.taus, self.LM1.T, cmap=get_cmap(cmap), shading="nearest", rasterized=True)
            fig.colorbar(im, ax=ax, label=r"$LM_1(\lambda,\tau)$", pad=0.02)
            if self.regimes:
                for b in self.regimes.get("breaks", []):
                    ax.axvline(x[min(b, self.n - 1)], color="w", ls="--", lw=1.0)
            ax.set_ylabel(r"Quantile $\tau$")
            ax.set_title("Time-quantile map of the instability detector")
        return fig, ax

    def plot_coefficients(self, ax=None, figsize=(6.8, 3.6)):
        """Unrestricted Granger coefficients gamma_j(tau) of the lagged cause (supWald fit)."""
        import matplotlib.pyplot as plt
        if self.gamma_hat is None:
            raise ValueError("re-run with sup_wald=True")
        from .._style import LINE_CYCLE
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            for j in range(self.gamma_hat.shape[1]):
                c, ls = LINE_CYCLE[j % len(LINE_CYCLE)]
                ax.plot(self.taus, self.gamma_hat[:, j], color=c, ls=ls, label=f"{self.names[1]} lag {j + 1}")
            ax.axhline(0, color="0.3", lw=0.8)
            ax.set_xlabel(r"Quantile $\tau$")
            ax.set_ylabel(r"$\hat\gamma(\tau)$")
            ax.legend(fontsize=8, ncol=2)
            ax.set_title("Quantile Granger coefficients")
        return fig, ax

    def plot(self, kind: str = "cusum", **kw):
        """Dispatcher: cusum, lm, cusum_map, coefficients."""
        f = {"cusum": self.plot_cusum, "lm": self.plot_lm, "cusum_map": self.plot_cusum_map,
             "coefficients": self.plot_coefficients}.get(kind)
        if f is None:
            raise ValueError(f"unknown plot kind '{kind}'")
        return f(**kw)


def qgc_map(y, x, lags: Sequence[int] = (1, 2, 3, 4, 5, 6), taus=np.round(np.arange(0.05, 0.951, 0.05), 2),
            **kwargs) -> QQGrid:
    """Lag x quantile map of LM(tau) with p-values (extension): one ``qgc_instability`` per lag order."""
    from ._core import qgc_instability
    kwargs.setdefault("regimes", False)
    kwargs.setdefault("sup_wald", False)
    V, P = [], []
    for q in lags:
        r = qgc_instability(y, x, lags=q, taus=taus, **kwargs)
        V.append(r.LM_tau)
        P.append(r.p_tau)
    nm = (getattr(y, "name", None) or "y", getattr(x, "name", None) or "x")
    return QQGrid(np.array(V), np.asarray(lags, dtype=float), taus, row_label="Lags of cause q",
                  col_label=rf"Quantile $\tau$ of {nm[0]}", value_label=r"LM($\tau$)",
                  title=f"Instability-robust quantile Granger causality: {nm[1]} -> {nm[0]}",
                  pvalues=np.array(P), row_names=[str(q) for q in lags], decimals=2,
                  note="Notes: LM(tau) of Mayer, Wied and Troster (2025, Sec. 3.1); stars from p-values.")
