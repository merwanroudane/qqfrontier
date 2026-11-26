"""Summary-statistics table in the layout of the quantile-connectedness literature.

Columns follow Gabauer and Stenfors (2024, Table 1): mean, variance, skewness,
excess kurtosis, Jarque-Bera, ERS (DF-GLS) unit-root test, and weighted
portmanteau tests Q(m) and Q2(m) (Fisher and Gallagher, 2012) on levels and
squares, each with significance stars, plus the unconditional correlation matrix.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ._style import latex_escape, stars

__all__ = ["descriptive_table", "weighted_portmanteau", "DescriptiveTable"]


def weighted_portmanteau(x, lags: int = 20, squared: bool = False):
    """Weighted Ljung-Box test of Fisher and Gallagher (2012).

    Q_W = n(n+2) sum_{k=1}^{m} (m-k+1)/m * r_k^2/(n-k), with the gamma
    approximation Gamma(shape=3m(m+1)/[4(2m^2+3m+1)], scale=2(2m^2+3m+1)/[3m(m+1)]).

    Returns (statistic, p-value).
    """
    x = np.asarray(x, dtype=float)
    if squared:
        x = x ** 2
    x = x - x.mean()
    n, m = len(x), int(lags)
    den = (x ** 2).sum()
    r = np.array([(x[k:] * x[:-k]).sum() / den for k in range(1, m + 1)])
    k = np.arange(1, m + 1)
    Q = n * (n + 2) * ((m - k + 1) / m * r ** 2 / (n - k)).sum()
    shape = 3 * m * (m + 1) / (4 * (2 * m ** 2 + 3 * m + 1))
    scale = 2 * (2 * m ** 2 + 3 * m + 1) / (3 * m * (m + 1))
    return float(Q), float(stats.gamma.sf(Q, a=shape, scale=scale))


def _ers(x):
    try:
        from arch.unitroot import DFGLS
        r = DFGLS(x, trend="c")
        return float(r.stat), float(r.pvalue)
    except Exception:  # pragma: no cover - fallback without arch
        from statsmodels.tsa.stattools import adfuller
        a = adfuller(x, regression="c", autolag="AIC")
        return float(a[0]), float(a[1])


class DescriptiveTable:
    """Container with ``.frame`` (values), ``.pvalues`` and journal renderings."""

    def __init__(self, frame: pd.DataFrame, pvalues: pd.DataFrame, corr: pd.DataFrame, lags: int,
                 unit_root: str):
        self.frame, self.pvalues, self.corr, self.lags, self.unit_root = frame, pvalues, corr, lags, unit_root

    def _cell(self, r, c):
        v = self.frame.loc[r, c]
        p = self.pvalues.loc[r, c]
        return f"{v:.3f}" + (stars(p) if np.isfinite(p) else "")

    def summary(self) -> str:
        cols = list(self.frame.columns)
        w0, w = 14, 13
        W = w0 + w * len(cols)
        out = ["=" * W, "Summary statistics".center(W), "=" * W,
               f"{'':<{w0}}" + "".join(f"{c:>{w}}" for c in cols), "-" * W]
        for r in self.frame.index:
            out.append(f"{r:<{w0}}" + "".join(f"{self._cell(r, c):>{w}}" for c in cols))
        out.append("-" * W)
        out.append("Unconditional correlations")
        for r in self.corr.index:
            out.append(f"{r:<{w0}}" + "".join(f"{self.corr.loc[r, c]:>{w}.3f}" for c in self.corr.columns))
        out += ["=" * W,
                "Notes: *** p<0.01, ** p<0.05, * p<0.10. Skewness: D'Agostino (1970) test; Ex.Kurtosis:",
                f"Anscombe-Glynn (1983) test; JB: Jarque-Bera (1980); {self.unit_root}: unit-root test with",
                f"a constant; Q({self.lags}) and Q2({self.lags}): Fisher-Gallagher (2012) weighted portmanteau",
                "tests on the series and on their squares."]
        return "\n".join(out)

    __repr__ = summary

    def to_latex(self, caption: str = "Summary statistics", label: str = "tab:descriptive") -> str:
        cols = list(self.frame.columns)
        L = [r"\begin{table}[htbp]\centering", r"\caption{" + caption + r"}\label{" + label + "}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(latex_escape(c) for c in cols) + r" \\", r"\midrule"]
        for r in self.frame.index:
            cells = []
            for c in cols:
                v, p = self.frame.loc[r, c], self.pvalues.loc[r, c]
                s = stars(p) if np.isfinite(p) else ""
                cells.append(f"{v:.3f}" + (r"\sym{" + s + "}" if s else ""))
            L.append(latex_escape(r) + " & " + " & ".join(cells) + r" \\")
        L += [r"\midrule", r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{\textit{Unconditional correlations}} \\"]
        for r in self.corr.index:
            L.append(latex_escape(r) + " & " + " & ".join(f"{self.corr.loc[r, c]:.3f}" for c in self.corr.columns)
                     + " & " * (len(cols) - len(self.corr.columns)) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\begin{minipage}{0.95\linewidth}\footnotesize\textit{Notes:} \sym{***}, \sym{**}, "
              r"\sym{*} denote significance at 1\%, 5\%, 10\%. JB: Jarque-Bera; " + self.unit_root +
              r": unit-root test with a constant; Q(" + str(self.lags) + r") and Q$^2$(" + str(self.lags) +
              r"): Fisher-Gallagher (2012) weighted portmanteau tests.\end{minipage}", r"\end{table}"]
        return "\n".join(L)

    def to_markdown(self) -> str:
        cols = list(self.frame.columns)
        L = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for r in self.frame.index:
            L.append(f"| **{r}** | " + " | ".join(self._cell(r, c) for c in cols) + " |")
        return "\n".join(L)


def descriptive_table(data, lags: int = 20, unit_root: str = "ERS",
                      columns: Optional[list] = None) -> DescriptiveTable:
    """Summary statistics with tests (Gabauer and Stenfors, 2024, Table 1 layout).

    Parameters
    ----------
    data : DataFrame or ndarray
        Series in columns (non-numeric columns are dropped).
    lags : int
        Lags of the weighted portmanteau tests (default 20).
    unit_root : {"ERS", "ADF"}
        Elliott-Rothenberg-Stock DF-GLS (needs the ``arch`` package) or ADF.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.standard_t(5, size=(300, 2)), columns=["A", "B"])
    >>> list(descriptive_table(df).frame.columns)
    ['Mean', 'Variance', 'Skewness', 'Ex.Kurtosis', 'JB', 'ERS', 'Q(20)', 'Q2(20)']
    """
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(np.asarray(data, dtype=float))
    df = df.select_dtypes(include=[np.number])
    df = df.loc[:, [c for c in df.columns if df[c].dtype != bool]]
    if columns:
        df = df[columns]
    rows, prow = {}, {}
    for c in df.columns:
        x = df[c].dropna().to_numpy(dtype=float)
        sk = stats.skew(x)
        ku = stats.kurtosis(x)
        p_sk = stats.skewtest(x).pvalue if len(x) >= 8 else np.nan
        p_ku = stats.kurtosistest(x).pvalue if len(x) >= 20 else np.nan
        jb = stats.jarque_bera(x)
        if unit_root.upper() == "ERS":
            ur, p_ur = _ers(x)
        else:
            from statsmodels.tsa.stattools import adfuller
            a = adfuller(x, regression="c", autolag="AIC")
            ur, p_ur = float(a[0]), float(a[1])
        q, pq = weighted_portmanteau(x, lags)
        q2, pq2 = weighted_portmanteau(x, lags, squared=True)
        rows[str(c)] = [x.mean(), x.var(ddof=1), sk, ku, jb.statistic, ur, q, q2]
        prow[str(c)] = [np.nan, np.nan, p_sk, p_ku, jb.pvalue, p_ur, pq, pq2]
    names = ["Mean", "Variance", "Skewness", "Ex.Kurtosis", "JB", unit_root.upper(), f"Q({lags})", f"Q2({lags})"]
    frame = pd.DataFrame(rows, index=names).T
    pv = pd.DataFrame(prow, index=names).T
    corr = df.corr()
    corr.index = corr.index.map(str)
    corr.columns = corr.columns.map(str)
    return DescriptiveTable(frame, pv, corr, lags, unit_root.upper())
