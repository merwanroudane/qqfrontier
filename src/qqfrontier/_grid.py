"""QQGrid: one quantile-on-quantile surface with every journal rendering.

Every module of qqfrontier reduces its main result to one or more surfaces
indexed by two quantile levels (a TCI surface, a cross-quantilogram at a lag,
a causality statistic, a coherency at a frequency, a QQARDL coefficient).
``QQGrid`` turns such a surface into

* an annotated heatmap (Parula by default, stars when p-values exist),
* a static 3D surface (matplotlib) and an interactive one (plotly),
* a contour map and cross-section ("slice") profiles,
* a journal text table, LaTeX (booktabs) and Markdown tables,
* a descriptive block (quadrant averages, peak, share significant) and a
  plain-language interpretation.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ._colors import bluered_colors, get_cmap, resolve_colorscale
from ._style import (LINE_CYCLE, SYM_DEF, fmt_q, journal_style, latex_escape,
                     stars)

__all__ = ["QQGrid"]


@dataclass
class QQGrid:
    """A surface ``values[i, j]`` over ``row_q[i]`` x ``col_q[j]``.

    Parameters
    ----------
    values : ndarray (m, n)
    row_q, col_q : array-like
        Quantile levels on the rows (vertical axis) and columns (horizontal axis).
    row_label, col_label : str
        Axis titles, e.g. ``"US2Y quantile"``.
    value_label : str
        Name of the plotted quantity, e.g. ``"TCI (%)"``.
    title : str
    pvalues : ndarray (m, n), optional
        P-values; adds stars to heatmaps and tables.
    se : ndarray (m, n), optional
        Standard errors shown under the values in tables.
    note : str
        Footer note (method, settings, source equation).
    center : float, optional
        Reference value for a diverging map (e.g. 0 for coefficients).
    """

    values: np.ndarray
    row_q: Sequence[float]
    col_q: Sequence[float]
    row_label: str = "tau_1"
    col_label: str = "tau_2"
    value_label: str = "value"
    title: str = ""
    pvalues: Optional[np.ndarray] = None
    se: Optional[np.ndarray] = None
    note: str = ""
    center: Optional[float] = None
    decimals: int = 2
    meta: Dict[str, object] = field(default_factory=dict)
    row_names: Optional[Sequence[str]] = None
    col_names: Optional[Sequence[str]] = None

    def _rl(self, i: int) -> str:
        return str(self.row_names[i]) if self.row_names is not None else fmt_q(self.row_q[i])

    def _cl(self, j: int) -> str:
        return str(self.col_names[j]) if self.col_names is not None else fmt_q(self.col_q[j])

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.row_q = np.asarray(self.row_q, dtype=float)
        self.col_q = np.asarray(self.col_q, dtype=float)
        if self.values.shape != (len(self.row_q), len(self.col_q)):
            raise ValueError("values must be len(row_q) x len(col_q)")
        if self.pvalues is not None:
            self.pvalues = np.asarray(self.pvalues, dtype=float)
        if self.se is not None:
            self.se = np.asarray(self.se, dtype=float)

    # ------------------------------------------------------------------ data
    def to_matrix(self) -> pd.DataFrame:
        """Wide DataFrame: rows = row quantiles (high at the top), columns = col quantiles."""
        df = pd.DataFrame(self.values, index=[self._rl(i) for i in range(len(self.row_q))],
                          columns=[self._cl(j) for j in range(len(self.col_q))])
        df.index.name, df.columns.name = self.row_label, self.col_label
        return df.iloc[::-1]

    def to_frame(self) -> pd.DataFrame:
        """Tidy long DataFrame with one row per quantile pair."""
        R, C = np.meshgrid(self.row_q, self.col_q, indexing="ij")
        out = pd.DataFrame({self.row_label: R.ravel(), self.col_label: C.ravel(),
                            self.value_label: self.values.ravel()})
        if self.se is not None:
            out["se"] = self.se.ravel()
        if self.pvalues is not None:
            out["p_value"] = self.pvalues.ravel()
            out["sig"] = [stars(p) for p in self.pvalues.ravel()]
        return out

    # ------------------------------------------------------------ describe
    def describe(self) -> pd.DataFrame:
        """Quadrant averages, extremes and share of significant cells.

        Low = quantiles <= 0.25, middle = 0.25 < q < 0.75, high = q >= 0.75.
        """
        V = self.values
        rq, cq = self.row_q, self.col_q
        bands = {"low": lambda q: q <= 0.25 + 1e-12, "mid": lambda q: (q > 0.25 + 1e-12) & (q < 0.75 - 1e-12),
                 "high": lambda q: q >= 0.75 - 1e-12}
        rows = []
        for rb, rf in bands.items():
            for cb, cf in bands.items():
                sub = V[np.ix_(rf(rq), cf(cq))]
                if sub.size:
                    rows.append((f"{self.row_label} {rb} / {self.col_label} {cb}",
                                 np.nanmean(sub)))
        i, j = np.unravel_index(np.nanargmax(V), V.shape)
        a, b = np.unravel_index(np.nanargmin(V), V.shape)
        rows += [("Grid mean", np.nanmean(V)), ("Grid std. dev.", np.nanstd(V)),
                 (f"Maximum at ({fmt_q(rq[i])}, {fmt_q(cq[j])})", V[i, j]),
                 (f"Minimum at ({fmt_q(rq[a])}, {fmt_q(cq[b])})", V[a, b])]
        if len(rq) == len(cq) and np.allclose(rq, cq):
            m = len(rq)
            rows += [("Directly related quantiles (diagonal) mean",
                      np.nanmean(np.diag(V))),
                     ("Reversely related quantiles (anti-diagonal) mean",
                      np.nanmean([V[m - 1 - l, l] for l in range(m)]))]
        if self.pvalues is not None:
            P = self.pvalues
            rows += [("Share significant at 10% (%)", 100 * np.nanmean(P < 0.10)),
                     ("Share significant at 5% (%)", 100 * np.nanmean(P < 0.05)),
                     ("Share significant at 1% (%)", 100 * np.nanmean(P < 0.01))]
        return pd.DataFrame(rows, columns=["statistic", self.value_label]).set_index("statistic")

    def interpret(self) -> str:
        """A plain-language reading of the surface for the results section."""
        d = self.describe()
        V = self.values
        rq, cq = self.row_q, self.col_q
        i, j = np.unravel_index(np.nanargmax(V), V.shape)
        a, b = np.unravel_index(np.nanargmin(V), V.shape)
        lab = self.value_label
        lines = [f"{self.title or lab}: the surface averages {np.nanmean(V):.{self.decimals}f} "
                 f"over {V.size} quantile pairs (range {np.nanmin(V):.{self.decimals}f} to "
                 f"{np.nanmax(V):.{self.decimals}f})."]
        lines.append(f"The largest value ({V[i, j]:.{self.decimals}f}) occurs when {self.row_label} is at "
                     f"its {fmt_q(rq[i])} quantile and {self.col_label} at its {fmt_q(cq[j])} quantile; "
                     f"the smallest ({V[a, b]:.{self.decimals}f}) at ({fmt_q(rq[a])}, {fmt_q(cq[b])}).")
        tails = [k for k in d.index if (" low / " in k or " high / " in k) and
                 (k.endswith(" low") or k.endswith(" high"))]
        mid = [k for k in d.index if " mid / " in k and k.endswith(" mid")]
        if tails and mid:
            tail_mean = d.loc[tails].iloc[:, 0].mean()
            mid_mean = d.loc[mid].iloc[:, 0].mean()
            word = "stronger" if tail_mean > mid_mean else "weaker"
            lines.append(f"Values in the joint tails (average {tail_mean:.{self.decimals}f}) are {word} "
                         f"than around the joint median (average {mid_mean:.{self.decimals}f}), "
                         + ("pointing to tail-dependent (asymmetric) behaviour." if word == "stronger"
                            else "so the relation is concentrated in normal market states."))
        dr = [k for k in d.index if k.startswith("Directly")]
        if dr:
            dv = d.loc[dr[0]].iloc[0]
            rv = d.loc[[k for k in d.index if k.startswith("Reversely")][0]].iloc[0]
            which = "reversely related (opposite-tail)" if rv > dv else "directly related (same-tail)"
            lines.append(f"On average, {which} quantile pairs dominate "
                         f"(reverse {rv:.{self.decimals}f} vs direct {dv:.{self.decimals}f}).")
        if self.pvalues is not None:
            s5 = 100 * np.nanmean(self.pvalues < 0.05)
            lines.append(f"{s5:.1f}% of the quantile pairs are significant at the 5% level.")
        return " ".join(lines)

    # ---------------------------------------------------------------- tables
    def _cell(self, i: int, j: int) -> str:
        v = self.values[i, j]
        if not np.isfinite(v):
            return "."
        s = f"{v:.{self.decimals}f}"
        if self.pvalues is not None:
            s += stars(self.pvalues[i, j])
        return s

    def summary(self, max_cols: int = 11) -> str:
        """Journal-style text table of the surface (rows: high quantile at the top)."""
        cols = list(range(len(self.col_q)))
        if len(cols) > max_cols:
            cols = sorted(set(np.round(np.linspace(0, len(cols) - 1, max_cols)).astype(int)))
        w0, w = 12, 10
        width = w0 + w * len(cols)
        out = ["=" * width, (self.title or self.value_label).center(width), "=" * width,
               f"Rows: {self.row_label}   Columns: {self.col_label}   Cells: {self.value_label}",
               "-" * width,
               f"{'':>{w0}}" + "".join(f"{self._cl(j):>{w}}" for j in cols),
               "-" * width]
        for i in range(len(self.row_q) - 1, -1, -1):
            out.append(f"{self._rl(i):>{w0}}" + "".join(f"{self._cell(i, j):>{w}}" for j in cols))
            if self.se is not None:
                out.append(f"{'':>{w0}}" + "".join(
                    f"{'(' + format(self.se[i, j], f'.{self.decimals}f') + ')':>{w}}" for j in cols))
        out.append("=" * width)
        if self.pvalues is not None:
            out.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")
        if self.note:
            out.append(self.note)
        return "\n".join(out)

    def to_markdown(self) -> str:
        """GitHub/Pages Markdown table of the surface."""
        head = f"| {self.row_label} \\ {self.col_label} | " + " | ".join(fmt_q(q) for q in self.col_q) + " |"
        sep = "|---|" + "---|" * len(self.col_q)
        body = [f"| **{self._rl(i)}** | " + " | ".join(self._cell(i, j) for j in range(len(self.col_q))) + " |"
                for i in range(len(self.row_q) - 1, -1, -1)]
        return "\n".join([head, sep] + body)

    def to_latex(self, caption: Optional[str] = None, label: str = "tab:qq",
                 notes: bool = True, small: bool = True) -> str:
        """booktabs LaTeX table (stars via ``\\sym``; definition printed in a comment)."""
        n = len(self.col_q)
        cap = latex_escape(caption or self.title or self.value_label)
        lines = [SYM_DEF, r"\begin{table}[htbp]\centering", r"\caption{" + cap + "}" + r"\label{" + label + "}"]
        if small:
            lines.append(r"\small")
        lines += [r"\begin{tabular}{l" + "c" * n + "}", r"\toprule",
                  latex_escape(self.row_label) + r" & \multicolumn{" + str(n) + "}{c}{" + latex_escape(self.col_label) + r"} \\",
                  r"\cmidrule(lr){2-" + str(n + 1) + "}",
                  " & " + " & ".join(self._cl(j) for j in range(n)) + r" \\", r"\midrule"]
        for i in range(len(self.row_q) - 1, -1, -1):
            cells = []
            for j in range(n):
                v = self.values[i, j]
                s = "." if not np.isfinite(v) else f"{v:.{self.decimals}f}"
                if self.pvalues is not None and np.isfinite(v):
                    st = stars(self.pvalues[i, j])
                    if st:
                        s += r"\sym{" + st + "}"
                cells.append(s)
            lines.append(self._rl(i) + " & " + " & ".join(cells) + r" \\")
            if self.se is not None:
                lines.append(" & " + " & ".join(f"({self.se[i, j]:.{self.decimals}f})" for j in range(n)) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if notes:
            txt = latex_escape(self.note) if self.note else ""
            if self.pvalues is not None:
                txt += r" \sym{***} $p<0.01$, \sym{**} $p<0.05$, \sym{*} $p<0.10$."
            if txt:
                lines.append(r"\par\smallskip\begin{minipage}{0.95\linewidth}\footnotesize\textit{Notes:} " + txt + r"\end{minipage}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ plots
    def _cmap_norm(self, cmap):
        import matplotlib.colors as mcolors
        V = self.values
        if self.center is not None:
            m = np.nanmax(np.abs(V - self.center)) or 1.0
            cm = get_cmap(cmap) if cmap not in (None, "auto") else mcolors.LinearSegmentedColormap.from_list(
                "bluered", bluered_colors(256))
            return cm, mcolors.TwoSlopeNorm(vcenter=self.center, vmin=self.center - m, vmax=self.center + m)
        return get_cmap(cmap if cmap not in (None, "auto") else "Parula"), None

    def heatmap(self, annotate: bool = True, show_stars: bool = True, cmap: str = "auto",
                ax=None, fontsize: Optional[float] = None, figsize=(6.4, 5.2),
                colorbar: bool = True):
        """Annotated heatmap, high quantiles at the top (Gabauer and Stenfors, 2024, Fig. 2 layout).

        Returns ``(fig, ax)``.
        """
        import matplotlib.pyplot as plt
        with journal_style({"axes.grid": False}):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            cm, norm = self._cmap_norm(cmap)
            im = ax.imshow(self.values, origin="lower", cmap=cm, norm=norm, aspect="auto",
                           interpolation="nearest")
            ax.set_xticks(range(len(self.col_q)))
            ax.set_xticklabels([self._cl(j) for j in range(len(self.col_q))],
                               rotation=45 if len(self.col_q) > 10 else 0)
            ax.set_yticks(range(len(self.row_q)))
            ax.set_yticklabels([self._rl(i) for i in range(len(self.row_q))])
            ax.set_xlabel(self.col_label)
            ax.set_ylabel(self.row_label)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xticks(np.arange(-.5, len(self.col_q), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(self.row_q), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", length=0)
            if annotate:
                n = max(self.values.shape)
                fs = fontsize or (8 if n <= 10 else 6 if n <= 19 else 4.5)
                vals = self.values[np.isfinite(self.values)]
                lo, hi = (vals.min(), vals.max()) if vals.size else (0, 1)
                for i in range(len(self.row_q)):
                    for j in range(len(self.col_q)):
                        v = self.values[i, j]
                        if not np.isfinite(v):
                            continue
                        txt = f"{v:.{max(self.decimals - 1, 0) if n > 10 else self.decimals}f}"
                        if show_stars and self.pvalues is not None:
                            txt += stars(self.pvalues[i, j])
                        if norm is not None:
                            dark = abs(norm(v) - 0.5) > 0.3
                        else:
                            dark = (v - lo) / (hi - lo + 1e-300) < 0.45
                        ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                                color="white" if dark else "black")
            if colorbar:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cb.set_label(self.value_label)
            if self.title:
                ax.set_title(self.title)
        return fig, ax

    def _smooth(self, resolution: int = 60):
        """Cubic (or linear) interpolation of the surface on a fine grid."""
        from scipy.interpolate import RegularGridInterpolator
        V = self.values
        if np.isnan(V).any() or min(V.shape) < 2:
            C, R = np.meshgrid(self.col_q, self.row_q)
            return C, R, V
        method = "cubic" if min(V.shape) >= 4 else "linear"
        f = RegularGridInterpolator((self.row_q, self.col_q), V, method=method)
        rr = np.linspace(self.row_q[0], self.row_q[-1], resolution)
        cc = np.linspace(self.col_q[0], self.col_q[-1], resolution)
        R, C = np.meshgrid(rr, cc, indexing="ij")
        Z = f(np.column_stack([R.ravel(), C.ravel()])).reshape(R.shape)
        return C, R, Z

    def surface3d(self, cmap: str = "auto", elev: float = 28, azim: float = -135,
                  mark_significant: bool = True, ax=None, figsize=(7.2, 5.6),
                  smooth: bool = True, resolution: int = 60, show_nodes: bool = True):
        """Static 3D surface (matplotlib), Parula by default. Returns ``(fig, ax)``.

        ``smooth=True`` draws a cubic interpolation through the estimated nodes
        (shown as small points); significant nodes (p < 0.05) are drawn in black.
        """
        import matplotlib.pyplot as plt
        with journal_style({"axes.grid": True}):
            if ax is None:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig = ax.figure
            C, R = np.meshgrid(self.col_q, self.row_q)
            cm, norm = self._cmap_norm(cmap)
            if smooth:
                Cs, Rs, Zs = self._smooth(resolution)
            else:
                Cs, Rs, Zs = C, R, self.values
            Zs = np.ma.masked_invalid(Zs)
            surf = ax.plot_surface(Cs, Rs, Zs, cmap=cm, norm=norm, linewidth=0 if smooth else 0.25,
                                   edgecolor="none" if smooth else "0.25", antialiased=True,
                                   alpha=0.97, rstride=1, cstride=1)
            if smooth and show_nodes and not (mark_significant and self.pvalues is not None):
                ax.scatter(C, R, self.values, color="0.15", s=4, depthshade=False)
            ax.set_box_aspect(None, zoom=0.88)
            if mark_significant and self.pvalues is not None:
                sig = self.pvalues < 0.05
                ax.scatter(C[sig], R[sig], self.values[sig], color="k", s=8, depthshade=False,
                           label="p < 0.05")
                ax.legend(loc="upper left")
            ax.set_xlabel(self.col_label, labelpad=8)
            ax.set_ylabel(self.row_label, labelpad=8)
            ax.set_zlabel(self.value_label, labelpad=8)
            ax.view_init(elev=elev, azim=azim)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_facecolor((1, 1, 1, 0))
            fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=self.value_label)
            if self.title:
                ax.set_title(self.title)
        return fig, ax

    def contour(self, levels: int = 14, cmap: str = "auto", ax=None, figsize=(6.4, 5.0)):
        """Filled contour map with labelled iso-lines. Returns ``(fig, ax)``."""
        import matplotlib.pyplot as plt
        with journal_style({"axes.grid": False}):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            cm, norm = self._cmap_norm(cmap)
            C, R = np.meshgrid(self.col_q, self.row_q)
            Cs, Rs, Zs = self._smooth(80)
            cf = ax.contourf(Cs, Rs, Zs, levels=levels, cmap=cm, norm=norm)
            cs = ax.contour(Cs, Rs, Zs, levels=cf.levels[::2], colors="k", linewidths=0.4, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=7, fmt=f"%.{max(self.decimals - 1, 0)}f")
            if self.pvalues is not None:
                sig = self.pvalues < 0.05
                ax.scatter(C[sig], R[sig], marker=".", color="k", s=10, label="p < 0.05")
                ax.legend(loc="upper right", frameon=True)
            ax.set_xlabel(self.col_label)
            ax.set_ylabel(self.row_label)
            fig.colorbar(cf, ax=ax, label=self.value_label)
            if self.title:
                ax.set_title(self.title)
        return fig, ax

    def slices(self, at: Optional[Sequence[float]] = None, along: str = "col", ax=None,
               figsize=(6.5, 4.0)):
        """Cross-section profiles of the surface.

        ``along="col"`` draws the value against ``col_q`` for selected row
        quantiles ``at`` (default: lowest, median, highest). Returns ``(fig, ax)``.
        """
        import matplotlib.pyplot as plt
        if along not in ("col", "row"):
            raise ValueError("along must be 'col' or 'row'")
        fixed = self.row_q if along == "col" else self.col_q
        xs = self.col_q if along == "col" else self.row_q
        if at is None:
            at = [fixed[0], fixed[len(fixed) // 2], fixed[-1]]
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            for k, q in enumerate(at):
                idx = int(np.argmin(np.abs(fixed - q)))
                y = self.values[idx] if along == "col" else self.values[:, idx]
                col, ls = LINE_CYCLE[k % len(LINE_CYCLE)]
                fixed_lab = self.row_label if along == "col" else self.col_label
                ax.plot(xs, y, color=col, ls=ls, marker="o", ms=3,
                        label=f"{fixed_lab} = {fmt_q(fixed[idx])}")
                if self.pvalues is not None:
                    p = self.pvalues[idx] if along == "col" else self.pvalues[:, idx]
                    sig = p < 0.05
                    ax.scatter(xs[sig], y[sig], s=40, facecolors="none", edgecolors=col)
            if self.center is not None:
                ax.axhline(self.center, lw=0.8, color="0.3")
            ax.set_xlabel(self.col_label if along == "col" else self.row_label)
            ax.set_ylabel(self.value_label)
            ax.legend()
            if self.title:
                ax.set_title(self.title)
        return fig, ax

    def plot_surface(self, colorscale: str = "Parula", show_significance: bool = True,
                     height: int = 620):
        """Interactive 3D surface (plotly). Returns a ``plotly.graph_objects.Figure``.

        Use ``fig.write_html("surface.html")`` to embed it in a web page.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as e:  # pragma: no cover
            raise ImportError("plot_surface requires plotly: pip install qqfrontier[plotly]") from e
        cs = resolve_colorscale(colorscale if self.center is None else
                                ("BlueRed" if colorscale == "Parula" else colorscale))
        hover = None
        if self.pvalues is not None:
            hover = np.array([[f"p = {p:.3f}{stars(p)}" for p in row] for row in self.pvalues])
        surf = go.Surface(z=self.values, x=self.col_q, y=self.row_q, colorscale=cs,
                          colorbar=dict(title=self.value_label), text=hover,
                          hovertemplate=(f"{self.col_label}: %{{x:.2f}}<br>{self.row_label}: %{{y:.2f}}"
                                         f"<br>{self.value_label}: %{{z:.3f}}<br>%{{text}}<extra></extra>"
                                         if hover is not None else None),
                          contours=dict(z=dict(show=True, usecolormap=True, project_z=True)))
        data = [surf]
        if show_significance and self.pvalues is not None:
            C, R = np.meshgrid(self.col_q, self.row_q)
            sig = self.pvalues < 0.05
            data.append(go.Scatter3d(x=C[sig], y=R[sig], z=self.values[sig], mode="markers",
                                     marker=dict(size=3, color="black"), name="p < 0.05"))
        fig = go.Figure(data=data)
        fig.update_layout(template="plotly_white", height=height, title=self.title,
                          font=dict(family="Times New Roman, serif", size=13),
                          scene=dict(xaxis_title=self.col_label, yaxis_title=self.row_label,
                                     zaxis_title=self.value_label,
                                     camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.0))))
        return fig

    def panel(self, figsize=(16.5, 5.0)):
        """Three-panel figure: annotated heatmap | 3D surface | contour. Returns ``(fig, axes)``."""
        import matplotlib.pyplot as plt
        with journal_style({"axes.grid": False}):
            fig = plt.figure(figsize=figsize)
            gsp = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.32)
            a1 = fig.add_subplot(gsp[0])
            a2 = fig.add_subplot(gsp[1], projection="3d")
            a3 = fig.add_subplot(gsp[2])
        self.heatmap(ax=a1, fontsize=5.5)
        self.surface3d(ax=a2)
        self.contour(ax=a3)
        for a, t in zip((a1, a2, a3), ("(a) Heatmap", "(b) 3D surface", "(c) Contour")):
            a.set_title(t)
        if self.title:
            fig.suptitle(self.title, y=1.02)
        return fig, (a1, a2, a3)

    def __repr__(self) -> str:
        return self.summary()
