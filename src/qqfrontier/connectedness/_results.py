"""Results object for quantile-on-quantile connectedness.

Reproduces the outputs of the source papers and adds interpretation aids:

* averaged TCI heatmap (Gabauer and Stenfors, 2024, Fig. 2; Bonaccolto et al., 2026, Fig. 1),
* directly vs reversely related TCI and their difference (GS 2024, Fig. 3; BCG 2026, Fig. 2),
* averaged NET heatmaps per series (Gabauer and Stenfors, 2024b, NAJEF, Fig. 6),
* additions: 3D surfaces (static and interactive), contour maps, quantile cross-sections,
  a time x quantile 3D surface of the diagonal / anti-diagonal TCI, TCI paths at selected
  configurations, Diebold-Yilmaz connectedness tables at any configuration, a journal
  summary table and a plain-language interpretation.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._grid import QQGrid
from .._style import LINE_CYCLE, fmt_q, journal_style, latex_escape

__all__ = ["QQConnResult", "decompose_tci", "compare_heatmaps"]


def _parse_dates(d: Sequence[str]):
    try:
        return pd.to_datetime(pd.Series(d)).values
    except Exception:
        try:
            return np.asarray(d, dtype=float)
        except Exception:
            return np.asarray(d)


@dataclass
class QQConnResult:
    """Output of :func:`qqc_gs`, :func:`qqc_kernel` and :func:`qqc_tvp`.

    Attributes
    ----------
    TCI : ndarray (D, C)
        Total connectedness index (%) per date and quantile configuration.
    TO, FROM, NET : ndarray (D, k, C)
        Directional measures (%) per date, series and configuration.
    taus : ndarray (C, k)
        Quantile configuration of every column (series order = ``names``).
    dates : list of str
    names : list of str
    method : str
    config : dict
        Settings actually used.
    fevd : ndarray (k, k, C, D) or None
        Stored variance decompositions (``keep_fevd=True``).
    """

    TCI: np.ndarray
    TO: np.ndarray
    FROM: np.ndarray
    NET: np.ndarray
    taus: np.ndarray
    dates: List[str]
    names: List[str]
    method: str
    config: Dict[str, object]
    fevd: Optional[np.ndarray] = None
    coef: Optional[Dict[str, np.ndarray]] = None
    coef_taus: Optional[list] = None
    pit: Optional[pd.DataFrame] = None
    extra: Dict[str, object] = field(default_factory=dict)

    # ---------------------------------------------------------------- basics
    @property
    def k(self) -> int:
        return len(self.names)

    def _is_full_grid(self) -> bool:
        if self.taus.shape[1] != 2:
            return False
        u1, u2 = np.unique(self.taus[:, 0]), np.unique(self.taus[:, 1])
        return len(self.taus) == len(u1) * len(u2)

    def _note(self) -> str:
        c = self.config
        bits = [self.method]
        if "window" in c:
            bits.append(f"{c['window']}-obs rolling window")
        bits.append(f"QVAR({c.get('nlag', 1)})")
        bits.append(f"{c.get('horizon')}-step-ahead GFEVD")
        if "bandwidth" in c:
            bits.append(f"{c['kernel']} kernel, h = {c['bandwidth']}")
        if "delta" in c:
            bits.append(f"delta = {c['delta']}, kappa = {c['kappa']}")
        if "cov_type" in c:
            bits.append(f"{c['cov_type']} residual matrix")
        bits.append(f"{len(self.dates)} dates ({self.dates[0]} to {self.dates[-1]})")
        return "Notes: " + "; ".join(bits) + "."

    def _measure(self, measure: str, series=0) -> np.ndarray:
        measure = measure.upper()
        if measure == "TCI":
            return self.TCI
        if measure not in ("TO", "FROM", "NET"):
            raise ValueError("measure must be 'TCI', 'TO', 'FROM' or 'NET'")
        s = self.names.index(series) if isinstance(series, str) else int(series)
        return getattr(self, measure)[:, s, :]

    # -------------------------------------------------------------- averages
    def grid(self, measure: str = "TCI", series=0, stat: str = "mean",
             dates: Optional[slice] = None) -> QQGrid:
        """Time-averaged measure on the (tau_1, tau_2) grid as a :class:`QQGrid`.

        Parameters
        ----------
        measure : {"TCI", "NET", "TO", "FROM"}
        series : int or str
            Series for directional measures.
        stat : {"mean", "median", "std"}
            Statistic over dates.
        dates : slice, optional
            Sub-period (positional slice of the dates), e.g. a crisis window.
        """
        if not self._is_full_grid():
            raise ValueError("grid() needs a bivariate model on a full quantile grid; "
                             "use average() for other designs")
        M = self._measure(measure, series)
        if dates is not None:
            M = M[dates]
        f = {"mean": np.nanmean, "median": np.nanmedian, "std": np.nanstd}[stat]
        v = f(M, axis=0)
        u1, u2 = np.unique(self.taus[:, 0]), np.unique(self.taus[:, 1])
        G = np.full((len(u1), len(u2)), np.nan)
        G[np.searchsorted(u1, self.taus[:, 0]), np.searchsorted(u2, self.taus[:, 1])] = v
        lab = f"{measure.upper()} (%)" if measure.upper() == "TCI" else \
            f"{measure.upper()} of {self.names[series] if not isinstance(series, str) else series} (%)"
        title = {"TCI": "Averaged dynamic quantile-on-quantile total connectedness",
                 "NET": "Averaged net quantile-on-quantile connectedness",
                 "TO": "Averaged TO connectedness", "FROM": "Averaged FROM connectedness"}[measure.upper()]
        return QQGrid(G, u1, u2, row_label=f"{self.names[0]} quantile",
                      col_label=f"{self.names[1]} quantile", value_label=lab,
                      title=title, note=self._note(), decimals=1,
                      center=0.0 if measure.upper() == "NET" else None)

    def average(self, measure: str = "TCI", series=0, digits: Optional[int] = 1) -> pd.DataFrame:
        """Averaged measure: wide matrix (high quantile of series 1 on top) or long table."""
        if self._is_full_grid():
            out = self.grid(measure, series).to_matrix()
        else:
            M = self._measure(measure, series)
            out = pd.DataFrame(self.taus, columns=self.names)
            out["value"] = np.nanmean(M, axis=0)
        return out.round(digits) if digits is not None else out

    average_tci = average

    def direct_reverse(self) -> pd.DataFrame:
        """Directly (tau_1 = tau_2) vs reversely (tau_1 = 1 - tau_2) related quantiles TCI.

        Returns a DataFrame with ``date``, ``reverse``, ``direct`` and ``delta`` =
        reverse - direct (Gabauer and Stenfors, 2024, Fig. 3).
        """
        if not self._is_full_grid():
            raise ValueError("A bivariate model on a full quantile grid is required")
        u1, u2 = np.unique(self.taus[:, 0]), np.unique(self.taus[:, 1])
        if len(u1) != len(u2) or np.abs(u1 - u2).max() > 1e-10:
            raise ValueError("Both series must use the same quantile levels")
        m = len(u1)

        def find(a, b):
            return int(np.where((np.abs(self.taus[:, 0] - a) < 1e-10) &
                                (np.abs(self.taus[:, 1] - b) < 1e-10))[0][0])
        d_id = [find(u1[l], u2[l]) for l in range(m)]
        r_id = [find(u1[m - 1 - l], u2[l]) for l in range(m)]
        direct = np.nanmean(self.TCI[:, d_id], axis=1)
        reverse = np.nanmean(self.TCI[:, r_id], axis=1)
        return pd.DataFrame({"date": _parse_dates(self.dates), "reverse": reverse,
                             "direct": direct, "delta": reverse - direct})

    def tci_path(self, taus: Optional[Sequence[Sequence[float]]] = None) -> pd.DataFrame:
        """TCI over time at selected configurations, e.g. ``[(0.05, 0.95), (0.5, 0.5)]``."""
        ids = range(len(self.taus)) if taus is None else []
        if taus is not None:
            for t in taus:
                w = np.where(np.abs(self.taus - np.asarray(t)).max(axis=1) < 1e-10)[0]
                if not len(w):
                    raise ValueError(f"Configuration {tuple(t)} not estimated")
                ids.append(int(w[0]))
        out = pd.DataFrame({"date": _parse_dates(self.dates)})
        for i in ids:
            out["(" + ",".join(fmt_q(q) for q in self.taus[i]) + ")"] = self.TCI[:, i]
        return out

    def to_frame(self) -> pd.DataFrame:
        """Long DataFrame: one row per date x configuration with TCI, NET, TO, FROM."""
        D, C = self.TCI.shape
        out = pd.DataFrame({"date": np.tile(_parse_dates(self.dates), C)})
        for j, nm in enumerate(self.names):
            out[f"tau_{nm}"] = np.repeat(self.taus[:, j], D)
        out["TCI"] = self.TCI.T.ravel()
        for j, nm in enumerate(self.names):
            out[f"NET_{nm}"] = self.NET[:, j, :].T.ravel()
            out[f"TO_{nm}"] = self.TO[:, j, :].T.ravel()
            out[f"FROM_{nm}"] = self.FROM[:, j, :].T.ravel()
        return out

    def connectedness_table(self, at: Sequence[float], dates: Optional[slice] = None) -> pd.DataFrame:
        """Time-averaged Diebold-Yilmaz table at one configuration (needs ``keep_fevd=True``)."""
        if self.fevd is None:
            raise ValueError("Re-estimate with keep_fevd=True to obtain connectedness tables")
        from ._fevd import connectedness_table as _ct
        w = np.where(np.abs(self.taus - np.asarray(at)).max(axis=1) < 1e-10)[0]
        if not len(w):
            raise ValueError(f"Configuration {tuple(at)} not estimated")
        F = self.fevd[:, :, w[0], :]
        if dates is not None:
            F = F[:, :, dates]
        return _ct(np.nanmean(F, axis=2), names=self.names)["table"].round(2)

    # ---------------------------------------------------------------- tables
    def dynamic_stats(self) -> pd.DataFrame:
        """Mean, MA, min, P5, P95, max of reverse / direct / delta TCI (BCG 2026, Table A.2 layout)."""
        dr = self.direct_reverse()
        rows = {}
        for c in ("reverse", "direct", "delta"):
            v = dr[c].to_numpy()
            rows[c] = [np.nanmean(v), np.nanmean(np.abs(v)), np.nanmin(v),
                       np.nanpercentile(v, 5), np.nanpercentile(v, 95), np.nanmax(v)]
        return pd.DataFrame(rows, index=["Mean", "MA", "Min", "P5", "P95", "Max"]).T.rename(
            index={"reverse": "Reverse TCI", "direct": "Direct TCI", "delta": "Delta TCI (reverse - direct)"})

    def net_transmitter_table(self) -> pd.DataFrame:
        """For each configuration: averaged TCI, NET of each series and the net transmitter."""
        out = pd.DataFrame(self.taus, columns=[f"tau_{n}" for n in self.names])
        out["TCI"] = np.nanmean(self.TCI, axis=0)
        for j, nm in enumerate(self.names):
            out[f"NET_{nm}"] = np.nanmean(self.NET[:, j, :], axis=0)
        nets = out[[f"NET_{n}" for n in self.names]].to_numpy()
        out["net_transmitter"] = [self.names[i] for i in np.nanargmax(nets, axis=1)]
        out["share_dates_transmitter_%"] = [100 * np.nanmean(self.NET[:, i, c] > 0)
                                            for c, i in enumerate(np.nanargmax(nets, axis=1))]
        return out.round(2)

    def summary(self) -> str:
        """Journal summary: settings, averaged TCI grid, NET grids, direct/reverse statistics."""
        W = 78
        c = self.config
        lines = ["=" * W, "Quantile-on-Quantile Connectedness".center(W), self.method.center(W), "=" * W,
                 f"{'Series':<22}: {', '.join(self.names)}",
                 f"{'Dates':<22}: {len(self.dates)}  [{self.dates[0]} ... {self.dates[-1]}]",
                 f"{'Configurations':<22}: {len(self.taus)}",
                 f"{'Quantile levels':<22}: " + " ".join(fmt_q(q) for q in np.unique(self.taus))]
        for key, val in c.items():
            lines.append(f"{key:<22}: {val}")
        avg = np.nanmean(self.TCI, axis=0)
        lines += ["-" * W, f"Average TCI across configurations: min {np.nanmin(avg):.2f} | "
                           f"mean {np.nanmean(avg):.2f} | max {np.nanmax(avg):.2f}", "=" * W]
        txt = "\n".join(lines)
        if self._is_full_grid():
            txt += "\n\n" + self.grid("TCI").summary()
            for j, nm in enumerate(self.names):
                txt += "\n\n" + self.grid("NET", j).summary()
            try:
                ds = self.dynamic_stats()
                tab = ["", "=" * W, "Directly vs reversely related quantiles TCI over time".center(W),
                       "-" * W, f"{'':<30}" + "".join(f"{h:>8}" for h in ds.columns), "-" * W]
                for idx, row in ds.iterrows():
                    tab.append(f"{idx:<30}" + "".join(f"{v:8.2f}" for v in row.values))
                tab += ["=" * W, "MA: mean absolute value; P5 / P95: 5th / 95th percentiles."]
                txt += "\n" + "\n".join(tab)
            except ValueError:
                pass
        return txt

    def to_latex(self, measure: str = "TCI", series=0, caption: Optional[str] = None,
                 label: str = "tab:qqconn") -> str:
        """booktabs LaTeX table of the averaged measure grid."""
        return self.grid(measure, series).to_latex(caption=caption, label=label)

    def to_markdown(self, measure: str = "TCI", series=0) -> str:
        return self.grid(measure, series).to_markdown()

    def interpret(self) -> str:
        """Plain-language reading of the results for the results section of a paper."""
        parts = []
        if self._is_full_grid():
            parts.append(self.grid("TCI").interpret())
            try:
                ds = self.dynamic_stats()
                share = 100 * np.nanmean(self.direct_reverse()["delta"] > 0)
                parts.append(f"Over time, reversely related quantiles exceed directly related ones on "
                             f"{share:.1f}% of the dates (mean difference {ds.loc[ds.index[2], 'Mean']:.2f} "
                             f"points, 5th-95th percentile range {ds.loc[ds.index[2], 'P5']:.2f} to "
                             f"{ds.loc[ds.index[2], 'P95']:.2f}).")
            except ValueError:
                pass
            nt = self.net_transmitter_table()
            for j, nm in enumerate(self.names):
                sh = 100 * np.mean(nt["net_transmitter"] == nm)
                parts.append(f"{nm} is the net transmitter of shocks in {sh:.0f}% of the quantile "
                             f"configurations.")
            g = self.grid("NET", 0)
            i, jj = np.unravel_index(np.nanargmax(np.abs(g.values)), g.values.shape)
            v = g.values[i, jj]
            parts.append(f"The strongest net position of {self.names[0]} ({v:+.1f}%) arises at "
                         f"({fmt_q(g.row_q[i])}, {fmt_q(g.col_q[jj])}), where it is a net "
                         f"{'transmitter' if v > 0 else 'receiver'}.")
        else:
            avg = np.nanmean(self.TCI, axis=0)
            i = int(np.nanargmax(avg))
            parts.append(f"Average TCI ranges from {np.nanmin(avg):.1f}% to {np.nanmax(avg):.1f}%, "
                         f"peaking at configuration {tuple(np.round(self.taus[i], 3))}.")
        return " ".join(parts)

    def __repr__(self) -> str:
        avg = np.nanmean(self.TCI, axis=0)
        return (f"QQConnResult({self.method}; series={self.names}; dates={len(self.dates)}; "
                f"configurations={len(self.taus)}; average TCI {np.nanmin(avg):.2f}-{np.nanmax(avg):.2f})")

    # ----------------------------------------------------------------- plots
    def plot(self, kind: str = "heatmap", **kw):
        """Dispatcher: heatmap, net, dynamic, surface, surface_plotly, contour, slices,
        panel, paths, time_surface, time_heatmap."""
        f = {"heatmap": lambda: self.grid("TCI").heatmap(**kw),
             "net": lambda: self.plot_net(**kw),
             "dynamic": lambda: self.plot_dynamic(**kw),
             "surface": lambda: self.grid("TCI").surface3d(**kw),
             "surface_plotly": lambda: self.grid("TCI").plot_surface(**kw),
             "contour": lambda: self.grid("TCI").contour(**kw),
             "slices": lambda: self.grid("TCI").slices(**kw),
             "panel": lambda: self.grid("TCI").panel(**kw),
             "paths": lambda: self.plot_paths(**kw),
             "time_surface": lambda: self.plot_time_surface(**kw),
             "time_heatmap": lambda: self.plot_time_heatmap(**kw)}.get(kind)
        if f is None:
            raise ValueError(f"unknown plot kind '{kind}'")
        return f()

    def plot_dynamic(self, ax=None, figsize=(7.0, 3.6), smooth: Optional[int] = None):
        """Reverse (red), direct (green) and delta = reverse - direct (blue) TCI over time.

        Layout of Gabauer and Stenfors (2024, Fig. 3) and Bonaccolto et al. (2026, Fig. 2).
        ``smooth`` applies a centered moving average for readability (off by default).
        """
        import matplotlib.pyplot as plt
        dr = self.direct_reverse()
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            for col, (c, ls), lab in zip(("reverse", "direct", "delta"),
                                         [("#D62728", "-"), ("#2CA02C", "-"), ("#1F4E9E", "-")],
                                         ("Reverse TCI", "Direct TCI", r"$\Delta$TCI")):
                v = dr[col]
                if smooth:
                    v = v.rolling(smooth, center=True, min_periods=1).mean()
                ax.plot(dr["date"], v, color=c, ls=ls, lw=0.9, label=lab)
            ax.axhline(0, color="0.3", lw=0.8, ls=":")
            ax.set_ylabel("Connectedness (%)")
            ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        return fig, ax

    def plot_net(self, figsize=None, annotate: bool = True):
        """Averaged NET heatmaps, one panel per series (diverging map centred at zero)."""
        import matplotlib.pyplot as plt
        with journal_style({"axes.grid": False}):
            fig, axes = plt.subplots(1, self.k, figsize=figsize or (6.2 * self.k, 5.0))
            for j, ax in enumerate(np.atleast_1d(axes)):
                g = self.grid("NET", j)
                g.heatmap(ax=ax, annotate=annotate, fontsize=5.5)
                ax.set_title(f"({chr(97 + j)}) NET connectedness of {self.names[j]}")
            fig.tight_layout()
        return fig, axes

    def plot_paths(self, taus: Optional[Sequence[Sequence[float]]] = None, ax=None,
                   figsize=(7.0, 3.6)):
        """TCI paths at selected configurations (default: both opposite tails and the median)."""
        import matplotlib.pyplot as plt
        if taus is None:
            u = np.unique(self.taus)
            taus = [(u[0], u[-1]), (u[-1], u[0]), (u[len(u) // 2], u[len(u) // 2]), (u[0], u[0]), (u[-1], u[-1])]
            taus = [t for t in taus if np.any(np.abs(self.taus - np.asarray(t)).max(axis=1) < 1e-10)]
        df = self.tci_path(taus)
        with journal_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            for i, col in enumerate(df.columns[1:]):
                c, ls = LINE_CYCLE[i % len(LINE_CYCLE)]
                ax.plot(df["date"], df[col], color=c, ls=ls, lw=0.9, label=f"TCI {col}")
            ax.set_ylabel("Connectedness (%)")
            ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        return fig, ax

    def plot_time_surface(self, which: str = "diagonal", blocks: int = 120,
                          cmap: str = "Parula", elev: float = 32, azim: float = -58,
                          figsize=(8.0, 5.6)):
        """3D surface of TCI over time x quantile along the diagonal (directly related,
        tau_1 = tau_2) or the anti-diagonal (reversely related, tau_1 = 1 - tau_2).

        Dates are averaged within ``blocks`` consecutive blocks so the surface stays
        readable with daily data."""
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from .._colors import get_cmap
        u = np.unique(self.taus[:, 0])
        cols = []
        for l, q in enumerate(u):
            b = q if which == "diagonal" else u[len(u) - 1 - l]
            cols.append(int(np.where((np.abs(self.taus[:, 0] - q) < 1e-10) &
                                     (np.abs(self.taus[:, 1] - b) < 1e-10))[0][0]))
        Z = self.TCI[:, cols]
        D = Z.shape[0]
        nb = int(min(blocks, D))
        edges = np.linspace(0, D, nb + 1).astype(int)
        Z = np.vstack([np.nanmean(Z[a:b], axis=0) for a, b in zip(edges[:-1], edges[1:]) if b > a])
        t_all = _parse_dates(self.dates)
        is_date = np.issubdtype(np.asarray(t_all).dtype, np.datetime64)
        mid = [(a + b - 1) // 2 for a, b in zip(edges[:-1], edges[1:]) if b > a]
        tnum = mdates.date2num(pd.to_datetime(np.asarray(t_all)[mid])) if is_date \
            else np.asarray(mid, dtype=float)
        Tm, Qm = np.meshgrid(tnum, u, indexing="ij")
        with journal_style():
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(Tm, Qm, Z, cmap=get_cmap(cmap), linewidth=0, antialiased=True,
                                   rstride=1, cstride=1)
            if is_date:
                ax.xaxis.set_major_locator(mdates.YearLocator(base=max(1, int(np.ptp(tnum) / 365 / 6))))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xlabel("Date", labelpad=8)
            ax.set_ylabel(r"Quantile $\tau_1 = \tau_2$" if which == "diagonal" else
                          r"Quantile $\tau_1 = 1 - \tau_2$", labelpad=8)
            ax.set_zlabel("TCI (%)", labelpad=6)
            ax.set_box_aspect(None, zoom=0.88)
            ax.view_init(elev=elev, azim=azim)
            fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="TCI (%)")
            ax.set_title(("Directly" if which == "diagonal" else "Reversely") +
                         " related quantiles: TCI over time")
        return fig, ax

    def plot_time_heatmap(self, which: str = "diagonal", cmap: str = "Parula", ax=None,
                          figsize=(8.0, 3.6)):
        """2D time x quantile map of TCI along the diagonal or the anti-diagonal."""
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from .._colors import get_cmap
        u = np.unique(self.taus[:, 0])
        cols = []
        for l, q in enumerate(u):
            b = q if which == "diagonal" else u[len(u) - 1 - l]
            cols.append(int(np.where((np.abs(self.taus[:, 0] - q) < 1e-10) &
                                     (np.abs(self.taus[:, 1] - b) < 1e-10))[0][0]))
        Z = self.TCI[:, cols].T
        t = _parse_dates(self.dates)
        is_date = np.issubdtype(np.asarray(t).dtype, np.datetime64)
        x = mdates.date2num(pd.to_datetime(t)) if is_date else np.arange(len(t), dtype=float)
        with journal_style({"axes.grid": False}):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            im = ax.pcolormesh(x, u, Z, cmap=get_cmap(cmap), shading="nearest", rasterized=True)
            if is_date:
                ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_ylabel(r"$\tau_1 = \tau_2$" if which == "diagonal" else r"$\tau_1 = 1 - \tau_2$")
            ax.set_yticks(u)
            ax.set_yticklabels([fmt_q(q) for q in u])
            fig.colorbar(im, ax=ax, label="TCI (%)", pad=0.02)
            ax.set_title(("Directly" if which == "diagonal" else "Reversely") +
                         " related quantiles: TCI over time")
        return fig, ax


def decompose_tci(gs_u: QQConnResult, gs_c: QQConnResult, qoq_u: QQConnResult,
                  qoq_c: QQConnResult) -> pd.DataFrame:
    """Method vs centering decomposition of Bonaccolto, Caporin and Gupta (2026, Eqs. 3-4, Table A.2).

    QoQ_C - GS_U = (QoQ_C - GS_C) + (GS_C - GS_U)        [Eq. 3: method + centering]
                 = (QoQ_C - QoQ_U) + (QoQ_U - GS_U)      [Eq. 4: centering + method]

    computed for direct and reverse TCI on the common dates; returns Mean, MA,
    Min, P5, P95 and Max for every component.
    """
    frames = {}
    for nm, r in (("gsU", gs_u), ("gsC", gs_c), ("qU", qoq_u), ("qC", qoq_c)):
        d = r.direct_reverse().set_index("date")[["direct", "reverse"]]
        frames[nm] = d
    common = frames["gsU"].index
    for d in frames.values():
        common = common.intersection(d.index)
    f = {k: v.loc[common] for k, v in frames.items()}
    rows = []
    for panel in ("direct", "reverse"):
        comps = {"Total difference": f["qC"][panel] - f["gsU"][panel],
                 "(3) Method": f["qC"][panel] - f["gsC"][panel],
                 "(3) Centering": f["gsC"][panel] - f["gsU"][panel],
                 "(4) Method": f["qU"][panel] - f["gsU"][panel],
                 "(4) Centering": f["qC"][panel] - f["qU"][panel]}
        for k, v in comps.items():
            v = v.to_numpy()
            rows.append((f"{panel.capitalize()} TCI", k, np.nanmean(v), np.nanmean(np.abs(v)),
                         np.nanmin(v), np.nanpercentile(v, 5), np.nanpercentile(v, 95), np.nanmax(v)))
    return pd.DataFrame(rows, columns=["Panel", "Component", "Mean", "MA", "Min", "P5", "P95", "Max"]
                        ).set_index(["Panel", "Component"]).round(2)


def compare_heatmaps(measure: str = "TCI", series=0, ncols: int = 2, common_scale: bool = True,
                     cmap: str = "Parula", figsize=None, **results):
    """Side-by-side averaged heatmaps of several estimators on a common colour scale.

    Layout of Bonaccolto, Caporin and Gupta (2026, Fig. 1). Pass results as keyword
    arguments, e.g. ``compare_heatmaps(**{"(a) GS: second moments": gs_u, "(b) GS: covariance": gs_c})``.
    Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt
    if not results:
        raise ValueError("pass at least one QQConnResult as a keyword argument")
    grids = {k: r.grid(measure, series) for k, r in results.items()}
    n = len(grids)
    nrows = int(np.ceil(n / ncols))
    vmin = min(np.nanmin(g.values) for g in grids.values())
    vmax = max(np.nanmax(g.values) for g in grids.values())
    with journal_style({"axes.grid": False}):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (6.0 * ncols, 5.0 * nrows), squeeze=False)
        for ax, (title, g) in zip(axes.flat, grids.items()):
            g.heatmap(ax=ax, cmap=cmap, fontsize=5.5, colorbar=False)
            if common_scale:
                ax.images[0].set_clim(vmin, vmax)
            ax.set_title(title)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        im = axes.flat[0].images[0]
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label=list(grids.values())[0].value_label)
    return fig, axes
