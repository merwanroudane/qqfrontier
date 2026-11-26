"""qqfrontier: frontier quantile-on-quantile econometrics in Python.

Modules
-------
connectedness  Quantile-on-quantile connectedness: Gabauer and Stenfors (2024),
               Bonaccolto, Caporin and Gupta (2026), TVP-QQVAR (2026).
quantilogram   Cross-quantilogram of Han, Linton, Oka and Whang (2016).
causality      Quantile Granger causality robust to instability (Mayer, Wied and
               Troster, 2025), joint sup-Wald / KS tests, multi-lag, variance.
coherency      Quantile coherency of Barunik and Kley (2019).
qqardl         Quantile-on-quantile ARDL (Destek et al., 2025).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane/qqfrontier
"""
from importlib.metadata import PackageNotFoundError, version as _v

from . import connectedness
from ._colors import (bluered_colors, get_cmap, list_colorscales, matlab_jet_colors,
                      parula_cmap, parula_colors, resolve_colorscale, sinha_colors,
                      turbo_colors)
from ._descriptive import descriptive_table, weighted_portmanteau
from ._grid import QQGrid
from ._qr import rq_fit, rq_noncross
from ._style import journal_style, save_figure
from .connectedness import (QQConnResult, cond_pit, connectedness_table, decompose_tci,
                            dqlm, gfevd, mc_qqvar, qqc_gs, qqc_kernel, qqc_tvp, qqconnect,
                            sim_qqvar)
from .datasets import load_us_yields

try:
    __version__ = _v("qqfrontier")
except PackageNotFoundError:  # pragma: no cover - source tree
    __version__ = "0.1.0"
__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

__all__ = [
    "connectedness", "qqc_gs", "qqc_kernel", "qqc_tvp", "qqconnect", "cond_pit", "dqlm",
    "gfevd", "connectedness_table", "QQConnResult", "decompose_tci", "sim_qqvar", "mc_qqvar",
    "QQGrid", "descriptive_table", "weighted_portmanteau", "rq_fit", "rq_noncross",
    "parula_colors", "parula_cmap", "matlab_jet_colors", "turbo_colors", "bluered_colors",
    "sinha_colors", "resolve_colorscale", "get_cmap", "list_colorscales", "journal_style",
    "save_figure", "load_us_yields",
]
