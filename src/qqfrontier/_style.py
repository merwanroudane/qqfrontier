"""Journal figure style and small table helpers shared by every module.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import contextlib
from typing import Iterable, Optional

import numpy as np

JOURNAL_RC = {
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
}

#: Line colours + styles that survive greyscale printing.
LINE_CYCLE = [("#D62728", "-"), ("#2CA02C", "--"), ("#1F4E9E", "-."),
              ("#FF7F0E", ":"), ("#6A3D9A", "-"), ("#8C564B", "--")]


@contextlib.contextmanager
def journal_style(rc: Optional[dict] = None):
    """Context manager applying the house journal style without leaking it."""
    import matplotlib.pyplot as plt
    params = dict(JOURNAL_RC)
    if rc:
        params.update(rc)
    with plt.rc_context(params):
        yield


def save_figure(fig, stem: str, formats: Iterable[str] = ("pdf", "png")) -> None:
    """Save a figure as vector PDF (manuscript) and 300-dpi PNG (README / web)."""
    for f in formats:
        fig.savefig(f"{stem}.{f}", dpi=300 if f == "png" else None, bbox_inches="tight")


def stars(p: float) -> str:
    """Significance stars: *** p<0.01, ** p<0.05, * p<0.10."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_q(q: float) -> str:
    """Quantile label: 0.05 -> '0.05'."""
    return f"{q:.2f}" if abs(q * 100 - round(q * 100)) < 1e-9 else f"{q:.3f}"


def latex_escape(s: str) -> str:
    """Escape the characters that break LaTeX in user-supplied names."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in str(s))


SYM_DEF = r"% Preamble: \newcommand{\sym}[1]{\ensuremath{^{#1}}}  \usepackage{booktabs}"
