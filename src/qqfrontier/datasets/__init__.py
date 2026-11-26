"""Real datasets shipped with qqfrontier (public domain, see SOURCES.md)."""
from __future__ import annotations

from importlib import resources

import pandas as pd

__all__ = ["load_us_yields"]


def _read(name: str) -> pd.DataFrame:
    with resources.files(__name__).joinpath(name).open("rb") as f:
        return pd.read_csv(f, compression="gzip")


def load_us_yields(main_only: bool = False) -> pd.DataFrame:
    """Daily changes (basis points) of the US 2-year Treasury yield and the 10Y-2Y spread.

    Source: FRED, series DGS2 and T10Y2Y (Board of Governors of the Federal Reserve
    System, public domain), 1998-02-19 to 2023-07-11, first differences x 100.
    The flag ``main`` marks the 6,253 changes after 1998-07-13 used by Gabauer and
    Stenfors (2024) and Bonaccolto, Caporin and Gupta (2026); the first 100 rows are
    the pre-sample needed by the kernel estimator.

    Columns: ``date`` (datetime), ``US2Y``, ``US2Y10Y``, ``main`` (bool).

    Examples
    --------
    >>> d = load_us_yields()
    >>> d.shape
    (6353, 4)
    """
    d = _read("us_yields.csv.gz")
    d["date"] = pd.to_datetime(d["date"])
    d["main"] = d["main"].astype(str).str.upper().eq("TRUE")
    return d.loc[d["main"]].reset_index(drop=True) if main_only else d
