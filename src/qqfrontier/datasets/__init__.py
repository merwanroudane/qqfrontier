"""Real datasets shipped with qqfrontier (see SOURCES.md for sources and licences)."""
from __future__ import annotations

from importlib import resources
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["load_us_yields", "load_wdi_co2_gdp"]


def _read(name: str) -> pd.DataFrame:
    with resources.files(__name__).joinpath(name).open("rb") as f:
        return pd.read_csv(f, compression="gzip")


def load_us_yields(main_only: bool = False) -> pd.DataFrame:
    """Daily changes (basis points) of the US 2-year Treasury yield and the 10Y-2Y spread.

    Source: FRED, series DGS2 and T10Y2Y (Board of Governors of the Federal Reserve
    System; public domain), 1998-02-19 to 2023-07-11, first differences x 100.
    The flag ``main`` marks the 6,253 changes after 1998-07-13 used by Gabauer and
    Stenfors (2024) and Bonaccolto, Caporin and Gupta (2026); the first 100 rows are
    the pre-sample needed by the kernel estimator.

    Columns: ``date`` (datetime), ``US2Y``, ``US2Y10Y``, ``main`` (bool).

    Examples
    --------
    >>> load_us_yields().shape
    (6353, 4)
    """
    d = _read("us_yields.csv.gz")
    d["date"] = pd.to_datetime(d["date"])
    d["main"] = d["main"].astype(str).str.upper().eq("TRUE")
    return d.loc[d["main"]].reset_index(drop=True) if main_only else d


def load_wdi_co2_gdp(country: Optional[str] = None, logs: bool = True) -> pd.DataFrame:
    """Annual GDP per capita and CO2 emissions per capita for Algeria and Egypt.

    Source: World Bank, World Development Indicators (CC BY 4.0): NY.GDP.PCAP.KD
    (GDP per capita, constant 2015 US$) and EN.GHG.CO2.PC.CE.AR5 (CO2 emissions per
    capita, t CO2e), 1960-2024 (CO2 from 1970).  ``logs=True`` adds ``lnGDP`` and
    ``lnCO2`` and drops years with a missing value, as in the qqardl example.

    Examples
    --------
    >>> d = load_wdi_co2_gdp("Algeria")
    >>> int(d.year.min()), int(d.year.max()), len(d)
    (1970, 2024, 55)
    """
    d = _read("wdi_co2_gdp.csv.gz")
    if logs:
        d["lnGDP"] = np.log(d["gdppc"])
        d["lnCO2"] = np.log(d["co2pc"])
        d = d.dropna(subset=["lnGDP", "lnCO2"])
    if country is not None:
        d = d[d["country"] == country]
    return d.sort_values(["country", "year"]).reset_index(drop=True)
