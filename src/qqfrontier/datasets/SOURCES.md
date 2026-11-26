# Data sources

| File | Content | Source | Licence |
|---|---|---|---|
| `us_yields.csv.gz` | Daily changes (basis points) of the 2-year US Treasury yield (`US2Y`, FRED `DGS2`) and of the 10Y-2Y spread (`US2Y10Y`, FRED `T10Y2Y`), 1998-02-19 to 2023-07-11; `main` flags the 6,253 changes after 1998-07-13 | Board of Governors of the Federal Reserve System via FRED, Federal Reserve Bank of St. Louis | Public domain (U.S. government data) |
| `wdi_co2_gdp.csv.gz` | Annual GDP per capita, constant 2015 US$ (`gdppc`, NY.GDP.PCAP.KD) and CO2 emissions per capita, t CO2e (`co2pc`, EN.GHG.CO2.PC.CE.AR5) for Algeria and Egypt, 1960-2024 | World Bank, World Development Indicators | CC BY 4.0 (attribution: The World Bank) |

Construction: `us_yields` follows `data-raw/us_yields.R` of the author's R package
`qqconnect` (first differences x 100 of the FRED series; the first 100 rows are the
pre-sample of Bonaccolto, Caporin and Gupta, 2026). `wdi_co2_gdp` is the WDI extract
used in the example of the author's Stata command `qqardl`.
