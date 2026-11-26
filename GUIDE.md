# qqfrontier — a complete, step-by-step guide

*For researchers who know the econometrics and have never written Python.*
Author: **Dr Merwan Roudane** · merwanroudane920@gmail.com · https://github.com/merwanroudane/qqfrontier

Every block below can be copied as it is. Lines that start with `#` are comments: Python
ignores them, they are there to explain.

---

## Part A — Getting ready (do this once)

### A.1 Install Python
1. Download **Anaconda** from <https://www.anaconda.com/download> and install it with the
   default options. Anaconda contains Python, Jupyter Notebook and Spyder.
2. Windows: press the `Win` key, type **Anaconda Prompt**, press Enter.
   macOS: open **Terminal**.
3. Check Python:

```bash
python --version
```

### A.2 Install qqfrontier

```bash
pip install "qqfrontier[full]"
```

`[full]` also installs `plotly` (interactive 3D figures), `numba` (fast quantile
regressions), `joblib` (parallel computing) and `arch` (the ERS unit-root test).

### A.3 Choose where to write your code

| Tool | When to use it | How to open |
|---|---|---|
| Jupyter Notebook | step by step, figures inside the page | type `jupyter notebook` in the terminal |
| Spyder | an RStudio / MATLAB-like window | type `spyder` |
| a `.py` file | a reproducible script for the paper | save `analysis.py`, run `python analysis.py` |

### A.4 The first two lines of every script

```python
import numpy as np            # numbers and arrays
import pandas as pd           # tables (data frames)
import qqfrontier as qf       # this package
```

---

## Part B — Loading data

### B.1 The datasets shipped with the package

```python
yields = qf.load_us_yields(main_only=True)   # daily US 2Y yield and 10Y-2Y spread changes
print(yields.head())

wdi = qf.load_wdi_co2_gdp("Algeria")         # annual ln CO2 and ln GDP per capita
print(wdi.head())
```

### B.2 Your own Excel or CSV file

```python
df = pd.read_excel("C:/Users/you/Desktop/mydata.xlsx")        # Excel
# df = pd.read_csv("C:/Users/you/Desktop/mydata.csv")          # or CSV
df["date"] = pd.to_datetime(df["date"])                         # if you have a date column
df = df.dropna()                                                # remove missing values
print(df.describe())
```

Use forward slashes `/` in Windows paths, or write `r"C:\Users\you\mydata.xlsx"`.

### B.3 Transforming variables

```python
df["lnCO2"] = np.log(df["co2"])                 # logarithm
df["r_oil"] = 100 * np.log(df["oil"]).diff()    # % log return
df["d_rate"] = df["rate"].diff()                # first difference
df = df.dropna()
```

### B.4 The summary-statistics table of the empirical section

```python
tab = qf.descriptive_table(df[["r_oil", "r_stock"]], lags=20)
print(tab)                                         # journal text table
open("table1.tex", "w").write(tab.to_latex())      # LaTeX (booktabs)
```

It reports mean, variance, skewness and excess kurtosis (with tests), Jarque–Bera, the
ERS unit-root test and the weighted portmanteau tests Q(20) and Q²(20), with stars, plus
the correlation matrix: the layout of Gabauer and Stenfors (2024, Table 1).

---

## Part C — The five modules, one by one

Every result object has the same tools:

| Command | What you get |
|---|---|
| `print(res.summary())` | the journal tables in the console |
| `res.to_latex()` | a LaTeX table (booktabs, stars, notes) |
| `res.to_markdown()` | a Markdown table (GitHub, Word via pandoc) |
| `res.to_frame()` | every number in a pandas table (save with `.to_excel("x.xlsx")`) |
| `res.interpret()` | a paragraph in plain English describing the results |
| `res.plot("...")` | the figures (see each module) |

Save any figure in vector (manuscript) and 300-dpi PNG (slides, web) formats:

```python
fig, ax = res.plot("heatmap")
qf.save_figure(fig, "figure2")        # writes figure2.pdf and figure2.png
```

### C.1 Quantile-on-quantile connectedness (M1)

**Question.** How strongly do shocks spill over between two (or more) markets when each
market is in a given part of its distribution (e.g. the 2-year yield in its lower tail and
the spread in its upper tail)?

```python
x = yields[["date", "US2Y", "US2Y10Y"]]           # a date column + the series
taus = np.arange(0.05, 1.0, 0.10)                  # 0.05, 0.15, ..., 0.95

# (1) Gabauer and Stenfors (2024): rolling quantile VAR
gs = qf.qqc_gs(x, taus=taus, window=200, horizon=20)

# (2) Bonaccolto, Caporin and Gupta (2026): kernel-weighted local Q-on-Q-QVAR
#     (needs 100 extra observations at the start: use the full data set)
full = qf.load_us_yields()[["date", "US2Y", "US2Y10Y"]]
qoq = qf.qqc_kernel(full, taus=taus, window=200, horizon=20, bandwidth=0.15,
                    presample=100, n_jobs=8)       # n_jobs = CPU cores to use

# (3) TVP-QQVAR (2026): no rolling window, time-varying parameters
tvp = qf.qqc_tvp(x, taus=taus, horizon=30, delta=0.99, kappa=0.99)
```

Tables:

```python
print(gs.summary())                     # averaged TCI grid, NET grids, direct/reverse statistics
print(gs.average())                     # the averaged TCI matrix (Fig. 2 of GS 2024)
print(gs.dynamic_stats())               # mean / MA / min / P5 / P95 / max of reverse, direct, delta
print(gs.net_transmitter_table())       # who transmits shocks in each quantile configuration
print(qf.decompose_tci(gs_u, gs_c, qoq_u, qoq_c))   # BCG (2026) Table A.2
```

Figures:

```python
gs.plot("heatmap")          # averaged TCI heatmap with values (Parula colours)
gs.plot("dynamic")          # reverse / direct / delta TCI over time (GS Fig. 3)
gs.plot("net")              # NET heatmap of each series
gs.plot("surface")          # 3D surface of the averaged TCI
gs.plot("panel")            # heatmap + 3D surface + contour in one figure
gs.plot("time_heatmap", which="antidiagonal")   # time x quantile map
gs.plot("paths")            # TCI over time at selected quantile pairs
gs.grid().plot_surface().write_html("tci_3d.html")   # interactive 3D (open in a browser)
qf.compare_heatmaps(**{"GS": gs, "Q-on-Q": qoq, "TVP": tvp})   # side by side
```

**Reading the heatmap.** Each cell is the average total connectedness index (in %) when
series 1 is at the row quantile and series 2 at the column quantile. Large values in the
corners mean that spillovers are concentrated in extreme market states; larger values on
the anti-diagonal (opposite tails) than on the diagonal (same tails) mean that spillovers
are strongest when the markets move in opposite directions.

### C.2 Cross-quantilogram (M2)

**Question.** Does being in the lower (upper) tail of series 2 today predict being in the
lower (upper) tail of series 1 after k periods?

```python
y1 = yields.set_index("date")["US2Y10Y"]      # predicted series
y2 = yields.set_index("date")["US2Y"]         # predictor
cq = qf.cross_quantilogram_test(
        y1, y2,
        tau1=[0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95],   # quantiles of y1
        tau2=[0.10, 0.90],                                              # quantiles of y2
        max_lag=20, n_boot=1000, seed=1,
        sup_range=np.arange(0.10, 0.31, 0.02))      # sup-test over a range of quantiles
print(cq.summary())
cq.plot("correlogram", tau2=0.10)     # bars + 95% band for no predictability (HLOW Fig. 2a)
cq.plot("portmanteau", tau2=0.10)     # Box-Ljung Q(p) against its critical value (Fig. 2b)
cq.plot("sup")                        # sup-Q test (Fig. 1)
cq.plot("heatmap", k=1)               # (tau1, tau2) map at lag 1 with stars
cq.plot("lag_surface", tau2=0.10)     # 3D: lag x tau1
```

Partial cross-quantilogram (controlling for a third variable):

```python
res = qf.partial_cross_quantilogram(y1, y2, controls=np.abs(y1), tau1=0.1, tau2=0.1,
                                    tau_z=[0.5], k=1, n_boot=1000)
print(res)
```

**Reading.** A positive significant value at (0.05, 0.10) and lag 1 means: when y2 is below
its 10% quantile today, y1 is more likely to fall below its 5% quantile tomorrow.

### C.3 Quantile Granger causality robust to instability (M3)

**Question.** Does x Granger-cause y in quantiles, allowing the effect to change over time
(breaks), and when?

```python
gc = qf.qgc_instability(y1, y2, lags=1, n_boot=499, seed=3, alpha=0.10)
print(gc.summary())                 # supWald / expLM / supLM p-values by quantile range,
                                    # LM(tau) by quantile, dated regimes (Algorithm 2)
gc.plot("cusum")                    # CUSUM curve, break dates, causality regimes shaded
gc.plot("lm")                       # LM(tau) across quantiles
gc.plot("cusum_map")                # time x quantile map of the instability detector
gc.plot("coefficients")             # unrestricted Granger coefficients gamma(tau)

# several lag orders in one table (MWT Table 3 layout)
print(qf.qgc_table(y1, y2, lags=(1, 3, 6, 9), n_boot=499, seed=4))
# causality in variance (squared series)
gv = qf.qgc_instability(y1, y2, lags=1, moment=2, n_boot=499)
```

**Reading.** If expLM rejects but supWald does not, the causal effect exists but changes
over time, which a standard test misses. The regime table tells you on which sub-periods
causality holds.

### C.4 Quantile coherency (M4)

**Question.** Are the two series connected over short, medium or long cycles, separately
in their lower tails, centres and upper tails?

```python
coh = qf.quantile_coherency(yields["US2Y"], yields["US2Y10Y"],
                            taus=np.arange(0.05, 0.96, 0.05),
                            periods={"W": 5, "M": 22, "Y": 250})   # daily data
print(coh.summary())
coh.plot("frequency", pairs=[(0.5, 0.5), (0.05, 0.05), (0.95, 0.95)])   # BK Fig. 3 left
coh.plot("quantile")                    # along tau1 = tau2 at the weekly, monthly, yearly periods
coh.plot("heatmap", period="M")         # quantile-on-quantile map at the monthly period
coh.plot("surface", period="M")         # its 3D surface
coh.plot("freq_surface")                # 3D: frequency x quantile
```

For monthly data use `periods={"Q": 3, "Y": 12, "BC": 48}`; for annual data
`periods={"5Y": 5, "10Y": 10}`.

**Reading.** Re R close to 1 at the 0.05|0.05 pair and the yearly period means the two
series tend to be in their lower tails together over yearly cycles. Stars (or a shaded band
not crossing zero) mark pairs whose 95% interval excludes zero.

### C.5 Quantile-on-quantile ARDL (M5)

**Question.** Is there a long-run (cointegrating) relation between the quantiles of y and
the quantiles of x, and how large are the long- and short-run effects?

```python
d = qf.load_wdi_co2_gdp("Algeria").set_index("year")
r = qf.qqardl_fit(d["lnCO2"], d["lnGDP"])          # default grid 0.05, 0.10, ..., 0.95
print(r.summary())                 # bounds-test F, long run, short run, ECT (with stars)
r.plot("heatmap", stat="lr")       # long-run coefficients (cointegrated pairs only)
r.plot("surface", stat="lr")       # 3D surface of the long run
r.plot("coint")                    # F heatmap, cointegrated pairs framed
r.plot("heatmap", stat="sr")       # short run
r.plot("heatmap", stat="ect")      # error-correction term
r.plot("qser")                     # observed series and its quantile series
r.plot("diagonal", stat="lr")      # long run along tau = theta with 95% band
```

Options: `lags=(p, q)`, `btest="ecm"`, `vce="robust"`, `lrtype="beta"`, `lrall=True`.
Bounds-test p-values use the Kripfganz–Schneider (2020) response surfaces. They are read
from the Stata `ardl` package if it is installed (`ssc install ardl`), or from a file you
point to with `coefs_path=...`; otherwise the PSS (2001) 10% bound decides cointegration.

---

## Part D — Putting results in your article

```python
# one table for the paper
open("tab_lr.tex", "w").write(r.to_latex("lr", caption="Long-run effects of GDP on CO2"))
# everything as Excel
r.to_frame().to_excel("qqardl_results.xlsx", index=False)
# a paragraph to start the discussion
print(r.interpret())
```

In LaTeX add to the preamble:

```latex
\usepackage{booktabs}
\newcommand{\sym}[1]{\ensuremath{^{#1}}}
```

---

## Part E — Practical advice

* **Sample size.** M1 needs a few hundred observations per window (daily data); M2 and M3
  work from about 300 observations; M4 needs long series (> 500) to separate cycles;
  M5 is designed for annual data (n >= 30).
* **Quantile grids.** 0.05–0.95 by 0.10 (10 x 10) is the standard grid in the
  connectedness papers; use 0.05 steps for smoother surfaces.
* **Bootstrap.** Use the paper values for final tables (CQ: 1,000; MWT: 499 or 1,999) and
  small values (99) while exploring.
* **Speed.** Install `numba` and `joblib`, and pass `n_jobs=` (number of CPU cores) to the
  kernel connectedness estimator.
* **Reproducibility.** Always pass `seed=` to functions with a bootstrap.

The complete application of every module is in `examples/qqfrontier_empirical.py`.
