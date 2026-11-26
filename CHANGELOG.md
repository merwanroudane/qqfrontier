# Changelog

## [0.1.0] - 2026-11-27

### Added
- **M1 connectedness**: quantile-on-quantile connectedness of Gabauer and Stenfors (2024),
  the kernel-weighted Q-on-Q-QVAR of Bonaccolto, Caporin and Gupta (2026) with Bondell et al.
  (2010) non-crossing probability levels, and the TVP-QQVAR of Candelo-Viafara and
  Orrego-Reyes (2026); a faithful port of the R package qqconnect (GFEVD, TO/FROM/NET/TCI,
  direct vs reverse TCI, method-vs-centering decomposition, Monte Carlo design).
- **M2 quantilogram**: cross-quantilogram of Han, Linton, Oka and Whang (2016) with the
  stationary bootstrap, Box-Ljung and sup-Box-Ljung tests, self-normalised statistic and
  partial cross-quantilogram.
- **M3 causality**: quantile Granger causality robust to instability of Mayer, Wied and
  Troster (2025): LM(tau), supLM, expLM, bootstrap of Algorithm 1, regime dating of
  Algorithm 2, supWald benchmark; multi-lag tables, causality in variance, lag x quantile map.
- **M4 coherency**: quantile coherency of Barunik and Kley (2019) with pointwise CIs.
- **M5 qqardl**: quantile-on-quantile ARDL of Destek et al. (2025), a port of the Stata command
  qqardl 1.1.0, with Kripfganz-Schneider (2020) bounds-test p-values.
- Shared journal layer: annotated heatmaps, static and interactive 3D surfaces, contours,
  cross-sections, booktabs LaTeX / Markdown tables, plain-language `interpret()`.
- Quantile-regression engine: numba Frisch-Newton (quantreg rq.fit.fnb), HiGHS simplex,
  structured interior point for Bondell non-crossing curves.
- Datasets: FRED US yields (public domain) and World Bank WDI CO2 / GDP (CC BY 4.0).
