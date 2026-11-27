# qqfrontier — methods and step → equation maps

Author: Dr Merwan Roudane · merwanroudane920@gmail.com

This page lists, for every module, which block of code implements which equation of the
source paper, the choices made where the papers are silent, and how each implementation
was validated.

---

## Quantile-regression engine (`qqfrontier._qr`)

| Function | Algorithm | Reference / check |
|---|---|---|
| `rq_fit(method="fn")`, `rq_fit_batch` | Frisch–Newton primal-dual interior point on the bounded dual LP, Mehrotra predictor-corrector, `beta = 0.99995`, `eps = 1e-6`, start a = 1 − τ | port of `quantreg::rq.fit.fnb` (Koenker and Portnoy, 1997); numba-compiled, batch-parallel; agrees with R `quantreg` to 1e-7 |
| `rq_fit(method="br")` | exact simplex vertex (HiGHS dual simplex) | analogue of `quantreg` "br"; same optimum as R to 1e-7 |
| `rq_noncross` | Bondell, Reich and Wang (2010) joint fit with non-crossing at the vertices of the regressor box; structured infeasible-start Mehrotra IPM on the bounded dual (Newton system pK × pK); HiGHS fallback | objective equal to the HiGHS solution to 1e-11; equal to `qqconnect::rq_noncross` |

---

## M1 — Quantile-on-quantile connectedness (`qqfrontier.connectedness`)

Port of the author's R package **qqconnect 0.1.0**.

| Step | Paper | Code |
|---|---|---|
| QVAR(p) equation j at its own τⱼ, rolling window | Gabauer and Stenfors (2024), Sec. 2 | `qqc_gs` |
| residual matrix: uncentered second moments R'R/n (GS) or centered covariance (BCG) | GS (2024); Bonaccolto, Caporin and Gupta (2026), Sec. 2 | `_resid_cov` |
| scaled GFEVD θᵢⱼ = σⱼⱼ⁻¹ Σₕ (eᵢ'ΨₕΣeⱼ)², rows normalised, h = 0..H−1 | Koop et al. (1996); Pesaran and Shin (1998) | `fevd_batch` (port of `connectedness.cpp`) |
| TO, FROM, NET, adjusted TCI = k/(k−1)·mean(TO) | Diebold and Yilmaz (2012); Chatziantoniou et al. (2021) | `conn_summary` |
| conditional probability levels uᵢₜ from rolling non-crossing QR on (1, yₜ₋₁) | BCG (2026), step 1 | `cond_pit` |
| kernel weights ∏ᵢ≠ⱼ K((uᵢₜ − τᵢ)/h), weighted QR per equation | BCG (2026), step 2 | `qqc_kernel` |
| AL-DQLM with discount factor δ, VB/EM for σ, discount Kalman filter/smoother | Candelo-Viafara and Orrego-Reyes (2026) | `dqlm` |
| Hₜ = κHₜ₋₁ + (1−κ)uₜuₜ' | TVP-QQVAR (2026) | `qqc_tvp` |
| direct (τ₁ = τ₂) vs reverse (τ₁ = 1 − τ₂) TCI | GS (2024), Fig. 3 | `direct_reverse` |
| method / centering decomposition | BCG (2026), Eqs. (3)–(4), Table A.2 | `decompose_tci` |

**Validation.** Averaged TCI grid of `qqc_gs` on the US yield data equals Fig. 1(a) of
BCG (2026) cell by cell; per-window TCI equal to `qqconnect` (median abs. difference 1.5e-8;
the remaining cells differ only where the quantile LP has several optimal solutions, which
happens with the integer-valued basis-point data); `qqc_tvp` equal to `qqconnect` to 1e-12.

---

## M2 — Cross-quantilogram (`qqfrontier.quantilogram`)

| Step | Han, Linton, Oka and Whang (2016) | Code |
|---|---|---|
| ψₐ(u) = 1[u < 0] − a | Eq. (1) | `_hits` (`hit="le"` reproduces the R package) |
| sample or QR conditional quantiles | Sec. 2.1 | `_hits(X=...)` |
| ρ̂τ(k) = Σψψ / √(Σψ² Σψ²), t = k+1..T | Eq. (2) | `cross_quantilogram` |
| Box–Ljung Q̌(p) = T(T+2) Σ ρ̂²/(T−k); sup over a quantile range | Sec. 2.1 | `cross_quantilogram_test` |
| stationary bootstrap of the (T−p) tuples, quantiles re-estimated, CI [ρ̂ + c₁, ρ̂ + c₂] | Sec. 3.1.1, Thm 2 | `cross_quantilogram_test` |
| mean block length: Politis–White (2004) with the Patton et al. (2009) correction, averaged | Sec. 6.1, fn. 16 | `block_length` |
| self-normalised S(p) = T ρ̂'V̂⁻¹ρ̂ with recursive subsamples; simulated critical values | Sec. 3.1.2, Thm 4 | `_self_normalized`, `self_normalized_cv` |
| partial cross-quantilogram −p₁₂/√(p₁₁p₂₂) | Eq. (10) | `partial_cross_quantilogram` |

**Validation.** ρ̂ equal to `quantilogram::crossq` to 1e-10; block length equal to `np::b.star`.

---

## M3 — Quantile Granger causality robust to instability (`qqfrontier.causality`)

| Step | Mayer, Wied and Troster (2025) | Code |
|---|---|---|
| xᵢ = (zᵢ', wᵢ')', H₀ = {γᵢ(τ) constant} ∩ {γ = 0} | Eq. (1) | `build_design` |
| restricted fit αₙ(τ) of y on w | Eq. (4) | `_paths` |
| Sₙ(λ,τ) = n^{-1/2} Σ xᵢψτ(yᵢ − wᵢ'α), ψτ(u) = 1{u ≤ 0} − τ | Eq. (3) | `_paths` |
| Hₙ = (X'X/n)^{-1/2} Sₙ with the Cholesky factor of the inverse | Eq. (5) | `_inv_sqrt_upper` |
| LM₁ (CUSUM), LM₂ (LM), LM(τ) = h(τ)(sup LM₁ + LM₂) | Sec. 3.1 | `qgc_instability` |
| supLM, expLM = ∫ exp(½(sup LM₁ + LM₂)) dτ (trapezoid on the grid) | Eq. (8) | `_interval_stats` |
| pivotal null under Assumption D (uniform-hit partial sums) | Cor. 2–3 | `simulate_null` |
| bootstrap: x resampled, y* = αₙ(U)'w, U ~ U(0,1) | Algorithm 1 | `_bootstrap_A` |
| break dating and regime-wise tests with Šidák levels | Algorithm 2, Eq. (9) | `_algorithm2` |
| supWald with pairs-bootstrap covariance, Bessel-process critical values | Eq. (10) | `_sup_wald` |

*Extensions (not in the paper):* several lag orders in one table (`qgc_table`), causality in
variance (`moment=2`), the lag × quantile map (`qgc_map`) and the time × quantile CUSUM map.

**Validation.** Simulated break at n/2 dated within 1% of the sample; empirical size under
the null at nominal level; empirical application in the example script.

---

## M4 — Quantile coherency (`qqfrontier.coherency`)

| Step | Baruník and Kley (2019) | Code |
|---|---|---|
| rank-based clipped DFT d(ω;τ) = Σₜ 1{Rₜ ≤ nτ} e^{−iωt} | Eq. (2.4) | `quantile_coherency` |
| CCR-periodogram (2πn)⁻¹ d₁(ω) d₂(−ω) | Eq. (2.4) | idem |
| smoothing over Fourier frequencies s = 1..n−1 (FFT circular convolution), normalised by Wᵏ | Eq. (2.6), Supp. S5.1 | `_circ_conv` |
| Wₙ(u) = Σⱼ bₙ⁻¹W(bₙ⁻¹(u + 2πj)), Epanechnikov on [−π, π], bₙ = 0.5 n^{-1/4} | Ass. 4.2, Sec. 5 | `kernel_weights` |
| R̂ = Ĝ¹²/√(Ĝ¹¹Ĝ²²) | Eq. (2.8) | idem |
| pointwise CIs for Re and Im via Cov(Hab, Hcd) of Eq. (S.10) and the delta expansion of L | Supp. S5.2 | `_coherency_sd` |

**Validation.** Coherency of a series with itself equals 1; the smoother equals a direct
evaluation of Eq. (2.6); Monte Carlo coverage of the 95% intervals under independence:
94–96% for real and imaginary parts at every frequency tested. (The development version of
the R package `quantspec` could not be used as a numerical reference: under R ≥ 4 its
frequency-to-index conversion floors 2πk/n·n/2π, so some Fourier frequencies are duplicated
and others dropped.)

---

## M5 — Quantile-on-quantile ARDL (`qqfrontier.qqardl`)

Port of the author's Stata command **qqardl 1.1.0**.

| Step | Destek et al. (2025) | Code |
|---|---|---|
| S1 quantile series QSERτ(t) = q̂τ + Σₖ [c₁cos(2πks/n) + c₂sin(2πks/n)] + Nyquist term | Li (2022/2025), as `qfa::qser` | `qser` |
| S2 bounds regression ΔY = c + Σwᵢ ΔYₜ₋ᵢ + Σvⱼ ΔXₜ₋ⱼ + g₁Yₜ₋₁ + g₂Xₜ₋₁ + e, Wald F | Eqs. (1)–(3) | `_btest` |
| bounds p-values: Kripfganz–Schneider (2020) response surfaces (case III coefficients shipped with the package) + MacKinnon local regression; cointegration if p < 0.10 | Kripfganz and Schneider (2020); Destek et al. (2025), p. 3 | `ardl_bounds_pvalue` |
| S3 ECM ΔY = a₀ + d₀Yₜ₋₁ + d₁ΔXₜ + b₁Xₜ₋₁ + e | Eq. (4) | `_ecm` |
| S4 ECT d₀, short run d₁, long run −b₁/d₀ (delta method) | p. 4 | `_ecm` |
| S5 grid τ, θ ∈ {0.05, 0.10, 0.20, …, 0.90, 0.95} | Eq. (5), fn. 1 | `qqardl` |

**Validation.** All four 11 × 11 tables of the Stata example (Algeria, 1970–2024) reproduced
to the third decimal, with identical significance stars and the same 53 cointegrated pairs.

---

## References

* Baruník, J. and Kley, T. (2019). *The Econometrics Journal* 22(2), 131–152. doi:10.1093/ectj/utz002
* Bonaccolto, G., Caporin, M. and Gupta, R. (2026). Rethinking quantile-on-quantile connectedness. *Finance Research Letters* 114, 111095.
* Bondell, H. D., Reich, B. J. and Wang, H. (2010). *Biometrika* 97(4), 825–838. doi:10.1093/biomet/asq048
* Candelo-Viafara, J. M. and Orrego-Reyes, J. E. (2026). SSRN 7434976. doi:10.2139/ssrn.7434976
* Chatziantoniou, I., Gabauer, D. and Stenfors, A. (2021). *Economics Letters* 204, 109891. doi:10.1016/j.econlet.2021.109891
* Destek, M. A., Radulescu, M., Ozkan, O. and Balsalobre-Lorente, D. (2025). *J. Environmental Management* 389, 126089. doi:10.1016/j.jenvman.2025.126089
* Diebold, F. X. and Yilmaz, K. (2012). *International Journal of Forecasting* 28(1), 57–66. doi:10.1016/j.ijforecast.2011.02.006
* Gabauer, D. and Stenfors, A. (2024). *Finance Research Letters* 60, 104852. doi:10.1016/j.frl.2023.104852
* Han, H., Linton, O., Oka, T. and Whang, Y.-J. (2016). *Journal of Econometrics* 193(1), 251–270. doi:10.1016/j.jeconom.2016.03.001
* Kripfganz, S. and Schneider, D. C. (2020). *Oxford Bulletin of Economics and Statistics* 82(6), 1456–1481. doi:10.1111/obes.12377
* Li, T.-H. (2022). Quantile Fourier transform, quantile series, and nonparametric estimation of quantile spectra. arXiv:2211.05844
* Mayer, A., Wied, D. and Troster, V. (2025). *Journal of Econometrics* 249, 105992. doi:10.1016/j.jeconom.2025.105992
* Patton, A., Politis, D. N. and White, H. (2009). *Econometric Reviews* 28(4), 372–375. doi:10.1080/07474930802459016
* Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). *Journal of Applied Econometrics* 16(3), 289–326. doi:10.1002/jae.616
* Politis, D. N. and Romano, J. P. (1994). *JASA* 89(428), 1303–1313. doi:10.1080/01621459.1994.10476870
