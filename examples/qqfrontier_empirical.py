"""qqfrontier - complete empirical application on real data.

Reproduces, with one script, every table and figure shown in the README and on the
documentation site:

  0. Summary statistics (Gabauer and Stenfors, 2024, Table 1 layout)
  1. M1  Quantile-on-quantile connectedness of the US 2-year yield and the 10Y-2Y spread
         (GS 2024; Bonaccolto, Caporin and Gupta 2026; TVP-QQVAR 2026)
  2. M2  Cross-quantilogram: does the 2-year yield predict the spread? (Han et al., 2016)
  3. M3  Quantile Granger causality robust to instability (Mayer, Wied and Troster, 2025)
  4. M4  Quantile coherency of the two yield series (Barunik and Kley, 2019)
  5. M5  Quantile-on-quantile ARDL: CO2 and GDP in Algeria and Egypt (Destek et al., 2025)

Run:  python examples/qqfrontier_empirical.py          (about 15-25 minutes on 8 cores)
      python examples/qqfrontier_empirical.py --quick  (smaller bootstrap / grids, ~3 minutes)

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import argparse
import pickle
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import qqfrontier as qf  # noqa: E402

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "assets"))
ap.add_argument("--cache", default="", help="folder to cache fitted results (off by default)")
ap.add_argument("--jobs", type=int, default=8)
args = ap.parse_args()
OUT = Path(args.out)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
CACHE = Path(args.cache) if args.cache else None
if CACHE:
    CACHE.mkdir(parents=True, exist_ok=True)
B_CQ, B_GC = (200, 99) if args.quick else (1000, 499)


def cached(name, fun):
    f = CACHE / f"{name}{'_quick' if args.quick else ''}.pkl" if CACHE else None
    if f is not None and f.exists():
        return pickle.load(open(f, "rb"))
    t = time.time()
    r = fun()
    print(f"  [{name}] {time.time() - t:6.1f}s", flush=True)
    if f is not None:
        pickle.dump(r, open(f, "wb"))
    return r


def fig(f, stem):
    qf.save_figure(f, str(OUT / stem))
    plt.close(f)


def table(text, stem, latex=None, md=None):
    (OUT / "tables" / f"{stem}.txt").write_text(text, encoding="utf-8")
    if latex:
        (OUT / "tables" / f"{stem}.tex").write_text(latex, encoding="utf-8")
    if md:
        (OUT / "tables" / f"{stem}.md").write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- data
print("0. Data and summary statistics")
yl_all = qf.load_us_yields()                                     # with 100-day pre-sample
yl = qf.load_us_yields(main_only=True)                            # 6,253 daily changes (GS 2024)
x2 = yl[["date", "US2Y", "US2Y10Y"]]
desc = qf.descriptive_table(yl[["US2Y", "US2Y10Y"]])
print(desc)
table(desc.summary(), "descriptive", desc.to_latex(), desc.to_markdown())

# ------------------------------------------------------------------------- M1
print("1. Quantile-on-quantile connectedness")
taus = np.round(np.arange(0.05, 1.0, 0.10), 2)
gs_u = cached("gs_u", lambda: qf.qqc_gs(x2, taus=taus, window=200, horizon=20, keep_fevd=True))
gs_c = cached("gs_c", lambda: qf.qqc_gs(x2, taus=taus, window=200, horizon=20, cov_type="centered"))
xk = yl_all[["date", "US2Y", "US2Y10Y"]]
qoq_c = cached("qoq_c", lambda: qf.qqc_kernel(xk, taus=taus, window=200, horizon=20, bandwidth=0.15,
                                               presample=100, n_jobs=args.jobs, progress=True))
qoq_u = cached("qoq_u", lambda: qf.qqc_kernel(xk, taus=taus, window=200, horizon=20, bandwidth=0.15,
                                               cov_type="uncentered", pit=qoq_c.pit.to_numpy(),
                                               presample=100))
tvp = cached("tvp", lambda: qf.qqc_tvp(x2, taus=taus, horizon=30, delta=0.99, kappa=0.99, n_jobs=args.jobs))

print(gs_u.summary()[:1500])
table(gs_u.summary(), "m1_gs_summary", gs_u.to_latex(caption="Averaged dynamic quantile-on-quantile total "
                                                              "connectedness (Gabauer and Stenfors, 2024)"),
      gs_u.to_markdown())
table(qoq_c.summary(), "m1_qoq_summary", qoq_c.to_latex(caption="Averaged TCI, Q-on-Q-QVAR (Bonaccolto et al., 2026)"),
      qoq_c.to_markdown())
table(tvp.summary(), "m1_tvp_summary", tvp.to_latex(caption="Averaged TCI, TVP-QQVAR"), tvp.to_markdown())
dec = qf.decompose_tci(gs_u, gs_c, qoq_u, qoq_c)
table(dec.to_string(), "m1_decomposition", dec.to_latex(float_format="%.2f"), dec.to_markdown())
ct = gs_u.connectedness_table(at=(0.05, 0.95))
table(ct.to_string(), "m1_dy_table_005_095", ct.to_latex(float_format="%.2f"), ct.to_markdown())
(OUT / "tables" / "m1_interpretation.txt").write_text(gs_u.interpret(), encoding="utf-8")

fig(gs_u.plot("heatmap")[0], "m1_gs_heatmap")
fig(gs_u.plot("dynamic")[0], "m1_gs_dynamic")
fig(gs_u.plot("net")[0], "m1_gs_net")
fig(gs_u.plot("panel")[0], "m1_gs_panel")
fig(gs_u.plot("surface")[0], "m1_gs_surface3d")
fig(gs_u.plot("time_heatmap", which="antidiagonal")[0], "m1_gs_time_heatmap")
fig(gs_u.plot("paths")[0], "m1_gs_paths")
fig(qf.compare_heatmaps(**{"(a) GS: second moments": gs_u, "(b) GS: covariance": gs_c,
                           "(c) Q-on-Q: second moments": qoq_u, "(d) Q-on-Q: covariance": qoq_c})[0],
    "m1_compare_bcg_fig1")
fig(qoq_c.plot("dynamic")[0], "m1_qoq_dynamic")
fig(tvp.plot("heatmap")[0], "m1_tvp_heatmap")
fig(tvp.plot("dynamic")[0], "m1_tvp_dynamic")
gs_u.grid().plot_surface().write_html(str(OUT / "m1_gs_surface.html"), include_plotlyjs="cdn")

# ------------------------------------------------------------------------- M2
print("2. Cross-quantilogram")
y1 = yl.set_index("date")["US2Y10Y"]
y2 = yl.set_index("date")["US2Y"]
t1 = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
cq = cached("cq", lambda: qf.cross_quantilogram_test(y1, y2, t1, [0.10, 0.90], max_lag=20, n_boot=B_CQ,
                                                     sup_range=np.round(np.arange(0.10, 0.301, 0.02), 2), seed=1))
qgrid = np.round(np.arange(0.05, 0.951, 0.10), 2)
cq_grid = cached("cq_grid", lambda: qf.cross_quantilogram_test(y1, y2, qgrid, qgrid, max_lag=5,
                                                               n_boot=max(B_CQ // 2, 100), seed=2))
print(cq.summary(tau2=[0.10], lags=[1, 2, 5, 10, 20])[:2500])
table(cq.summary(), "m2_summary", cq.to_latex(tau2=0.10), cq.to_markdown(tau2=0.10))
(OUT / "tables" / "m2_interpretation.txt").write_text(cq_grid.interpret(), encoding="utf-8")
fig(cq.plot("correlogram", tau2=0.10)[0], "m2_correlogram_tau2_010")
fig(cq.plot("correlogram", tau2=0.90)[0], "m2_correlogram_tau2_090")
fig(cq.plot("portmanteau", tau2=0.10)[0], "m2_portmanteau_tau2_010")
fig(cq.plot("sup")[0], "m2_sup")
fig(cq_grid.plot("heatmap", k=1)[0], "m2_heatmap_lag1")
fig(cq_grid.plot("surface", k=1)[0], "m2_surface_lag1")
fig(cq.plot("lag_surface", tau2=0.10)[0], "m2_lag_surface")

# ------------------------------------------------------------------------- M3
print("3. Quantile Granger causality robust to instability")
gc = cached("gc", lambda: qf.qgc_instability(y1, y2, lags=1, n_boot=B_GC, seed=3, alpha=0.10))
print(gc.summary())
gct = cached("gc_table", lambda: qf.qgc_table(y1, y2, lags=(1, 3, 6), n_boot=max(B_GC // 2, 99), seed=4))
table(gc.summary(), "m3_summary", gc.to_latex(), None)
table(gct.to_string(), "m3_table3_layout", gct.to_latex(float_format="%.3f"), gct.to_markdown())
(OUT / "tables" / "m3_interpretation.txt").write_text(gc.interpret(), encoding="utf-8")
fig(gc.plot("cusum")[0], "m3_cusum")
fig(gc.plot("lm")[0], "m3_lm_tau")
fig(gc.plot("cusum_map")[0], "m3_cusum_map")
fig(gc.plot("coefficients")[0], "m3_coefficients")

# ------------------------------------------------------------------------- M4
print("4. Quantile coherency")
coh = cached("coh", lambda: qf.quantile_coherency(yl["US2Y"], yl["US2Y10Y"],
                                                  taus=np.round(np.arange(0.05, 0.951, 0.05), 2),
                                                  periods={"W": 5, "M": 22, "Y": 250}))
print(coh.summary())
table(coh.summary(), "m4_summary", coh.to_latex(), None)
(OUT / "tables" / "m4_interpretation.txt").write_text(coh.interpret(), encoding="utf-8")
fig(coh.plot("frequency", pairs=[(0.5, 0.5), (0.05, 0.05), (0.95, 0.95)])[0], "m4_frequency_diag")
fig(coh.plot("frequency", pairs=[(0.05, 0.95), (0.95, 0.05)])[0], "m4_frequency_opposite")
fig(coh.plot("quantile")[0], "m4_quantile")
fig(coh.plot("heatmap", period="M")[0], "m4_heatmap_monthly")
fig(coh.plot("surface", period="M")[0], "m4_surface_monthly")
fig(coh.plot("freq_surface", max_freq=0.25)[0], "m4_freq_surface")

# ------------------------------------------------------------------------- M5
print("5. Quantile-on-quantile ARDL")
for country, code in (("Algeria", "DZ"), ("Egypt", "EG")):
    d = qf.load_wdi_co2_gdp(country).set_index("year")
    r = cached(f"qqardl_{code}", lambda d=d: qf.qqardl_fit(d["lnCO2"], d["lnGDP"]))
    print(r.summary()[:1200])
    table(r.summary(), f"m5_{code}_summary", r.to_latex("lr", caption=f"QQARDL long-run coefficients, {country}"),
          r.to_markdown("lr"))
    (OUT / "tables" / f"m5_{code}_interpretation.txt").write_text(r.interpret(), encoding="utf-8")
    fig(r.plot("heatmap", stat="lr")[0], f"m5_{code}_lr_heatmap")
    fig(r.plot("surface", stat="lr")[0], f"m5_{code}_lr_surface")
    fig(r.plot("coint")[0], f"m5_{code}_coint")
    fig(r.plot("heatmap", stat="sr")[0], f"m5_{code}_sr_heatmap")
    fig(r.plot("heatmap", stat="ect")[0], f"m5_{code}_ect_heatmap")
    fig(r.plot("qser")[0], f"m5_{code}_qser")
    fig(r.plot("diagonal", stat="lr")[0], f"m5_{code}_lr_diagonal")
print("done ->", OUT)
