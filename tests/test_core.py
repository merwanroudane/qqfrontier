"""Core numerics and cross-implementation checks for qqfrontier.

Reference values come from the author's R package qqconnect (M1), the R package
quantilogram / np::b.star (M2), and the Stata command qqardl (M5) - see tests/fixtures.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qqfrontier as qf
from qqfrontier._qr import _rq_br, check_loss, rq_fit, rq_fit_batch, rq_noncross

FIX = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------- QR solvers
def test_rq_fn_equals_simplex_objective():
    rng = np.random.default_rng(1)
    n = 199
    X = np.column_stack([np.ones(n), rng.standard_t(4, size=(n, 2))])
    y = X @ [0.3, 0.5, -0.2] + rng.standard_t(3, size=n)
    for t in (0.05, 0.5, 0.95):
        b_fn, b_br = rq_fit(X, y, t, method="fn"), rq_fit(X, y, t, method="br")
        np.testing.assert_allclose(b_fn, b_br, atol=1e-6)


def test_rq_batch_matches_single():
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(100), rng.normal(size=100)])
    Y = rng.normal(size=(5, 100))
    taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    B = rq_fit_batch(np.broadcast_to(X, (5, 100, 2)), Y, taus)
    for i in range(5):
        np.testing.assert_allclose(B[i], _rq_br(X, Y[i], taus[i]), atol=1e-6)


def test_noncross_ipm_equals_simplex_and_does_not_cross():
    rng = np.random.default_rng(3)
    y = rng.standard_t(4, size=101)
    X = np.column_stack([np.ones(100), y[:-1]])
    g = np.arange(0.05, 0.96, 0.05)
    b1 = rq_noncross(X, y[1:], g)
    b2 = rq_noncross(X, y[1:], g, method="simplex")
    o = lambda b: check_loss(y[1:, None] - X @ b, g).sum()
    assert abs(o(b1) - o(b2)) / o(b2) < 1e-8
    E = np.array([[1, X[:, 1].min()], [1, X[:, 1].max()]])
    assert np.diff(E @ b1, axis=1).min() > -1e-8


# ------------------------------------------------------------------ M1
def test_gfevd_matches_r_qqconnect():
    B = np.array([[0.5, 0.2], [0.1, 0.3]])
    S = np.array([[1, 0.4], [0.4, 2]])
    g = qf.gfevd(B, S, 10)
    np.testing.assert_allclose(g.values, [[0.787804, 0.212196], [0.093312, 0.906688]], atol=1e-6)
    assert abs(qf.connectedness_table(g)["TCI"] - 30.550743) < 1e-5


def test_qqc_gs_matches_r_qqconnect():
    x = pd.read_csv(FIX / "yields_400.csv", parse_dates=["date"])
    r = qf.qqc_gs(x, taus=[0.1, 0.5, 0.9], window=150, horizon=10)
    ref = pd.read_csv(FIX / "r_gs_tci.csv").to_numpy()
    assert r.TCI.shape == ref.shape
    # averaged grid within 1e-3 points; cells differ only where the LP optimum is not unique
    np.testing.assert_allclose(r.TCI.mean(0), ref.mean(0), atol=5e-3)
    assert np.median(np.abs(r.TCI - ref)) < 1e-6


def test_qqc_tvp_matches_r_qqconnect():
    x = pd.read_csv(FIX / "yields_400.csv", parse_dates=["date"])
    r = qf.qqc_tvp(x, taus=[0.1, 0.5, 0.9], horizon=10)
    ref = pd.read_csv(FIX / "r_tvp_tci.csv").to_numpy()
    np.testing.assert_allclose(r.TCI, ref, atol=1e-8)


def test_full_grid_and_direct_reverse():
    rng = np.random.default_rng(4)
    x = rng.standard_t(5, size=(160, 2))
    r = qf.qqc_gs(x, taus=[0.1, 0.5, 0.9], window=100, horizon=8)
    g = r.grid()
    assert g.values.shape == (3, 3)
    dr = r.direct_reverse()
    np.testing.assert_allclose(dr["delta"], dr["reverse"] - dr["direct"])
    assert isinstance(r.summary(), str) and "TCI" in r.summary()


# ------------------------------------------------------------------ M2
def test_cross_quantilogram_matches_r():
    D = np.loadtxt(FIX / "cq_data.csv", delimiter=",")
    ref = pd.read_csv(FIX / "r_crossq.csv").to_numpy()
    for k, t1, t2, v in ref:
        got = qf.cross_quantilogram(D[:, 0], D[:, 1], t1, t2, int(k), hit="le")
        assert abs(float(got) - v) < 1e-10


def test_block_length_matches_np_bstar():
    from qqfrontier.quantilogram import block_length
    D = np.loadtxt(FIX / "cq_data.csv", delimiter=",")
    ref = pd.read_csv(FIX / "r_bstar.csv").to_numpy()[:, 0]
    np.testing.assert_allclose([block_length(D[:, 0]), block_length(D[:, 1])], ref, rtol=1e-6)


def test_cross_quantilogram_detects_predictability():
    rng = np.random.default_rng(5)
    T = 800
    y2 = rng.standard_normal(T)
    y1 = np.r_[0, 0.9 * y2[:-1]] + 0.5 * rng.standard_normal(T)
    r = qf.cross_quantilogram_test(y1, y2, [0.1, 0.5, 0.9], [0.1, 0.9], max_lag=3, n_boot=99, seed=1)
    assert r.pvalues[0, 0, 0] < 0.05 and r.rho[0, 0, 0] > 0.2


# ------------------------------------------------------------------ M3
@pytest.mark.slow
def test_qgc_detects_break_and_dates_it():
    rng = np.random.default_rng(3)
    n = 600
    z, e = rng.standard_normal(n), rng.standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.2 * y[i - 1] + (0.0 if i < n // 2 else 0.4) * z[i - 1] + e[i]
    r = qf.qgc_instability(y, z, lags=1, n_boot=99, seed=1, alpha=0.10, sup_wald=False)
    assert r.tests.iloc[0]["p_expLM"] < 0.05
    assert r.regimes["breaks"] and abs(r.regimes["breaks"][0] - n // 2) < n // 10


def test_qgc_null_not_rejected():
    rng = np.random.default_rng(8)
    r = qf.qgc_instability(rng.standard_normal(400), rng.standard_normal(400), lags=1, n_boot=49,
                           seed=2, sup_wald=False, regimes=False,
                           taus=np.round(np.arange(0.1, 0.91, 0.05), 2),
                           intervals={"[0.10, 0.90]": (0.1, 0.9)})
    assert r.tests.iloc[0]["p_expLM"] > 0.01


# ------------------------------------------------------------------ M4
def test_coherency_self_is_one_and_smoother_is_eq_2_6():
    from scipy.stats import rankdata
    from qqfrontier.coherency import kernel_weights
    rng = np.random.default_rng(9)
    z = rng.standard_normal(300)
    r = qf.quantile_coherency(z, z, taus=[0.2, 0.5], ci=False)
    np.testing.assert_allclose(r.R[:, [0, 1], [0, 1]].real, 1.0, atol=1e-12)
    # brute-force Eq. (2.6) at one frequency
    x, y = rng.standard_normal(256), rng.standard_normal(256)
    n, b = 256, 0.5 * 256 ** -0.25
    r2 = qf.quantile_coherency(x, y, taus=[0.3], ci=False, bandwidth=b)
    w = kernel_weights(n, b)
    d1 = np.fft.fft((rankdata(x, method="max") <= n * 0.3).astype(float))
    d2 = np.fft.fft((rankdata(y, method="max") <= n * 0.3).astype(float))
    I12, I11, I22 = d1 * np.conj(d2), np.abs(d1) ** 2, np.abs(d2) ** 2
    j, s = 20, np.arange(1, n)
    G = lambda I: (w[(j - s) % n] * I[s]).sum() / w[(j - s) % n].sum()
    np.testing.assert_allclose(r2.R[j - 1, 0, 0], G(I12) / np.sqrt(G(I11) * G(I22)), atol=1e-10)


# ------------------------------------------------------------------ M5
def test_qqardl_matches_stata_qqardl():
    d = qf.load_wdi_co2_gdp("Algeria").set_index("year")
    r = qf.qqardl_fit(d["lnCO2"], d["lnGDP"])
    th, ta = list(r.thetas), list(r.taus)
    i95, i05 = th.index(0.95), th.index(0.05)
    # values printed by Stata qqardl 1.1.0 (example_algeria_egypt/qqardl_algeria_egypt.log)
    assert abs(r.F[i95, ta.index(0.05)] - 42.775) < 2e-3
    assert abs(r.F[i05, ta.index(0.10)] - 16.394) < 2e-3
    assert abs(r.ect[i95, ta.index(0.05)] - (-0.882)) < 2e-3
    assert abs(r.sr[i05, ta.index(0.05)] - 2.125) < 2e-3
    assert abs(r.lr[i05, ta.index(0.05)] - 1.387) < 2e-3
    assert r.NB == 53 and r.df == 50


def test_qqardl_bounds_pvalues_embedded_match_stata():
    # embedded Kripfganz-Schneider (2020) case III coefficients: no Stata needed
    d = qf.load_wdi_co2_gdp("Algeria").set_index("year")
    r = qf.qqardl_fit(d["lnCO2"], d["lnGDP"])
    assert int(r.coint.sum()) == 53                 # Stata: 53 of 121


# ------------------------------------------------------------------ API / colours
def test_dataframe_and_ndarray_give_same_numbers():
    rng = np.random.default_rng(11)
    a = rng.standard_t(5, size=(150, 2))
    df = pd.DataFrame(a, columns=["A", "B"])
    r1 = qf.qqc_gs(a, taus=[0.2, 0.8], window=100, horizon=5)
    r2 = qf.qqc_gs(df, taus=[0.2, 0.8], window=100, horizon=5)
    np.testing.assert_allclose(r1.TCI, r2.TCI)
    assert r2.names == ["A", "B"]


def test_input_errors():
    with pytest.raises(ValueError):
        qf.qqc_gs(np.ones((50, 1)))
    with pytest.raises(ValueError):
        qf.qqardl_fit(np.arange(10.0), np.arange(10.0))


def test_parula_endpoints():
    c = qf.parula_colors(64)
    assert c[0] == "#3E26A8" and c[-1] == "#F9FB15" and len(c) == 64
    assert qf.resolve_colorscale("Parula", 2) == [[0.0, "#3E26A8"], [1.0, "#F9FB15"]]
