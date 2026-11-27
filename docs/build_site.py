"""Build docs/index.html from the outputs written by examples/qqfrontier_empirical.py.

Run after the example:  python docs/build_site.py
Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import datetime as _dt
import html
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent
A = DOCS / "assets"
T = A / "tables"
VERSION = "0.1.1"
REPO = "https://github.com/merwanroudane/qqfrontier"
PYPI = "https://pypi.org/project/qqfrontier/0.1.1/"
SITE = "https://merwanroudane.github.io/qqfrontier/"


def read(name: str) -> str:
    p = T / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def md_table(md: str) -> str:
    """Pipe-table Markdown -> HTML table."""
    rows = [r.strip() for r in md.strip().splitlines() if r.strip().startswith("|")]
    if len(rows) < 2:
        return ""
    cells = lambda r: [c.strip().replace("**", "") for c in r.strip("|").split("|")]
    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    star = lambda s: re.sub(r"(\*+)$", r"<sup>\1</sup>", html.escape(s))
    out = ["<div class='tbl'><table><thead><tr>"] + [f"<th>{html.escape(h)}</th>" for h in head] + ["</tr></thead><tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td>{star(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def pre(text: str, cls: str = "out") -> str:
    return f"<pre class='{cls}'>{html.escape(text)}</pre>" if text else ""


def code(text: str) -> str:
    return f"<pre><code class='language-python'>{html.escape(text.strip())}</code></pre>"


def fig(stem: str, cap: str) -> str:
    if not (A / f"{stem}.png").exists():
        return ""
    return (f"<figure><img loading='lazy' src='assets/{stem}.png' alt='{html.escape(cap)}'>"
            f"<figcaption>{cap}</figcaption></figure>")


def interp(name: str) -> str:
    t = read(name)
    return f"<div class='interp'><b>Reading the results.</b> {html.escape(t)}</div>" if t else ""


MODULES = [
    ("M1", "--m1", "Quantile-on-quantile connectedness",
     "Spillovers between quantiles of several markets: Gabauer–Stenfors (2024), kernel Q-on-Q-QVAR (2026), TVP-QQVAR (2026)."),
    ("M2", "--m2", "Cross-quantilogram",
     "Directional predictability between quantiles with stationary-bootstrap inference (Han, Linton, Oka & Whang, 2016)."),
    ("M3", "--m3", "Robust quantile Granger causality",
     "Causality in quantiles robust to structural breaks, with dated regimes (Mayer, Wied & Troster, 2025)."),
    ("M4", "--m4", "Quantile coherency",
     "Dependence between quantiles across short, medium and long cycles (Baruník & Kley, 2019)."),
    ("M5", "--m5", "Quantile-on-quantile ARDL",
     "Bounds test, long run, short run and ECT on quantile series (Destek et al., 2025)."),
]

css_mod = lambda c: f"style='--c:var({c})'"

parts = []
parts.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>qqfrontier — frontier quantile-on-quantile econometrics in Python</title>
<meta name="description" content="qqfrontier: QQ connectedness, cross-quantilogram, robust quantile Granger causality, quantile coherency and QQARDL in Python. By Dr Merwan Roudane.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&family=JetBrains+Mono&family=Source+Serif+4:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<link rel="stylesheet" href="assets/style.css">
</head><body>
<div class="top"><div class="wrap"><span class="brand">qqfrontier</span>
<nav class="nav"><a href="#modules">Modules</a><a href="#install">Install</a><a href="#theory">Theory</a>
<a href="#example">Examples</a><a href="#api">Syntax</a><a href="#refs">References</a><a href="#author">Author</a>
<a href="{REPO}">GitHub</a><a href="{PYPI}">PyPI</a></nav></div></div>
<header class="hero"><div class="wrap">
<div class="pills"><span>v{VERSION}</span><span>MIT licence</span><span>Python ≥ 3.9</span><span>5 modules</span><span>validated against R &amp; Stata</span></div>
<h1>qqfrontier</h1>
<p class="lead">Frontier quantile-on-quantile econometrics in Python: QQ connectedness, the cross-quantilogram,
quantile Granger causality robust to instability, quantile coherency and the quantile-on-quantile ARDL —
with journal-quality tables, heatmaps and 3D surfaces.</p>
<div class="install"><span id="pipcmd">pip install "qqfrontier[full]"</span><button onclick="navigator.clipboard.writeText(document.getElementById('pipcmd').innerText)">copy</button></div><br>
<a class="btn" href="{REPO}">GitHub repository</a><a class="btn" href="{PYPI}">PyPI package</a>
<a class="btn ghost" href="https://github.com/merwanroudane/qqfrontier/blob/main/GUIDE.md">Step-by-step guide</a>
<div class="author-line">By <b>Dr Merwan Roudane</b> · <a style="color:#fff;text-decoration:underline" href="mailto:merwanroudane920@gmail.com">merwanroudane920@gmail.com</a> · <a style="color:#fff;text-decoration:underline" href="https://github.com/merwanroudane">github.com/merwanroudane</a></div>
</div></header>
<main class="wrap"><section style="padding-top:34px"><a href="{PYPI}"><img src="assets/banner.png" alt="qqfrontier banner"
 style="width:100%;border-radius:18px;box-shadow:var(--shadow);border:1px solid var(--border)"></a></section>""")

# modules
parts.append("<section id='modules'><h2><span class='kick'>What it does</span>Five methods, one consistent interface</h2>"
             "<p class='sub'>Every result object prints journal tables, exports LaTeX and Markdown, draws Parula heatmaps, "
             "static and interactive 3D surfaces, and writes a plain-language interpretation.</p><div class='grid g5'>")
for tag, c, name, desc in MODULES:
    parts.append(f"<div class='card mod' {css_mod(c)}><span class='tag'>{tag}</span><h4>{name}</h4><p>{desc}</p></div>")
parts.append("</div></section>")

# install
parts.append("""<section id="install"><h2><span class="kick">Installation</span>Install in one line</h2>
<div class="grid g2"><div class="card">""" + code('''pip install qqfrontier            # core
pip install "qqfrontier[plotly]"  # + interactive 3D surfaces
pip install "qqfrontier[fast]"    # + numba / joblib acceleration
pip install "qqfrontier[full]"    # everything (recommended)''') + """</div><div class="card">
<p>New to Python? Install <a href="https://www.anaconda.com/download">Anaconda</a>, open <b>Anaconda Prompt</b>
and type the line above. The <a href="https://github.com/merwanroudane/qqfrontier/blob/main/GUIDE.md">step-by-step guide</a>
shows how to load your own Excel or CSV file and write every line of code.</p>
<div class="note">Requirements: numpy, scipy, pandas, matplotlib, statsmodels (installed automatically).
Bounds-test p-values of M5 use the Kripfganz–Schneider (2020) critical values (case III), shipped with the package, as in Destek et al. (2025).</div>
</div></div></section>""")

# theory
parts.append(r"""<section id="theory"><h2><span class="kick">Theory</span>The five estimators in brief</h2>
<div class="grid g2 theory">
<div class="card mod" style="--c:var(--m1)"><span class="tag">M1</span><h4>QQ connectedness</h4>
Each equation of the QVAR is estimated at its own quantile, \(Q_{\tau_j}(y_{jt}\mid y_{t-1}) = c_j(\tau_j) + A_j(\tau_j) y_{t-1}\).
The generalized FEVD
$$\theta_{ij}(H)=\frac{\sigma_{jj}^{-1}\sum_{h=0}^{H-1}(e_i'\Psi_h\Sigma e_j)^2}{\sum_j \sigma_{jj}^{-1}\sum_{h}(e_i'\Psi_h\Sigma e_j)^2}$$
gives TO, FROM, NET and the adjusted total connectedness \(TCI=\tfrac{k}{k-1}\overline{TO}\) for every quantile configuration \((\tau_1,\tau_2)\).</div>
<div class="card mod" style="--c:var(--m2)"><span class="tag">M2</span><h4>Cross-quantilogram</h4>
$$\hat\rho_\tau(k)=\frac{\sum_{t}\psi_{\tau_1}(y_{1t}-\hat q_{1t})\,\psi_{\tau_2}(y_{2,t-k}-\hat q_{2,t-k})}{\sqrt{\sum_t\psi^2_{\tau_1}}\sqrt{\sum_t\psi^2_{\tau_2}}},\qquad \psi_a(u)=\mathbb 1[u<0]-a$$
with the Box–Ljung test \(\check Q^{(p)}_\tau=T(T+2)\sum_{k\le p}\hat\rho_\tau^2(k)/(T-k)\) and stationary-bootstrap critical values.</div>
<div class="card mod" style="--c:var(--m3)"><span class="tag">M3</span><h4>Robust quantile Granger causality</h4>
From the subgradient \(H_n(\lambda,\tau)=(X'X/n)^{-1/2}n^{-1/2}\sum_{i\le\lambda n}x_i\psi_\tau(y_i-w_i'\hat\alpha(\tau))\):
$$\mathrm{expLM}=\int_{\mathcal T}\exp\Big[\tfrac12\Big(\sup_\lambda\big\|R'(H_n(\lambda)-\lambda H_n(1))\big\|_\infty+\|R'H_n(1)\|_\infty\Big)\Big]d\tau$$
jointly tests no causality and parameter stability.</div>
<div class="card mod" style="--c:var(--m4)"><span class="tag">M4</span><h4>Quantile coherency</h4>
$$R(\omega;\tau_1,\tau_2)=\frac{f_{12}(\omega;\tau_1,\tau_2)}{\sqrt{f_{11}(\omega;\tau_1,\tau_1)f_{22}(\omega;\tau_2,\tau_2)}}$$
estimated from smoothed rank-based copula cross-periodograms, with pointwise confidence intervals.</div>
<div class="card mod" style="--c:var(--m5)"><span class="tag">M5</span><h4>QQARDL</h4>
On the quantile series of Li (2022), \(\Delta Y_t=a_0+d_0Y_{t-1}+d_1\Delta X_t+b_1X_{t-1}+e_t\) for every pair \((\tau,\theta)\):
bounds test of \(g_1=g_2=0\), long run \(-b_1/d_0\), short run \(d_1\), error correction \(d_0\).</div>
<div class="card"><h4>Faithful to the papers</h4><p>Every code step is mapped to the equation it implements in
<a href="https://github.com/merwanroudane/qqfrontier/blob/main/docs/methods.md">docs/methods.md</a>. M1 reproduces the R package
<i>qqconnect</i> and Fig. 1(a) of Bonaccolto et al. (2026) cell by cell; M2 equals <i>quantilogram</i>; M5 reproduces the Stata command
<i>qqardl</i> table by table.</p></div>
</div></section>""")

# examples
ex = ["<section id='example'><h2><span class='kick'>Complete applications on real data</span>Every table and figure, generated by one script</h2>"
      "<p class='sub'>Data shipped with the package: daily FRED changes of the US 2-year Treasury yield (US2Y) and the 10Y–2Y spread "
      "(US2Y10Y), 1998–2023 (public domain); World Bank WDI CO₂ and GDP per capita of Algeria and Egypt (CC BY 4.0). "
      "Script: <a href='https://github.com/merwanroudane/qqfrontier/blob/main/examples/qqfrontier_empirical.py'>examples/qqfrontier_empirical.py</a>.</p>"]
ex.append("<h3>Summary statistics</h3>" + code('''import qqfrontier as qf
yl = qf.load_us_yields(main_only=True)
print(qf.descriptive_table(yl[["US2Y", "US2Y10Y"]]))''') + pre(read("descriptive.txt")))

ex.append(f"<h3 class='mtitle' {css_mod('--m1')}><span class='tag'>M1</span> Quantile-on-quantile connectedness</h3>" + code('''x = yl[["date", "US2Y", "US2Y10Y"]]
taus = np.arange(0.05, 1.0, 0.10)
gs  = qf.qqc_gs(x, taus=taus, window=200, horizon=20)                  # Gabauer & Stenfors (2024)
qoq = qf.qqc_kernel(qf.load_us_yields()[["date","US2Y","US2Y10Y"]], taus=taus,
                    window=200, horizon=20, bandwidth=0.15, presample=100, n_jobs=8)   # BCG (2026)
tvp = qf.qqc_tvp(x, taus=taus, horizon=30)                               # TVP-QQVAR (2026)
print(gs.summary()); print(gs.interpret())
gs.plot("heatmap"); gs.plot("dynamic"); gs.plot("net"); gs.plot("surface"); gs.plot("panel")'''))
ex.append(md_table(read("m1_gs_summary.md")))
ex.append(interp("m1_interpretation.txt"))
ex.append("<div class='grid g2'>" + fig("m1_gs_heatmap", "Averaged dynamic quantile-on-quantile TCI (GS 2024, Fig. 2 layout).")
          + fig("m1_gs_dynamic", "Reverse, direct and ΔTCI over time (GS 2024, Fig. 3).") + "</div>")
ex.append(fig("m1_gs_panel", "Heatmap, 3D surface and contour of the averaged TCI."))
ex.append("<div class='grid g2'>" + fig("m1_gs_surface3d", "3D surface of the averaged TCI (Parula).")
          + fig("m1_gs_net", "NET connectedness of each series (blue: receiver, red: transmitter).") + "</div>")
if (A / "m1_gs_surface.html").exists():
    ex.append("<figure><iframe class='plotly' src='assets/m1_gs_surface.html' title='Interactive 3D TCI surface'></iframe>"
              "<figcaption>Interactive 3D surface: drag to rotate, hover for values.</figcaption></figure>")
ex.append(fig("m1_compare_bcg_fig1", "GS vs Q-on-Q, second moments vs covariance (Bonaccolto, Caporin & Gupta 2026, Fig. 1 layout)."))
ex.append("<h4>Decomposition of TCI differences (BCG 2026, Table A.2)</h4>" + md_table(read("m1_decomposition.md")))
ex.append("<div class='grid g2'>" + fig("m1_gs_time_heatmap", "Reversely related quantiles: TCI over time.")
          + fig("m1_tvp_dynamic", "TVP-QQVAR: reverse vs direct TCI.") + "</div>")

ex.append(f"<h3 class='mtitle' {css_mod('--m2')}><span class='tag'>M2</span> Cross-quantilogram: does the 2-year yield predict the spread?</h3>" + code('''y1 = yl.set_index("date")["US2Y10Y"]; y2 = yl.set_index("date")["US2Y"]
cq = qf.cross_quantilogram_test(y1, y2, tau1=[.05,.1,.2,.3,.5,.7,.8,.9,.95], tau2=[.1,.9],
                                max_lag=20, n_boot=1000, sup_range=np.arange(.10,.31,.02), seed=1)
print(cq.summary()); cq.plot("correlogram", tau2=0.10); cq.plot("heatmap", k=1)'''))
ex.append(md_table(read("m2_summary.md")))
ex.append(interp("m2_interpretation.txt"))
ex.append(fig("m2_correlogram_tau2_010", "Cross-quantilogram with the 95% bootstrap band for no predictability (HLOW Fig. 2a layout)."))
ex.append("<div class='grid g2'>" + fig("m2_portmanteau_tau2_010", "Box–Ljung Q(p) and bootstrap critical values.")
          + fig("m2_sup", "Sup-Box–Ljung test over τ ∈ [0.10, 0.30].") + "</div>")
ex.append("<div class='grid g2'>" + fig("m2_heatmap_lag1", "Quantile-on-quantile map of the cross-quantilogram at lag 1 (stars: bootstrap p-values).")
          + fig("m2_lag_surface", "3D surface over lag × τ₁.") + "</div>")

ex.append(f"<h3 class='mtitle' {css_mod('--m3')}><span class='tag'>M3</span> Quantile Granger causality robust to instability</h3>" + code('''gc = qf.qgc_instability(y1, y2, lags=1, n_boot=499, seed=3, alpha=0.10)
print(gc.summary()); gc.plot("cusum"); gc.plot("lm"); gc.plot("cusum_map")
print(qf.qgc_table(y1, y2, lags=(1, 3, 6), n_boot=249))      # MWT Table 3 layout'''))
ex.append(pre(read("m3_summary.txt")))
ex.append("<h4>p-values (%) by lag order and quantile range (MWT 2025, Table 3 layout)</h4>" + md_table(read("m3_table3_layout.md")))
ex.append(interp("m3_interpretation.txt"))
ex.append("<div class='grid g2'>" + fig("m3_cusum", "CUSUM detector with break dates; Granger-causality regimes shaded (MWT Fig. 2).")
          + fig("m3_lm_tau", "LM(τ) and its instability and causality components.") + "</div>")
ex.append("<div class='grid g2'>" + fig("m3_cusum_map", "Time × quantile map of the instability detector.")
          + fig("m3_coefficients", "Quantile Granger coefficients γ(τ).") + "</div>")

ex.append(f"<h3 class='mtitle' {css_mod('--m4')}><span class='tag'>M4</span> Quantile coherency</h3>" + code('''coh = qf.quantile_coherency(yl["US2Y"], yl["US2Y10Y"], taus=np.arange(.05,.96,.05),
                            periods={"W": 5, "M": 22, "Y": 250})
print(coh.summary()); coh.plot("frequency"); coh.plot("quantile"); coh.plot("surface", period="M")'''))
ex.append(pre(read("m4_summary.txt")))
ex.append(interp("m4_interpretation.txt"))
ex.append("<div class='grid g2'>" + fig("m4_frequency_diag", "Re R across frequencies with 95% CIs; top axis: weekly, monthly, yearly (BK Fig. 3).")
          + fig("m4_quantile", "Re R along τ₁ = τ₂ at fixed periods (BK Fig. 3, right).") + "</div>")
ex.append("<div class='grid g2'>" + fig("m4_heatmap_monthly", "Quantile-on-quantile coherency at the monthly period.")
          + fig("m4_surface_monthly", "3D surface of the monthly coherency.") + "</div>")
ex.append(fig("m4_freq_surface", "3D surface over frequency × quantile."))

ex.append(f"<h3 class='mtitle' {css_mod('--m5')}><span class='tag'>M5</span> Quantile-on-quantile ARDL: CO₂ and growth in Algeria</h3>" + code('''d = qf.load_wdi_co2_gdp("Algeria").set_index("year")
r = qf.qqardl_fit(d["lnCO2"], d["lnGDP"])
print(r.summary()); r.plot("heatmap", stat="lr"); r.plot("surface", stat="lr"); r.plot("coint")'''))
ex.append(pre(read("m5_DZ_summary.txt")))
ex.append(interp("m5_DZ_interpretation.txt"))
ex.append("<div class='grid g2'>" + fig("m5_DZ_lr_heatmap", "Long-run coefficients, cointegrated pairs (stars: p-values).")
          + fig("m5_DZ_lr_surface", "3D surface of the long-run coefficients.") + "</div>")
ex.append("<div class='grid g2'>" + fig("m5_DZ_coint", "Bounds-test F; framed cells are cointegrated (p < 0.10).")
          + fig("m5_DZ_qser", "Observed ln CO₂ and its quantile series (Li, 2022).") + "</div>")
ex.append("<div class='grid g2'>" + fig("m5_DZ_sr_heatmap", "Short-run coefficients.") + fig("m5_DZ_ect_heatmap", "Error-correction term.") + "</div>")
ex.append(f"<h4>Egypt</h4>" + interp("m5_EG_interpretation.txt") +
          "<div class='grid g2'>" + fig("m5_EG_lr_heatmap", "Egypt: long-run coefficients.") + fig("m5_EG_lr_surface", "Egypt: 3D long-run surface.") + "</div>")
ex.append("</section>")
parts += ex

# API
api = [("connectedness", "qqc_gs(x, taus, nlag=1, window=200, horizon=20, cov_type='uncentered')",
        "qqc_kernel(x, taus, window=200, horizon=20, bandwidth=0.15, presample=100, noncross='bondell', n_jobs=1)",
        "qqc_tvp(x, taus, horizon=30, delta=0.99, kappa=0.99)", "gfevd, connectedness_table, decompose_tci, compare_heatmaps, cond_pit, dqlm, sim_qqvar, mc_qqvar"),
       ("quantilogram", "cross_quantilogram(y1, y2, tau1, tau2, k)",
        "cross_quantilogram_test(y1, y2, tau1, tau2, max_lag=20, n_boot=1000, sup_range=None, self_normalized=False)",
        "partial_cross_quantilogram(y1, y2, controls, tau1, tau2, tau_z, k=1)", "block_length, stationary_bootstrap_index"),
       ("causality", "qgc_instability(y, x, lags=1, taus=0.05..0.95, n_boot=499, method='bootstrap', moment=1, regimes=True)",
        "qgc_table(y, x, lags=(1,3,6,9))", "qgc_map(y, x, lags=(1,...,6))", "simulate_null"),
       ("coherency", "quantile_coherency(x, y, taus, bandwidth=0.5 n^-1/4, kernel='epanechnikov', ci=True, periods={'W':5,'M':22,'Y':250})",
        "", "", "kernel_weights"),
       ("qqardl", "qqardl_fit(y, x, tau=(.05,.10,...,.95), theta=None, lags=(1,1), btest='eq1', vce='ols', lrtype='multiplier')",
        "qser(y, taus)", "ardl_bounds_pvalue(F, n, k=1, sr=2)", "")]
parts.append("<section id='api'><h2><span class='kick'>Syntax</span>Main functions</h2><p class='sub'>Every result has "
             "<code>.summary()</code>, <code>.to_latex()</code>, <code>.to_markdown()</code>, <code>.to_frame()</code>, "
             "<code>.interpret()</code> and <code>.plot(kind)</code>. Full argument lists are in the docstrings "
             "(<code>help(qf.qqc_gs)</code>) and in the <a href='https://github.com/merwanroudane/qqfrontier/blob/main/GUIDE.md'>guide</a>.</p>"
             "<div class='grid g2'>")
for (mod, a, b, c, d), (tag, col, name, _) in zip(api, MODULES):
    body = "\n".join(s for s in (a, b, c) if s)
    parts.append(f"<div class='card mod' {css_mod(col)}><span class='tag'>{tag}</span><h4>qqfrontier.{mod}</h4>{code(body)}"
                 + (f"<p>Also: <code>{d}</code></p>" if d else "") + "</div>")
parts.append("</div></section>")

# references
refs = [("Baruník, J. and Kley, T. (2019). Quantile coherency. The Econometrics Journal, 22(2), 131–152.", "10.1093/ectj/utz002"),
        ("Bonaccolto, G., Caporin, M. and Gupta, R. (2026). Rethinking quantile-on-quantile connectedness. Finance Research Letters, 114, 111095.", None),
        ("Bondell, H. D., Reich, B. J. and Wang, H. (2010). Noncrossing quantile regression curve estimation. Biometrika, 97(4), 825–838.", "10.1093/biomet/asq048"),
        ("Candelo-Viáfara, J. M. and Orrego-Reyes, J. E. (2026). A TVP quantile-on-quantile VAR approach. SSRN.", "10.2139/ssrn.7434976"),
        ("Chatziantoniou, I., Gabauer, D. and Stenfors, A. (2021). Economics Letters, 204, 109891.", "10.1016/j.econlet.2021.109891"),
        ("Destek, M. A., Radulescu, M., Ozkan, O. and Balsalobre-Lorente, D. (2025). Journal of Environmental Management, 389, 126089.", "10.1016/j.jenvman.2025.126089"),
        ("Diebold, F. X. and Yilmaz, K. (2012). International Journal of Forecasting, 28(1), 57–66.", "10.1016/j.ijforecast.2011.02.006"),
        ("Gabauer, D. and Stenfors, A. (2024). Finance Research Letters, 60, 104852.", "10.1016/j.frl.2023.104852"),
        ("Han, H., Linton, O., Oka, T. and Whang, Y.-J. (2016). The cross-quantilogram. Journal of Econometrics, 193(1), 251–270.", "10.1016/j.jeconom.2016.03.001"),
        ("Kripfganz, S. and Schneider, D. C. (2020). Oxford Bulletin of Economics and Statistics, 82(6), 1456–1481.", "10.1111/obes.12377"),
        ("Li, T.-H. (2022). Quantile Fourier transform, quantile series, and nonparametric estimation of quantile spectra. arXiv:2211.05844.", "10.48550/arXiv.2211.05844"),
        ("Mayer, A., Wied, D. and Troster, V. (2025). Quantile Granger causality in the presence of instability. Journal of Econometrics, 249, 105992.", "10.1016/j.jeconom.2025.105992"),
        ("Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). Journal of Applied Econometrics, 16(3), 289–326.", "10.1002/jae.616"),
        ("Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. JASA, 89(428), 1303–1313.", "10.1080/01621459.1994.10476870")]
parts.append("<section id='refs'><h2><span class='kick'>References</span>Papers implemented</h2><div class='card'><ol>")
for t, d in refs:
    parts.append(f"<li>{html.escape(t)}" + (f" <a href='https://doi.org/{d}'>doi:{d}</a>" if d else "") + "</li>")
parts.append("</ol></div></section>")

# author
bib = """@software{roudane2026qqfrontier,
  author  = {Roudane, Merwan},
  title   = {qqfrontier: Frontier Quantile-on-Quantile Econometrics in Python},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/merwanroudane/qqfrontier}
}"""
parts.append(f"""<section id="author"><h2><span class="kick">Author</span>Dr Merwan Roudane</h2>
<div class="card author"><div class="avatar">MR</div><div>
<p style="margin:0"><b>Dr Merwan Roudane</b> — econometrician, author of R, Python and Stata packages for quantile-on-quantile and
frequency-domain econometrics.<br>✉ <a href="mailto:merwanroudane920@gmail.com">merwanroudane920@gmail.com</a> ·
<a href="https://github.com/merwanroudane">github.com/merwanroudane</a></p>
<div class="links"><a href="{REPO}">qqfrontier on GitHub</a><a href="{PYPI}">qqfrontier on PyPI</a><a href="{SITE}">Documentation</a>
<a href="https://cran.r-project.org/package=QuantileOnQuantile">R: QuantileOnQuantile</a><a href="https://cran.r-project.org/package=mqqr">R: mqqr</a>
<a href="https://cran.r-project.org/package=qqkrls">R: qqkrls</a><a href="https://cran.r-project.org/package=mqqcause">R: mqqcause</a>
<a href="https://pypi.org/project/wqr/">Python: wqr</a><a href="https://pypi.org/project/mqqr/">Python: mqqr</a>
<a href="https://pypi.org/project/qqkrls/">Python: qqkrls</a></div></div></div>
<h3>Cite this package</h3>{code(bib)}<p>Please also cite the original paper of each method you use (see References).</p></section>""")
parts.append(f"""</main><footer>© 2026 Dr Merwan Roudane · MIT Licence · qqfrontier {VERSION} · updated {_dt.date.today():%d %B %Y}</footer>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}}]}})"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>hljs.highlightAll();</script></body></html>""")

(DOCS / "index.html").write_text("\n".join(parts), encoding="utf-8")
(DOCS / ".nojekyll").write_text("", encoding="utf-8")
print("wrote", DOCS / "index.html")
