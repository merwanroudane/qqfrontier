"""Draw the README / site banner (docs/assets/banner.png) from real qqfrontier output.

Light, colourful design: soft gradient background, the averaged quantile-on-quantile
TCI surface of the US yield application (Parula) and module chips.
Run:  python docs/make_banner.py
Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from qqfrontier import parula_cmap  # noqa: E402

DOCS = Path(__file__).resolve().parent
OUT = DOCS / "assets" / "banner.png"

# averaged QQ total connectedness of US2Y and US2Y10Y (rows: US2Y 0.05..0.95), from the M1 example
md = (DOCS / "assets" / "tables" / "m1_gs_summary.md")
if md.exists():
    rows = [r for r in md.read_text(encoding="utf-8").splitlines() if r.startswith("| **")]
    Z = np.array([[float(c.strip().rstrip("*")) for c in r.strip("|").split("|")[1:]] for r in rows])[::-1]
else:  # fallback: same values as Fig. 1(a) of Bonaccolto, Caporin and Gupta (2026)
    Z = np.array([[64.9, 47.5, 31.3, 17.0, 9.9, 11.3, 23.7, 40.1, 58.5, 73.0],
                  [46.8, 33.2, 22.6, 14.1, 11.9, 13.2, 21.4, 32.7, 46.0, 57.2],
                  [30.2, 22.2, 17.0, 13.4, 14.0, 15.2, 19.9, 26.2, 34.1, 40.7],
                  [13.8, 12.0, 12.1, 12.9, 15.4, 16.8, 18.8, 20.8, 22.5, 23.1],
                  [7.2, 9.4, 11.8, 14.2, 16.3, 17.2, 17.0, 15.9, 13.7, 10.6],
                  [12.0, 14.2, 15.7, 16.8, 17.0, 16.6, 15.1, 12.7, 9.8, 7.4],
                  [22.9, 22.1, 20.6, 18.8, 16.5, 15.4, 13.9, 12.3, 11.9, 13.4],
                  [40.9, 33.9, 27.0, 20.4, 14.7, 13.4, 14.1, 16.6, 22.7, 31.2],
                  [56.4, 45.2, 33.7, 22.3, 12.9, 11.1, 14.4, 21.4, 33.7, 47.7],
                  [71.6, 56.6, 40.7, 24.4, 11.3, 9.5, 17.2, 30.3, 49.3, 66.8]])
q = np.round(np.arange(0.05, 1.0, 0.10), 2)

W, H = 16.0, 5.4
fig = plt.figure(figsize=(W, H), dpi=100)
# --- soft light gradient background (white -> lavender -> mint) ---
bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
gx = np.linspace(0, 1, 512)
gy = np.linspace(0, 1, 160)
GX, GY = np.meshgrid(gx, gy)
c1, c2, c3 = np.array([1.0, 1.0, 1.0]), np.array([0.93, 0.92, 1.0]), np.array([0.88, 0.97, 0.96])
t = GX[..., None]
img = np.where(t < 0.55, c1 + (c2 - c1) * (t / 0.55), c2 + (c3 - c2) * ((t - 0.55) / 0.45))
img = img * (1 - 0.03 * GY[..., None])
bg.imshow(img, aspect="auto", extent=[0, 1, 0, 1], origin="lower")
for (x, y, r, col, a) in [(0.06, 0.95, 0.22, "#a5b4fc", .25), (0.42, 0.02, 0.18, "#99f6e4", .22),
                          (0.97, 0.9, 0.2, "#fde68a", .28)]:
    bg.add_patch(plt.Circle((x, y), r, color=col, alpha=a, transform=bg.transAxes, lw=0))
bg.set_axis_off()

# --- text block ---
fig.text(0.045, 0.80, "qqfrontier", fontsize=58, fontweight="bold", color="#3730a3", family="DejaVu Sans")
fig.text(0.047, 0.70, "Frontier quantile-on-quantile econometrics in Python", fontsize=18.5,
         color="#1e293b", family="DejaVu Sans")
fig.text(0.047, 0.635, "Journal-ready tables  •  Parula heatmaps  •  3D surfaces  •  validated against R & Stata",
         fontsize=12.5, color="#475569", family="DejaVu Sans")
chips = [("M1  QQ connectedness", "#4f46e5"), ("M2  Cross-quantilogram", "#0284c7"),
         ("M3  Robust QQ Granger causality", "#e11d48"), ("M4  Quantile coherency", "#0d9488"),
         ("M5  QQARDL", "#d97706")]
x0, y0 = 0.047, 0.47
layout = [[0, 1, 2], [3, 4]]
for row, idxs in enumerate(layout):
    x = x0
    y = y0 - 0.115 * row
    for i in idxs:
        lab, col = chips[i]
        w = 0.016 + 0.0068 * len(lab)
        fig.patches.append(FancyBboxPatch((x, y), w, 0.075, boxstyle="round,pad=0.004,rounding_size=0.018",
                                          transform=fig.transFigure, fc=col, ec="none", alpha=0.95))
        fig.text(x + 0.009, y + 0.024, lab, fontsize=11.5, color="white", fontweight="bold", family="DejaVu Sans")
        x += w + 0.014
# install pill
fig.patches.append(FancyBboxPatch((0.047, 0.115), 0.255, 0.085, boxstyle="round,pad=0.004,rounding_size=0.02",
                                  transform=fig.transFigure, fc="white", ec="#6366f1", lw=1.8))
fig.text(0.058, 0.143, "$ pip install qqfrontier", fontsize=15, color="#3730a3", family="monospace",
         fontweight="bold")
fig.patches.append(FancyBboxPatch((0.393, 0.345), 0.088, 0.075, boxstyle="round,pad=0.004,rounding_size=0.018",
                                  transform=fig.transFigure, fc="white", ec="#0d9488", lw=1.6))
fig.text(0.401, 0.369, "on PyPI", fontsize=11.5, color="#0d9488", fontweight="bold", family="DejaVu Sans")
fig.text(0.32, 0.143, "Dr Merwan Roudane  ·  github.com/merwanroudane/qqfrontier", fontsize=12.5,
         color="#334155", family="DejaVu Sans")

# --- 3D surface (smoothed real TCI grid) ---
ax = fig.add_axes([0.605, 0.04, 0.265, 0.94], projection="3d")
f = RegularGridInterpolator((q, q), Z, method="cubic")
g = np.linspace(q[0], q[-1], 70)
R, C = np.meshgrid(g, g, indexing="ij")
S = f(np.column_stack([R.ravel(), C.ravel()])).reshape(R.shape)
ax.plot_surface(C, R, S, cmap=parula_cmap(), linewidth=0, antialiased=True, alpha=0.98)
ax.set_box_aspect((1, 1, 0.62), zoom=1.08)
ax.view_init(elev=26, azim=-128)
ax.set_axis_off()
ax.patch.set_alpha(0)

# --- small heatmap card ---
hx = fig.add_axes([0.865, 0.30, 0.115, 0.42])
hx.imshow(Z, origin="lower", cmap=parula_cmap(), aspect="auto")
hx.set_xticks([])
hx.set_yticks([])
for s in hx.spines.values():
    s.set_edgecolor("white")
    s.set_linewidth(3)
fig.text(0.866, 0.225, "Averaged TCI (%)\nUS 2Y vs 10Y–2Y spread", fontsize=9.5, color="#334155",
         family="DejaVu Sans")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
