"""Schematic Fig. 1 of BI-QMN paper 1 (no data): the history, a mask, and the support-equivalent circle.

Run from the repository root:  python manuscripts/paper1/figures/make_schematic.py
Writes fig0_schematic.pdf next to this file.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 7.5,
                     "text.color": INK, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def panel_history(ax):
    ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1.05); ax.axis("off")
    ax.set_title("(a) a history", loc="left", fontsize=8)
    active = {0: 0, 1: 1, 3: 2}
    ys = {r: 0.05 + 0.11 * r for r in range(8)}
    for r, y in ys.items():
        on = r in active
        ax.plot([0.08, 0.30], [y, y], color=S1 if on else AXIS, lw=2.0 if on else 1.0, solid_capstyle="butt")
        ax.text(0.05, y, rf"$\omega_{r}$", ha="right", va="center", fontsize=6.5, color=INK if on else MUTED)
    ax.text(0.19, 0.98, "clock", ha="center", fontsize=7, color=INK2)
    for r, j in active.items():
        yb = 0.18 + 0.27 * j
        ax.add_patch(Rectangle((0.66, yb - 0.07), 0.22, 0.14, facecolor="white", edgecolor=S1, lw=1.0))
        ax.text(0.77, yb, rf"$c_{j+1}\,u_{j+1}$", ha="center", va="center", fontsize=7)
        ax.add_patch(FancyArrowPatch((0.31, ys[r]), (0.655, yb), arrowstyle="-", color=S1, lw=0.8, connectionstyle="arc3,rad=0.0"))
    ax.text(0.77, 0.98, "system", ha="center", fontsize=7, color=INK2)
    ax.text(0.5, -0.09, r"$|\Psi\rangle=\sum_j c_j|\omega_{m_j}\rangle|u_j\rangle$, $H_{\rm tot}|\Psi\rangle=0$", ha="center", fontsize=7)


def panel_mask(ax):
    ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1.05); ax.axis("off")
    ax.set_title("(b) observed contexts (a low mask)", loc="left", fontsize=8)
    d_in = ["XI", "YI", "ZI", "IX", "IY", "IZ", "XX", "YY", "ZZ", "XZ", "ZX"]
    f_in = ["ZI", "IZ", "XX", "YY", "ZZ"]
    d_obs = {"IX", "XI", "XX", "YI", "ZI", "ZX"}
    f_obs = {"XX", "YY"}
    w, h = 0.082, 0.13

    def cell(x, y, label, kind):
        fc = {"obs": S1, "unobs": "white", "held": GRID}[kind]
        ax.add_patch(Rectangle((x, y), w * 0.9, h, facecolor=fc, edgecolor=S1 if kind != "held" else AXIS, lw=0.8))
        ax.text(x + w * 0.45, y + h / 2, label, ha="center", va="center", fontsize=5.6, color="white" if kind == "obs" else INK2, rotation=90)

    ax.text(0.0, 0.83, "$D$", fontsize=8, va="center")
    for k, p in enumerate(d_in):
        cell(0.07 + w * k, 0.76, p, "obs" if p in d_obs else "unobs")
    ax.text(0.0, 0.43, "$F$", fontsize=8, va="center")
    for k, p in enumerate(f_in):
        cell(0.07 + w * k, 0.36, p, "obs" if p in f_obs else "unobs")
    for k, (b, p) in enumerate([("D", "XY"), ("F", "XZ"), ("F", "ZX"), ("F", "XY")]):
        cell(0.07 + w * (6 + k), 0.36, f"{b},{p}", "held")
    for x, fc, ec, lab in ((0.07, S1, S1, "observed"), (0.38, "white", S1, "unobserved"), (0.79, GRID, AXIS, "held out")):
        ax.add_patch(Rectangle((x, 0.05), 0.04, 0.07, facecolor=fc, edgecolor=ec))
        ax.text(x + 0.06, 0.085, lab, va="center", fontsize=6.5)
    ax.text(0.5, -0.09, r"observed $D$ Paulis share the commutant $R=IX$", ha="center", fontsize=7)


def panel_circle(ax):
    """The support-equivalent circle only: observed tables identical, unobserved tables differ."""
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.05, 1.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(c) a circle of histories", loc="left", fontsize=8)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=S1, lw=1.4))

    def pt(t):
        a = math.pi / 2 - 2 * t      # tables have period pi in t: draw angle 2t; t = pi/2 is opposite the truth
        return math.cos(a), math.sin(a)

    x0, y0 = pt(0.0)
    ax.plot([x0], [y0], "o", color=INK, ms=6, zorder=3)
    ax.text(x0, y0 + 0.16, r"$\Psi$", ha="center", va="bottom", fontsize=7.5)
    x1, y1 = pt(math.pi / 2)
    ax.plot([x1], [y1], "o", mfc="white", mec=INK, ms=6, zorder=3)
    ax.text(x1, y1 - 0.16, r"$(I\otimes iR)\Psi$, orthogonal to $\Psi$", ha="center", va="top", fontsize=6.3)
    # direction of motion e^{itR}
    ax.add_patch(FancyArrowPatch(pt(0.18), pt(0.52), connectionstyle="arc3,rad=-0.18", arrowstyle="-|>",
                                 mutation_scale=7, color=S1, lw=0.9))
    xm, ym = pt(0.37)
    ax.text(xm + 0.12, ym + 0.05, r"$e^{itR}$", ha="left", va="center", fontsize=7, color=S1)
    ax.text(0, 0.16, "observed tables:", ha="center", fontsize=6.2, color=INK2)
    ax.text(0, -0.08, "identical", ha="center", fontsize=6.2, color=INK)
    ax.text(0, -1.62, "unobserved tables differ along the circle;", ha="center", fontsize=6.2, color=INK2)
    ax.text(0, -1.88, "the presence of dynamics does not", ha="center", fontsize=6.2, color=INK2)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [0.85, 1.25, 1.0]})
    panel_history(axes[0]); panel_mask(axes[1]); panel_circle(axes[2])
    fig.savefig(OUT / "fig0_schematic.pdf")
    fig.savefig(OUT / "fig0_schematic.png", dpi=150)


if __name__ == "__main__":
    main()
