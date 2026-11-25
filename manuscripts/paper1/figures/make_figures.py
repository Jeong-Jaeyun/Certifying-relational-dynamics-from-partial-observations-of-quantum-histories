"""Figures of BI-QMN paper 1, regenerated from frozen artifacts (SHA-256 checked) and the independent Born engine.

Run from the repository root with PYTHONPATH=src:
    python manuscripts/paper1/figures/make_figures.py
Writes fig0_overview.pdf, fig1_example.pdf ... fig7_design2.pdf and fig5_sealed.pdf next to this file.

Colour: the validated reference categorical palette, unmodified, in fixed slot order
(blue, orange, aqua, yellow). Scatter plots use at most the first three slots, which are validated for
all pairs; line plots use at most four, validated for adjacent pairs. Every series also carries a
marker or line style, and multi-series panels carry a legend or direct labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).resolve().parent

SOURCES = {
    "kit": ("outputs/m1_evidence_package/m1_reproduction_kit_3a.json", "c041d4d8b9bbb8ef4b9120849551660d161bd8e2b4705ca9c7f428ffdaf87767"),
    "dev": ("outputs/m1_added_measurement/m1_added_measurement.json", "bb61f9c6f6fc20d20cb3a783a890d514a8f8f71f6b648d9b4857fb7070d3b776"),
    "posthoc": ("outputs/m1_added_measurement_posthoc/m1_added_measurement_posthoc.json", "46ae1d70d79ebef7c74c819c227ad3057694d53da9424c5b1a5d40c3d8616909"),
    "ordn_sealed": ("outputs/observed_rdn_v2/observed_rdn_v2_sealed.json", "752c66d7707369652f7cf132451f1202a61322eaaba6f51ca5737205f23d1d85"),
    "d2_bound": ("outputs/design2/bound_heldout.json", "c3e23b0a2568e1d7430284573a216a980986f290205c3ed6c26a40ae36a61358"),
    "d2_pilot": ("outputs/design2/pilot.json", "bc2e1ea62016d343d4a8d7da8e476ed3818eeb91c7142f1c38cb89ea3df358e8"),
    "e1": ("outputs/m1_e1_confirmatory/m1_e1_confirmatory.json", "cf8330dd4353bec9a8956a4c2433ed3ca27f398c146a6bc1ceaceb3c8402f441"),
    "d2_confirm": ("outputs/design2_confirmatory/design2_confirmatory.json", "3b583c152d389790a46399fb592e147f2fd32f00161522947cac96b7f4ec8aa4"),
}

# reference palette (light), fixed order; chrome
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STATUS = {"primal_verified": (S1, "o", "BI verified"), "unresolved": (S2, "s", "BI unresolved"), "dual_refuted": (S3, "^", "BI refuted")}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL, WIDE = 3.4, 7.0


def load(key: str):
    rel, sha = SOURCES[key]
    path = ROOT / rel
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        raise RuntimeError(f"{rel}: SHA-256 {digest} differs from the recorded {sha}")
    return json.loads(path.read_text(encoding="utf-8"))


def _example_histories():
    """The orthogonal pair of Sec. IV D from the reproduction kit, recomputed with the independent Born engine."""
    from bi_qmn.p3 import independent_born as ib
    kit = load("kit")
    A = np.asarray(kit["history_A_real_imag"][0]) + 1j * np.asarray(kit["history_A_real_imag"][1])
    TA = ib.tables(A, kit["support_contexts"] + kit["heldout_contexts"])
    B = np.asarray(kit["history_B_real_imag"][0]) + 1j * np.asarray(kit["history_B_real_imag"][1])
    TB = ib.tables(B, list(TA))
    # the kit tables must be reproduced by the independent engine
    assert ib.max_table_difference(TA, kit["tables_A"], list(TA)) < 1e-14 and ib.max_table_difference(TB, kit["tables_B"], list(TB)) < 1e-14
    return kit, A, TA, TB


def _orthogonal_scatter(ax, title="(d) the two histories"):
    kit, _, TA, TB = _example_histories()
    obs = [(TA[c][o], TB[c][o]) for c in kit["support_contexts"] for o in TA[c]]
    hel = [(TA[c][o], TB[c][o]) for c in kit["heldout_contexts"] for o in TA[c]]
    lim = max(max(x, y) for x, y in obs + hel) * 1.05
    ax.plot([0, lim], [0, lim], color=AXIS, lw=0.8, zorder=1)
    ax.scatter(*zip(*hel), s=12, marker="^", color=S2, edgecolor="white", linewidth=0.4, label="unobserved", zorder=2)
    ax.scatter(*zip(*obs), s=12, marker="o", color=S1, edgecolor="white", linewidth=0.4, label="observed", zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"probability under $\Psi$", fontsize=7)
    ax.set_ylabel(r"probability under $(I\otimes iR)\Psi$", fontsize=7)
    ax.tick_params(labelsize=6)
    d_obs = max(abs(x - y) for x, y in obs)
    d_hel = max(abs(x - y) for x, y in hel)
    ax.legend(loc="upper left", fontsize=6, handletextpad=0.2, borderaxespad=0.2)
    obs_txt = "< 1e-15" if d_obs < 1e-15 else f"{d_obs:.1e}"
    ax.text(0.97, 0.04, f"max |difference|\nobserved {obs_txt}\nunobserved {d_hel:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color=INK2)
    ax.set_title(title, loc="left", fontsize=8)
    return d_obs, d_hel


def fig0_overview():
    """Fig. 1: schematic panels (a)-(c) from make_schematic.py and the data panel (d) of the orthogonal pair."""
    sys.path.insert(0, str(OUT))
    import make_schematic as ms
    fig = plt.figure(figsize=(WIDE, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.28)
    top = gs[0].subgridspec(1, 2, width_ratios=[0.75, 1.25], wspace=0.08)
    bot = gs[1].subgridspec(1, 3, width_ratios=[1.05, 0.12, 0.95], wspace=0.0)
    axes = [fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1]), fig.add_subplot(bot[0, 0]), fig.add_subplot(bot[0, 2])]
    ms.panel_history(axes[0]); ms.panel_mask(axes[1]); ms.panel_circle(axes[2])
    d_obs, d_hel = _orthogonal_scatter(axes[3])
    axes[3].grid(False)
    fig.savefig(OUT / "fig0_overview.pdf")
    plt.close(fig)
    return {"observed_max": d_obs, "unobserved_max": d_hel}


def fig1_example():
    """Fig. 3: support response along exp(itR), R = IX, for one state and three masks."""
    from bi_qmn.p3 import independent_born as ib
    from bi_qmn.p3.runs.m1_evidence_package import MASKS
    from scipy.linalg import expm
    _, A, _, _ = _example_histories()
    fig, bx = plt.subplots(figsize=(COL, 2.5))
    ts = np.logspace(-4, math.log10(math.pi / 2), 60)
    G = ib.pauli("IX")
    curves = (("type3a", S1, "-", "3a: exact circle"), ("type3b", S2, "--", r"3b: $\propto t^2$"), ("normal", S3, ":", r"regular: $\propto t$"))
    out = {}
    for name, color, ls, label in curves:
        mask = list(MASKS[name])
        T0 = ib.tables(A, mask)
        res = [max(ib.max_table_difference(ib.tables(ib.rotate_system(A, expm(1j * t * G)), mask), T0, mask), 1e-17) for t in ts]
        bx.loglog(ts, res, color=color, ls=ls, label=label)
        out[name] = res[-1]
    bx.set_xlabel(r"position on the circle, $t$")
    bx.set_ylabel("largest change of an\nobserved probability")
    bx.set_ylim(1e-17, 1)
    bx.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_example.pdf")
    plt.close(fig)
    return out


def fig2_tails():
    dev = load("dev")
    rows = [r for r in dev["records"] if r["class"] == "3a"]
    n = len(rows)
    m = max(1, math.ceil(round(0.1 * n, 9)))
    arms = (("A0", S1, "-", "A0 support only"), ("C", S2, "--", "C same shots on support"),
            ("A_f", S3, "-.", "A$_f$ fixed-order context"), ("A_s", S4, "-", "A$_s$ support-chosen context"))
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.axvspan(0.5, m + 0.5, color=GRID, alpha=0.6, lw=0)
    ax.text(m / 2 + 0.5, 0.0105, "worst 10%\n(CVaR$_{0.9}$)", ha="center", va="bottom", fontsize=6.5, color=INK2)
    cv = {}
    for arm, color, ls, label in arms:
        e = np.sort([r["errors"][arm]["E"]["diag"] for r in rows])[::-1]
        cv[arm] = float(np.mean(e[:m]))
        ax.semilogy(np.arange(1, n + 1), e, color=color, ls=ls, label=f"{label}: {cv[arm]:.3f}")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.007, 0.7)
    ax.set_xlabel("records ranked by error (type 3a, n = 128)")
    ax.set_ylabel("unobserved-table error ($D$ class)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.14, 0.29), title="arm: CVaR$_{0.9}$", title_fontsize=7, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tails.pdf")
    plt.close(fig)
    return cv


def fig5_sealed():
    """Fig. 5: the preregistered same-budget contrast of both designs on sealed banks, with the declared pass lines."""
    e1 = load("e1")["decision"]
    d2 = load("d2_confirm")["analysis"]["endpoints"]
    rows_a = [(lab, e1[h]["point"], e1[h]["lower"], e1[h]["upper"]) for h, lab in (("H1", "H1: type 3a"), ("H2", "H2: flagged masks"))]
    mode = {"golomb": r"$\{0,1,4,6\}$", "collision": r"$\{0,1,2,3\}$"}
    fam = {"exact": "exact", "near_0.03": r"$\varepsilon=0.03$"}
    rows_b = [(f"{mode[d2[g]['design']]}, {fam[d2[g]['family']]}", d2[g]["point"], d2[g]["q025"], d2[g]["q975"])
              for g in ("G1", "G2", "G3", "G4")]
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.55), sharex=True, gridspec_kw={"height_ratios": [2, 4]})
    panels = ((axes[0], rows_a, float(e1["delta_min"]), "(a) main design, sealed banks"),
              (axes[1], rows_b, 0.1, "(b) second design, sealed bank"))
    out = {}
    for ax, rows, line, title in panels:
        ys = np.arange(len(rows))[::-1].astype(float)
        ax.axvline(0.0, color=AXIS, lw=0.8)
        ax.axvline(line, color=INK2, lw=0.9, ls="--")
        ax.text(line + 0.006, ys.max() + 0.55, f"declared pass line {line:.3g}", fontsize=6, color=INK2, va="center")
        for y, (lab, pt, lo, hi) in zip(ys, rows):
            ax.plot([lo, hi], [y, y], color=S1, lw=1.6, solid_capstyle="round")
            ax.plot([pt], [y], "o", color=S1, ms=4.5, mec="white", mew=0.6)
            ax.text(hi + 0.008, y, f"{pt:.3f}", fontsize=6, color=INK2, va="center")
            out[lab] = (pt, lo, hi)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
        ax.set_ylim(-0.6, ys.max() + 0.95)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
    axes[1].set_xlim(-0.02, 0.5)
    axes[1].set_xlabel("$\\Delta$: reduction of the worst-decile\nmean error on unobserved tables")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_sealed.pdf")
    plt.close(fig)
    return out


def fig3_record771():
    from bi_qmn.p3.runs import m1_added_measurement as am
    from bi_qmn.p3.runs import m1_added_measurement_posthoc as ph
    from bi_qmn.p3.runs import m1_support_identifiability_map as m1
    from bi_qmn.p3.runs.p3_d1_v6_development import _true_branch_vectors
    post = load("posthoc")
    rec = next(r for r in post["records_3a"] if r["index"] == 771)
    ctx = am._prepare_light(ROOT / "config")
    V = _true_branch_vectors(ctx["instances"][771])
    G = m1.pauli_matrix(rec["R"])
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ts = np.linspace(0, math.pi, 400)
    out = {}
    for q, color, ls, label in ((rec["Q_s"], S1, "-", f"{rec['Q_s']} (s$_Q$ choice)"), (rec["Q_f"], S2, "--", f"{rec['Q_f']} (fixed order)")):
        cm = ph.coefficient_matrix(ctx["fw"], V, G, q, float(ctx["n_shot"]))
        sep = [ph.separation(cm, t, float(ctx["n_shot"]))["l2"] for t in ts]
        ax.plot(ts, sep, color=color, ls=ls, label=label + rf", $\sigma_{{\min}}$={cm['sigma_min']:.4f}")
        out[q] = {"sigma_min": cm["sigma_min"], "noise": cm["noise_l2"]}
    noise = out[rec["Q_f"]]["noise"]
    ax.axhline(noise, color=MUTED, lw=0.8, ls=":")
    ax.text(0.03, noise * 1.04, "shot-noise scale", fontsize=6.5, color=INK2, va="bottom")
    for arm, lab in (("A0", "A0 fit"), ("A_f", "A$_f$ fit"), ("A_s", "A$_s$ fit")):
        t = rec["fitted_orbit_point"][arm]["t"]
        ax.axvline(t, color=MUTED, lw=0.6)
        ax.text(t + 0.025, 0.42, lab, rotation=90, fontsize=6.5, color=INK2, ha="left", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, math.pi)
    ax.set_ylim(0, 0.205)
    ax.set_xticks([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi], ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_xlabel(r"position on the circle, $t$")
    ax.set_ylabel(r"$\|q_Q(t)-q_Q(0)\|_2$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_record771.pdf")
    plt.close(fig)
    return out


def _status_scatter(ax, xs, ys, statuses):
    for status, (color, marker, label) in STATUS.items():
        sel = [k for k, s in enumerate(statuses) if s == status]
        ax.scatter([xs[k] for k in sel], [ys[k] for k in sel], s=16, marker=marker, color=color, edgecolor="white", linewidth=0.4, label=f"{label} ({len(sel)})", zorder=3)


def fig4_energy():
    post = load("posthoc")
    rows = post["records_3a"]
    xs = [r["fitted_orbit_point"]["A0"]["angle_from_truth"] for r in rows]
    ys = [r["energy"]["B"] * math.sin(x) ** 2 for r, x in zip(rows, xs)]
    st = [r["a0_bi_status"] for r in rows]
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.axhline(1e-6, color=MUTED, lw=0.8, ls=":")
    ax.text(0.02, 1.3e-6, r"registered bound $\theta_{\mathrm{con}}$", fontsize=6.5, color=INK2, va="bottom")
    _status_scatter(ax, xs, ys, st)
    ax.set_yscale("log")
    ax.set_xlim(0, math.pi / 2 * 1.02)
    ax.set_ylim(3e-7, 1.5)
    ax.set_xticks([0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2], ["0", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"])
    ax.set_xlabel(r"distance of the support-only fit from the truth, $|t|$")
    ax.set_ylabel(r"registered energy $\beta\sin^2 t$")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.33))
    fig.tight_layout()
    fig.savefig(OUT / "fig4_energy.pdf")
    plt.close(fig)


def fig5_fiber():
    post = load("posthoc")
    rows = post["records_3a"]
    xs = [max(r["fiber_A0"]["truth_ratio"].values()) for r in rows]
    ys = [r["fiber_A0"]["fiber_min_ratio"]["joint"] for r in rows]
    st = [r["a0_bi_status"] for r in rows]
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    ax.axvline(1, color=MUTED, lw=0.8, ls=":")
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    _status_scatter(ax, xs, ys, st)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 40)
    ax.set_ylim(0.2, 1.35)
    ax.text(38, 0.22, "truth outside the body,\nsome circle point inside", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xlabel("body score of the true point / radius")
    ax.set_ylabel("best body score on the circle / radius")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_fiber.pdf")
    plt.close(fig)
    return {"truth_covered": sum(x <= 1 for x in xs), "fiber_covered": sum(y <= 1 for y in ys)}


def _cp(k: int, n: int) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def fig6_observed():
    """Observed test on the sealed bank: (a) false certification on static histories, (b) detection against true kappa."""
    art = load("ordn_sealed")
    rows = art["records"]
    alpha = 0.01
    cert = lambda r, v: float(r["tests"][v]["p_monte_carlo"]) <= alpha   # noqa: E731
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDE, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})
    nulls = (("tau_flat", "static"), ("tau_flat_readout_0.05", "symmetric\nreadout"),
             ("nonuniform_clock_static_0.5", "non-uniform\nclock\nmarginal"), ("tau_flat_clock_skew_0.1", "asymmetric\nclock\nreadout"))
    out = {}
    for k, (fam, _label) in enumerate(nulls):
        sr = [r for r in rows if r["family"] == fam]
        for v, color, marker, dx, lab in (("v2", S1, "o", -0.12, "both margins (Prop. 2)"), ("v1", S2, "s", 0.12, "uniform clock")):
            kk = sum(cert(r, v) for r in sr)
            rate = kk / len(sr)
            lo, hi = _cp(kk, len(sr))
            shown = max(rate, 1.2e-4)
            ax.errorbar(k + dx, shown, yerr=[[shown - max(lo, 1.2e-4)], [hi - shown]], fmt=marker, color=color, ms=5, mec="white", mew=0.4,
                        elinewidth=1.0, capsize=0, label=lab if k == 0 else None, zorder=3)
            out[f"{fam}:{v}"] = (kk, len(sr))
    ax.axhline(alpha, color=MUTED, lw=0.8, ls=":")
    ax.text(0.5, alpha * 1.15, r"$\alpha=0.01$", fontsize=6.5, color=INK2, ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.6)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xticks(range(len(nulls)), [lab for _, lab in nulls], fontsize=6.5)
    ax.set_ylabel("false certification rate")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.85))
    ax.set_title("(a) static conditional states (n = 1024 each)", loc="left")
    # (b) one point per family and stratum at its median true kappa (Clopper-Pearson 95% intervals)
    fams = ("dephased_0.9", "dephased_0.75", "pure", "readout_0.05", "pure_clock_skew_0.1", "registry_violating")
    for d, filled in (("medium", True), ("low", False)):
        for v, color, marker, dx, vlabel in (("v2", S1, "o", 1.0, "both margins"), ("v1", S2, "s", 1.06, "uniform clock")):
            for j, fam in enumerate(fams):
                sr = [r for r in rows if r["family"] == fam and r["deficit"] == d]
                kk = sum(cert(r, v) for r in sr)
                lo, hi = _cp(kk, len(sr))
                x = float(np.median([r["truth"]["kappa"] for r in sr])) * dx
                y = kk / len(sr)
                bx.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=marker, color=color, ms=4.5, mfc=color if filled else "white", mec=color, mew=1.0,
                            elinewidth=1.0, capsize=0, zorder=3, label=f"{vlabel}, {d}" if j == 0 else None)
    bx.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    bx.text(1.03, 0.04, r"true $\kappa=1$", fontsize=6.5, color=INK2, ha="left", va="bottom")
    bx.text(1.3, 0.62, r"dephased $\nu=0.9$", fontsize=6.5, color=INK2, ha="left")
    bx.set_xscale("log")
    bx.set_xlim(0.6, 30)
    bx.set_ylim(-0.03, 1.05)
    bx.set_xlabel(r"median true model-based margin $\kappa$ of the family")
    bx.set_ylabel("certification rate")
    bx.legend(loc="lower right", fontsize=6.5)
    bx.set_title("(b) histories with relational dynamics (n = 512 each)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_observed.pdf")
    plt.close(fig)
    return out


def fig7_design2():
    """Second design: estimator-independent two-point bound (lines) against the pilot's estimator errors (markers)."""
    bound = load("d2_bound")
    pilot = load("d2_pilot")
    rows = [r for r in pilot["rows"] if r["kind"] == "truth"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.8), sharey=True)
    Ns = [r["N"] for r in bound["golomb|specified-given|0.0"]["arms"]["A0"]]
    series = (("0.0", "exact", S1, "o"), ("0.03", "near_0.03", S2, "s"), ("0.1", "near_0.1", S3, "^"))
    out = {}
    for ax, design, title in ((axes[0], "golomb", "(a) modes {0,1,4,6}"), (axes[1], "collision", "(b) modes {0,1,2,3}")):
        for eps, fam, color, marker in series:
            keys = [k for k in bound if k.startswith(design + "|") and k.endswith("|" + eps) and ("constructed" in k or "specified-given" in k)]
            med = np.median([[r["bound"] for r in bound[k]["arms"]["A0"]] for k in keys], axis=0)
            lab = "exact circle" if eps == "0.0" else rf"$\varepsilon={eps}$"
            ax.plot(Ns, med, color=color, lw=1.2, label=f"{lab}: bound, support only")
            pts = [(N, np.median([r["errors"]["A0"] for r in rows if r["design"] == design and r["family"] == fam and r["N"] == N])) for N in (512, 2048, 8192)]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker, color=color, ms=5, mec="white", mew=0.4, ls="none",
                    label=f"{lab}: estimator, support only")
            out[f"{design}:{fam}"] = pts
        keys = [k for k in bound if k.startswith(design + "|") and ("constructed" in k or "specified-given" in k)]
        med_add = np.median([[r["bound"] for r in bound[k]["arms"]["+best"]] for k in keys], axis=0)
        ax.plot(Ns, med_add, color=INK2, lw=1.0, ls="--", label=r"bound, fixed added context (largest true $\sigma_{\min}$)")
        pts = [(N, np.median([r["errors"]["A_sig"] for r in rows if r["design"] == design and r["family"] in ("exact", "near_0.03", "near_0.1") and r["N"] == N])) for N in (512, 2048, 8192)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "D", color=INK2, ms=4, mfc="white", ls="none", label=r"estimator, context chosen from data ($\sigma_{\min}$ rule)")
        eq = bound[f"{design}|specified-equal|0.0"]["arms"]["+IZ"]
        ax.plot(Ns, [r["bound"] for r in eq], color=S4, lw=1.0, ls=":", label=r"bound, rank-1 context ($IZ$, equal weights)")
        if design == "collision":
            zz = bound["collision|specified-equal|0.0"]["arms"]["+ZZ"]
            ax.plot(Ns, [r["bound"] for r in zz], color=S4, lw=1.0, ls="-.", label=r"bound, weak context ($ZZ$, $\sigma_{\min}=0.005$)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(5e-5, 1.5)
        ax.set_xlabel("shots per context")
        ax.set_title(title, loc="left")
        # direct labels: what is a bound and what is an estimate, and how the added context was chosen
        ax.text(0.98, 0.97, "lines: two-point lower bounds\nmarkers: pilot estimator errors", transform=ax.transAxes,
                ha="right", va="top", fontsize=6, color=INK2)
        n_mid = Ns.index(32768)
        ax.annotate("bound, context fixed\nwith the true history", xy=(Ns[n_mid], med_add[n_mid]), xytext=(140, 1.3e-4),
                    fontsize=6, color=INK2, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
        ax.annotate("estimator, context\nchosen from data", xy=pts[1], xytext=(140, 1.1e-3),
                    fontsize=6, color=INK2, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    axes[0].set_ylabel("unobserved-table error ($L_2$)")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(OUT / "fig7_design2.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("fig0", fig0_overview())
    print("fig5_sealed", fig5_sealed())
    print("fig1", fig1_example())
    print("fig2", fig2_tails())
    print("fig3", fig3_record771())
    fig4_energy()
    print("fig5", fig5_fiber())
    print("fig6", fig6_observed())
    print("fig7", fig7_design2())
