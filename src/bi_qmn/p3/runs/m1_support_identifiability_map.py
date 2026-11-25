"""M1 (roadmap Q1): where does the third support non-identifiability come from?

Development diagnostic for roadmap milestone M1 ([roadmap](../../../../documents/BI-QMN-roadmap.md)).
Design and hypotheses: documents/M1-third-mechanism-design.md.

D1-v7 found screened-in low records whose exact support does not identify the state: the support
Jacobian of the shared branch-pair (SBP) model has one null direction beyond gauge at the true
parameters, and exact-support alternatives move the unobserved tables by 0.07-0.52 (P3-d1-v7-run.md §5).
Neither screen test (Pauli commutant, disconnected branch) sees it.

This module opens no observation bank and samples no shots. Everything is computed from exact tables
at true parameters.

* **E1 structural map.** For every system unitary of the generator and every support mask of the low
  and medium deficits, and for several draws of the continuous parameters (branch probabilities and
  phases), count the support-Jacobian null directions beyond gauge. A combination is *deficient* when
  every draw has one, *identifiable* when none has, and *mixed* otherwise.
* **E2 mechanism.** For deficient combinations, and for deficient records of earlier runs, split the
  null direction into branch-amplitude, branch-phase, within-branch-subspace rotation and rotation into
  the unused system direction. Write its rotation part as a system generator G. Count the rotation-only
  first-order kernel (the restricted commutant of the observed Paulis on the branch vectors). Follow
  the family with a predictor-corrector on the exact support to test whether it is exact, and measure
  how far it moves the unobserved tables and the state.
* **E3 retrospective join.** Rebuild (construction and mask only) every record of D1-v7, T1 run 2 and
  A1. Place each on the map at its own true parameters, and join it with the run artifacts (diag score,
  selector error bound).

The model's tables are ``q_o = x^H A_o x`` with ``x = sum_j |m_j> (x) c_j u_j``. A D context reads, per
Pauli ``P``, the pair coherences ``conj(c_j) c_k <u_j|P|u_k>`` and the weighted diagonal
``sum_j p_j <u_j|P|u_j>``. An F context reads each ``p_j <u_j|P|u_j>``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.stats import spearmanr
import yaml

from bi_qmn.core.config import load_p1_config
from bi_qmn.p1.model.p1_banks import support_contexts_for_deficit
from bi_qmn.p1.model.p1_generator import seeded_construction_bank_p1, system_unitary_from_config
from bi_qmn.p1.model.p1_measurement import contexts_from_p1_measurement_config, full_p1_context_list, p1_effect_map
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairEstimator
from bi_qmn.p3.p3_support_screen import pauli_commutant, support_paulis
from bi_qmn.p3.runs.p3_d1_v6_tail_diagnostics import theta_from_branch_vectors


METHOD = "m1_support_identifiability_map_v1"
OUTPUT_SUBDIR = "m1_third_mechanism"
ARTIFACT = "m1_support_identifiability_map.json"
DEFICITS = ("low", "medium")

# Continuous-parameter draws for the structural map: the D1-v7 generator law (Dirichlet 8, floor 0.15,
# uniform phases). Development seed outside every declared bank and the reserved 53500-53999 range.
DRAW_SEED = 55101
DRAWS = 16
DIRICHLET_CONCENTRATION = 8.0
MINIMUM_MODE_PROBABILITY = 0.15

FD_STEP = 1.0e-6
NULL_RELATIVE_TOL = 1.0e-7          # singular value / largest singular value of the same matrix
ORACLE_PAIR_THRESHOLD = 1.0e-9      # TRUE_MASK_THRESHOLD of D1-v6/v7, T1 and A1
KERNEL_RELATIVE_TOL = 1.0e-9

E2_CAP_PER_CELL = 80                # synthetic deficient combinations analysed per (unitary, deficit)
E2_CAP_RECORDS = 120                # deficient records of earlier runs analysed
CURVE_STEP = 0.05
CURVE_STEPS = 60
CURVE_RESIDUAL_MAX = 1.0e-8
CURVE_MIN_PROBABILITY = 0.01
ORBIT_T = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)

BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

D1V7_CONFIG = "p3_d1_v7.yaml"
T1_CONFIG = "p1_sparse_selector_d1_v4.yaml"
A1_CONFIG = "p1_sparse_atlas_a1.yaml"
D1V7_ARTIFACT = Path("outputs/p3_d1_v7/p3_d1_v7_development.json")
T1_ARTIFACT = Path("outputs/p1_sparse_selector_d1_v4_run2/p1_sparse_selector_d1_v4.json")
A1_ARTIFACT = Path("outputs/p1_sparse_atlas_a1/p1_sparse_atlas_a1.json")
SELECTOR_BUDGET = 0.007644142507474676   # T1 run 2 operational error budget (fixed input of A1)

_PAULI_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.asarray([[0, 1], [1, 0]], dtype=complex),
    "Y": np.asarray([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.asarray([[1, 0], [0, -1]], dtype=complex),
}
PAULI_LABELS = tuple(a + b for a in "IXYZ" for b in "IXYZ")


def pauli_matrix(label: str) -> np.ndarray:
    return np.kron(_PAULI_1Q[label[0]], _PAULI_1Q[label[1]])


# ---------------------------------------------------------------------------
# Model context
# ---------------------------------------------------------------------------


def _keep_count(n: int, coverage: float) -> int:
    """Number of contexts ``support_contexts_for_deficit`` keeps (``_select_contexts``)."""
    if coverage >= 1.0:
        return n
    if coverage <= 0.0:
        return 0
    return max(1, int(round(n * coverage)))


def deficit_masks(measurement: Mapping[str, Any], deficit: str) -> list[tuple[str, ...]]:
    """Every support mask the deficit can draw, as sorted context-id tuples."""
    grouped = contexts_from_p1_measurement_config(dict(measurement))
    diag = [c.context_id for c in grouped["input"] if c.clock == "D"]
    coh = [c.context_id for c in grouped["input"] if c.clock == "F"]
    cfg = measurement["observation_deficit"][deficit]
    nd, nc = _keep_count(len(diag), float(cfg["cov_diag"])), _keep_count(len(coh), float(cfg["cov_coh"]))
    return [tuple(sorted(d + c)) for d in itertools.combinations(diag, nd) for c in itertools.combinations(coh, nc)]


def draw_truths(n_modes: int, seed: int, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Branch probabilities and phases under the D1-v7 generator law."""
    rng = np.random.default_rng(int(seed))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    while len(out) < count:
        p = rng.dirichlet(np.full(n_modes, DIRICHLET_CONCENTRATION))
        if float(np.min(p)) < MINIMUM_MODE_PROBABILITY:
            continue
        phases = np.zeros(n_modes)
        phases[1:] = rng.uniform(0.0, 2.0 * math.pi, size=n_modes - 1)
        out.append((p, phases))
    return out


def model_context(config_dir: str | Path = "config", draws: int = DRAWS) -> dict[str, Any]:
    config_dir = Path(config_dir)
    p1 = load_p1_config(config_dir)
    contexts = full_p1_context_list(p1.measurement)
    dims = p1.generator["dimensions"]
    K, d_S = int(dims["K"]), int(dims["d_S"])
    effects = {cid: {o: np.asarray(op.data, dtype=complex) for o, op in row.items()} for cid, row in p1_effect_map(contexts, K=K).items()}
    sbp = SharedBranchPairEstimator(p1.generator["active_modes"], K, d_S, effects)
    unitaries = dict(system_unitary_from_config(item) for item in p1.generator["construction_grid"]["system_unitaries"])
    return {
        "config_dir": str(config_dir),
        "p1": p1,
        "sbp": sbp,
        "fw": sbp.forward,
        "ids": [c.context_id for c in contexts],
        "unitaries": unitaries,
        "masks": {deficit: deficit_masks(p1.measurement, deficit) for deficit in DEFICITS},
        "draws": draw_truths(len(p1.generator["active_modes"]), DRAW_SEED, int(draws)),
        "bank_cache": {},
    }


# ---------------------------------------------------------------------------
# Exact-table Jacobians
# ---------------------------------------------------------------------------


def truth_branch_vectors(U: np.ndarray, probabilities: Sequence[float], phases: Sequence[float]) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    p = p / float(np.sum(p))
    c = np.sqrt(p) * np.exp(1j * np.asarray(phases, dtype=float))
    return np.asarray(U, dtype=complex)[:, : c.size] * c[None, :]


def branch_derivatives(sbp: SharedBranchPairEstimator, theta: np.ndarray, step: float = FD_STEP) -> np.ndarray:
    """``d vec(V) / d theta`` by central differences, shape ``(n d_S, len(theta))``."""
    fw = sbp.forward
    cols = []
    for k in range(theta.size):
        e = np.zeros_like(theta)
        e[k] = step
        cols.append((fw.vector(sbp.branch_vectors(theta + e)) - fw.vector(sbp.branch_vectors(theta - e))) / (2.0 * step))
    return np.stack(cols, axis=1)


def context_jacobians(fw, x: np.ndarray, dx: np.ndarray, ids: Iterable[str]) -> dict[str, np.ndarray]:
    """``d q_o / d theta = 2 Re(x^H A_o dx)`` per context, each of shape ``(outcomes, len(theta))``."""
    out = {}
    for cid in ids:
        Ax = np.einsum("oij,j->oi", fw.A[cid], x)
        out[cid] = 2.0 * np.real(Ax.conj() @ dx)
    return out


def split_gauge(J_all: np.ndarray, tol: float = NULL_RELATIVE_TOL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Null basis of the all-context Jacobian (gauge) and an orthonormal basis of its complement."""
    _, s, Vt = np.linalg.svd(J_all)
    rank = int(np.sum(s > tol * s[0]))
    return Vt[rank:].T, Vt[:rank].T, s


def masked_singular_values(Jc: Mapping[str, np.ndarray], Q: np.ndarray, mask: Sequence[str]) -> np.ndarray:
    return np.linalg.svd(np.concatenate([Jc[c] for c in mask], axis=0) @ Q, compute_uv=False)


def extra_null(s: np.ndarray, tol: float = NULL_RELATIVE_TOL) -> int:
    return int(np.sum(s <= tol * s[0]))


def truth_point(ctx: Mapping[str, Any], V: np.ndarray) -> dict[str, Any]:
    sbp, fw = ctx["sbp"], ctx["fw"]
    theta = theta_from_branch_vectors(sbp, V)
    x = fw.vector(sbp.branch_vectors(theta))
    dx = branch_derivatives(sbp, theta)
    Jc = context_jacobians(fw, x, dx, ctx["ids"])
    gauge, Q, _ = split_gauge(np.concatenate([Jc[c] for c in ctx["ids"]], axis=0))
    return {"theta": theta, "x": x, "dx": dx, "Jc": Jc, "gauge": gauge, "Q": Q, "gauge_dim": int(gauge.shape[1])}


def mask_readout(tp: Mapping[str, Any], mask: Sequence[str]) -> dict[str, Any]:
    s = masked_singular_values(tp["Jc"], tp["Q"], mask)
    e = extra_null(s)
    return {
        "extra_null": e,
        "smin_rel": float(s[-1] / s[0]),
        "smallest_nonnull_rel": float(s[-1 - e] / s[0]) if e < s.size else 0.0,
    }


def oracle_screen(fw, V: np.ndarray, mask: Sequence[str]) -> dict[str, Any]:
    """The two failure types the frozen screen targets, from the true basis."""
    commutant = pauli_commutant(support_paulis(mask))
    obs = fw.pair_observability(V, [c for c in mask if c.startswith("D,")], ORACLE_PAIR_THRESHOLD)
    return {
        "pauli_commutant": list(commutant),
        "true_mask_class": obs["mask_class"],
        "oracle_screened_out": bool(commutant) or obs["mask_class"] == "branch_disconnected",
    }


# ---------------------------------------------------------------------------
# Mechanism analysis (E2)
# ---------------------------------------------------------------------------


def _branch_frame(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Branch amplitudes ``c`` (real, positive) and a unitary frame whose first columns are ``u_j``."""
    d, n = V.shape
    c = np.linalg.norm(V, axis=0)
    u = V / c[None, :]
    w, vecs = np.linalg.eigh(np.eye(d) - u @ u.conj().T)
    comp = vecs[:, w > 0.5]
    return c, np.concatenate([u, comp], axis=1)


def decompose_direction(V: np.ndarray, dV: np.ndarray) -> dict[str, Any]:
    """Split a tangent direction of the branch vectors into amplitude, phase and rotation parts.

    With ``c_k`` real and ``u_k = V[:, k] / c_k``: ``M = U^H dV / c`` has amplitude ``Re M_kk``, phase
    ``Im M_kk``, within-branch rotation ``M_jk`` (``j != k < n``) and rotation into the unused system
    directions ``M_jk`` (``j >= n``). The global phase is removed first.
    """
    d, n = V.shape
    dV = dV - (np.vdot(V, dV).imag / float(np.vdot(V, V).real)) * 1j * V
    c, Uf = _branch_frame(V)
    rel = (Uf.conj().T @ dV) / c[None, :]
    w = c**2
    amp = np.real(np.diag(rel[:n]))
    pha = np.imag(np.diag(rel[:n]))
    within = rel[:n].copy()
    np.fill_diagonal(within, 0.0)
    perp = rel[n:]
    parts = {
        "amplitude": float(np.sum(w * amp**2)),
        "phase": float(np.sum(w * pha**2)),
        "within_rotation": float(np.sum(w[None, :] * np.abs(within) ** 2)),
        "unused_direction_rotation": float(np.sum(w[None, :] * np.abs(perp) ** 2)),
    }
    total = sum(parts.values()) or 1.0
    # Rotation generator iG in the branch frame. A tangent vector keeps u orthonormal, so the
    # within-branch block is anti-Hermitian; its defect is reported.
    iG = np.zeros((d, d), dtype=complex)
    iG[:, :n] = rel
    iG[np.arange(n), np.arange(n)] = 1j * pha
    iG[:n, n:] = -rel[n:].conj().T
    anti_defect = float(np.max(np.abs(iG[:n, :n] + iG[:n, :n].conj().T))) if n > 1 else 0.0
    iG[:n, :n] = 0.5 * (iG[:n, :n] - iG[:n, :n].conj().T)
    G = -1j * (Uf @ iG @ Uf.conj().T)
    G = 0.5 * (G + G.conj().T)
    coeffs = {lab: float(np.real(np.trace(pauli_matrix(lab) @ G)) / 4.0) for lab in PAULI_LABELS if lab != "II"}
    norm = math.sqrt(sum(v * v for v in coeffs.values())) or 1.0
    top = sorted(((lab, v / norm) for lab, v in coeffs.items() if abs(v) / norm > 1e-6), key=lambda t: -abs(t[1]))
    return {
        "fractions": {k: v / total for k, v in parts.items()},
        "amplitude_by_branch": amp.tolist(),
        "phase_by_branch": pha.tolist(),
        "within_rotation_antihermitian_defect": anti_defect,
        "generator_pauli_weights": [[lab, round(v, 6)] for lab, v in top[:8]],
        "generator": G,
    }


def _commutator_constraints(V: np.ndarray, mask: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Real linear constraints on a Hermitian generator (Pauli coordinates) for first-order invisibility.

    A pure rotation ``u_k -> u_k + i G u_k`` (``c`` fixed) changes ``<u_j|P|u_k>`` by ``i <u_j|[P,G]|u_k>``.
    D contexts read every pair ``j < k`` and the ``p``-weighted diagonal; F contexts read each diagonal.
    """
    d, n = V.shape
    c, Uf = _branch_frame(V)
    u = Uf[:, :n]
    p = c**2
    basis = [pauli_matrix(lab) for lab in PAULI_LABELS]
    rows, labels = [], []
    for cid in mask:
        clock, lab = cid.split(",")
        P = pauli_matrix(lab)
        comms = [P @ B - B @ P for B in basis]
        if clock == "D":
            for j in range(n):
                for k in range(j + 1, n):
                    vals = np.asarray([u[:, j].conj() @ C @ u[:, k] for C in comms])
                    rows.append(vals.real)
                    rows.append(vals.imag)
                    labels += [f"{cid}:pair{j}{k}:re", f"{cid}:pair{j}{k}:im"]
            vals = np.asarray([sum(p[j] * (u[:, j].conj() @ C @ u[:, j]) for j in range(n)) for C in comms])
            rows.append(vals.imag)
            labels.append(f"{cid}:diag")
        else:
            for j in range(n):
                vals = np.asarray([u[:, j].conj() @ C @ u[:, j] for C in comms])
                rows.append(vals.imag)
                labels.append(f"{cid}:branch{j}")
    return np.asarray(rows, dtype=float), labels


def rotation_kernel(V: np.ndarray, mask: Sequence[str]) -> dict[str, Any]:
    """Dimension of the rotation-only first-order kernel (restricted commutant of the observed Paulis).

    Two directions are always in it and change nothing: the identity and the projector onto the unused
    system direction(s) (``d_S - n`` of them, here one).
    """
    d, n = V.shape
    C, _ = _commutator_constraints(V, mask)
    _, s, Vt = np.linalg.svd(C)
    rank = int(np.sum(s > KERNEL_RELATIVE_TOL * max(float(s[0]), 1.0)))
    kernel = len(PAULI_LABELS) - rank
    trivial = 1 + (d - n) ** 2
    return {"kernel_dim": kernel, "trivial_dim": trivial, "nontrivial_dim": kernel - trivial, "basis": Vt[rank:].T}


def generator_in_kernel(V: np.ndarray, mask: Sequence[str], G: np.ndarray) -> float:
    """Relative residual of a generator in the rotation-only constraints."""
    C, _ = _commutator_constraints(V, mask)
    g = np.asarray([np.real(np.trace(pauli_matrix(lab) @ G)) / 4.0 for lab in PAULI_LABELS])
    scale = float(np.linalg.norm(C, 2) * np.linalg.norm(g)) or 1.0
    return float(np.linalg.norm(C @ g) / scale)


def _tables(ctx: Mapping[str, Any], theta: np.ndarray, ids: Sequence[str]) -> np.ndarray:
    T = ctx["fw"].tables(ctx["sbp"].branch_vectors(theta), ids)
    return np.concatenate([T[c] for c in ids])


def _state_distance(x0: np.ndarray, x1: np.ndarray) -> float:
    """Trace distance between the pure histories ``x0`` and ``x1``."""
    f = abs(np.vdot(x0, x1)) ** 2 / (np.vdot(x0, x0).real * np.vdot(x1, x1).real)
    return float(math.sqrt(max(0.0, 1.0 - f)))


def orbit_test(ctx: Mapping[str, Any], V: np.ndarray, G: np.ndarray, mask: Sequence[str]) -> list[dict[str, float]]:
    """Move every branch by ``exp(i t G)`` (amplitudes fixed) and read the support and unobserved tables."""
    fw = ctx["fw"]
    unobs = [c for c in ctx["ids"] if c not in mask]
    G = G / (np.linalg.norm(G) or 1.0)
    T0 = fw.tables(V, ctx["ids"])
    rows = []
    for t in ORBIT_T:
        Vt = expm(1j * t * G) @ V
        T = fw.tables(Vt, ctx["ids"])
        rows.append({
            "t": float(t),
            "support_max_abs": max(float(np.max(np.abs(T[c] - T0[c]))) for c in mask),
            "unobserved_max_abs": max(float(np.max(np.abs(T[c] - T0[c]))) for c in unobs),
            "state_distance": _state_distance(fw.vector(V), fw.vector(Vt)),
        })
    return rows


def follow_family(ctx: Mapping[str, Any], theta0: np.ndarray, mask: Sequence[str], eta0: np.ndarray, sign: float) -> dict[str, Any]:
    """Predictor-corrector along the exact-support null direction, from the truth, in one orientation."""
    sbp, fw = ctx["sbp"], ctx["fw"]
    mask = list(mask)
    unobs = [c for c in ctx["ids"] if c not in mask]
    target = _tables(ctx, theta0, mask)
    unobs0 = _tables(ctx, theta0, unobs)
    x0 = fw.vector(sbp.branch_vectors(theta0))

    def residual(t: np.ndarray) -> np.ndarray:
        return _tables(ctx, t, mask) - target

    def jac(t: np.ndarray) -> np.ndarray:
        x = fw.vector(sbp.branch_vectors(t))
        Jc = context_jacobians(fw, x, branch_derivatives(sbp, t), mask)
        return np.concatenate([Jc[c] for c in mask], axis=0)

    theta, prev = theta0.copy(), sign * eta0 / np.linalg.norm(eta0)
    arc, stop, traj, completed = 0.0, "max_steps", [], 0
    max_res = max_unobs = max_state = 0.0
    for step in range(CURVE_STEPS):
        x = fw.vector(sbp.branch_vectors(theta))
        dx = branch_derivatives(sbp, theta)
        Jc = context_jacobians(fw, x, dx, ctx["ids"])
        _, Q, _ = split_gauge(np.concatenate([Jc[c] for c in ctx["ids"]], axis=0))
        _, s, Wt = np.linalg.svd(np.concatenate([Jc[c] for c in mask], axis=0) @ Q)
        e = extra_null(s)
        if e == 0:
            stop = "null_vanished"
            break
        N = Q @ Wt[-e:].T                       # null directions in theta coordinates
        eta = N @ (N.T @ prev)
        if np.linalg.norm(eta) < 1e-12:
            eta = N[:, 0]
        eta = eta / np.linalg.norm(eta)
        if float(eta @ prev) < 0:
            eta = -eta
        pred = theta + CURVE_STEP * eta
        sol = least_squares(residual, pred, jac=jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200)
        new = sol.x
        res = float(np.max(np.abs(residual(new))))
        arc += float(np.linalg.norm(new - theta))
        x_new = fw.vector(sbp.branch_vectors(new))
        dev = float(np.max(np.abs(_tables(ctx, new, unobs) - unobs0)))
        dist = _state_distance(x0, x_new)
        probs = np.linalg.norm(sbp.branch_vectors(new), axis=0) ** 2
        point = {"step": step + 1, "arc_theta": arc, "support_residual": res, "unobserved_max_abs": dev, "state_distance": dist, "min_probability": float(np.min(probs))}
        left = res > CURVE_RESIDUAL_MAX
        floor = float(np.min(probs)) < CURVE_MIN_PROBABILITY
        if step % 5 == 4 or left or floor or step == CURVE_STEPS - 1:
            traj.append(point)
        if left:
            stop = "left_family"
            break
        completed = step + 1
        max_res, max_unobs, max_state = max(max_res, res), max(max_unobs, dev), max(max_state, dist)
        if floor:
            stop = "probability_floor"
            break
        prev, theta = eta, new
    return {
        "sign": sign,
        "steps_on_family": completed,
        "stop": stop,
        "arc_theta": arc,
        "max_support_residual": max_res,
        "max_unobserved_deviation": max_unobs,
        "max_state_distance": max_state,
        "trajectory": traj,
    }


def analyse_combination(ctx: Mapping[str, Any], unitary: str, mask: Sequence[str], probabilities, phases, tag: Mapping[str, Any]) -> dict[str, Any]:
    V0 = truth_branch_vectors(ctx["unitaries"][unitary], probabilities, phases)
    tp = truth_point(ctx, V0)
    V = ctx["sbp"].branch_vectors(tp["theta"])
    J = np.concatenate([tp["Jc"][c] for c in mask], axis=0) @ tp["Q"]
    _, s, Wt = np.linalg.svd(J)
    e = extra_null(s)
    out: dict[str, Any] = {**tag, "unitary": unitary, "mask": list(mask), "gauge_dim": tp["gauge_dim"], "extra_null": e,
                           "smin_rel": float(s[-1] / s[0]), "oracle": oracle_screen(ctx["fw"], V, mask)}
    kern = rotation_kernel(V, mask)
    out["rotation_kernel"] = {k: v for k, v in kern.items() if k != "basis"}
    directions = []
    for k in range(e):
        eta = tp["Q"] @ Wt[-1 - k]
        dV = (tp["dx"] @ eta).reshape(V.shape[1], V.shape[0]).T
        dec = decompose_direction(V, dV)
        G = dec.pop("generator")
        rot = 1.0 - dec["fractions"]["amplitude"]
        entry = {**dec, "rotation_fraction": rot, "generator_kernel_residual": generator_in_kernel(V, mask, G)}
        if dec["fractions"]["amplitude"] < 1e-8:
            entry["orbit"] = orbit_test(ctx, V, G, mask)
        directions.append(entry)
    out["null_directions"] = directions
    if e >= 1:
        eta0 = tp["Q"] @ Wt[-1]
        out["family"] = [follow_family(ctx, tp["theta"], mask, eta0, sgn) for sgn in (1.0, -1.0)]
    return out


# ---------------------------------------------------------------------------
# Retrospective records (E3)
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def retrospective_banks(config_dir: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    """(source, role) -> bank spec and deficits, read from the frozen configs (read only)."""
    config_dir = Path(config_dir)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    v7 = _read_yaml(config_dir / D1V7_CONFIG)["development_banks"]
    for role, spec in v7.items():
        if role == "deficits":
            continue
        out[("d1_v7", role)] = {"spec": dict(spec), "deficits": list(v7["deficits"])}
    t1 = _read_yaml(config_dir / T1_CONFIG)
    out[("t1_run2", "D_cal")] = {"spec": dict(t1["selector_calibration"]["bank"]), "deficits": ["low"]}
    for role in ("D_val_A", "D_val_B"):
        out[("t1_run2", role)] = {"spec": dict(t1["validation_banks"][role]), "deficits": ["low"]}
    a1 = _read_yaml(config_dir / A1_CONFIG)
    for role in ("D_atlas_cal", "D_atlas_eval"):
        out[("a1", role)] = {"spec": dict(a1["banks"][role]), "deficits": ["low"]}
    return out


def _bank_instances(ctx: dict[str, Any], key: tuple[str, str], spec: Mapping[str, Any]):
    cache = ctx.setdefault("bank_cache", {})
    if key not in cache:
        cache[key] = seeded_construction_bank_p1(ctx["p1"].generator, spec)
    return cache[key]


def retrospective_rows(ctx: dict[str, Any], source: str, role: str, deficit: str, spec: Mapping[str, Any], indices: Sequence[int]) -> list[dict[str, Any]]:
    instances = _bank_instances(ctx, (source, role), spec)
    rows = []
    for idx in indices:
        inst = instances[idx]
        mask = tuple(sorted(c.context_id for c in support_contexts_for_deficit(ctx["p1"].measurement, deficit, seed_key=f"p3v2:{spec['bank_id']}:{deficit}:{idx}")))
        coeffs = np.asarray(inst.coefficients, dtype=complex)
        probs, phases = np.abs(coeffs) ** 2, np.angle(coeffs)
        V = truth_branch_vectors(inst.system_unitary, probs, phases)
        tp = truth_point(ctx, V)
        rows.append({
            "source": source, "role": role, "deficit": deficit, "block": int(idx), "unitary": inst.system_unitary_label,
            "mask": list(mask), "probabilities": probs.tolist(), "phases": phases.tolist(), "gauge_dim": tp["gauge_dim"],
            **mask_readout(tp, mask), **oracle_screen(ctx["fw"], V, mask),
        })
    return rows


# ---------------------------------------------------------------------------
# Worker jobs
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(root: str, config_dir: str, draws: int) -> None:
    os.chdir(root)
    _WORKER.clear()
    _WORKER.update(model_context(config_dir, draws))


def e1_job(job: tuple[str, int]) -> dict[str, Any]:
    ctx = _WORKER
    label, r = job
    p, ph = ctx["draws"][r]
    V = truth_branch_vectors(ctx["unitaries"][label], p, ph)
    tp = truth_point(ctx, V)
    out: dict[str, Any] = {"unitary": label, "draw": r, "gauge_dim": tp["gauge_dim"], "cells": {}}
    for deficit in DEFICITS:
        reads = [mask_readout(tp, m) for m in ctx["masks"][deficit]]
        cell: dict[str, Any] = {
            "extra": [x["extra_null"] for x in reads],
            "smin_rel": [x["smin_rel"] for x in reads],
            "smallest_nonnull_rel": [x["smallest_nonnull_rel"] for x in reads],
        }
        if r == 0:
            cell["oracle"] = [oracle_screen(ctx["fw"], V, m) for m in ctx["masks"][deficit]]
        out["cells"][deficit] = cell
    return out


def e2_job(job: dict[str, Any]) -> dict[str, Any]:
    ctx = _WORKER
    started = time.perf_counter()
    out = analyse_combination(ctx, job["unitary"], job["mask"], job["probabilities"], job["phases"], job["tag"])
    out["seconds"] = time.perf_counter() - started
    return out


def e3_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    return retrospective_rows(_WORKER, job["source"], job["role"], job["deficit"], job["spec"], job["indices"])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def classify_draws(extra_by_draw: Sequence[int]) -> str:
    if all(e >= 1 for e in extra_by_draw):
        return "deficient"
    if all(e == 0 for e in extra_by_draw):
        return "identifiable"
    return "mixed"


def _histogram(values: Iterable[float], lo: float = -14.0, hi: float = 0.5, width: float = 0.5) -> dict[str, int]:
    edges = np.arange(lo, hi + width, width)
    logs = np.log10(np.clip(np.asarray(list(values), dtype=float), 1e-300, None))
    counts, _ = np.histogram(np.clip(logs, lo, hi - 1e-9), bins=edges)
    return {f"{edges[i]:.1f}": int(c) for i, c in enumerate(counts) if c}


def aggregate_map(ctx: Mapping[str, Any], e1: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by = {(r["unitary"], r["draw"]): r for r in e1}
    labels = sorted(ctx["unitaries"])
    draws = sorted({r["draw"] for r in e1})
    cells: dict[str, Any] = {}
    combos: list[dict[str, Any]] = []
    smin_identifiable, nonnull_deficient = [], []
    for deficit in DEFICITS:
        masks = ctx["masks"][deficit]
        for label in labels:
            oracle = by[(label, 0)]["cells"][deficit]["oracle"]
            counts = {"deficient": 0, "identifiable": 0, "mixed": 0}
            counts_in = {"deficient": 0, "identifiable": 0, "mixed": 0}
            min_smin = []
            for i, mask in enumerate(masks):
                extras = [by[(label, r)]["cells"][deficit]["extra"][i] for r in draws]
                cls = classify_draws(extras)
                counts[cls] += 1
                screened_in = not oracle[i]["oracle_screened_out"]
                if screened_in:
                    counts_in[cls] += 1
                if cls == "identifiable":
                    vals = [by[(label, r)]["cells"][deficit]["smin_rel"][i] for r in draws]
                    smin_identifiable += vals
                    if screened_in:
                        min_smin.append(min(vals))
                else:
                    nonnull_deficient += [by[(label, r)]["cells"][deficit]["smallest_nonnull_rel"][i] for r in draws]
                    combos.append({"deficit": deficit, "unitary": label, "mask_index": i, "mask": list(mask), "class": cls,
                                   "extra_by_draw": extras, "oracle": oracle[i], "screened_in": screened_in})
            cells[f"{deficit}:{label}"] = {
                "masks": len(masks),
                "counts": counts,
                "screened_in": sum(counts_in.values()),
                "counts_screened_in": counts_in,
                "screened_in_identifiable_min_smin_rel_quantiles": {
                    q: float(np.quantile(min_smin, float(q))) for q in ("0.01", "0.05", "0.5")
                } if min_smin else {},
            }
    incidence = {}
    for deficit in DEFICITS:
        num = sum(cells[f"{deficit}:{u}"]["counts_screened_in"]["deficient"] for u in labels)
        mix = sum(cells[f"{deficit}:{u}"]["counts_screened_in"]["mixed"] for u in labels)
        den = sum(cells[f"{deficit}:{u}"]["screened_in"] for u in labels)
        incidence[deficit] = {
            "deficient_screened_in": num, "mixed_screened_in": mix, "screened_in": den,
            "predicted_rate_given_screened_in": num / den if den else None,
            "note": "unitary and mask are uniform and independent under the generator, so the rate is a ratio of combination counts",
        }
    gauge_dims = sorted({r["gauge_dim"] for r in e1})
    return {
        "draws": len(draws),
        "gauge_dims_seen": gauge_dims,
        "cells": cells,
        "predicted_incidence": incidence,
        "combinations_not_identifiable": combos,
        "log10_smin_rel_histogram_identifiable": _histogram(smin_identifiable),
        "log10_smallest_nonnull_rel_histogram_deficient": _histogram(nonnull_deficient),
        "separation": {
            "identifiable_min_smin_rel": float(min(smin_identifiable)) if smin_identifiable else None,
            "null_cut_rel": NULL_RELATIVE_TOL,
        },
    }


def _artifact_index() -> dict[str, dict[tuple[str, str, int], dict[str, Any]]]:
    out: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {"d1_v7": {}, "t1_run2": {}, "a1": {}}
    v7 = json.loads(D1V7_ARTIFACT.read_text(encoding="utf-8"))
    for row in v7["record_metrics"].values():
        out["d1_v7"][(row["role"], row["deficit"], int(row["block"]))] = {
            "unitary": row["unitary"], "true_mask_class": row["true_mask_class"], "screen_passes": row["screen_passes"],
            "diag_score": row["diag_score"], "coh_score": row["coh_score"], "coef_error_rms": row["coef_error_rms"],
        }
    for source, path in (("t1_run2", T1_ARTIFACT), ("a1", A1_ARTIFACT)):
        art = json.loads(path.read_text(encoding="utf-8"))
        for role, rows in art["records"].items():
            for row in rows:
                sel = row.get("selector") or {}
                out[source][(role, "low", int(row["block"]))] = {   # screened-out rows carry no unitary label
                    "unitary": row.get("unitary"), "true_mask_class": row["true_mask_class"], "screen_passes": row.get("screen_passes"),
                    "diag_score": row.get("diag_score"), "coef_error_rms": row.get("coef_error_rms"),
                    "selector_status": sel.get("status"), "selector_claim_status": sel.get("claim_status"),
                    "selector_error_bound": sel.get("error_bound"),
                    "bi_status": (row.get("bi") or {}).get("status"),
                }
    return out


def join_retrospective(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = _artifact_index()
    mismatches = []
    for row in rows:
        art = index[row["source"]].get((row["role"], row["deficit"], row["block"]))
        row["artifact"] = art
        if art is None:
            continue
        if (art["unitary"] is not None and art["unitary"] != row["unitary"]) or art["true_mask_class"] != row["true_mask_class"]:
            mismatches.append({k: row[k] for k in ("source", "role", "deficit", "block", "unitary", "true_mask_class")} | {"artifact": art})
    return {"rows_with_artifact": sum(1 for r in rows if r["artifact"] is not None), "rebuild_mismatches": mismatches}


def summarize_retrospective(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # D1-v7: is structural deficiency what made the wide tail?
    for deficit in DEFICITS:
        v7 = [r for r in rows if r["source"] == "d1_v7" and r["deficit"] == deficit and r["artifact"] is not None]
        inside = [r for r in v7 if r["artifact"]["screen_passes"]]
        deficient = [r for r in inside if r["extra_null"] >= 1]
        scores_def = sorted(float(r["artifact"]["diag_score"]) for r in deficient)
        scores_ok = [float(r["artifact"]["diag_score"]) for r in inside if r["extra_null"] == 0]
        out[f"d1_v7:{deficit}"] = {
            "records": len(v7),
            "screened_in": len(inside),
            "deficient_screened_in": len(deficient),
            "deficient_by_unitary": {u: sum(1 for r in deficient if r["unitary"] == u) for u in sorted({r["unitary"] for r in deficient})},
            "deficient_diag_scores": scores_def,
            "identifiable_diag_score_max": max(scores_ok) if scores_ok else None,
            "identifiable_diag_score_p99": float(np.quantile(scores_ok, 0.99)) if scores_ok else None,
            "deficient_records": [{k: r[k] for k in ("role", "block", "unitary", "extra_null")} | {"diag_score": r["artifact"]["diag_score"], "mask": r["mask"]} for r in deficient],
            "screened_in_with_diag_score_above_0.15": [{k: r[k] for k in ("role", "block", "unitary", "extra_null", "smin_rel")} | {"diag_score": r["artifact"]["diag_score"]} for r in inside if float(r["artifact"]["diag_score"]) > 0.15],
        }
    # T1 / A1: does the selector tail sit on or near the deficient set?
    for source in ("t1_run2", "a1"):
        sel = [r for r in rows if r["source"] == source and r["artifact"] is not None and isinstance(r["artifact"].get("selector_error_bound"), (int, float))]
        err = np.asarray([float(r["artifact"]["selector_error_bound"]) for r in sel])
        smin = np.asarray([float(r["smin_rel"]) for r in sel])
        rho = spearmanr(smin, err) if len(sel) >= 3 else None
        top = sorted(sel, key=lambda r: -float(r["artifact"]["selector_error_bound"]))[:10]
        out[source] = {
            "records_with_selector_error": len(sel),
            "deficient": sum(1 for r in sel if r["extra_null"] >= 1),
            "spearman_smin_rel_vs_error_bound": {"rho": float(rho[0]), "p": float(rho[1])} if rho is not None else None,
            "exceed_budget": [{k: r[k] for k in ("role", "block", "unitary", "extra_null", "smin_rel")} | {"error_bound": r["artifact"]["selector_error_bound"]} for r in sel if float(r["artifact"]["selector_error_bound"]) > SELECTOR_BUDGET],
            "top10_error": [{k: r[k] for k in ("role", "block", "unitary", "extra_null", "smin_rel")} | {"error_bound": r["artifact"]["selector_error_bound"]} for r in top],
            "smin_rel_quantiles_by_unitary": {
                u: {q: float(np.quantile([float(r["smin_rel"]) for r in sel if r["unitary"] == u], float(q))) for q in ("0.1", "0.5")}
                for u in sorted({r["unitary"] for r in sel})
            },
        }
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    return obj


def _chunks(seq: Sequence[int], size: int) -> list[list[int]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def plan_e2(aggregate: Mapping[str, Any], e3_rows: Sequence[Mapping[str, Any]], draws: Sequence[tuple[np.ndarray, np.ndarray]], cap_cell: int, cap_records: int) -> list[dict[str, Any]]:
    jobs = []
    per_cell: dict[tuple[str, str], int] = {}
    for combo in aggregate["combinations_not_identifiable"]:
        key = (combo["deficit"], combo["unitary"])
        if per_cell.get(key, 0) >= cap_cell:
            continue
        r = next(i for i, e in enumerate(combo["extra_by_draw"]) if e >= 1)
        p, ph = draws[r]
        jobs.append({"unitary": combo["unitary"], "mask": combo["mask"], "probabilities": p.tolist(), "phases": ph.tolist(),
                     "tag": {"kind": "synthetic", "deficit": combo["deficit"], "mask_index": combo["mask_index"], "draw": r, "class": combo["class"]}})
        per_cell[key] = per_cell.get(key, 0) + 1
    recs = [r for r in e3_rows if r["extra_null"] >= 1]
    recs.sort(key=lambda r: (r["source"] != "d1_v7", r["source"], r["role"], r["deficit"], r["block"]))
    for r in recs[:cap_records]:
        jobs.append({"unitary": r["unitary"], "mask": r["mask"], "probabilities": r["probabilities"], "phases": r["phases"],
                     "tag": {"kind": "record", "source": r["source"], "role": r["role"], "deficit": r["deficit"], "block": r["block"]}})
    return jobs


def summarize_mechanism(e2: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("synthetic", "record"):
        rows = [r for r in e2 if r["kind"] == kind and r["extra_null"] >= 1]
        if not rows:
            out[kind] = {"analysed": 0}
            continue
        dirs = [d for r in rows for d in r["null_directions"]]
        fam = [f for r in rows for f in r.get("family", [])]
        exact = [r for r in rows if all(f["stop"] != "left_family" or f["arc_theta"] > 0.5 for f in r.get("family", []))]
        out[kind] = {
            "analysed": len(rows),
            "extra_null_counts": {str(k): sum(1 for r in rows if r["extra_null"] == k) for k in sorted({r["extra_null"] for r in rows})},
            "rotation_kernel_nontrivial_equals_extra_null": sum(1 for r in rows if r["rotation_kernel"]["nontrivial_dim"] == r["extra_null"]),
            "rotation_kernel_nontrivial_dims": {str(k): sum(1 for r in rows if r["rotation_kernel"]["nontrivial_dim"] == k) for k in sorted({r["rotation_kernel"]["nontrivial_dim"] for r in rows})},
            "direction_fraction_means": {k: float(np.mean([d["fractions"][k] for d in dirs])) for k in dirs[0]["fractions"]},
            "directions_pure_rotation": sum(1 for d in dirs if d["fractions"]["amplitude"] < 1e-8),
            "directions_with_unused_direction_share_above_0.5": sum(1 for d in dirs if d["fractions"]["unused_direction_rotation"] > 0.5),
            "orbit_exact": sum(1 for d in dirs if "orbit" in d and max(o["support_max_abs"] for o in d["orbit"]) < 1e-10),
            "orbit_tested": sum(1 for d in dirs if "orbit" in d),
            "family_stop_reasons": {s: sum(1 for f in fam if f["stop"] == s) for s in sorted({f["stop"] for f in fam})},
            "family_arc_theta_quantiles": {q: float(np.quantile([f["arc_theta"] for f in fam], float(q))) for q in ("0.1", "0.5", "0.9")},
            "family_max_unobserved_deviation_quantiles": {q: float(np.quantile([f["max_unobserved_deviation"] for f in fam], float(q))) for q in ("0.1", "0.5", "0.9")},
            "family_max_state_distance_quantiles": {q: float(np.quantile([f["max_state_distance"] for f in fam], float(q))) for q in ("0.1", "0.5", "0.9")},
            "records_with_both_orientations_exact_beyond_arc_0.5": len(exact),
            "top_generator_paulis": _top_paulis(dirs),
        }
    return out


def _top_paulis(dirs: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    tally: dict[str, float] = {}
    for d in dirs:
        for lab, w in d["generator_pauli_weights"]:
            tally[lab] = tally.get(lab, 0.0) + float(w) ** 2
    return [[lab, round(v / max(len(dirs), 1), 4)] for lab, v in sorted(tally.items(), key=lambda t: -t[1])[:10]]


def run_m1_support_identifiability_map(
    config_dir: str | Path = "config",
    output_dir: str | Path = "outputs",
    *,
    workers: int = 1,
    draws: int = DRAWS,
    e2_cap_cell: int = E2_CAP_PER_CELL,
    e2_cap_records: int = E2_CAP_RECORDS,
    unitaries: Sequence[str] | None = None,
    retrospective: bool = True,
    progress: Callable[[str], None] | None = None,
    force: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    say = progress or (lambda line: None)
    root = Path.cwd()
    out_path = Path(output_dir) / OUTPUT_SUBDIR / ARTIFACT
    if write and out_path.exists() and not force:
        raise FileExistsError(f"{out_path} exists; pass force to overwrite")
    started = time.perf_counter()
    ctx = model_context(config_dir, draws)
    labels = sorted(unitaries) if unitaries else sorted(ctx["unitaries"])
    if unitaries:
        ctx["unitaries"] = {k: v for k, v in ctx["unitaries"].items() if k in labels}
    timings: dict[str, float] = {}

    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    pool = None
    try:
        if workers > 1:
            for name in BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(root), str(Path(config_dir).resolve()), int(draws)))
        else:
            _WORKER.clear()
            _WORKER.update(ctx)

        def run_map(fn, jobs, chunksize=1):
            if pool is None:
                return [fn(j) for j in jobs]
            return list(pool.map(fn, jobs, chunksize=chunksize))

        t = time.perf_counter()
        e1 = run_map(e1_job, [(u, r) for u in labels for r in range(int(draws))])
        e1 = [r for r in e1 if r["unitary"] in labels]
        agg = aggregate_map({**ctx, "unitaries": {u: ctx["unitaries"][u] for u in labels}}, e1)
        timings["e1"] = time.perf_counter() - t
        say(f"E1 done in {timings['e1']:.0f}s: not identifiable={len(agg['combinations_not_identifiable'])} "
            f"incidence={ {d: agg['predicted_incidence'][d]['predicted_rate_given_screened_in'] for d in DEFICITS} }")

        e3_rows: list[dict[str, Any]] = []
        joined: dict[str, Any] = {}
        if retrospective:
            t = time.perf_counter()
            banks = retrospective_banks(config_dir)
            jobs = []
            for (source, role), info in banks.items():
                n = int(info["spec"]["block_count"])
                for deficit in info["deficits"]:
                    for chunk in _chunks(list(range(n)), 32):
                        jobs.append({"source": source, "role": role, "deficit": deficit, "spec": info["spec"], "indices": chunk})
            for part in run_map(e3_job, jobs):
                e3_rows.extend(part)
            e3_rows = [r for r in e3_rows if r["unitary"] in labels]
            joined = join_retrospective(e3_rows)
            timings["e3"] = time.perf_counter() - t
            say(f"E3 done in {timings['e3']:.0f}s: rows={len(e3_rows)} deficient={sum(1 for r in e3_rows if r['extra_null'] >= 1)} mismatches={len(joined['rebuild_mismatches'])}")

        t = time.perf_counter()
        e2_jobs = plan_e2(agg, e3_rows, ctx["draws"], int(e2_cap_cell), int(e2_cap_records))
        e2 = run_map(e2_job, e2_jobs)
        for row, job in zip(e2, e2_jobs):
            row["kind"] = job["tag"]["kind"]
        timings["e2"] = time.perf_counter() - t
        say(f"E2 done in {timings['e2']:.0f}s: analysed={len(e2)}")
    finally:
        if pool is not None:
            pool.shutdown()
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    sources = [Path(__file__), Path("runners/p3/run_m1_support_identifiability_map.py"),
               Path(config_dir) / "p1_generator.yaml", Path(config_dir) / "p1_measurement.yaml",
               Path(config_dir) / D1V7_CONFIG, Path(config_dir) / T1_CONFIG, Path(config_dir) / A1_CONFIG,
               D1V7_ARTIFACT, T1_ARTIFACT, A1_ARTIFACT]
    artifact = {
        "artifact_type": METHOD,
        "status": "development_diagnostic",
        "not_declared": "roadmap M1 cause-identification experiment; opens no observation bank, samples no shots, changes no frozen result",
        "design_document": "documents/M1-third-mechanism-design.md",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "unitaries": labels, "deficits": list(DEFICITS), "draws": int(draws), "draw_seed": DRAW_SEED,
            "dirichlet_concentration": DIRICHLET_CONCENTRATION, "minimum_mode_probability": MINIMUM_MODE_PROBABILITY,
            "fd_step": FD_STEP, "null_relative_tol": NULL_RELATIVE_TOL, "oracle_pair_threshold": ORACLE_PAIR_THRESHOLD,
            "kernel_relative_tol": KERNEL_RELATIVE_TOL, "curve_step": CURVE_STEP, "curve_steps": CURVE_STEPS,
            "curve_residual_max": CURVE_RESIDUAL_MAX, "e2_cap_cell": int(e2_cap_cell), "e2_cap_records": int(e2_cap_records),
            "masks": {d: len(ctx["masks"][d]) for d in DEFICITS},
        },
        "inputs_sha256": {str(p).replace("\\", "/"): _sha256(p) for p in sources if p.exists()},
        "e1_structural_map": agg,
        "e2_mechanism_summary": summarize_mechanism(e2),
        "e2_mechanism": e2,
        "e3_retrospective_join": joined,
        "e3_retrospective_summary": summarize_retrospective(e3_rows) if retrospective else None,
        "e3_retrospective_rows": [{k: v for k, v in r.items() if k not in ("probabilities", "phases")} for r in e3_rows],
        "execution": {"workers": int(workers), "seconds": timings, "total_seconds": time.perf_counter() - started},
    }
    artifact = _jsonable(artifact)
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    return artifact
