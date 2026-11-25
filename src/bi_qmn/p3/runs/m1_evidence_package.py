"""M1.1 / M1.2 / E6: explicit identifiability cases, independent Born-rule cross-checks and a reproduction kit.

Plan: documents/BI-QMN-remaining-experiments-maximum-package.md §3 (M1.1, M1.2) and §4 E6. Development
only: it reads the already-opened M1 development bank (``m1_added_measurement_v1``) and its post-hoc
artifact, and opens nothing sealed.

Cases (one state for four masks, so that the mask alone decides the type):

* ``type1``  identity unitary, D {IX, XI, XX, YI, ZI, ZX} + F {XX, ZI}: every observed Pauli commutes
  with ``R = IX``; the orbit ``exp(itR)`` is exact for every state.
* ``type3a`` same state, F {XX, YY}: the D Paulis commute with ``IX``; the only anticommuting F Pauli
  ``YY`` has ``<YY>_j = <i YY IX>_j = <YZ>_j = 0`` on every branch, so the orbit is exact.
* ``type3b`` same state, F {ZI, ZZ}: ``<ZZ>_j = +-1`` but ``<i ZZ IX>_j = -<ZY>_j = 0``, so the support
  moves only at second order along the orbit (a first-order null).
* ``normal`` same state, D {IX, XI, XX, YI, ZI, ZZ} + F {XX, YY}: no D commutant, every branch pair
  observed.
* ``type2``  a cnot_h0 construction whose observed D Paulis leave one branch disconnected: its relative
  phase is free.

For every case: the support residual and the held-out difference along the alternative family (from the
independent Born engine), trace distances of the histories and of ``rho_{S|tau}``, the conjugation
identity, the registered energy, the branch-condition components, the first-order Jacobian rank, the
order of the support response (3b), and a multi-start global search on the exact support tables
(numerical evidence, with its scope stated). M1.2 adds the orbit coefficient matrices of the added
candidates and global searches with one added context, and the zero-energy effect-map rank on the
128 development 3a records.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_hash
from bi_qmn.p3 import independent_born as ib
from bi_qmn.p3.runs import m1_added_measurement as am
from bi_qmn.p3.runs import m1_added_measurement_posthoc as ph
from bi_qmn.p3.runs import m1_d_commutant_check as dc
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _true_branch_vectors


METHOD = "m1_evidence_package_v1"
OUTPUT_FOLDER = "m1_evidence_package"
ARTIFACT = "m1_evidence_package.json"
KIT = "m1_reproduction_kit_3a.json"
RUNNER = "runners/p3/run_m1_evidence_package.py"
SOURCE = Path("outputs") / "m1_added_measurement" / "m1_added_measurement.json"
STATE_INDEX = 469          # identity unitary, a declared 3a record of the development bank
TYPE2_INDEX = 7            # cnot_h0, screened out as branch-disconnected without a Pauli commutant
D_BASE = ("D,IX", "D,XI", "D,XX", "D,YI", "D,ZI", "D,ZX")
MASKS = {
    "type1": (*D_BASE, "F,XX", "F,ZI"),
    "type3a": (*D_BASE, "F,XX", "F,YY"),
    "type3b": (*D_BASE, "F,ZI", "F,ZZ"),
    "normal": ("D,IX", "D,XI", "D,XX", "D,YI", "D,ZI", "D,ZZ", "F,XX", "F,YY"),
}
ORBIT_T = tuple(float(t) for t in np.linspace(0.0, math.pi, 25)[1:-1])
ORDER_T = tuple(float(t) for t in np.logspace(-4, -1.5, 6))
EXACT_RESIDUAL = 1.0e-6
SUBSTANTIVE_HELDOUT = 1.0e-3


# ---------------------------------------------------------------------------
# Independent-engine helpers
# ---------------------------------------------------------------------------


def amplitude(instance, V: np.ndarray | None = None) -> np.ndarray:
    """``(K, d_S)`` amplitude array of the history, from branch vectors ``V`` (``c_j u_j``) or the instance."""
    if V is None:
        V = _true_branch_vectors(instance)
    return ib.history(instance.active_modes, np.ones(len(instance.active_modes)), np.asarray(V, dtype=complex), int(instance.K))


def branch_components(V: np.ndarray, mask: Sequence[str], R: str) -> dict[str, list[dict[str, float]]]:
    """``<P>_j`` and ``<iPR>_j`` for every observed F Pauli ``P`` that anticommutes with ``R``."""
    G = ib.pauli(R)
    c = np.linalg.norm(V, axis=0)
    u = V / c[None, :]
    out = {}
    for cid in mask:
        clock, label = cid.split(",")
        P = ib.pauli(label)
        if clock != "F" or np.linalg.norm(P @ G + G @ P) > 1e-12:
            continue
        iPR = 1j * P @ G
        out[label] = [{"weight": float(c[j] ** 2), "P": float(np.real(u[:, j].conj() @ P @ u[:, j])), "iPR": float(np.real(u[:, j].conj() @ iPR @ u[:, j]))} for j in range(u.shape[1])]
    return out


def family_point(A0: np.ndarray, family: Mapping[str, Any], t: float) -> np.ndarray:
    if family["kind"] == "orbit":
        return ib.rotate_system(A0, expm(1j * t * ib.pauli(family["R"])))
    if family["kind"] == "branch_phase":
        A = np.array(A0, copy=True)
        A[int(family["mode"])] *= np.exp(1j * t)
        return A
    raise ValueError(family["kind"])


def family_readout(A0: np.ndarray, family: Mapping[str, Any], mask: Sequence[str], ids: Sequence[str], ts: Sequence[float], H_tot: np.ndarray | None) -> dict[str, Any]:
    support0 = ib.tables(A0, mask)
    held = [c for c in ids if c not in set(mask)]
    held0 = ib.tables(A0, held)
    p0, rho0 = ib.conditional_states(A0)
    rows = []
    for t in ts:
        A = family_point(A0, family, t)
        p, rho = ib.conditional_states(A)
        row = {"t": float(t), "support_residual": ib.max_table_difference(ib.tables(A, mask), support0, mask),
               "heldout_max_abs": ib.max_table_difference(ib.tables(A, held), held0, held),
               "history_trace_distance": ib.pure_trace_distance(A0, A),
               "p_tau_max_change": float(np.max(np.abs(p - p0))),
               "conditional_state_trace_distance_max": max(ib.trace_distance(rho0[l], rho[l]) for l in range(len(p0)))}
        if family["kind"] == "orbit":
            Ut = expm(1j * t * ib.pauli(family["R"]))
            row["conjugation_identity_error"] = float(max(np.max(np.abs(rho[l] - Ut @ rho0[l] @ Ut.conj().T)) for l in range(len(p0))))
        if H_tot is not None:
            row["registered_energy"] = ib.normalized_energy(A, H_tot)
        rows.append(row)
    best = max(rows, key=lambda r: r["heldout_max_abs"])
    out = {"points": rows, "max_support_residual": max(r["support_residual"] for r in rows), "max_heldout_difference": best["heldout_max_abs"], "t_at_max_heldout": best["t"],
           "history_trace_distance_at_that_t": best["history_trace_distance"], "conditional_state_trace_distance_at_that_t": best["conditional_state_trace_distance_max"]}
    if H_tot is not None and family["kind"] == "orbit":
        B = [r["registered_energy"] / math.sin(r["t"]) ** 2 for r in rows if abs(math.sin(r["t"])) > 1e-3]
        out["energy_B"] = {"mean": float(np.mean(B)), "relative_spread": float((max(B) - min(B)) / max(abs(np.mean(B)), 1e-300))}
    return out


def response_order(A0: np.ndarray, family: Mapping[str, Any], mask: Sequence[str], ts: Sequence[float] = ORDER_T) -> dict[str, Any]:
    """Slope of log(support residual) against log(t) near the truth: 1 = first order, 2 = first-order null."""
    support0 = ib.tables(A0, mask)
    res = [ib.max_table_difference(ib.tables(family_point(A0, family, t), mask), support0, mask) for t in ts]
    ok = [(math.log(t), math.log(r)) for t, r in zip(ts, res) if r > 1e-13]
    slope = float(np.polyfit([a for a, _ in ok], [b for _, b in ok], 1)[0]) if len(ok) >= 3 else None
    return {"t": list(ts), "support_residual": res, "log_slope": slope}


# ---------------------------------------------------------------------------
# Global multi-start search on exact tables (numerical evidence)
# ---------------------------------------------------------------------------


def global_search(sbp, exact: Mapping[str, Mapping[str, float]], observed: Sequence[str], ids: Sequence[str], starts: int, seed: int, R: str | None, V_true: np.ndarray) -> dict[str, Any]:
    """Minimize the SBP cross-entropy to the exact observed tables from ``starts`` random points."""
    fw = sbp.forward
    support = {c: exact[c] for c in observed}
    ids_obs, A, counts = sbp._support_arrays(support, 1.0e6)
    rng = np.random.default_rng(seed)
    d, n = fw.d_S, fw.n
    held = [c for c in ids if c not in set(observed)]
    true_tables = {c: np.asarray([float(exact[c][o]) for o in fw.outcomes]) for c in ids}
    rows = []
    for _ in range(int(starts)):
        theta0 = np.concatenate([rng.normal(0.0, 1.0, d * d), np.zeros(n - 1), rng.uniform(0.0, 2.0 * math.pi, n - 1)])
        res = minimize(sbp._objective, theta0, args=(A, counts), jac=True, method="L-BFGS-B", options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10})
        V = sbp.branch_vectors(res.x)
        T = fw.tables(V, ids)
        row = {"objective": float(res.fun), "support_residual": float(max(np.max(np.abs(T[c] - true_tables[c])) for c in observed)),
               "heldout_difference": float(max(np.max(np.abs(T[c] - true_tables[c])) for c in held)) if held else 0.0}
        if R is not None:
            p = ph.nearest_orbit_point(fw, V, V_true, m1.pauli_matrix(R), points=180)
            row["orbit_t"], row["orbit_distance"] = p["t"], p["distance"]
        rows.append(row)
    exact_rows = [r for r in rows if r["support_residual"] < EXACT_RESIDUAL]
    different = [r for r in rows if r["heldout_difference"] > SUBSTANTIVE_HELDOUT]
    out = {
        "starts": int(starts), "exact_fits": len(exact_rows),
        "exact_fits_with_substantive_heldout_difference": sum(r["heldout_difference"] > SUBSTANTIVE_HELDOUT for r in exact_rows),
        "max_heldout_difference_among_exact_fits": max((r["heldout_difference"] for r in exact_rows), default=None),
        "min_support_residual_among_substantively_different": min((r["support_residual"] for r in different), default=None),
        "support_residual_quantiles": {q: float(np.quantile([r["support_residual"] for r in rows], float(q))) for q in ("0.0", "0.5", "1.0")},
        "thresholds": {"exact_residual": EXACT_RESIDUAL, "substantive_heldout": SUBSTANTIVE_HELDOUT},
    }
    if R is not None and exact_rows:
        out["orbit_t_of_exact_fits"] = sorted(round(r["orbit_t"], 3) for r in exact_rows)
        out["max_orbit_distance_of_exact_fits"] = max(r["orbit_distance"] for r in exact_rows)
    return out


# ---------------------------------------------------------------------------
# M1.2: orbit coefficient matrices (independent engine) and the zero-energy effect rank
# ---------------------------------------------------------------------------


def coefficient_matrix_independent(A0: np.ndarray, R: str, q: str) -> dict[str, float]:
    T = lambda t: np.asarray(list(ib.tables(family_point(A0, {"kind": "orbit", "R": R}, t), [q])[q].values()))   # noqa: E731
    T0, Tq, Th = T(0.0), T(math.pi / 4), T(math.pi / 2)
    c, a = (T0 + Th) / 2.0, (T0 - Th) / 2.0
    b = Tq - c
    s = np.linalg.svd(np.column_stack([a, b]), compute_uv=False)
    resid = max(float(np.max(np.abs(T(t) - (c + a * math.cos(2 * t) + b * math.sin(2 * t))))) for t in (0.3, 1.1, 2.4))
    return {"sigma_max": float(s[0]), "sigma_min": float(s[-1]), "rank": int(np.sum(s > 1e-12 * max(s[0], 1e-300))), "sinusoid_residual": resid}


def zero_energy_effect_rank(H_tot: np.ndarray, K: int, d_S: int, modes: Sequence[int], context_ids: Sequence[str], tol: float = 1e-10) -> dict[str, int]:
    """Real rank of ``rho -> (Tr E rho)`` on Hermitian operators supported on ``ker(H_tot)`` within the active modes."""
    B = np.zeros((K * d_S, len(modes) * d_S), dtype=complex)
    for j, m in enumerate(modes):
        B[m * d_S:(m + 1) * d_S, j * d_S:(j + 1) * d_S] = np.eye(d_S)
    _, s, vh = np.linalg.svd(np.asarray(H_tot) @ B)
    rank_H = int(np.sum(s > tol * s[0]))
    N = B @ vh[rank_H:].conj().T                     # kernel basis in the full clock-major space
    k = N.shape[1]
    # Hermitian basis of k x k (k^2 real directions)
    basis = []
    for a in range(k):
        for b in range(a, k):
            E = np.zeros((k, k), dtype=complex)
            if a == b:
                E[a, a] = 1.0
                basis.append(E)
            else:
                E[a, b] = E[b, a] = 1.0
                basis.append(E.copy())
                E = np.zeros((k, k), dtype=complex)
                E[a, b], E[b, a] = 1j, -1j
                basis.append(E)
    rows = []
    for cid in context_ids:
        clock, label = cid.split(",")
        P = ib.pauli(label)
        for l in range(K):
            v = ib.clock_vector(K, clock, l)
            for sgn in (1, -1):
                eff = np.kron(np.outer(v, v.conj()), (np.eye(d_S) + sgn * P) / 2.0)
                Ek = N.conj().T @ eff @ N
                rows.append([float(np.real(np.trace(Ek @ X))) for X in basis])
    M = np.asarray(rows)
    sv = np.linalg.svd(M, compute_uv=False)
    return {"kernel_dimension": k, "hermitian_dimension": k * k, "rank": int(np.sum(sv > tol * sv[0]))}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _jacobian_readout(ctx: Mapping[str, Any], V: np.ndarray, mask: Sequence[str]) -> dict[str, Any]:
    tp = m1.truth_point({"sbp": ctx["sbp"], "fw": ctx["fw"], "ids": ctx["ids"]}, V)
    r = m1.mask_readout(tp, mask)
    return {"gauge_dim": int(tp["gauge_dim"]), "extra_null": int(r["extra_null"]), "smin_rel": float(r["smin_rel"]), "smallest_nonnull_rel": float(r["smallest_nonnull_rel"])}


def run_m1_evidence_package(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, starts: int = 128, seed: int = 20261122,
                            progress=print, write: bool = True, rank_audit: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    config_dir = Path(config_dir)
    folder = Path(output_dir) / OUTPUT_FOLDER
    if write and (folder / ARTIFACT).exists():
        raise FileExistsError(f"evidence package already exists: {folder / ARTIFACT}")
    ctx = am._prepare_light(config_dir)
    ids, K, d_S = ctx["ids"], ctx["K"], ctx["d_S"]
    say = progress or (lambda _: None)

    # -- E6: independent engine against the operational tables -------------------------------
    cross = {}
    for i in (STATE_INDEX, TYPE2_INDEX, 0, 771):
        inst = ctx["instances"][i]
        A0 = amplitude(inst)
        T = ib.tables(A0, ids)
        r = am.build_construction(ctx, i)
        fwT = ctx["fw"].tables(_true_branch_vectors(inst), ids)
        cross[str(i)] = {"unitary": inst.system_unitary_label,
                         "vs_p1_measurement_table": ib.max_table_difference(T, r["exact"], ids),
                         "vs_sbp_forward": float(max(abs(T[c][o] - fwT[c][k]) for c in ids for k, o in enumerate(ctx["fw"].outcomes))),
                         "vs_instance_statevector": float(np.max(np.abs(A0.reshape(-1) - np.asarray(inst.statevector.data)))),
                         "registered_energy_of_truth": ib.normalized_energy(A0, np.asarray(inst.H_tot.data))}
    say(f"E6 cross-check: {cross}")

    # -- M1.1 cases ------------------------------------------------------------------------------
    inst = ctx["instances"][STATE_INDEX]
    V = _true_branch_vectors(inst)
    A0 = amplitude(inst)
    H = np.asarray(inst.H_tot.data)
    exact = ib.tables(A0, ids)
    cases: dict[str, Any] = {}
    for name, mask in MASKS.items():
        D_paulis = [c for c in mask if c.startswith("D,")]
        Rs = dc.d_commutant(tuple(mask))
        R = Rs[0] if Rs else None
        oracle = m1.oracle_screen(ctx["fw"], V, list(mask))
        case = {"construction_index": STATE_INDEX, "unitary": inst.system_unitary_label, "active_modes": list(inst.active_modes),
                "branch_weights": (np.abs(np.asarray(inst.coefficients)) ** 2).tolist(), "branch_phases": np.angle(np.asarray(inst.coefficients)).tolist(),
                "mask": list(mask), "D": D_paulis, "F": [c for c in mask if c.startswith("F,")], "R": R,
                "full_pauli_commutant": oracle["pauli_commutant"], "branch_connectivity": oracle["true_mask_class"],
                "jacobian": _jacobian_readout(ctx, V, mask)}
        if R is not None:
            fam = {"kind": "orbit", "R": R}
            case["branch_components"] = branch_components(V, mask, R)
            case["family"] = family_readout(A0, fam, mask, ids, ORBIT_T, H)
            case["response_order"] = response_order(A0, fam, mask)
        say(f"case {name}: R={R} jacobian={case['jacobian']}")
        case["global_search"] = global_search(ctx["sbp"], exact, mask, ids, starts, seed + len(cases), R, V)
        say(f"  global search: {case['global_search']['exact_fits']} exact fits, substantive alternatives {case['global_search']['exact_fits_with_substantive_heldout_difference']}")
        cases[name] = case

    # type 2: a branch-disconnected construction without a Pauli commutant
    r2 = am.build_construction(ctx, TYPE2_INDEX)
    inst2 = ctx["instances"][TYPE2_INDEX]
    V2 = _true_branch_vectors(inst2)
    A2 = amplitude(inst2)
    oracle2 = m1.oracle_screen(ctx["fw"], V2, r2["mask"])
    obs2 = ctx["fw"].pair_observability(V2, [c for c in r2["mask"] if c.startswith("D,")], 0.05)
    dis = obs2["disconnected_modes"]
    fam2 = {"kind": "branch_phase", "mode": int(dis[0])} if dis else None
    case2 = {"construction_index": TYPE2_INDEX, "unitary": inst2.system_unitary_label, "active_modes": list(inst2.active_modes), "mask": r2["mask"],
             "full_pauli_commutant": oracle2["pauli_commutant"], "branch_connectivity": oracle2["true_mask_class"], "pair_strength": obs2["strength"],
             "disconnected_modes": dis, "jacobian": _jacobian_readout(ctx, V2, r2["mask"]), "screen_reasons": r2["screen_reasons"]}
    if fam2 is not None:
        case2["family"] = family_readout(A2, fam2, r2["mask"], ids, ORBIT_T, np.asarray(inst2.H_tot.data))
    case2["global_search"] = global_search(ctx["sbp"], ib.tables(A2, ids), r2["mask"], ids, starts, seed + 99, None, V2)
    cases["type2"] = case2
    say(f"case type2: disconnected={dis} global={case2['global_search']['exact_fits_with_substantive_heldout_difference']}")

    # -- M1.2: the 3a case with one added context -------------------------------------------------
    mask3a, R3a = MASKS["type3a"], cases["type3a"]["R"]
    cands = am.mac.candidates(list(mask3a), R3a, ctx["pool"])
    coef = {q: coefficient_matrix_independent(A0, R3a, q) for q in cands}
    for q in cands:   # operational cross-check of the singular values
        op = ph.coefficient_matrix(ctx["fw"], V, m1.pauli_matrix(R3a), q, float(ctx["n_shot"]))
        coef[q]["operational_sigma_min_difference"] = abs(op["sigma_min"] - coef[q]["sigma_min"])
        coef[q]["t_weak"] = op["t_weak"]
        coef[q]["noise_l2"] = op["noise_l2"]
    best = max(cands, key=lambda q: coef[q]["sigma_min"])
    weak = min(cands, key=lambda q: coef[q]["sigma_min"])
    added = {}
    for q in dict.fromkeys([best, weak]):
        added[q] = global_search(ctx["sbp"], exact, [*mask3a, q], ids, starts, seed + 200 + cands.index(q), R3a, V)
        say(f"  3a + {q}: exact fits {added[q]['exact_fits']}, alternatives {added[q]['exact_fits_with_substantive_heldout_difference']}")
    m12 = {"candidates": cands, "coefficient_matrices": coef, "best_sigma_min": best, "weakest_sigma_min": weak, "global_search_with_added_context": added}

    # -- M1.2: zero-energy effect rank on the development 3a records (level ii) -------------------
    rank_rows = []
    if rank_audit:
        src = json.loads(Path(SOURCE).read_text(encoding="utf-8"))
        for row in src["records"]:
            if row["class"] != "3a":
                continue
            ins = ctx["instances"][int(row["index"])]
            rk_s = zero_energy_effect_rank(np.asarray(ins.H_tot.data), K, d_S, ins.active_modes, row["mask"])
            rank_rows.append({"index": int(row["index"]), **{f"support_{k}": v for k, v in rk_s.items()}})
        rank_all = zero_energy_effect_rank(H, K, d_S, inst.active_modes, ids)
        m12["zero_energy_effect_rank"] = {"records": len(rank_rows), "support_rank_counts": am._counts([f"{r['support_rank']}/{r['support_hermitian_dimension']}" for r in rank_rows]),
                                         "all_contexts_rank_state_469": rank_all}
        say(f"zero-energy rank: {m12['zero_energy_effect_rank']['support_rank_counts']}")

    # -- reproduction kit (3a) -------------------------------------------------------------------
    t_star = cases["type3a"]["family"]["t_at_max_heldout"]
    A_star = family_point(A0, {"kind": "orbit", "R": R3a}, t_star)
    kit = {
        "description": "Two histories with identical support tables (M1 type 3a). Recompute with any Born-rule code: clock |tau> = K^-1/2 sum_r exp(-2 pi i r tau/K)|omega_r>, effects |clock_l><clock_l| (x) (I + (-1)^b P)/2, clock factor first, outcome key c{l}_s{b}.",
        "K": K, "d_S": d_S, "active_modes": list(inst.active_modes), "omega": list(inst.omega),
        "history_A_real_imag": [A0.real.tolist(), A0.imag.tolist()], "history_B_real_imag": [A_star.real.tolist(), A_star.imag.tolist()],
        "B_equals": f"(I_C (x) exp(i t R)) A with R = {R3a}, t = {t_star}",
        "support_contexts": list(mask3a), "heldout_contexts": [c for c in ids if c not in set(mask3a)],
        "tables_A": exact, "tables_B": ib.tables(A_star, ids),
        "H_tot_real_imag": [H.real.tolist(), H.imag.tolist()], "theta_con": ctx["theta_con"],
        "frozen_low_radii": ctx["bodies"]["radii"],
        "expected": {"support_max_abs_difference": cases["type3a"]["family"]["max_support_residual"], "heldout_max_abs_difference": cases["type3a"]["family"]["max_heldout_difference"]},
    }
    artifact = {
        "artifact_type": "m1_evidence_package", "method": METHOD, "status": "development_only",
        "claim_boundary": {"development_only": True, "sealed_bank_opened": False,
                           "evidence_levels": {"analytic": "orbit invariance, conjugation identity, branch-condition components, B sin^2 t closed form",
                                               "computational": "Jacobian ranks, response order, multi-start global search (numerical, not a proof of global uniqueness)",
                                               "independent_engine": "every table, trace distance and energy above is from bi_qmn.p3.independent_born"}},
        "settings": {"starts": int(starts), "seed": int(seed), "orbit_t": list(ORBIT_T), "order_t": list(ORDER_T), "state_index": STATE_INDEX, "type2_index": TYPE2_INDEX},
        "e6_cross_check": cross,
        "cases": cases,
        "m1_2": m12,
        "zero_energy_rank_records": rank_rows,
        "execution": {"wall_seconds": time.perf_counter() - started},
    }
    snapshot = import_closure_snapshot(Path.cwd(), extra_files=[p for p in (str(config_dir / am.CONFIG_FILE), RUNNER) if Path(p).exists()])
    artifact.update({"source_snapshot_hash": snapshot_hash(snapshot), "source_snapshot": snapshot, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "source_artifact_sha256": file_sha256(Path(SOURCE)) if Path(SOURCE).exists() else None})
    artifact = m1._jsonable(artifact)
    kit = m1._jsonable(kit)
    if write:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ARTIFACT).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (folder / KIT).write_text(json.dumps(kit, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
