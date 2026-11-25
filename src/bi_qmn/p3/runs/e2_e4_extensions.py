"""E2 (design-scope extension of the exact orbit and its repair) and E4 (Hamiltonian information), development.

Protocols: documents/E2-scope-extension-protocol.md and documents/E4-hamiltonian-information-protocol.md, each
frozen by its own declaration.

**E2.** The exact (type-3a) circles of the registered law sit mostly on one of four registered unitaries. E2 builds
system bases *outside* that set: for a flagged low mask with D commutant ``R`` it draws a Haar unitary and projects
its branch columns onto the exact-circle condition ``<u_j|P|u_j> = <u_j|iPR|u_j> = 0`` for every observed F Pauli
``P`` that anticommutes with ``R`` (least squares on U(4), residual below 1e-12). It also perturbs those bases by
``exp(i eps G)`` (near-exact cases) and draws unconstrained Haar bases (generic cases). The estimator and the
selection rule are never told the basis or the class. Arms as in the M1 run (A0, C, A_s, A_f) on the common
held-out set; certificates for the constructed-3a records.

**E4.** On a fresh enriched type-3a bank of the registered law, the same support is fitted at several levels of
Hamiltonian information: none (A0, level i), the exact zero-energy kernel of the registered ``H_tot`` (level ii),
a certified tolerance ``<H^2>/s_H^2 <= theta`` (soft constraint), a registry whose system eigenbasis is rotated by
``exp(i delta G)``, a registry whose spectrum is shifted by ``epsilon`` (the exact kernel is then empty), and level
ii plus the support-chosen added context. Fits at level ii and above are done in the active-mode coordinates
``y`` (mode ``j``, system index ``a``), in which every table is ``y^dag A_o y / y^dag y`` with the estimator's own
effect matrices and the registered energy is ``y^dag C y / y^dag y`` with ``C`` block diagonal.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares, minimize
import yaml

from bi_qmn.core import identifiability as ident
from bi_qmn.core.config import load_p1_config
from bi_qmn.core.observations import sample_shot_batch
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.ilgc.certificate import _compute_existence_claim
from bi_qmn.ilgc.model import calibration_bodies_from_radii, measurement_model_from_p1, p1_constraint_registry
from bi_qmn.ilgc.relational_dynamics import adjudicate_relational_dynamics, energy_block_candidate, optimal_classical_candidate, tau_flat_distance
from bi_qmn.p1.model.p1_generator import build_p1_instance, construction_bank_p1, seeded_construction_bank_p1, system_unitary_from_config
from bi_qmn.p1.model.p1_measurement import full_p1_context_list, p1_effect_map, p1_measurement_table
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairEstimator
from bi_qmn.p3.models.p3_zero_energy_history import feasible_half_width
from bi_qmn.p3.p3_ilgc_facing import load_ilgc_targets
from bi_qmn.p3.p3_support_screen import SupportIdentifiabilityScreen
from bi_qmn.p3.runs import e5_rdn_specificity as e5
from bi_qmn.p3.runs import m1_added_context_check as mac
from bi_qmn.p3.runs import m1_added_measurement as am
from bi_qmn.p3.runs import m1_added_measurement_posthoc as ph
from bi_qmn.p3.runs import m1_d_commutant_check as dc
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _keys, _table_dict
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint


CONFIGS = {"e2": "e2_scope_extension.yaml", "e4": "e4_hamiltonian_information.yaml"}
DECLARATIONS = {s: f"{s}_declaration.json" for s in CONFIGS}
ARTIFACTS = {"e2": "e2_scope_extension.json", "e4": "e4_hamiltonian_information.json"}
RUNNER = "runners/p3/run_e2_e4.py"
CLASSES = ("diag", "coh")
RESERVED = (53500, 53999)
EXACT_TOL = 1.0e-10
_W: dict[str, Any] = {}


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:16], 16)


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Configuration and shared context
# ---------------------------------------------------------------------------


def _load(config_dir: Path, stage: str) -> tuple[dict[str, Any], str]:
    path = Path(config_dir) / CONFIGS[stage]
    values = _yaml(path)
    if values.get("status") not in ("development_only", "test_only") or values.get("stage") != stage:
        raise ValueError(f"{stage} is a development protocol")
    bank = values["bank"]
    seeds = [int(bank[k]) for k in ("seed", "observation_seed", "mask_seed")] + [int(values["evaluation"]["bootstrap"]["seed"])]
    if len(set(seeds)) != len(seeds) or any(RESERVED[0] <= s <= RESERVED[1] for s in seeds):
        raise ValueError("seeds must be distinct and outside the reserved confirmatory namespace")
    dev = _yaml(Path(config_dir) / values["design_source"]["development_config"])
    if values["estimator"] != dev["estimator"]:
        raise ValueError("estimator settings differ from D1-v7")
    ref = dev["development_banks"]["evaluation"]
    if any(bank[k] != ref[k] for k in ("dirichlet_concentration", "minimum_mode_probability", "phase_range")):
        raise ValueError("the bank must use the D1-v7 weight and phase law")
    return values, file_sha256(path)


def prepare(config_dir: Path, stage: str) -> dict[str, Any]:
    config_dir = Path(config_dir)
    values, _ = _load(config_dir, stage)
    p1 = load_p1_config(config_dir)
    dev = _yaml(config_dir / values["design_source"]["development_config"])
    contexts = full_p1_context_list(p1.measurement)
    dims = p1.generator["dimensions"]
    K, d_C, d_S = int(dims["K"]), int(dims["d_C"]), int(dims["d_S"])
    effects_by_context = {cid: {o: np.asarray(op.data, dtype=complex) for o, op in row.items()} for cid, row in p1_effect_map(contexts, K=K).items()}
    cfg = values["estimator"]
    sbp = SharedBranchPairEstimator(p1.generator["active_modes"], K, d_S, effects_by_context, starts=int(cfg["starts"]), seed=int(cfg["seed"]),
                                    probability_floor=float(cfg["probability_floor"]), likelihood_floor=float(cfg["likelihood_floor"]),
                                    near_optimal_nats=float(cfg["near_optimal_nats"]), max_iterations=int(cfg["max_iterations"]),
                                    pair_observability_threshold=float(cfg["pair_observability_threshold"]))
    bodies = e5.frozen_s3_bodies()
    return {"values": values, "stage": stage, "p1": p1, "K": K, "d_C": d_C, "d_S": d_S, "modes": tuple(int(m) for m in p1.generator["active_modes"]),
            "n": len(p1.generator["active_modes"]), "omega": tuple(float(w) for w in p1.generator["omega"]), "eta": 0.0,
            "theta_con": float(p1.generator["thresholds"]["theta_con"]), "contexts": contexts, "context_by_id": {c.context_id: c for c in contexts},
            "ids": [c.context_id for c in contexts], "effects": {(cid, o): M for cid, row in effects_by_context.items() for o, M in row.items()},
            "sbp": sbp, "fw": sbp.forward,
            "screen": SupportIdentifiabilityScreen(p1.generator["active_modes"], K, epsilon=float(dev["screen"]["disconnected_branch_null_tail_epsilon"])),
            "bodies": {"radii": bodies["radii"]["low"], "calibration_hash": bodies["calibration_hash"]},
            "stratum": load_ilgc_targets(config_dir, "p1_ilgc_operational_r4.yaml")["strata"]["low"],
            "flagged": am.flagged_masks(p1.measurement),
            "pool": sorted({cid for m in m1.deficit_masks(p1.measurement, "low") for cid in m if cid.startswith("D,")}),
            "n_shot": int(p1.measurement["observation_deficit"]["low"]["n_shot"]), "diag_keys": _keys(contexts, "diag", K),
            "registered_unitaries": [system_unitary_from_config(item) for item in p1.generator["construction_grid"]["system_unitaries"]]}


def make_instance(ctx: Mapping[str, Any], label: str, probs: Sequence[float], phases: Sequence[float], U: np.ndarray):
    return build_p1_instance(coefficient_label=label, probabilities=list(probs), phase_label=label, phases=list(phases), system_unitary_label=label,
                             system_unitary=np.asarray(U, dtype=complex), active_modes=ctx["modes"], omega=ctx["omega"], d_C=ctx["d_C"], d_S=ctx["d_S"], K=ctx["K"])


def _cplx(m) -> np.ndarray:
    return np.asarray(m[0]) + 1j * np.asarray(m[1])


def _pack(m: np.ndarray) -> list:
    m = np.asarray(m, dtype=complex)
    return [m.real.tolist(), m.imag.tolist()]


# ---------------------------------------------------------------------------
# Exact-circle geometry (pure)
# ---------------------------------------------------------------------------


def anticommuting_f(mask: Sequence[str], R: str) -> list[str]:
    return [c.split(",")[1] for c in mask if c.startswith("F,") and not dc.paulis_commute(c.split(",")[1], R)]


def branch_components(U: np.ndarray, n: int, R: str, Fs: Sequence[str]) -> tuple[float, float]:
    """``max |<P>_j|`` and ``max |<iPR>_j|`` over anticommuting observed F Paulis and branches."""
    u = np.asarray(U, dtype=complex)[:, :n]
    Rm = m1.pauli_matrix(R)
    a = b = 0.0
    for P in Fs:
        Pm = m1.pauli_matrix(P)
        for j in range(n):
            a = max(a, abs(u[:, j].conj() @ Pm @ u[:, j]))
            b = max(b, abs(u[:, j].conj() @ (1j * Pm @ Rm) @ u[:, j]))
    return float(a), float(b)


def classify(U: np.ndarray, n: int, R: str, Fs: Sequence[str]) -> str:
    a, b = branch_components(U, n, R, Fs)
    if a < EXACT_TOL and b < EXACT_TOL:
        return "3a"
    return "3b" if b < EXACT_TOL else "regular"


def _ah(theta: np.ndarray, d: int) -> np.ndarray:
    X = np.zeros((d, d), dtype=complex)
    iu = np.triu_indices(d, 1)
    k = len(iu[0])
    X[iu] = theta[:k] + 1j * theta[k:2 * k]
    X = X - X.conj().T
    X[np.diag_indices(d)] = 1j * theta[2 * k:]
    return X


def project_3a(U0: np.ndarray, R: str, Fs: Sequence[str], n: int, tries: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Least squares on U(d) for ``<u_j|P|u_j> = <u_j|iPR|u_j> = 0``; returns the best unitary and its residual."""
    d = U0.shape[0]
    Rm = m1.pauli_matrix(R)
    ops = [m1.pauli_matrix(P) for P in Fs] + [1j * m1.pauli_matrix(P) @ Rm for P in Fs]
    best = (U0, math.inf)
    for t in range(int(tries)):
        start = U0 if t == 0 else e5.haar_unitary(d, rng)

        def residual(theta: np.ndarray) -> np.ndarray:
            U = start @ expm(_ah(theta, d))
            return np.asarray([float(np.real(U[:, j].conj() @ A @ U[:, j])) for A in ops for j in range(n)])

        res = least_squares(residual, np.zeros(d * d), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        U = start @ expm(_ah(res.x, d))
        r = float(np.max(np.abs(residual(res.x))))
        if r < best[1]:
            best = (U, r)
        if r < 1e-13:
            break
    return best


def support_circle_deviation(fw, V: np.ndarray, R: str, mask: Sequence[str], points: int = 16) -> float:
    G = m1.pauli_matrix(R)
    T0 = fw.tables(V, list(mask))
    worst = 0.0
    for t in np.linspace(0.0, math.pi, points, endpoint=False)[1:]:
        Tt = fw.tables(expm(1j * t * G) @ V, list(mask))
        worst = max(worst, max(float(np.max(np.abs(Tt[c] - T0[c]))) for c in mask))
    return worst


def distance_to_registered(ctx: Mapping[str, Any], U: np.ndarray) -> float:
    """``min`` over registered unitaries of ``max_j (1 - |<u_j|u'_j>|^2)`` (0 when a branch basis coincides up to phases)."""
    n = ctx["n"]
    return float(min(max(1.0 - abs(np.vdot(np.asarray(Ur)[:, j], np.asarray(U)[:, j])) ** 2 for j in range(n)) for _, Ur in ctx["registered_unitaries"]))


# ---------------------------------------------------------------------------
# Active-mode coordinates for level ii and the soft constraint (pure)
# ---------------------------------------------------------------------------


def energy_matrix(H_S: np.ndarray, modes: Sequence[int], omega: Sequence[float], s2: float) -> np.ndarray:
    d = H_S.shape[0]
    blocks = [(H_S + float(omega[m]) * np.eye(d)) @ (H_S + float(omega[m]) * np.eye(d)) / float(s2) for m in modes]
    C = np.zeros((len(modes) * d, len(modes) * d), dtype=complex)
    for j, Bk in enumerate(blocks):
        C[j * d:(j + 1) * d, j * d:(j + 1) * d] = Bk
    return C


def h_scale_sq(H_S: np.ndarray, omega: Sequence[float]) -> float:
    d = H_S.shape[0]
    K = len(omega)
    H = np.kron(np.diag(omega), np.eye(d)) + np.kron(np.eye(K), H_S)
    return float(np.real(np.trace(H @ H)) / (K * d))


def kernel_basis(C: np.ndarray, rel_tol: float = 1e-9) -> np.ndarray:
    w, Q = np.linalg.eigh(C)
    return Q[:, w <= rel_tol * max(1.0, float(np.max(np.abs(w))))]


def _nll(yr: np.ndarray, A: np.ndarray, counts: np.ndarray, floor: float) -> tuple[float, np.ndarray]:
    m = yr.size // 2
    y = yr[:m] + 1j * yr[m:]
    nrm = float(np.real(np.vdot(y, y)))
    Ay = A @ y
    q = np.real(np.einsum("i,oi->o", y.conj(), Ay)) / nrm
    qf = q + floor
    value = -float(counts @ np.log(qf))
    w = -((counts / qf)[:, None] * (Ay - q[:, None] * y[None, :])).sum(axis=0) / nrm
    return value, np.concatenate([2 * w.real, 2 * w.imag])


def _energy(yr: np.ndarray, C: np.ndarray) -> tuple[float, np.ndarray]:
    m = yr.size // 2
    y = yr[:m] + 1j * yr[m:]
    nrm = float(np.real(np.vdot(y, y)))
    Cy = C @ y
    e = float(np.real(np.vdot(y, Cy))) / nrm
    w = (Cy - e * y) / nrm
    return e, np.concatenate([2 * w.real, 2 * w.imag])


def support_arrays(fw, tables: Mapping[str, Mapping[str, float]], n_by_context: Mapping[str, float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(tables)
    A = np.concatenate([fw.A[cid] for cid in ids])
    counts = np.concatenate([[float(tables[cid].get(o, 0.0)) * float(n_by_context[cid]) for o in fw.outcomes] for cid in ids])
    return ids, A, counts


def fit_kernel(fw, tables, n_by_context, Kb: np.ndarray, starts: int, seed_text: str, floor: float) -> dict[str, Any]:
    """Maximum likelihood over unit vectors in the span of ``Kb`` (active-mode coordinates)."""
    k = Kb.shape[1]
    if k == 0:
        return {"feasible": False, "reason": "empty_kernel", "kernel_dimension": 0}
    ids, A, counts = support_arrays(fw, tables, n_by_context)
    Ak = np.einsum("ia,oij,jb->oab", Kb.conj(), A, Kb)
    rng = np.random.default_rng(_seed(seed_text))
    best = (math.inf, None)
    for _ in range(int(starts)):
        w0 = rng.normal(size=k) + 1j * rng.normal(size=k)
        res = minimize(_nll, np.concatenate([w0.real, w0.imag]), args=(Ak, counts, floor), jac=True, method="L-BFGS-B", options={"maxiter": 4000, "ftol": 1e-13, "gtol": 1e-9})
        if float(res.fun) < best[0]:
            best = (float(res.fun), res.x.copy())
    w = best[1][:k] + 1j * best[1][k:]
    y = Kb @ w
    return {"feasible": True, "kernel_dimension": int(k), "nll": best[0], "y": y / np.linalg.norm(y)}


def fit_soft(fw, tables, n_by_context, C: np.ndarray, theta: float, starts_y: Sequence[np.ndarray], extra: int, seed_text: str, floor: float) -> dict[str, Any]:
    """Maximum likelihood over active-mode pure states with registered energy at most ``theta`` (SLSQP)."""
    ids, A, counts = support_arrays(fw, tables, n_by_context)
    m = A.shape[1]
    rng = np.random.default_rng(_seed(seed_text))
    starts = [np.asarray(y, dtype=complex) for y in starts_y] + [rng.normal(size=m) + 1j * rng.normal(size=m) for _ in range(int(extra))]
    cons = {"type": "ineq", "fun": lambda yr: (float(theta) - _energy(yr, C)[0]) / max(float(theta), 1e-12), "jac": lambda yr: -_energy(yr, C)[1] / max(float(theta), 1e-12)}
    best = (math.inf, None, None)
    for y0 in starts:
        y0 = y0 / np.linalg.norm(y0)
        res = minimize(_nll, np.concatenate([y0.real, y0.imag]), args=(A, counts, floor), jac=True, method="SLSQP", constraints=[cons], options={"maxiter": 800, "ftol": 1e-12})
        e = _energy(res.x, C)[0]
        if e <= float(theta) * (1 + 1e-6) + 1e-15 and float(res.fun) < best[0]:
            best = (float(res.fun), res.x.copy(), e)
    if best[1] is None:
        return {"feasible": False, "reason": "no_feasible_optimum"}
    y = best[1][:m] + 1j * best[1][m:]
    return {"feasible": True, "nll": best[0], "energy": best[2], "y": y / np.linalg.norm(y)}


def y_from_V(V: np.ndarray) -> np.ndarray:
    y = np.asarray(V, dtype=complex).T.reshape(-1)
    return y / np.linalg.norm(y)


def V_from_y(y: np.ndarray, n: int, d: int) -> np.ndarray:
    return np.asarray(y, dtype=complex).reshape(n, d).T


def output_table(ctx: Mapping[str, Any], V: np.ndarray) -> dict[str, dict[str, float]]:
    fw, floor = ctx["fw"], ctx["sbp"].probability_floor
    T = fw.tables(V, ctx["ids"])
    out = {}
    for cid in ctx["ids"]:
        v = np.clip(T[cid], 0.0, None) + floor
        v = v / v.sum()
        out[cid] = dict(zip(fw.outcomes, v.tolist(), strict=True))
    return out


# ---------------------------------------------------------------------------
# Worker jobs
# ---------------------------------------------------------------------------


def _job_sbp(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    support = MeasurementTable(probabilities=job["tables"], contexts=[ctx["context_by_id"][c] for c in sorted(job["tables"])], metadata={"n_shots": float(job["n_shots"])}, source_density=None)
    prediction = ctx["sbp"].predict_table(support, list(ctx["contexts"]))
    fit = prediction.metadata["shared_branch_pair_fit"]
    return {"key": tuple(job["key"]), "table": _table_dict(prediction), "V": fit["branch_vectors_real_imag"], "nll": float(fit["objective_nats"]),
            "near": int(fit["starts_within_near_optimal"]), "seconds": time.perf_counter() - started}


def _job_y(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    n_by = {c: float(job["n_by_context"][c]) for c in job["tables"]}
    C = _cplx(job["C"])
    floor = ctx["sbp"].likelihood_floor
    if job["mode"] == "kernel":
        out = fit_kernel(ctx["fw"], job["tables"], n_by, kernel_basis(C), int(job["starts"]), job["seed_text"], floor)
    else:
        out = fit_soft(ctx["fw"], job["tables"], n_by, C, float(job["theta"]), [_cplx(y) for y in job["start_y"]], int(job["extra"]), job["seed_text"], floor)
    res = {"key": tuple(job["key"]), "feasible": out["feasible"], "seconds": time.perf_counter() - started}
    for k in ("reason", "kernel_dimension", "nll", "energy"):
        if k in out:
            res[k] = out[k]
    if out["feasible"]:
        V = V_from_y(out["y"], ctx["n"], ctx["d_S"])
        res["table"] = output_table(ctx, V)
        res["V"] = _pack(V)
        res["energy_C"] = float(np.real(np.vdot(out["y"], C @ out["y"])))
    return res


def _job_cert(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    """BI existence (registry of ``registry`` instance) and rdn_v1 on the frozen low bodies; exact-state candidate from ``truth``."""
    started = time.perf_counter()
    K, d_C, d_S, contexts = ctx["K"], ctx["d_C"], ctx["d_S"], ctx["contexts"]
    cert = ctx["values"]["certificate"]
    truth = make_instance(ctx, "truth", job["truth"]["probs"], job["truth"]["phases"], _cplx(job["truth"]["U"]))
    reg = make_instance(ctx, "registry", job["truth"]["probs"], job["truth"]["phases"], _cplx(job["registry_U"]))
    table = job["table"]
    radii = ctx["bodies"]["radii"]
    calibration_hash = f"p3_s3_calibration:{ctx['bodies']['calibration_hash']}:low"
    model = measurement_model_from_p1(MeasurementTable(probabilities=table, contexts=list(contexts), metadata={"eta": 0.0}, source_density=None), K, stratum=ctx["stratum"],
                                      estimator_checkpoint_hash=ctx["sbp"].checkpoint_hash, support_query_hash=str(job["key"]))
    bodies = calibration_bodies_from_radii(model, radii, calibration_hash=calibration_hash, stratum=ctx["stratum"])
    registry = p1_constraint_registry(reg, ctx["theta_con"], calibration_hash=calibration_hash)
    claim, _ = _compute_existence_claim(model, bodies, registry, precision_dps=int(cert["precision_dps"]), proposal_backends=tuple(cert["proposal_backends"]))
    bi = {"status": claim.status, "verified_upper_bound": claim.verified_upper_bound, "verified_lower_bound": claim.verified_lower_bound}
    class_keys = {cls: _keys(contexts, cls, K) for cls in CLASSES}
    targets = {cls: ident.class_vector(table, contexts, cls, K) for cls in CLASSES}
    rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, d_S=d_S, precision_dps=int(cert["precision_dps"]))
    if rdn["status"] != "certified":
        H = np.asarray(reg.H_tot.data, dtype=complex)
        constraints = [(H @ H / (reg.h_scale ** 2), None, ctx["theta_con"])]
        rho_bi = np.asarray(claim.primal_evidence.density.data, dtype=complex) if claim.status == "primal_verified" else None
        candidates = [] if rho_bi is None else [("energy_pinched_bi_primal_state", energy_block_candidate(rho_bi, d_C, d_S))]
        candidates.append(("energy_pinched_exact_state", energy_block_candidate(np.asarray(truth.density.data), d_C, d_S)))
        rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], candidates=candidates, constraints=constraints, d_S=d_S)
        if rdn["status"] == "unresolved" and cert.get("refine_unresolved_with_optimal_classical_state", True):
            proposal = optimal_classical_candidate(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], constraints=constraints, d_C=d_C, d_S=d_S)
            if proposal["blocks"] is not None:
                rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"],
                                                     candidates=[*candidates, ("optimal_classical_quantum_state", proposal["blocks"])], constraints=constraints, d_S=d_S)
    return {"key": tuple(job["key"]), "bi": bi, "rdn": {"status": rdn["status"], "kappa": rdn["kappa"]}, "seconds": time.perf_counter() - started}


_JOBS = {"sbp": _job_sbp, "y": _job_y, "cert": _job_cert}


def _worker_init(root: str, config_dir: str, stage: str) -> None:
    os.chdir(root)
    _W.clear()
    _W.update(prepare(Path(config_dir), stage))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    return _JOBS[job["kind"]](_W, job)


class _Runner:
    def __init__(self, ctx: Mapping[str, Any], config_dir: Path, stage: str, workers: int, progress):
        self.ctx, self.progress = ctx, progress
        self.saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
        self.pool = None
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            self.pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(Path.cwd()), str(Path(config_dir).resolve()), stage))
        else:
            _W.clear()
            _W.update(ctx)

    def wave(self, jobs: list[dict], label: str) -> dict:
        return _execute(self.pool, _worker_job, jobs, label, self.progress) if jobs else {}

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _probs_phases(rng: np.random.Generator, bank: Mapping[str, Any], n: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(10000):
        p = rng.dirichlet(np.full(n, float(bank["dirichlet_concentration"])))
        if float(np.min(p)) >= float(bank["minimum_mode_probability"]):
            break
    phases = np.zeros(n)
    phases[1:] = rng.uniform(float(bank["phase_range"][0]), float(bank["phase_range"][1]), size=n - 1)
    return p, phases


def build_record(ctx: Mapping[str, Any], rid: int, family: str, probs, phases, U: np.ndarray, mask: Sequence[str], R: str) -> dict[str, Any]:
    bank = ctx["values"]["bank"]
    inst = make_instance(ctx, f"{bank['bank_id']}:{family}:{rid}", probs, phases, U)
    exact = p1_measurement_table(inst.density, ctx["contexts"], K=ctx["K"], eta=0.0, metadata={"rid": rid, "family": family}).copy_without_source()
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bank['bank_id']}:{family}:{rid}"))
    support = sample_shot_batch(exact, [ctx["context_by_id"][c] for c in mask], n_shots=ctx["n_shot"], rng=rng, shot_start=rid * 100_000_000, role=f"{ctx['stage']}_support").table.probabilities
    screen = ctx["screen"].evaluate(support, float(ctx["n_shot"]))
    Fs = anticommuting_f(mask, R)
    a, b = branch_components(U, ctx["n"], R, Fs)
    V = np.asarray(U, dtype=complex)[:, :ctx["n"]] * np.asarray(inst.coefficients)[None, :]
    return {"index": int(rid), "family": family, "mask": list(mask), "R": R, "probs": list(map(float, probs)), "phases": list(map(float, phases)), "U": _pack(U),
            "class": classify(U, ctx["n"], R, Fs), "branch_a": a, "branch_b": b, "screen_passes": bool(screen["passes"]), "screen_reasons": list(screen["reasons"]),
            "support_circle_deviation": support_circle_deviation(ctx["fw"], V, R, mask), "distance_to_registered": distance_to_registered(ctx, U),
            "exact": {c: dict(row) for c, row in exact.probabilities.items()}, "support": {c: dict(row) for c, row in support.items()}, "n_shot": ctx["n_shot"],
            "V_true": _pack(V), "H_tot": None}


def attach(ctx: Mapping[str, Any], r: dict[str, Any]) -> None:
    sub = {"values": {"bank": {"bank_id": f"{ctx['values']['bank']['bank_id']}:{r['family']}", "observation_seed": ctx["values"]["bank"]["observation_seed"]},
                      "control": {"extra_shots_per_support_context": int(ctx["values"]["control"]["extra_shots_per_support_context"])}},
           "contexts": ctx["contexts"], "context_by_id": ctx["context_by_id"], "pool": ctx["pool"], "n_shot": ctx["n_shot"]}
    am.attach_samples(sub, r)


def generate_e2(ctx: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bank, gen = ctx["values"]["bank"], ctx["values"]["generation"]
    n, bid = ctx["n"], bank["bank_id"]
    ledger: dict[str, Any] = {}
    out: list[dict[str, Any]] = []
    constructed = []
    led = {"generated": 0, "type1_mask": 0, "projection_failed": 0, "screened_out": 0, "not_3a": 0, "accepted": 0}
    for i in range(int(gen["constructed_3a"]["cap"])):
        if len(constructed) >= int(gen["constructed_3a"]["quota"]):
            break
        rng = np.random.default_rng(_seed(f"{bank['seed']}:{bid}:construct:{i}"))
        led["generated"] += 1
        p, ph_ = _probs_phases(rng, bank, n)
        mask = ctx["flagged"][int(np.random.default_rng(_seed(f"{bank['mask_seed']}:{bid}:construct:{i}")).integers(0, len(ctx["flagged"])))]
        R = dc.d_commutant(mask)[0]
        Fs = anticommuting_f(mask, R)
        if not Fs:
            led["type1_mask"] += 1
            continue
        U, res = project_3a(e5.haar_unitary(ctx["d_S"], rng), R, Fs, n, int(gen["constructed_3a"]["projection_tries"]), rng)
        if res > float(gen["constructed_3a"]["projection_tolerance"]):
            led["projection_failed"] += 1
            continue
        r = build_record(ctx, i, "constructed_3a", p, ph_, U, mask, R)
        r["projection_residual"] = res
        if not r["screen_passes"]:
            led["screened_out"] += 1
            continue
        if r["class"] != "3a":
            led["not_3a"] += 1
            continue
        led["accepted"] += 1
        constructed.append(r)
    ledger["constructed_3a"] = led
    out.extend(constructed)
    for k, eps in enumerate(gen["perturbed_3a"]["epsilons"]):
        led = {"generated": 0, "screened_out": 0, "classes": {}}
        for r0 in constructed[: int(gen["perturbed_3a"]["base_records"])]:
            rng = np.random.default_rng(_seed(f"{bank['seed']}:{bid}:perturb:{r0['index']}"))
            G = rng.normal(size=(ctx["d_S"], ctx["d_S"])) + 1j * rng.normal(size=(ctx["d_S"], ctx["d_S"]))
            G = (G + G.conj().T) / 2.0
            G = G / np.linalg.norm(G)
            U = expm(1j * float(eps) * G) @ _cplx(r0["U"])
            r = build_record(ctx, 100_000 * (k + 1) + int(r0["index"]), f"perturbed_{eps:g}", r0["probs"], r0["phases"], U, r0["mask"], r0["R"])
            r["base_index"] = int(r0["index"])
            led["generated"] += 1
            led["classes"][r["class"]] = led["classes"].get(r["class"], 0) + 1
            if not r["screen_passes"]:
                led["screened_out"] += 1
            out.append(r)
        ledger[f"perturbed_{eps:g}"] = led
    led = {"generated": 0, "type1_mask": 0, "screened_out": 0, "accepted": 0, "classes": {}}
    count = 0
    for i in range(int(gen["haar"]["cap"])):
        if count >= int(gen["haar"]["quota"]):
            break
        rng = np.random.default_rng(_seed(f"{bank['seed']}:{bid}:haar:{i}"))
        led["generated"] += 1
        p, ph_ = _probs_phases(rng, bank, n)
        mask = ctx["flagged"][int(np.random.default_rng(_seed(f"{bank['mask_seed']}:{bid}:haar:{i}")).integers(0, len(ctx["flagged"])))]
        R = dc.d_commutant(mask)[0]
        r = build_record(ctx, 900_000 + i, "haar", p, ph_, e5.haar_unitary(ctx["d_S"], rng), mask, R)
        led["classes"][r["class"]] = led["classes"].get(r["class"], 0) + 1
        if not r["screen_passes"]:
            led["screened_out"] += 1
            continue
        led["accepted"] += 1
        count += 1
        out.append(r)
    ledger["haar"] = led
    return out, ledger


def generate_e4(ctx: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enriched screened-in type-3a records of the registered law (as in the M1 run)."""
    bank = ctx["values"]["bank"]
    p1 = ctx["p1"]
    instances = seeded_construction_bank_p1(p1.generator, dict(bank))
    sub = {**ctx, "values": {**ctx["values"], "bank": {**bank, "exact_class_tolerance": EXACT_TOL}, "deficit": "low"}, "instances": instances}
    out, led = [], {"generated": 0, "screened_out": 0, "screened_in_3b": 0, "accepted": 0}
    for i in range(int(bank["block_count"])):
        if len(out) >= int(bank["quota"]):
            break
        r = am.build_construction(sub, i)
        led["generated"] += 1
        if not r["screen_passes"]:
            led["screened_out"] += 1
            continue
        if r["class"] != "3a":
            led["screened_in_3b"] += 1
            continue
        inst = instances[i]
        coeffs = np.asarray(inst.coefficients)
        U = np.asarray(getattr(inst.system_unitary, "data", inst.system_unitary), dtype=complex)
        V = U[:, :ctx["n"]] * coeffs[None, :]
        r.update(family="registered_3a", probs=list(map(float, np.abs(coeffs) ** 2)), phases=list(map(float, np.angle(coeffs) - np.angle(coeffs[0]))), U=_pack(U),
                 unitary=inst.system_unitary_label, n_shot=ctx["n_shot"], V_true=_pack(V))
        led["accepted"] += 1
        out.append(r)
    return out, {"registered_3a": led}


# ---------------------------------------------------------------------------
# Running the arms
# ---------------------------------------------------------------------------


def _sbp_job(r, variant: str) -> dict:
    tables, n = am.variant_tables(r, variant)
    return {"kind": "sbp", "key": (r["index"], variant), "tables": tables, "n_shots": n}


def pick_rules(ctx: Mapping[str, Any], r: dict[str, Any], V_hat: np.ndarray) -> None:
    tol = float(ctx["values"]["added_context"]["tie_relative_tolerance"])
    r["s_Q_fit"] = {q: mac.orbit_invariant_sensitivity(V_hat, q, r["R"]) for q in r["candidates"]}
    r["Q_s"], r["Q_s_tied"] = am.argmax_with_ties(r["s_Q_fit"], r["candidates"], tol)
    V = _cplx(r["V_true"])
    G = m1.pauli_matrix(r["R"])
    r["candidate_sigma_min"] = {q: float(ph.coefficient_matrix(ctx["fw"], V, G, q, float(ctx["n_shot"]))["sigma_min"]) for q in r["candidates"]}


def errors_on(ctx, table, r, ids) -> dict[str, float]:
    return am.class_errors(ctx, table, r["exact"], ids)


def coverage(ctx, table, r) -> dict[str, Any]:
    s = am.class_errors(ctx, table, r["exact"], ctx["ids"])
    radii = ctx["bodies"]["radii"]
    return {**{c: bool(s[c] <= radii[c]) for c in CLASSES}, "joint": bool(all(s[c] <= radii[c] for c in CLASSES)), "scores": {c: s[c] for c in CLASSES}}


def run_e2_arms(ctx: Mapping[str, Any], runner: _Runner, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live = [r for r in records if r["screen_passes"]]
    for r in live:
        attach(ctx, r)
    w1 = runner.wave([_sbp_job(r, v) for r in live for v in ("support", "control")], "E2 fits (A0, C)")
    for r in live:
        pick_rules(ctx, r, _cplx(w1[(r["index"], "support")]["V"]))
    jobs, seen = [], set()
    for r in live:
        for q in dict.fromkeys([r["Q_s"], r["Q_f"]]):
            if (r["index"], f"add:{q}") not in seen:
                seen.add((r["index"], f"add:{q}"))
                jobs.append(_sbp_job(r, f"add:{q}"))
    w2 = runner.wave(jobs, "E2 fits (A_s, A_f)")
    fits = {**w1, **w2}
    certify = set(ctx["values"]["certificate"]["families"])
    cjobs = []
    for r in live:
        if r["family"] in certify:
            truth = {"probs": r["probs"], "phases": r["phases"], "U": r["U"]}
            for arm, v in (("A0", "support"), ("A_s", f"add:{r['Q_s']}")):
                cjobs.append({"kind": "cert", "key": (r["index"], arm), "table": fits[(r["index"], v)]["table"], "truth": truth, "registry_U": r["U"]})
    w3 = runner.wave(cjobs, "E2 certificates")
    rows = []
    for r in records:
        row = {k: r[k] for k in ("index", "family", "mask", "R", "class", "branch_a", "branch_b", "screen_passes", "screen_reasons", "support_circle_deviation", "distance_to_registered")}
        row.update({k: r[k] for k in ("projection_residual", "base_index") if k in r})
        if r["screen_passes"]:
            variants = {"A0": "support", "C": "control", "A_s": f"add:{r['Q_s']}", "A_f": f"add:{r['Q_f']}"}
            E = [c for c in ctx["ids"] if c not in set(r["mask"]) | {r["Q_s"], r["Q_f"]}]
            row.update(candidates=r["candidates"], Q_s=r["Q_s"], Q_f=r["Q_f"], s_Q_fit=r["s_Q_fit"], candidate_sigma_min=r["candidate_sigma_min"],
                       errors={arm: errors_on(ctx, fits[(r["index"], v)]["table"], r, E) for arm, v in variants.items()},
                       coverage={arm: coverage(ctx, fits[(r["index"], v)]["table"], r) for arm, v in variants.items()},
                       a0_near_optimal_starts=fits[(r["index"], "support")]["near"])
            if r["class"] == "3a":
                row["fiber_A0"] = ph.fiber_coverage(ctx, fits[(r["index"], "support")]["table"], _cplx(r["V_true"]), r["R"])
            row["certificates"] = {arm: {k: w3[(r["index"], arm)][k] for k in ("bi", "rdn")} for arm in ("A0", "A_s") if (r["index"], arm) in w3}
        rows.append(row)
    return rows


def rotated(U: np.ndarray, delta: float, rng: np.random.Generator) -> np.ndarray:
    G = rng.normal(size=U.shape) + 1j * rng.normal(size=U.shape)
    G = (G + G.conj().T) / 2.0
    G = G / np.linalg.norm(G, 2)
    return expm(1j * float(delta) * G) @ U


def run_e4_arms(ctx: Mapping[str, Any], runner: _Runner, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = ctx["values"]["arms"]
    d, n, modes, omega = ctx["d_S"], ctx["n"], ctx["modes"], ctx["omega"]
    for r in records:
        attach(ctx, r)
        U = _cplx(r["U"])
        eigs = np.zeros(d)
        unused = [m for m in range(ctx["K"]) if m not in modes]
        for j, m in enumerate(modes):
            eigs[j] = -omega[m]
        for j in range(n, d):
            eigs[j] = -omega[unused[j - n]]
        H_S = U @ np.diag(eigs) @ U.conj().T
        s2 = h_scale_sq(H_S, omega)
        regs = {"H_exact": (U, H_S, s2)}
        rng = np.random.default_rng(_seed(f"{ctx['values']['bank']['seed']}:rot:{r['index']}"))
        for delta in cfg["rotation_deltas"]:
            Ud = rotated(U, float(delta), np.random.default_rng(rng.integers(0, 2**63)))
            regs[f"H_rot_{delta:g}"] = (Ud, Ud @ np.diag(eigs) @ Ud.conj().T, None)
        for eps in cfg["eigenvalue_shifts"]:
            shift = np.zeros(d)
            shift[:n] = float(eps) * np.asarray([1.0, -1.0, 1.0])[:n]
            regs[f"H_eig_{eps:g}"] = (None, U @ np.diag(eigs + shift) @ U.conj().T, None)
        r["registries"] = {}
        for name, (Ur, HS, s2r) in regs.items():
            s2r = s2r if s2r is not None else h_scale_sq(HS, omega)
            C = energy_matrix(HS, modes, omega, s2r)
            y_true = y_from_V(_cplx(r["V_true"]))
            r["registries"][name] = {"U": None if Ur is None else _pack(Ur), "C": _pack(C), "kernel_dimension": int(kernel_basis(C).shape[1]),
                                     "truth_energy": float(np.real(np.vdot(y_true, C @ y_true)))}
    n_by = lambda r, tables: {c: (float(r["n_shot"]) if c in r["support"] else float(r["n_shot"])) for c in tables}   # noqa: E731
    jobs = [_sbp_job(r, "support") for r in records]
    for r in records:
        for name in ["H_exact", *[f"H_rot_{x:g}" for x in cfg["rotation_deltas"]], *[f"H_eig_{x:g}" for x in cfg["eigenvalue_shifts"]]]:
            jobs.append({"kind": "y", "mode": "kernel", "key": (r["index"], name), "tables": r["support"], "n_by_context": n_by(r, r["support"]),
                         "C": r["registries"][name]["C"], "starts": int(cfg["kernel_starts"]), "seed_text": f"{r['index']}:{name}"})
    w1 = runner.wave(jobs, "E4 wave 1 (A0, kernels)")
    jobs = []
    for r in records:
        V_hat = _cplx(w1[(r["index"], "support")]["V"])
        pick_rules(ctx, r, V_hat)
        starts = [_pack(y_from_V(V_hat))]
        if w1[(r["index"], "H_exact")]["feasible"]:
            starts.insert(0, _pack(y_from_V(_cplx(w1[(r["index"], "H_exact")]["V"]))))
        for theta in cfg["soft_thetas"]:
            jobs.append({"kind": "y", "mode": "soft", "key": (r["index"], f"H_soft_{theta:g}"), "tables": r["support"], "n_by_context": n_by(r, r["support"]),
                         "C": r["registries"]["H_exact"]["C"], "theta": float(theta), "start_y": starts, "extra": int(cfg["soft_extra_starts"]), "seed_text": f"{r['index']}:soft:{theta}"})
        for eps in cfg["eigenvalue_shifts"]:
            name = f"H_eig_{eps:g}"
            jobs.append({"kind": "y", "mode": "soft", "key": (r["index"], f"{name}_soft"), "tables": r["support"], "n_by_context": n_by(r, r["support"]),
                         "C": r["registries"][name]["C"], "theta": float(cfg["eigen_soft_theta"]), "start_y": starts, "extra": int(cfg["soft_extra_starts"]), "seed_text": f"{r['index']}:eigsoft:{eps}"})
        added = {**r["support"], r["Q_s"]: r["added"][r["Q_s"]]}
        jobs.append(_sbp_job(r, f"add:{r['Q_s']}"))
        jobs.append({"kind": "y", "mode": "kernel", "key": (r["index"], "H_exact_added"), "tables": added, "n_by_context": n_by(r, added),
                     "C": r["registries"]["H_exact"]["C"], "starts": int(cfg["kernel_starts"]), "seed_text": f"{r['index']}:H_exact_added"})
    w2 = runner.wave(jobs, "E4 wave 2 (soft, added)")
    fits = {**w1, **w2}
    arm_keys = ["support", "H_exact", *[f"H_soft_{x:g}" for x in cfg["soft_thetas"]], *[f"H_rot_{x:g}" for x in cfg["rotation_deltas"]],
                *[f"H_eig_{x:g}" for x in cfg["eigenvalue_shifts"]], *[f"H_eig_{x:g}_soft" for x in cfg["eigenvalue_shifts"]], "H_exact_added"]
    cjobs = []
    for r in records[: int(ctx["values"]["certificate"]["records"])]:
        truth = {"probs": r["probs"], "phases": r["phases"], "U": r["U"]}
        for name in ["H_exact", *[f"H_rot_{x:g}" for x in cfg["rotation_deltas"]]]:
            f = fits[(r["index"], name)]
            if f["feasible"]:
                cjobs.append({"kind": "cert", "key": (r["index"], name), "table": f["table"], "truth": truth, "registry_U": r["registries"][name]["U"]})
        cjobs.append({"kind": "cert", "key": (r["index"], "support"), "table": fits[(r["index"], "support")]["table"], "truth": truth, "registry_U": r["U"]})
    w3 = runner.wave(cjobs, "E4 certificates")
    rows = []
    for r in records:
        E = [c for c in ctx["ids"] if c not in set(r["mask"]) | {r["Q_s"]}]
        row = {k: r[k] for k in ("index", "family", "mask", "R", "class", "unitary", "Q_s", "candidates")}
        row["registries"] = {name: {k: v for k, v in reg.items() if k in ("kernel_dimension", "truth_energy")} for name, reg in r["registries"].items()}
        y_true = y_from_V(_cplx(r["V_true"]))
        C_true = _cplx(r["registries"]["H_exact"]["C"])
        yR = y_from_V(m1.pauli_matrix(r["R"]) @ _cplx(r["V_true"]))
        row["orbit_energy_B"] = float(np.real(np.vdot(yR, C_true @ yR)))          # <H^2>/s_H^2 along the orbit is B sin^2 t
        row["arc_half_width"] = {f"{x:g}": feasible_half_width(row["orbit_energy_B"], float(x)) for x in [ctx["theta_con"], *cfg["soft_thetas"]]}
        row["truth_energy_true_registry"] = float(np.real(np.vdot(y_true, C_true @ y_true)))
        arms = {}
        nll_a0 = fits[(r["index"], "support")]["nll"]
        for key in arm_keys + [f"add:{r['Q_s']}"]:
            f = fits[(r["index"], key)]
            name = {"support": "A0", f"add:{r['Q_s']}": "A_s"}.get(key, key)
            if not f.get("feasible", True):
                arms[name] = {"feasible": False, "reason": f.get("reason"), "kernel_dimension": f.get("kernel_dimension")}
                continue
            a = {"feasible": True, "errors": errors_on(ctx, f["table"], r, E), "coverage": coverage(ctx, f["table"], r),
                 "nll_minus_A0": (float(f["nll"]) - nll_a0) if key not in ("H_exact_added", f"add:{r['Q_s']}") else None,
                 "energy_true_registry": float(np.real(np.vdot(y_from_V(_cplx(f["V"])), C_true @ y_from_V(_cplx(f["V"]))))),
                 "orbit_angle": ph.nearest_orbit_point(ctx["fw"], _cplx(f["V"]), _cplx(r["V_true"]), m1.pauli_matrix(r["R"]))["angle_from_truth"]}
            if (r["index"], key) in w3:
                a["certificate"] = {k: w3[(r["index"], key)][k] for k in ("bi", "rdn")}
            arms[name] = a
        row["arms"] = arms
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Analysis (pure)
# ---------------------------------------------------------------------------


def _paired(rows, a: str, b: str, ev, seed_text: str) -> dict[str, Any] | None:
    pairs = [(r["errors"][a]["diag"], r["errors"][b]["diag"]) for r in rows if "errors" in r]
    if len(pairs) < 2:
        return None
    x, y = np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])
    rng = np.random.default_rng(_seed(seed_text))
    idx = rng.integers(0, len(x), size=(int(ev["bootstrap"]["resamples"]), len(x)))
    return am.paired_cvar_difference(x, y, idx, float(ev["level"]), ev["bootstrap"]["interval"])


def _cp(k: int, n: int):
    return e5.clopper_pearson(int(k), int(n)) if n else None


def analyse_e2(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]) -> dict[str, Any]:
    ev = values["evaluation"]
    delta_min = float(e5.frozen_s3_bodies()["radii"]["low"]["diag"])
    out: dict[str, Any] = {"ledger": ledger, "delta_min": delta_min, "by_family": {}}
    for fam in dict.fromkeys(r["family"] for r in rows):
        fr = [r for r in rows if r["family"] == fam]
        live = [r for r in fr if r["screen_passes"]]
        block: dict[str, Any] = {"records": len(fr), "screened_in": len(live), "classes": e5._counts([r["class"] for r in fr]),
                                 "support_circle_deviation": e5._q([r["support_circle_deviation"] for r in fr]),
                                 "distance_to_registered": e5._q([r["distance_to_registered"] for r in fr])}
        if len(live) >= 2:
            block["cvar"] = {arm: am.cvar([r["errors"][arm]["diag"] for r in live], float(ev["level"])) for arm in ("A0", "C", "A_s", "A_f")}
            block["median"] = {arm: float(np.median([r["errors"][arm]["diag"] for r in live])) for arm in ("A0", "C", "A_s", "A_f")}
            block["contrasts"] = {name: _paired(live, a, b, ev, f"{ev['bootstrap']['seed']}:{fam}:{name}") for name, a, b in
                                  (("C_minus_A_s", "C", "A_s"), ("A0_minus_A_s", "A0", "A_s"), ("A0_minus_C", "A0", "C"), ("A_f_minus_A_s", "A_f", "A_s"))}
            c = block["contrasts"]["C_minus_A_s"]
            block["same_budget_gain_above_delta_min"] = bool(c["lower"] > delta_min)
            block["coverage_joint"] = {arm: sum(r["coverage"][arm]["joint"] for r in live) for arm in ("A0", "C", "A_s", "A_f")}
            block["pick_sigma_min"] = e5._q([r["candidate_sigma_min"][r["Q_s"]] for r in live])
            block["pick_below_noise"] = sum(2 * r["candidate_sigma_min"][r["Q_s"]] < 0.0214 for r in live)
            certs = [r for r in live if r.get("certificates")]
            if certs:
                block["certificates"] = {arm: {"bi": e5._counts([r["certificates"][arm]["bi"]["status"] for r in certs]),
                                               "rdn": e5._counts([r["certificates"][arm]["rdn"]["status"] for r in certs]),
                                               "success": sum(r["certificates"][arm]["bi"]["status"] == "primal_verified" and r["certificates"][arm]["rdn"]["status"] == "certified" for r in certs)}
                                         for arm in ("A0", "A_s")}
            fib = [r["fiber_A0"] for r in live if "fiber_A0" in r]
            if fib:
                block["fiber_A0"] = {"n": len(fib), "truth_joint": sum(f["covered"]["truth_joint"] for f in fib), "fiber_joint": sum(f["covered"]["fiber_joint"] for f in fib)}
        out["by_family"][fam] = block
    return out


def analyse_e4(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]) -> dict[str, Any]:
    ev = values["evaluation"]
    out: dict[str, Any] = {"ledger": ledger, "n": len(rows), "by_arm": {}}
    names = list(rows[0]["arms"]) if rows else []
    for name in names:
        feas = [r for r in rows if r["arms"][name]["feasible"]]
        b: dict[str, Any] = {"feasible": len(feas), "kernel_dimension": e5._counts([r["arms"][name].get("kernel_dimension") for r in rows if not r["arms"][name]["feasible"]])}
        if feas:
            errs = [r["arms"][name]["errors"]["diag"] for r in feas]
            b.update(cvar=am.cvar(errs, float(ev["level"])), median=float(np.median(errs)), maximum=float(np.max(errs)),
                     coverage_joint=sum(r["arms"][name]["coverage"]["joint"] for r in feas), coverage_ci95=_cp(sum(r["arms"][name]["coverage"]["joint"] for r in feas), len(feas)),
                     orbit_angle=e5._q([r["arms"][name]["orbit_angle"] for r in feas]),
                     energy_true_registry=e5._q([r["arms"][name]["energy_true_registry"] for r in feas]),
                     nll_minus_A0=e5._q([r["arms"][name]["nll_minus_A0"] for r in feas if r["arms"][name]["nll_minus_A0"] is not None]))
            certs = [r["arms"][name]["certificate"] for r in feas if "certificate" in r["arms"][name]]
            if certs:
                b["certificate"] = {"n": len(certs), "bi": e5._counts([c["bi"]["status"] for c in certs]), "rdn": e5._counts([c["rdn"]["status"] for c in certs])}
        out["by_arm"][name] = b
    reg_names = list(rows[0]["registries"]) if rows else []
    out["registries"] = {name: {"kernel_dimension": e5._counts([r["registries"][name]["kernel_dimension"] for r in rows]),
                                "truth_energy": e5._q([r["registries"][name]["truth_energy"] for r in rows]),
                                "truth_outside_registry": sum(r["registries"][name]["truth_energy"] > 1e-6 for r in rows)} for name in reg_names}
    out["orbit_energy_B"] = e5._q([r["orbit_energy_B"] for r in rows])
    both = [r for r in rows if all(r["arms"][x]["feasible"] for x in ("A0", "H_exact"))]
    out["contrasts"] = {}
    for name in names:
        if name == "H_exact":
            continue
        rs = [{"errors": {"x": r["arms"][name]["errors"], "H": r["arms"]["H_exact"]["errors"]}} for r in both if r["arms"][name]["feasible"]]
        out["contrasts"][f"{name}_minus_H_exact"] = _paired(rs, "x", "H", ev, f"{ev['bootstrap']['seed']}:e4:{name}")
    return out


# ---------------------------------------------------------------------------
# Declaration and run
# ---------------------------------------------------------------------------


def _folder(values: Mapping[str, Any], output_dir: str | Path) -> Path:
    return Path(output_dir) / Path(str(values["output_root"])).name


def _extra_files(config_dir: Path, values: Mapping[str, Any], stage: str) -> list[str]:
    names = [CONFIGS[stage], "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", "p1_ilgc_operational_r4.yaml", str(values["design_source"]["development_config"])]
    return [str(Path(config_dir) / x) for x in dict.fromkeys(names)] + [*values["design_source"]["dev_inputs"], str(values["protocol_document"]), RUNNER]


def declare(stage: str, config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir, stage)
    folder = _folder(values, output_dir)
    target = folder / DECLARATIONS[stage]
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    dev_inputs = am._dev_inputs(root, values)
    p1 = load_p1_config(config_dir)
    known = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked = {}
    for path, spec in harness.declared_bank_specs(config_dir, exclude=CONFIGS[stage]):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    prints = sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, values["bank"])) if stage == "e4" else []
    overlaps = {name: sorted(set(prints) & f) for name, f in known.items() if set(prints) & f}
    if overlaps:
        raise ValueError(f"the bank is not disjoint: {sorted(overlaps)}")
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values, stage))
    declaration = {"artifact_type": f"{stage}_declaration", "status": values["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "config_sha256": config_sha, "protocol_document": str(values["protocol_document"]), "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
                   "dev_inputs": dev_inputs, "bank": values["bank"], "registered_law_fingerprint_hash": hashlib.sha256("".join(prints).encode("ascii")).hexdigest()[:16] if prints else None,
                   "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps,
                                    "note": "E2 constructions use new system bases and cannot coincide with registered-law banks" if stage == "e2" else None},
                   "source_snapshot": snapshot, "source_snapshot_hash": snapshot_hash(snapshot)}
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return declaration


def run(stage: str, config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started = time.perf_counter()
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir, stage)
    folder = _folder(values, output_dir)
    declaration = json.loads((folder / DECLARATIONS[stage]).read_text(encoding="utf-8"))
    if (folder / ARTIFACTS[stage]).exists():
        raise FileExistsError("the artifact already exists and is immutable")
    if declaration["config_sha256"] != config_sha or declaration["protocol_document_sha256"] != file_sha256(root / str(values["protocol_document"])):
        raise RuntimeError("config or protocol drifted after declaration")
    extra = [p for p in declaration["source_snapshot"] if not p.startswith("src/")]
    drift = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    if drift["changed"] or drift["missing"]:
        raise RuntimeError(f"source drift after declaration: {drift}")
    ctx = prepare(config_dir, stage)
    records, ledger = (generate_e2 if stage == "e2" else generate_e4)(ctx)
    runner = _Runner(ctx, config_dir, stage, workers, progress)
    try:
        rows = (run_e2_arms if stage == "e2" else run_e4_arms)(ctx, runner, records)
    finally:
        runner.close()
    analysis = (analyse_e2 if stage == "e2" else analyse_e4)(values, rows, ledger)
    artifact = {"artifact_type": stage, "status": values["status"], "analysis": analysis, "records": rows,
                "claim_boundary": {"development_only": True, "sealed_bank_opened": False,
                                   "exact_states_used_for": "generation (E2 bases), class labels, the oracle-free rules' evaluation, errors, coverage and the orbit geometry"},
                "execution": {"workers": int(workers), "records": len(rows), "wall_seconds": time.perf_counter() - started}}
    artifact = m1._jsonable(artifact)
    artifact["declaration_sha256"] = file_sha256(folder / DECLARATIONS[stage])
    artifact["config_sha256"] = config_sha
    artifact["source_snapshot_hash_declared"] = declaration["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    (folder / ARTIFACTS[stage]).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
