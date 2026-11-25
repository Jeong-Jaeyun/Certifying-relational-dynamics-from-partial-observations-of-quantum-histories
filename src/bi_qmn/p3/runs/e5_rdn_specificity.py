"""E5 development pilot: specificity of the S3 certification route (RDN and BI) under controlled truths.

Plan: documents/BI-QMN-remaining-experiments-maximum-package.md §4 E5. Development only: a new development
bank (seeds 56101 / 56102, outside the reserved namespace) and the frozen S3 bodies, read only. Nothing
sealed is opened and no frozen artifact is changed.

The S3 route certifies relational dynamics on the **fitted** table of the pure-history SBP estimator. That
table includes the unobserved contexts the model extrapolates. The S3 claim was a sensitivity claim on
pure tau-dependent histories. This pilot asks what the same route does when the truth is not tau-dependent
or not in the model class.

Truth families (same construction, same mask and the same shot random numbers for each family):

* ``pure``          ``|Psi><Psi|``: the S3 population (positive control);
* ``dephased_0.5``, ``dephased_0.9``: ``(1-nu)|Psi><Psi| + nu E(|Psi><Psi|)``, where ``E`` pinches the clock
  energy blocks. The state stays zero-energy, and its tau-modulation scales by ``1-nu`` (weak positives);
* ``tau_flat``      ``E(|Psi><Psi|)`` (nu = 1): zero-energy and exactly tau-flat; **any RDN certification
  is false** (negative control);
* ``registry_violating``: ``(I (x) W)|Psi>`` with a Haar-random system unitary ``W``. The tau-dependence is
  kept, but the state is not zero-energy for the registered ``H_tot`` (BI control).

Each record runs the S3 route for its stratum (medium or low): screen v1 on the sampled support, the SBP
fit, BI existence with the constraint registry, and rdn_v1 with the energy-pinched candidates (the exact-state
candidate is the family's truth) and the optimal-classical refinement. Recorded per record:

* screen status, BI and RDN status, and the stage at which RDN decided;
* the fitted kappa and the true kappa (the truth's tau-flat distance over the radius);
* body coverage of the truth;
* the support misfit (the G statistic of the fit against the observed support), a candidate abstention
  diagnostic;
* the spurious tau-modulation the fit puts on unobserved D contexts.
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
from qiskit.quantum_info import DensityMatrix
from scipy.stats import beta as beta_dist
import yaml

from bi_qmn.core import identifiability as ident
from bi_qmn.core.config import load_p1_config
from bi_qmn.core.observations import sample_shot_batch
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.ilgc.certificate import _compute_existence_claim
from bi_qmn.ilgc.model import calibration_bodies_from_radii, measurement_model_from_p1, p1_constraint_registry
from bi_qmn.ilgc.relational_dynamics import RDN_VERSION, adjudicate_relational_dynamics, energy_block_candidate, optimal_classical_candidate, tau_flat_distance
from bi_qmn.p1.model.p1_banks import support_contexts_for_deficit
from bi_qmn.p1.model.p1_generator import seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import full_p1_context_list, p1_effect_map, p1_measurement_table
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairEstimator
from bi_qmn.p3.p3_ilgc_facing import load_ilgc_targets
from bi_qmn.p3.p3_support_screen import SupportIdentifiabilityScreen
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _keys, _summ, _table_dict
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint


METHOD = "e5_rdn_specificity_v1"
OUTPUT_FOLDER = "e5_rdn_specificity"
ARTIFACT = "e5_rdn_specificity.json"
RUNNER = "runners/p3/run_e5_rdn_specificity.py"
S3_CALIBRATION = Path("outputs") / "p3_s3_calibration" / "p3_s3_calibration.json"
S3_CALIBRATION_SHA256 = "2144a7c4cb2d321026f05c7a9b29022eb69f2f49e6cd42de22ea46d74e4673fb"
DEV_CONFIG = "p3_d1_v7.yaml"
FAMILIES = ("pure", "dephased_0.5", "dephased_0.9", "tau_flat", "registry_violating")
DEFICITS = ("medium", "low")
CLASSES = ("diag", "coh")
BANK = {"bank_id": "e5_rdn_dev_v1", "seed": 56101, "observation_seed": 56102, "haar_seed": 56103, "block_count": 48,
        "dirichlet_concentration": 8.0, "minimum_mode_probability": 0.15, "phase_range": [0.0, 6.283185307179586]}
RESERVED = (53500, 53999)
_W: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:16], 16)


def frozen_s3_bodies(path: Path = S3_CALIBRATION, sha256: str | None = S3_CALIBRATION_SHA256) -> dict[str, Any]:
    if sha256 is not None and file_sha256(path) != sha256:
        raise RuntimeError("the frozen S3 calibration differs from its recorded SHA-256")
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))["freeze"]
    if not freeze.get("frozen"):
        raise RuntimeError("the S3 calibration is not frozen")
    return {"radii": {d: {c: float(freeze["radii"][d][c]) for c in CLASSES} for d in DEFICITS}, "calibration_hash": freeze["calibration_hash"]}


def prepare(config_dir: str | Path = "config", bank: Mapping[str, Any] | None = None, *, estimator_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config_dir = Path(config_dir)
    bank = dict(bank or BANK)
    if any(RESERVED[0] <= int(bank[k]) <= RESERVED[1] for k in ("seed", "observation_seed", "haar_seed")):
        raise ValueError("E5 development seeds must lie outside the reserved confirmatory namespace")
    p1 = load_p1_config(config_dir)
    dev = yaml.safe_load((config_dir / DEV_CONFIG).read_text(encoding="utf-8"))
    reference = dev["development_banks"]["evaluation"]
    if any(bank[k] != reference[k] for k in ("dirichlet_concentration", "minimum_mode_probability", "phase_range")):
        raise ValueError("the E5 bank must use the D1-v7 generative specification")
    cfg = {**dev["estimator"], **dict(estimator_overrides or {})}
    contexts = full_p1_context_list(p1.measurement)
    dims = p1.generator["dimensions"]
    K, d_C, d_S = int(dims["K"]), int(dims["d_C"]), int(dims["d_S"])
    effects_by_context = {cid: {o: np.asarray(op.data, dtype=complex) for o, op in row.items()} for cid, row in p1_effect_map(contexts, K=K).items()}
    sbp = SharedBranchPairEstimator(
        p1.generator["active_modes"], K, d_S, effects_by_context,
        starts=int(cfg["starts"]), seed=int(cfg["seed"]), probability_floor=float(cfg["probability_floor"]), likelihood_floor=float(cfg["likelihood_floor"]),
        near_optimal_nats=float(cfg["near_optimal_nats"]), max_iterations=int(cfg["max_iterations"]), pair_observability_threshold=float(cfg["pair_observability_threshold"]),
    )
    instances = seeded_construction_bank_p1(p1.generator, bank)
    targets = load_ilgc_targets(config_dir, "p1_ilgc_operational_r4.yaml")
    cert = dict(dev["reporting"]["certificate"])
    return {
        "config_dir": str(config_dir), "bank": bank, "p1": p1, "K": K, "d_C": d_C, "d_S": d_S,
        "theta_con": float(p1.generator["thresholds"]["theta_con"]), "eta": float(p1.measurement["noise"]["readout_eta"]),
        "instances": instances, "instance_hash": hashlib.sha256("".join(physical_fingerprint(i) for i in instances).encode("ascii")).hexdigest()[:16],
        "contexts": contexts, "context_by_id": {c.context_id: c for c in contexts}, "ids": [c.context_id for c in contexts],
        "effects": {(cid, o): M for cid, row in effects_by_context.items() for o, M in row.items()},
        "sbp": sbp, "estimator": cfg,
        "screen": SupportIdentifiabilityScreen(p1.generator["active_modes"], K, epsilon=float(dev["screen"]["disconnected_branch_null_tail_epsilon"])),
        "bodies": frozen_s3_bodies(), "strata": {d: targets["strata"][d] for d in DEFICITS},
        "certificate": {"precision_dps": int(cert.get("precision_dps", 80)), "proposal_backends": tuple(cert["proposal_backends"]),
                        "refine": bool(cert.get("refine_unresolved_with_optimal_classical_state", True))},
        "diag_keys": _keys(contexts, "diag", K),
    }


# ---------------------------------------------------------------------------
# Truth families
# ---------------------------------------------------------------------------


def pinch(rho: np.ndarray, K: int, d_S: int) -> np.ndarray:
    """``E(rho) = sum_r (|omega_r><omega_r| (x) I) rho (|omega_r><omega_r| (x) I)``."""
    out = np.zeros_like(rho)
    for r in range(K):
        s = slice(r * d_S, (r + 1) * d_S)
        out[s, s] = rho[s, s]
    return out


def haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    Z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))[None, :]


def family_density(ctx: Mapping[str, Any], i: int, family: str) -> np.ndarray:
    inst = ctx["instances"][int(i)]
    psi = np.asarray(inst.statevector.data, dtype=complex)
    P = np.outer(psi, psi.conj())
    K, d_S = ctx["K"], ctx["d_S"]
    if family == "pure":
        return P
    if family.startswith("dephased_"):
        nu = float(family.split("_", 1)[1])
        return (1.0 - nu) * P + nu * pinch(P, K, d_S)
    if family == "tau_flat":
        return pinch(P, K, d_S)
    if family == "registry_violating":
        W = haar_unitary(d_S, np.random.default_rng(_seed(f"{ctx['bank']['haar_seed']}:{ctx['bank']['bank_id']}:{int(i)}")))
        psi2 = np.kron(np.eye(K), W) @ psi
        return np.outer(psi2, psi2.conj())
    raise ValueError(f"unknown family {family}")


def registered_energy(ctx: Mapping[str, Any], i: int, rho: np.ndarray) -> float:
    inst = ctx["instances"][int(i)]
    H = np.asarray(inst.H_tot.data, dtype=complex)
    return float(np.real(np.trace(rho @ H @ H)) / (float(inst.h_scale) ** 2))


def conditional_tau_spread(rho: np.ndarray, K: int, d_S: int) -> float:
    """Largest trace distance between conditional system states at two clock readings (0 iff tau-flat)."""
    states = []
    for l in range(K):
        v = np.exp(-2j * math.pi * np.arange(K) * l / K) / math.sqrt(K)
        Pi = np.kron(np.outer(v, v.conj()), np.eye(d_S))
        M = Pi @ rho @ Pi
        # partial trace over the clock of M
        Mr = M.reshape(K, d_S, K, d_S)
        rs = np.einsum("kakb->ab", Mr)
        p = float(np.real(np.trace(rs)))
        states.append(rs / p if p > 1e-14 else rs)
    return max(0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(states[a] - states[b])))) for a in range(K) for b in range(a + 1, K))


# ---------------------------------------------------------------------------
# The S3 route for one record
# ---------------------------------------------------------------------------


def certify(ctx: Mapping[str, Any], i: int, deficit: str, table: Mapping[str, Mapping[str, float]], rho_truth: np.ndarray, support_query_hash: str) -> dict[str, Any]:
    """BI existence and rdn_v1 on the frozen S3 bodies of ``deficit`` (the S3 confirmatory route)."""
    K, d_C, d_S, contexts = ctx["K"], ctx["d_C"], ctx["d_S"], ctx["contexts"]
    cert = ctx["certificate"]
    instance = ctx["instances"][int(i)]
    stratum, radii = ctx["strata"][deficit], ctx["bodies"]["radii"][deficit]
    calibration_hash = f"p3_s3_calibration:{ctx['bodies']['calibration_hash']}:{deficit}"
    model = measurement_model_from_p1(MeasurementTable(probabilities=table, contexts=list(contexts), metadata={"eta": 0.0}, source_density=None), K, stratum=stratum,
                                      estimator_checkpoint_hash=ctx["sbp"].checkpoint_hash, support_query_hash=support_query_hash)
    bodies = calibration_bodies_from_radii(model, radii, calibration_hash=calibration_hash, stratum=stratum)
    registry = p1_constraint_registry(instance, ctx["theta_con"], calibration_hash=calibration_hash)
    claim, _ = _compute_existence_claim(model, bodies, registry, precision_dps=cert["precision_dps"], proposal_backends=cert["proposal_backends"])
    bi = {"status": claim.status, "reason": claim.decision_reason}
    rho_bi = np.asarray(claim.primal_evidence.density.data, dtype=complex) if claim.status == "primal_verified" else None
    class_keys = {cls: _keys(contexts, cls, K) for cls in CLASSES}
    targets = {cls: ident.class_vector(table, contexts, cls, K) for cls in CLASSES}
    H = np.asarray(instance.H_tot.data, dtype=complex)
    constraints = [(H @ H / (instance.h_scale ** 2), None, ctx["theta_con"])]
    rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, d_S=d_S, precision_dps=cert["precision_dps"])
    stage = "distance"
    if rdn["status"] != "certified":
        stage = "candidates"
        candidates = [] if rho_bi is None else [("energy_pinched_bi_primal_state", energy_block_candidate(rho_bi, d_C, d_S))]
        candidates.append(("energy_pinched_exact_state", energy_block_candidate(np.asarray(rho_truth), d_C, d_S)))
        rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], candidates=candidates, constraints=constraints, d_S=d_S)
        if rdn["status"] == "unresolved" and cert["refine"]:
            stage = "optimal_classical"
            proposal = optimal_classical_candidate(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], constraints=constraints, d_C=d_C, d_S=d_S)
            if proposal["blocks"] is not None:
                rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], candidates=[*candidates, ("optimal_classical_quantum_state", proposal["blocks"])], constraints=constraints, d_S=d_S)
    return {"bi": bi, "rdn": {"status": rdn["status"], "kappa": rdn["kappa"], "tau_flat_distance": rdn["tau_flat_distance"], "refuting_candidate": rdn.get("refuting_candidate"), "stage": stage}}


def support_misfit(support: Mapping[str, Mapping[str, float]], fitted: Mapping[str, Mapping[str, float]], n_shots: float) -> dict[str, float]:
    """G statistic ``2 sum n p log(p / q)`` of the fitted support tables against the observed support."""
    G, cells = 0.0, 0
    for cid, row in support.items():
        for o, p in row.items():
            cells += 1
            if p > 0:
                G += 2.0 * n_shots * float(p) * math.log(float(p) / max(float(fitted[cid][o]), 1e-300))
    return {"G": G, "cells": cells, "contexts": len(support)}


def unobserved_modulation(ctx: Mapping[str, Any], table: Mapping[str, Mapping[str, float]], observed: Sequence[str]) -> float:
    """tau-modulation norm summed (in quadrature) over the D contexts that were not observed."""
    per = ident.tau_modulation(table, ctx["contexts"], ctx["K"])["per_context"]
    return float(math.sqrt(sum(v * v for cid, v in per.items() if cid not in set(observed))))


def record_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    i, deficit, family = int(job["index"]), str(job["deficit"]), str(job["family"])
    bank = ctx["bank"]
    inst = ctx["instances"][i]
    rho = family_density(ctx, i, family)
    exact = p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=inst.K, eta=ctx["eta"],
                                 metadata={"instance_index": i, "deficit": deficit, "family": family, "bank_id": bank["bank_id"]}).copy_without_source()
    exact_tables = {c: dict(row) for c, row in exact.probabilities.items()}
    n_shot = int(ctx["p1"].measurement["observation_deficit"][deficit]["n_shot"])
    support_contexts = support_contexts_for_deficit(ctx["p1"].measurement, deficit, seed_key=f"p3v2:{bank['bank_id']}:{deficit}:{i}")
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bank['bank_id']}:{deficit}:{i}"))      # common random numbers across families
    support = sample_shot_batch(exact, support_contexts, n_shots=n_shot, rng=rng, shot_start=(DEFICITS.index(deficit) * 1000 + i) * 100_000_000, role="e5_support").table.probabilities
    support = {c: dict(row) for c, row in support.items()}
    observed = sorted(support)
    screen = ctx["screen"].evaluate(support, float(n_shot))
    radii = ctx["bodies"]["radii"][deficit]
    true_distance = tau_flat_distance(ident.class_vector(exact_tables, ctx["contexts"], "diag", ctx["K"]), ctx["diag_keys"])
    row = {"index": i, "deficit": deficit, "family": family, "unitary": inst.system_unitary_label, "observed": observed,
           "screen_passes": bool(screen["passes"]), "screen_reasons": list(screen["reasons"]),
           "truth": {"kappa": true_distance / radii["diag"], "tau_flat_distance": true_distance, "registered_energy": registered_energy(ctx, i, rho),
                     "conditional_tau_spread": conditional_tau_spread(rho, ctx["K"], ctx["d_S"]), "purity": float(np.real(np.trace(rho @ rho)))}}
    prediction = ctx["sbp"].predict_table(MeasurementTable(probabilities=support, contexts=[ctx["context_by_id"][c] for c in observed], metadata={"n_shots": float(n_shot)}, source_density=None), list(ctx["contexts"]))
    table = _table_dict(prediction)
    fit = prediction.metadata["shared_branch_pair_fit"]
    full = {cls: float(np.linalg.norm(ident.class_vector(table, ctx["contexts"], cls, ctx["K"]) - ident.class_vector(exact_tables, ctx["contexts"], cls, ctx["K"]))) for cls in CLASSES}
    row["fit"] = {"objective_nats": fit["objective_nats"], "starts_within_near_optimal": fit["starts_within_near_optimal"], "mask_class": fit["pair_observability"]["mask_class"],
                  "kappa": tau_flat_distance(ident.class_vector(table, ctx["contexts"], "diag", ctx["K"]), ctx["diag_keys"]) / radii["diag"],
                  "misfit": support_misfit(support, table, float(n_shot)),
                  "unobserved_D_modulation": unobserved_modulation(ctx, table, observed), "unobserved_D_modulation_truth": unobserved_modulation(ctx, exact_tables, observed)}
    row["coverage"] = {**{cls: bool(full[cls] <= radii[cls]) for cls in CLASSES}, "scores": full}
    row["coverage"]["joint"] = all(row["coverage"][cls] for cls in CLASSES)
    if row["screen_passes"]:
        row["certificate"] = certify(ctx, i, deficit, table, rho, f"{bank['bank_id']}:{deficit}:{i}:{family}")
        row["success_s3_rule"] = row["certificate"]["bi"]["status"] == "primal_verified" and row["certificate"]["rdn"]["status"] == "certified"
    else:
        row["certificate"], row["success_s3_rule"] = None, False
    row["seconds"] = time.perf_counter() - started
    return {"key": (i, deficit, family), "row": row}


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


def _q(v: Sequence[float]) -> dict[str, float] | None:
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return {k: float(np.quantile(v, float(k))) for k in ("0.0", "0.1", "0.5", "0.9", "1.0")} if v else None


def _counts(values: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def auc(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """P(a positive score exceeds a negative one), ties half."""
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (pos.size * neg.size))


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for deficit in DEFICITS:
        for family in FAMILIES:
            sr = [r for r in rows if r["deficit"] == deficit and r["family"] == family]
            if not sr:
                continue
            inn = [r for r in sr if r["screen_passes"]]
            rdn_cert = sum(r["certificate"]["rdn"]["status"] == "certified" for r in inn)
            succ = sum(r["success_s3_rule"] for r in sr)
            out[f"{deficit}:{family}"] = {
                "n": len(sr), "screened_in": len(inn),
                "bi_status_counts": _counts([r["certificate"]["bi"]["status"] for r in inn]),
                "rdn_status_counts": _counts([r["certificate"]["rdn"]["status"] for r in inn]),
                "rdn_stage_counts": _counts([f"{r['certificate']['rdn']['stage']}:{r['certificate']['rdn']['status']}" for r in inn]),
                "rdn_certified_rate_screened_in": rdn_cert / len(inn) if inn else None, "rdn_certified_ci95": clopper_pearson(rdn_cert, len(inn)) if inn else None,
                "s3_success": succ, "s3_success_rate": succ / len(sr), "s3_success_ci95": clopper_pearson(succ, len(sr)),
                "truth_kappa": _q([r["truth"]["kappa"] for r in sr]), "fit_kappa": _q([r["fit"]["kappa"] for r in sr]),
                "truth_conditional_tau_spread": _q([r["truth"]["conditional_tau_spread"] for r in sr]),
                "truth_registered_energy": _q([r["truth"]["registered_energy"] for r in sr]),
                "coverage": {k: sum(r["coverage"][k] for r in sr) for k in ("diag", "coh", "joint")},
                "misfit_G": _q([r["fit"]["misfit"]["G"] for r in sr]),
                "unobserved_D_modulation_fit": _q([r["fit"]["unobserved_D_modulation"] for r in sr]),
                "unobserved_D_modulation_truth": _q([r["fit"]["unobserved_D_modulation_truth"] for r in sr]),
                "fit_mask_class_counts": _counts([r["fit"]["mask_class"] for r in sr]),
            }
    # misfit as a diagnostic: how well does G separate each family from the pure family (same deficit)
    sep = {}
    for deficit in DEFICITS:
        pure = [r["fit"]["misfit"]["G"] for r in rows if r["deficit"] == deficit and r["family"] == "pure"]
        for family in FAMILIES[1:]:
            other = [r["fit"]["misfit"]["G"] for r in rows if r["deficit"] == deficit and r["family"] == family]
            sep[f"{deficit}:{family}"] = {"auc_G_family_above_pure": auc(other, pure), "pure_G_p95": float(np.quantile(pure, 0.95)) if pure else None,
                                          "family_above_pure_p95": float(np.mean(np.asarray(other) > np.quantile(pure, 0.95))) if pure and other else None}
    out["misfit_separation"] = sep
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _worker_init(root: str, config_dir: str, bank: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> None:
    os.chdir(root)
    _W.clear()
    _W.update(prepare(config_dir, bank, estimator_overrides=overrides))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    out = record_job(_W, job)
    out["instance_hash"] = _W["instance_hash"]
    return out


def run_e5(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print,
           bank: Mapping[str, Any] | None = None, families: Sequence[str] = FAMILIES, deficits: Sequence[str] = DEFICITS,
           estimator_overrides: Mapping[str, Any] | None = None, write: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    config_dir = Path(config_dir)
    target = Path(output_dir) / OUTPUT_FOLDER / ARTIFACT
    if write and target.exists():
        raise FileExistsError(f"E5 artifact already exists: {target}")
    ctx = prepare(config_dir, bank, estimator_overrides=estimator_overrides)
    jobs = [{"index": i, "deficit": d, "family": f} for d in deficits for i in range(len(ctx["instances"])) for f in families]
    pool = None
    saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
    try:
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(Path.cwd()), str(config_dir.resolve()), dict(ctx["bank"]), dict(estimator_overrides or {})))
        else:
            _W.clear()
            _W.update(ctx)
        results = _execute(pool, _worker_job, jobs, "E5 records", progress)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if any(x.get("instance_hash") != ctx["instance_hash"] for x in results.values()):
        raise RuntimeError("worker-rebuilt constructions differ from the parent constructions")
    rows = [results[(j["index"], j["deficit"], j["family"])]["row"] for j in jobs]
    snapshot = import_closure_snapshot(Path.cwd(), extra_files=[p for p in (str(config_dir / DEV_CONFIG), RUNNER) if Path(p).exists()])
    artifact = {
        "artifact_type": "e5_rdn_specificity", "method": METHOD, "status": "development_only",
        "claim_boundary": {"development_only": True, "sealed_bank_opened": False, "frozen_artifacts_changed": False,
                           "exact_states_used_for": "the truth families, true kappa and spread, coverage, the energy-pinched exact-state candidate (as in S3), registered energy"},
        "settings": {"bank": ctx["bank"], "families": list(families), "deficits": list(deficits), "frozen_radii": ctx["bodies"]["radii"], "calibration_hash": ctx["bodies"]["calibration_hash"],
                     "estimator": ctx["estimator"], "certificate": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ctx["certificate"].items()}, "rdn_version": RDN_VERSION},
        "summary": summarize(rows),
        "records": rows,
        "execution": {"workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial", "records": len(rows),
                      "record_seconds": _summ([r["seconds"] for r in rows]), "wall_seconds": time.perf_counter() - started},
        "source_snapshot_hash": snapshot_hash(snapshot), "source_snapshot": snapshot, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    artifact = m1._jsonable(artifact)
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
