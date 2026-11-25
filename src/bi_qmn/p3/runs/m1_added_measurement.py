"""M1 Q1 verification: risk detection on a fixed support, and recovery by one added D context.

Protocol: documents/M1-added-measurement-design.md (frozen by the declaration). User decisions of
2026-11-22. Development only: the bank is new, built and screened only by the run, after the declaration.

* **Bank.** Constructions under the D1-v7 generative law, each with a low mask drawn uniformly from the
  flagged masks (observed D Paulis with a non-trivial Pauli commutant ``R``). Screen v1 on the sampled
  support; screened-in constructions are classified 3a (exact orbit: branch condition at the true branch
  directions) or 3b, and accepted in index order until each class quota is full or the cap is reached.
* **Arms.** A0 support only; A_s support + the candidate with the largest orbit-invariant ``s_Q`` at the
  A0 fit; A_f support + the first candidate in fixed order; C the same total budget as extra shots on the
  support; A_o support + the candidate with the largest truth ``smin_rel`` (fixed before any added-shot
  result). Every candidate's 2,048-shot sample is drawn before any fit, so no observation depends on the rule.
* **Evaluation.** Class scores on ``E = all - support - {Q_s, Q_f}`` (A0, A_s, A_f, C) and on
  ``E_o = E - {Q_o}`` (oracle block); ``CVaR_0.9`` per arm and a paired bootstrap over record indices.
* **Reports.** Rank stability of ``s_Q`` at the fit against the truth (with the fit's distance from the
  truth's orbit), A0 certificates on the frozen S3 low bodies (BI + rdn_v1), coverage, ``kappa`` for the
  exact table and every arm, and structural checks at the truth.
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
from scipy.stats import spearmanr
import yaml

from bi_qmn.core import identifiability as ident
from bi_qmn.core.config import load_p1_config
from bi_qmn.core.observations import sample_shot_batch
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.ilgc.certificate import _compute_existence_claim  # existence claim only: no selector, no atlas
from bi_qmn.ilgc.model import calibration_bodies_from_radii, measurement_model_from_p1, p1_constraint_registry
from bi_qmn.ilgc.relational_dynamics import RDN_VERSION, adjudicate_relational_dynamics, energy_block_candidate, optimal_classical_candidate, tau_flat_distance
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import full_p1_context_list, p1_effect_map, p1_measurement_table
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairEstimator
from bi_qmn.p3.p3_ilgc_facing import load_ilgc_targets
from bi_qmn.p3.p3_support_screen import SCREEN_VERSION, SupportIdentifiabilityScreen
from bi_qmn.p3.runs import m1_added_context_check as mac
from bi_qmn.p3.runs import m1_d_commutant_check as dc
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _keys, _summ, _table_dict, _true_branch_vectors
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint


CONFIG_FILE = "m1_added_measurement.yaml"
RUNNER = "runners/p3/run_m1_added_measurement.py"
DECLARATION = "m1_added_measurement_declaration.json"
ARTIFACT = "m1_added_measurement.json"
CLASSES = ("diag", "coh")
STRATA = ("3a", "3b")
ARMS = ("A0", "A_s", "A_f", "C", "A_o")
GENERATIVE_KEYS = ("dirichlet_concentration", "minimum_mode_probability", "phase_range")
COMPARISONS_E = (("P1", "A0", "A_s"), ("P2", "A_f", "A_s"), ("A0_minus_A_f", "A0", "A_f"), ("C_minus_A_s", "C", "A_s"), ("A0_minus_C", "A0", "C"))
COMPARISONS_E_O = (("A0_minus_A_o", "A0", "A_o"), ("A_s_minus_A_o", "A_s", "A_o"), ("A_f_minus_A_o", "A_f", "A_o"))
_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Configuration, bodies, declaration, guard
# ---------------------------------------------------------------------------


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:16], 16)


def _load(config_dir: Path) -> tuple[dict[str, Any], str]:
    path = Path(config_dir) / CONFIG_FILE
    values = _yaml(path)
    if values.get("status") != "development_only" or float(values["coherent_core_gamma"]) != 1.0:
        raise ValueError("the M1 added-measurement verification is a development-only gamma=1 run")
    dev = _yaml(Path(config_dir) / values["design_source"]["development_config"])
    if values["estimator"] != dev["estimator"]:
        raise ValueError("estimator settings differ from the D1-v7 development config")
    if values["screen"] != dev["screen"] or values["screen"]["method"] != SCREEN_VERSION or not values["screen"]["pauli_commutant"]:
        raise ValueError(f"the screen must be the frozen {SCREEN_VERSION} with the commutant check")
    if values["deficit"] != "low":
        raise ValueError("the flagged masks are low masks")
    bank = values["bank"]
    reference = dev["development_banks"]["evaluation"]
    if any(bank[key] != reference[key] for key in GENERATIVE_KEYS):
        raise ValueError("the bank must use the D1-v7 generative specification")
    if set(bank["quotas"]) != set(STRATA) or any(int(v) < 0 for v in bank["quotas"].values()):
        raise ValueError("quotas are declared for 3a and 3b")
    lo, hi = values["reserved_future_confirmatory_namespace"]["seed_range"]
    seeds = [bank["seed"], bank["observation_seed"], bank["mask_seed"], values["evaluation"]["bootstrap"]["seed"]]
    if any(int(lo) <= int(s) <= int(hi) for s in seeds):
        raise ValueError("development seeds intrude on the reserved confirmatory namespace")
    cert = values["certificate"]
    dev_cert = dev["reporting"]["certificate"]
    if (cert["existence"], cert["proposal_backends"]) != (dev_cert["existence"], dev_cert["proposal_backends"]) or cert["rdn_version"] != RDN_VERSION:
        raise ValueError("certificate settings differ from the D1-v7/S3 route")
    return values, file_sha256(path)


def _bodies(root: Path, values: Mapping[str, Any]) -> dict[str, Any]:
    spec = values["bodies"]
    path = Path(root) / str(spec["source_artifact"])
    if file_sha256(path) != spec["source_artifact_sha256"]:
        raise RuntimeError("the frozen S3 calibration artifact differs from its declared SHA-256")
    freeze = json.loads(path.read_text(encoding="utf-8"))["freeze"]
    if not freeze.get("frozen") or freeze["calibration_hash"] != spec["calibration_hash"]:
        raise RuntimeError("the body source is not the declared frozen S3 calibration")
    radii = {cls: float(freeze["radii"][values["deficit"]][cls]) for cls in CLASSES}
    return {"radii": radii, "calibration_hash": spec["calibration_hash"], "source_artifact": str(spec["source_artifact"]), "source_artifact_sha256": spec["source_artifact_sha256"]}


def _dev_inputs(root: Path, values: Mapping[str, Any]) -> dict[str, str]:
    out = {}
    for rel, sha in values["design_source"]["dev_inputs"].items():
        if file_sha256(Path(root) / rel) != sha:
            raise RuntimeError(f"development input differs from its declared SHA-256: {rel}")
        out[rel] = sha
    return out


def _extra_files(config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    names = [CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", str(values["ilgc_targets"]["operational_config"]), str(values["design_source"]["development_config"])]
    return [str(Path(config_dir) / n) for n in dict.fromkeys(names)] + [str(values["bodies"]["source_artifact"]), *values["design_source"]["dev_inputs"], str(values["protocol_document"]), RUNNER]


def flagged_masks(measurement: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """Low masks whose observed D Paulis have a non-trivial Pauli commutant (the threshold-free risk flag)."""
    return [m for m in m1.deficit_masks(measurement, "low") if dc.d_commutant(m)]


def declare_m1_added_measurement(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
    target = folder / DECLARATION
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    bodies = _bodies(root, values)
    dev_inputs = _dev_inputs(root, values)
    p1 = load_p1_config(config_dir)
    bank = values["bank"]
    fingerprints = sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, bank))
    known = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked = {}
    for path, spec in harness.declared_bank_specs(config_dir, exclude=CONFIG_FILE):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    overlaps = {name: sorted(set(fingerprints) & f) for name, f in known.items() if set(fingerprints) & f}
    if overlaps:
        raise ValueError(f"the M1 added-measurement bank is not disjoint: {sorted(overlaps)}")
    flagged = flagged_masks(p1.measurement)
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values))
    declaration = {
        "artifact_type": "m1_added_measurement_declaration",
        "status": values["status"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha,
        "protocol_document": str(values["protocol_document"]),
        "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
        "dev_inputs": dev_inputs,
        "bodies": bodies,
        "estimator": values["estimator"],
        "screen": values["screen"],
        "bank": {**bank, "construction_fingerprint_hash": hashlib.sha256("".join(fingerprints).encode("ascii")).hexdigest()[:16], "observation_tables_opened": False,
                 "flagged_mask_count": len(flagged), "flagged_mask_hash": hashlib.sha256(json.dumps([list(m) for m in flagged]).encode("ascii")).hexdigest()[:16]},
        "added_context": values["added_context"],
        "control": values["control"],
        "evaluation": values["evaluation"],
        "endpoints": values["endpoints"],
        "certificate": values["certificate"],
        "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps},
        "reserved_future_confirmatory_namespace": values["reserved_future_confirmatory_namespace"],
        "source_snapshot": snapshot,
        "source_snapshot_hash": snapshot_hash(snapshot),
    }
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return declaration


def _guard(config_dir: Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path.cwd()
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
    declaration = json.loads((folder / DECLARATION).read_text(encoding="utf-8"))
    if (folder / ARTIFACT).exists():
        raise FileExistsError("the M1 added-measurement artifact already exists and is immutable")
    if declaration["config_sha256"] != config_sha:
        raise RuntimeError("config drifted after declaration")
    if declaration["protocol_document_sha256"] != file_sha256(root / str(values["protocol_document"])):
        raise RuntimeError("protocol document drifted after declaration")
    extra = [p for p in declaration["source_snapshot"] if not p.startswith("src/")]
    drift = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    if drift["changed"] or drift["missing"]:
        raise RuntimeError(f"source drift after declaration: {drift}")
    return {"root": root, "values": values, "config_sha": config_sha, "declaration": declaration, "folder": folder, "extra_files": extra}


# ---------------------------------------------------------------------------
# Context, bank and samples
# ---------------------------------------------------------------------------


def _prepare_light(config_dir: Path) -> dict[str, Any]:
    values, _ = _load(config_dir)
    p1 = load_p1_config(config_dir)
    instances = seeded_construction_bank_p1(p1.generator, values["bank"])
    contexts = full_p1_context_list(p1.measurement)
    dims = p1.generator["dimensions"]
    K, d_C, d_S = int(dims["K"]), int(dims["d_C"]), int(dims["d_S"])
    effects_by_context = {cid: {o: np.asarray(op.data, dtype=complex) for o, op in row.items()} for cid, row in p1_effect_map(contexts, K=K).items()}
    cfg = values["estimator"]
    sbp = SharedBranchPairEstimator(
        p1.generator["active_modes"], K, d_S, effects_by_context,
        starts=int(cfg["starts"]), seed=int(cfg["seed"]), probability_floor=float(cfg["probability_floor"]), likelihood_floor=float(cfg["likelihood_floor"]),
        near_optimal_nats=float(cfg["near_optimal_nats"]), max_iterations=int(cfg["max_iterations"]), pair_observability_threshold=float(cfg["pair_observability_threshold"]),
    )
    n_shot = int(p1.measurement["observation_deficit"][values["deficit"]]["n_shot"])
    if int(values["added_context"]["n_shot"]) != n_shot:
        raise ValueError("the added context gets the support's per-context shots (user decision)")
    flagged = flagged_masks(p1.measurement)
    targets = load_ilgc_targets(config_dir, values["ilgc_targets"]["operational_config"])
    return {
        "values": values,
        "p1": p1,
        "K": K, "d_C": d_C, "d_S": d_S, "n": len(p1.generator["active_modes"]),
        "eta": float(p1.measurement["noise"]["readout_eta"]),
        "theta_con": float(p1.generator["thresholds"]["theta_con"]),
        "instances": instances,
        "instance_hash": hashlib.sha256("".join(physical_fingerprint(i) for i in instances).encode("ascii")).hexdigest()[:16],
        "contexts": contexts,
        "context_by_id": {c.context_id: c for c in contexts},
        "ids": [c.context_id for c in contexts],
        "effects": {(cid, o): M for cid, row in effects_by_context.items() for o, M in row.items()},
        "sbp": sbp,
        "fw": sbp.forward,
        "screen": SupportIdentifiabilityScreen(p1.generator["active_modes"], K, epsilon=float(values["screen"]["disconnected_branch_null_tail_epsilon"])),
        "bodies": _bodies(Path.cwd(), values),
        "stratum": targets["strata"][values["deficit"]],
        "flagged": flagged,
        "pool": sorted({cid for m in m1.deficit_masks(p1.measurement, "low") for cid in m if cid.startswith("D,")}),   # input D contexts
        "n_shot": n_shot,
        "diag_keys": _keys(contexts, "diag", K),
    }


def build_construction(ctx: Mapping[str, Any], i: int) -> dict[str, Any]:
    """Mask, exact table, support sample, screen and oracle class of construction ``i``."""
    bank = ctx["values"]["bank"]
    bid = bank["bank_id"]
    inst = ctx["instances"][i]
    rng = np.random.default_rng(_seed(f"{bank['mask_seed']}:{bid}:{i}"))
    mask = tuple(ctx["flagged"][int(rng.integers(0, len(ctx["flagged"])))])
    exact = p1_measurement_table(inst.density, ctx["contexts"], K=inst.K, eta=ctx["eta"],
                                 metadata={"instance": inst.coefficient_label, "instance_index": i, "gamma": 1.0, "deficit": ctx["values"]["deficit"], "eta": ctx["eta"], "bank_id": bid}).copy_without_source()
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bid}:{ctx['values']['deficit']}:{i}"))
    support = sample_shot_batch(exact, [ctx["context_by_id"][c] for c in mask], n_shots=ctx["n_shot"], rng=rng, shot_start=i * 100_000_000, role="m1am_support").table.probabilities
    screen = ctx["screen"].evaluate(support, float(ctx["n_shot"]))
    Rs = dc.d_commutant(mask)
    if len(Rs) != 1:
        raise ValueError(f"expected one D-commutant Pauli for {mask}, got {Rs}")
    U = np.asarray(getattr(inst.system_unitary, "data", inst.system_unitary), dtype=complex)
    cond = dc.branch_condition(U, ctx["n"], Rs[0], mask)
    return {
        "index": i, "unitary": inst.system_unitary_label, "mask": list(mask), "R": Rs[0], "branch_condition": float(cond),
        "class": "3a" if cond < float(bank["exact_class_tolerance"]) else "3b",
        "screen_passes": bool(screen["passes"]), "screen_reasons": list(screen["reasons"]),
        "exact": {c: dict(row) for c, row in exact.probabilities.items()}, "support": {c: dict(row) for c, row in support.items()},
    }


def open_bank(ctx: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate constructions in index order and accept screened-in ones until the quotas fill or the cap is reached."""
    bank = ctx["values"]["bank"]
    quotas = {c: int(bank["quotas"][c]) for c in STRATA}
    accepted: list[dict[str, Any]] = []
    counts = dict.fromkeys(STRATA, 0)
    ledger = {"cap": int(bank["block_count"]), "generated": 0, "screened_out": dict.fromkeys(STRATA, 0), "screened_in": dict.fromkeys(STRATA, 0),
              "turned_away_quota_full": dict.fromkeys(STRATA, 0), "accepted": counts, "screen_out_reasons": {}}
    for i in range(int(bank["block_count"])):
        if all(counts[c] >= quotas[c] for c in STRATA):
            break
        r = build_construction(ctx, i)
        ledger["generated"] += 1
        if not r["screen_passes"]:
            ledger["screened_out"][r["class"]] += 1
            for reason in r["screen_reasons"]:
                ledger["screen_out_reasons"][reason] = ledger["screen_out_reasons"].get(reason, 0) + 1
            continue
        ledger["screened_in"][r["class"]] += 1
        if counts[r["class"]] >= quotas[r["class"]]:
            ledger["turned_away_quota_full"][r["class"]] += 1
            continue
        counts[r["class"]] += 1
        accepted.append(r)
    ledger["quota"] = quotas
    ledger["quota_filled"] = {c: counts[c] >= quotas[c] for c in STRATA}
    ledger["last_index_generated"] = ledger["generated"] - 1
    return accepted, ledger


def attach_samples(ctx: Mapping[str, Any], r: dict[str, Any]) -> None:
    """Candidates, every candidate's added-context sample and the control sample, all drawn before any fit."""
    bank = ctx["values"]["bank"]
    bid, i, n_shot = bank["bank_id"], int(r["index"]), ctx["n_shot"]
    exact = MeasurementTable(probabilities=r["exact"], contexts=list(ctx["contexts"]), metadata={}, source_density=None)
    r["candidates"] = mac.candidates(r["mask"], r["R"], ctx["pool"])
    r["added"] = {}
    for k, q in enumerate(r["candidates"]):
        rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bid}:added:{i}:{q}"))
        r["added"][q] = sample_shot_batch(exact, [ctx["context_by_id"][q]], n_shots=n_shot, rng=rng, shot_start=i * 100_000_000 + 20_000_000 + k * n_shot, role="m1am_added").table.probabilities[q]
    extra_shots = int(ctx["values"]["control"]["extra_shots_per_support_context"])
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bid}:control:{i}"))
    extra = sample_shot_batch(exact, [ctx["context_by_id"][c] for c in r["mask"]], n_shots=extra_shots, rng=rng, shot_start=i * 100_000_000 + 40_000_000, role="m1am_control").table.probabilities
    total = n_shot + extra_shots
    r["control"] = {c: {o: (n_shot * float(r["support"][c][o]) + extra_shots * float(extra[c].get(o, 0.0))) / total for o in r["support"][c]} for c in r["mask"]}
    r["control_n_shots"] = total
    r["Q_f"] = r["candidates"][0]


def argmax_with_ties(scores: Mapping[str, float], order: Sequence[str], rel_tol: float) -> tuple[str, list[str]]:
    """Largest score; values within ``rel_tol`` of the maximum are tied and the first in ``order`` wins."""
    top = max(scores[q] for q in order)
    tied = [q for q in order if scores[q] >= top - rel_tol * abs(top)]
    return tied[0], tied


def variant_tables(r: Mapping[str, Any], variant: str) -> tuple[dict[str, dict[str, float]], float]:
    if variant == "support":
        return r["support"], float(r["n_shot"])
    if variant == "control":
        return r["control"], float(r["control_n_shots"])
    q = variant.split(":", 1)[1]
    return {**r["support"], q: r["added"][q]}, float(r["n_shot"])


# ---------------------------------------------------------------------------
# Worker jobs
# ---------------------------------------------------------------------------


def _truth_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    V = _true_branch_vectors(ctx["instances"][int(job["index"])])
    tp = m1.truth_point(ctx, V)
    base = m1.mask_readout(tp, job["mask"])
    cands = {}
    for q in job["candidates"]:
        r = m1.mask_readout(tp, list(job["mask"]) + [q])
        cands[q] = {"extra_null": int(r["extra_null"]), "smin_rel": float(r["smin_rel"]), "s_Q": mac.orbit_invariant_sensitivity(V, q, job["R"])}
    return {"key": ("truth", int(job["index"])), "support_extra_null": int(base["extra_null"]), "support_smin_rel": float(base["smin_rel"]), "gauge_dim": int(tp["gauge_dim"]), "candidates": cands}


def _fit_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    support = MeasurementTable(probabilities=job["tables"], contexts=[ctx["context_by_id"][c] for c in sorted(job["tables"])], metadata={"n_shots": float(job["n_shots"])}, source_density=None)
    prediction = ctx["sbp"].predict_table(support, list(ctx["contexts"]))
    fit = prediction.metadata["shared_branch_pair_fit"]
    keep = ("objective_nats", "starts_converged", "starts_within_near_optimal", "populations", "branch_vectors_real_imag", "n_shots", "support_context_ids")
    return {"key": (int(job["index"]), str(job["variant"])), "table": _table_dict(prediction), "fit": {k: fit[k] for k in keep}, "elapsed_seconds": time.perf_counter() - started}


def _cert_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    """BI existence and rdn_v1 for the A0 table on the frozen S3 low bodies (the S3 route)."""
    started = time.perf_counter()
    K, d_C, d_S, contexts = ctx["K"], ctx["d_C"], ctx["d_S"], ctx["contexts"]
    cert = ctx["values"]["certificate"]
    dps = int(cert["precision_dps"])
    instance = ctx["instances"][int(job["index"])]
    table = job["table"]
    radii = ctx["bodies"]["radii"]
    calibration_hash = f"p3_s3_calibration:{ctx['bodies']['calibration_hash']}:{ctx['values']['deficit']}"
    model = measurement_model_from_p1(MeasurementTable(probabilities=table, contexts=list(contexts), metadata={"eta": 0.0}, source_density=None), K, stratum=ctx["stratum"],
                                      estimator_checkpoint_hash=ctx["sbp"].checkpoint_hash, support_query_hash=job["support_query_hash"])
    bodies = calibration_bodies_from_radii(model, radii, calibration_hash=calibration_hash, stratum=ctx["stratum"])
    registry = p1_constraint_registry(instance, ctx["theta_con"], calibration_hash=calibration_hash)
    claim, _ = _compute_existence_claim(model, bodies, registry, precision_dps=dps, proposal_backends=tuple(cert["proposal_backends"]))
    bi = {"status": claim.status, "verified_upper_bound": claim.verified_upper_bound, "verified_lower_bound": claim.verified_lower_bound, "reason": claim.decision_reason}
    rho_bi = np.asarray(claim.primal_evidence.density.data, dtype=complex) if claim.status == "primal_verified" else None
    class_keys = {cls: _keys(contexts, cls, K) for cls in CLASSES}
    targets = {cls: ident.class_vector(table, contexts, cls, K) for cls in CLASSES}
    H = np.asarray(instance.H_tot.data, dtype=complex)
    constraints = [(H @ H / (instance.h_scale ** 2), None, ctx["theta_con"])]
    rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, d_S=d_S, precision_dps=dps)
    if rdn["status"] != "certified":
        candidates = [] if rho_bi is None else [("energy_pinched_bi_primal_state", energy_block_candidate(rho_bi, d_C, d_S))]
        candidates.append(("energy_pinched_exact_state", energy_block_candidate(np.asarray(instance.density.data), d_C, d_S)))
        rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], candidates=candidates, constraints=constraints, d_S=d_S)
        if rdn["status"] == "unresolved" and cert["refine_unresolved_with_optimal_classical_state"]:
            proposal = optimal_classical_candidate(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], constraints=constraints, d_C=d_C, d_S=d_S)
            if proposal["blocks"] is not None:
                rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], candidates=[*candidates, ("optimal_classical_quantum_state", proposal["blocks"])],
                                                     constraints=constraints, d_S=d_S) | {"optimal_classical_proposal_objective": proposal["objective"]}
    return {"key": ("cert", int(job["index"])), "bi": bi,
            "rdn": {"status": rdn["status"], "kappa": rdn["kappa"], "tau_flat_distance": rdn["tau_flat_distance"], "refuting_candidate": rdn.get("refuting_candidate")},
            "seconds": time.perf_counter() - started}


_JOBS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]] = {"truth": _truth_job, "fit": _fit_job, "cert": _cert_job}


def _worker_init(root: str, config_dir: str) -> None:
    os.chdir(root)
    _WORKER.clear()
    _WORKER.update(_prepare_light(Path(config_dir)))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    out = _JOBS[job["kind"]](_WORKER, job)
    out["instance_hash"] = _WORKER["instance_hash"]
    return out


# ---------------------------------------------------------------------------
# Metrics and statistics (pure)
# ---------------------------------------------------------------------------


def class_errors(ctx: Mapping[str, Any], table: Mapping[str, Mapping[str, float]], exact: Mapping[str, Mapping[str, float]], context_ids: Sequence[str]) -> dict[str, float]:
    """Per-class L2 error and the largest absolute table difference over ``context_ids``."""
    S = set(context_ids)
    out: dict[str, float] = {}
    for cls in CLASSES:
        subset = [c for c in ctx["contexts"] if c.context_class == cls and c.context_id in S]
        out[cls] = float(np.linalg.norm(ident.class_vector(table, subset, cls, ctx["K"]) - ident.class_vector(exact, subset, cls, ctx["K"]))) if subset else 0.0
    out["max_abs"] = max((abs(float(table[c][o]) - float(exact[c][o])) for c in S for o in exact[c]), default=0.0)
    return out


def tail_count(n: int, level: float) -> int:
    return max(1, math.ceil(round((1.0 - float(level)) * n, 9)))


def cvar(values: Sequence[float], level: float) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    return float(np.mean(v[-tail_count(v.size, level):]))


def paired_cvar_difference(a: Sequence[float], b: Sequence[float], index: np.ndarray, level: float, interval: Sequence[float]) -> dict[str, float]:
    """``CVaR(a) - CVaR(b)``, each arm's tail computed separately, on the same resampled record indices."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = tail_count(a.size, level)
    boot = np.mean(np.sort(a[index], axis=1)[:, -m:], axis=1) - np.mean(np.sort(b[index], axis=1)[:, -m:], axis=1)
    lo, hi = np.quantile(boot, [float(interval[0]), float(interval[1])])
    return {"point": cvar(a, level) - cvar(b, level), "lower": float(lo), "upper": float(hi), "cvar_first": cvar(a, level), "cvar_second": cvar(b, level), "tail_count": m, "n": int(a.size)}


def p2_outcome(d: Mapping[str, float]) -> str:
    if d["lower"] > 0.0:
        return "s_Q_lower_tail"
    if d["upper"] < 0.0:
        return "fixed_order_replaces_s_Q"
    return "not_distinguished_keep_s_Q"


def m1_statement(c: Mapping[str, Mapping[str, Any]]) -> str:
    """The protocol §10 reading of the judged and reported 3a comparisons."""
    p1 = c["E"]["P1"]["lower"] > 0.0
    if p1:
        return "recovery_demonstrated_fixed_order" if p2_outcome(c["E"]["P2"]) == "fixed_order_replaces_s_Q" else "recovery_demonstrated_s_Q"
    if c["E"]["A0_minus_A_f"]["lower"] > 0.0:
        return "fixed_order_reduced_tail_primary_rule_did_not"
    if c["E_o"]["A0_minus_A_o"]["lower"] > 0.0:
        return "oracle_only_no_support_only_rule_demonstrated"
    return "no_tail_reduction_demonstrated_on_this_bank"


def _quantiles(v: Sequence[float]) -> dict[str, float] | None:
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return {q: float(np.quantile(v, float(q))) for q in ("0.1", "0.5", "0.9", "1.0")} if v else None


def _arm_summary(rows: Sequence[Mapping[str, Any]], set_name: str, level: float) -> dict[str, Any]:
    out = {}
    for arm in ARMS:
        for key in (*CLASSES, "max_abs"):
            v = [r["errors"][arm][set_name][key] for r in rows]
            out[f"{arm}:{key}"] = {"cvar": cvar(v, level), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "max": float(np.max(v))} if v else None
    return out


# ---------------------------------------------------------------------------
# Per-record row (parent)
# ---------------------------------------------------------------------------


def _orbit_distance(ctx: Mapping[str, Any], V_hat: np.ndarray, V: np.ndarray, R: str, points: int) -> float:
    fw = ctx["fw"]
    G = m1.pauli_matrix(R)
    x_hat = fw.vector(V_hat)
    return float(min(m1._state_distance(x_hat, fw.vector(expm(1j * t * G) @ V)) for t in np.linspace(0.0, math.pi, int(points), endpoint=False)))


def _kappa(ctx: Mapping[str, Any], table: Mapping[str, Mapping[str, float]]) -> float:
    return tau_flat_distance(ident.class_vector(table, ctx["contexts"], "diag", ctx["K"]), ctx["diag_keys"]) / float(ctx["bodies"]["radii"]["diag"])


def _branch_vectors(fit: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(fit["branch_vectors_real_imag"][0]) + 1j * np.asarray(fit["branch_vectors_real_imag"][1])


def record_row(ctx: Mapping[str, Any], r: Mapping[str, Any], truth: Mapping[str, Any], fits: Mapping[tuple, Mapping[str, Any]], cert: Mapping[str, Any] | None) -> dict[str, Any]:
    i, ev = int(r["index"]), ctx["values"]["evaluation"]
    variants = {"A0": "support", "C": "control", "A_s": f"add:{r['Q_s']}", "A_f": f"add:{r['Q_f']}", "A_o": f"add:{r['Q_o']}"}
    tables = {arm: fits[(i, v)]["table"] for arm, v in variants.items()}
    rest = [c for c in ctx["ids"] if c not in set(r["mask"])]
    E = [c for c in rest if c not in {r["Q_s"], r["Q_f"]}]
    E_o = [c for c in E if c != r["Q_o"]]
    own = {"A0": rest, "C": rest, "A_s": [c for c in rest if c != r["Q_s"]], "A_f": [c for c in rest if c != r["Q_f"]], "A_o": [c for c in rest if c != r["Q_o"]]}
    errors = {arm: {"E": class_errors(ctx, tables[arm], r["exact"], E), "E_o": class_errors(ctx, tables[arm], r["exact"], E_o), "own": class_errors(ctx, tables[arm], r["exact"], own[arm])} for arm in ARMS}
    full = class_errors(ctx, tables["A0"], r["exact"], ctx["ids"])
    radii = ctx["bodies"]["radii"]
    # rank stability of s_Q at the A0 fit against the truth
    V_hat = _branch_vectors(fits[(i, "support")]["fit"])
    V = _true_branch_vectors(ctx["instances"][i])
    tol = float(ctx["values"]["added_context"]["tie_relative_tolerance"])
    s_fit = {q: float(r["s_Q_fit"][q]) for q in r["candidates"]}
    s_true = {q: float(truth["candidates"][q]["s_Q"]) for q in r["candidates"]}
    _, truth_tied = argmax_with_ties(s_true, r["candidates"], tol)
    rho = spearmanr([s_fit[q] for q in r["candidates"]], [s_true[q] for q in r["candidates"]])[0] if len(set(s_true.values())) > 1 and len(set(s_fit.values())) > 1 else None
    row = {
        "index": i, "class": r["class"], "unitary": r["unitary"], "mask": r["mask"], "R": r["R"], "branch_condition": r["branch_condition"],
        "candidates": r["candidates"], "Q_s": r["Q_s"], "Q_f": r["Q_f"], "Q_o": r["Q_o"], "Q_s_tied_at_fit": r["Q_s_tied"], "Q_o_tied": r["Q_o_tied"],
        "evaluation_sets": {"E": E, "E_o": E_o},
        "errors": errors,
        "kappa": {"exact": _kappa(ctx, r["exact"]), **{arm: _kappa(ctx, tables[arm]) for arm in ARMS}},
        "a0_full_scores": full,
        "a0_covered": {cls: bool(full[cls] <= radii[cls]) for cls in CLASSES},
        "rank_stability": {
            "s_Q_fit": s_fit, "s_Q_truth": s_true, "truth_tied_argmax": truth_tied,
            "agrees_with_truth": r["Q_s"] in truth_tied, "spearman": None if rho is None or not np.isfinite(rho) else float(rho),
            "max_relative_error": max(abs(s_fit[q] - s_true[q]) / s_true[q] for q in r["candidates"]),
            "orbit_distance": _orbit_distance(ctx, V_hat, V, r["R"], int(ev["orbit_distance_t_points"])),
        },
        "structural": {"support_extra_null": truth["support_extra_null"], "gauge_dim": truth["gauge_dim"],
                       "candidate_extra_null": {q: truth["candidates"][q]["extra_null"] for q in r["candidates"]},
                       "truth_smin_rel_with_candidate": {q: truth["candidates"][q]["smin_rel"] for q in r["candidates"]}},
        "fits": {arm: {k: fits[(i, v)]["fit"][k] for k in ("objective_nats", "starts_within_near_optimal")} | {"seconds": fits[(i, v)]["elapsed_seconds"]} for arm, v in variants.items()},
    }
    row["structural"]["unclassified"] = row["structural"]["support_extra_null"] != 1 or any(v != 0 for q, v in row["structural"]["candidate_extra_null"].items() if q in {r["Q_s"], r["Q_f"], r["Q_o"]})
    if cert is not None:
        row["certificate"] = {"bi": cert["bi"], "rdn": cert["rdn"], "seconds": cert["seconds"],
                              "success": cert["bi"]["status"] == "primal_verified" and cert["rdn"]["status"] == "certified"}
    return row


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ev = values["evaluation"]
    level = float(ev["statistic"]["level"])
    boot = ev["bootstrap"]
    primary = str(ev["primary_class"])
    out: dict[str, Any] = {}
    for stratum in STRATA:
        sr = [r for r in rows if r["class"] == stratum]
        n = len(sr)
        block: dict[str, Any] = {"n": n}
        if n == 0:
            out[stratum] = block
            continue
        rng = np.random.default_rng(_seed(f"{boot['seed']}:{stratum}"))
        index = rng.integers(0, n, size=(int(boot["resamples"]), n))
        comps: dict[str, dict[str, Any]] = {"E": {}, "E_o": {}}
        for set_name, table in (("E", COMPARISONS_E), ("E_o", COMPARISONS_E_O)):
            for name, a, b in table:
                comps[set_name][name] = paired_cvar_difference([r["errors"][a][set_name][primary] for r in sr], [r["errors"][b][set_name][primary] for r in sr], index, level, boot["interval"])
        # the judged comparisons on E_o too, so the oracle block is self-contained
        comps["E_o"]["A0_minus_A_s"] = paired_cvar_difference([r["errors"]["A0"]["E_o"][primary] for r in sr], [r["errors"]["A_s"]["E_o"][primary] for r in sr], index, level, boot["interval"])
        block["comparisons_primary_class"] = comps
        block["arm_summaries"] = {s: _arm_summary(sr, s, level) for s in ("E", "E_o", "own")}
        block["picks"] = {"Q_s_equals_Q_f": sum(r["Q_s"] == r["Q_f"] for r in sr), "Q_o_equals_Q_s": sum(r["Q_o"] == r["Q_s"] for r in sr), "Q_o_equals_Q_f": sum(r["Q_o"] == r["Q_f"] for r in sr),
                          "Q_s_tied_at_fit": sum(len(r["Q_s_tied_at_fit"]) > 1 for r in sr)}
        rs = [r["rank_stability"] for r in sr]
        dist = np.asarray([x["orbit_distance"] for x in rs])
        edges = np.quantile(dist, [0.25, 0.5, 0.75])
        bins = np.digitize(dist, edges)
        block["rank_stability"] = {
            "agreement_rate": float(np.mean([x["agrees_with_truth"] for x in rs])),
            "spearman": _quantiles([x["spearman"] for x in rs]),
            "max_relative_error": _quantiles([x["max_relative_error"] for x in rs]),
            "orbit_distance": _quantiles(dist.tolist()),
            "by_orbit_distance_quartile": [{"distance_range": [float(dist[bins == k].min()), float(dist[bins == k].max())] if np.any(bins == k) else None,
                                            "n": int(np.sum(bins == k)), "agreement_rate": float(np.mean([rs[j]["agrees_with_truth"] for j in np.flatnonzero(bins == k)])) if np.any(bins == k) else None}
                                           for k in range(4)],
        }
        certs = [r["certificate"] for r in sr if "certificate" in r]
        kap = {key: [r["kappa"][key] for r in sr] for key in ("exact", *ARMS)}
        spread = [(max(r["kappa"][a] for a in ARMS) - min(r["kappa"][a] for a in ARMS)) / r["kappa"]["exact"] for r in sr]
        block["certification_a0"] = {
            "bi_status_counts": _counts([c["bi"]["status"] for c in certs]),
            "rdn_status_counts": _counts([c["rdn"]["status"] for c in certs]),
            "success": sum(c["success"] for c in certs), "n": len(certs),
            "coverage": {**{cls: sum(r["a0_covered"][cls] for r in sr) for cls in CLASSES}, "joint": sum(all(r["a0_covered"].values()) for r in sr)},
            "certificate_seconds": _summ([c["seconds"] for c in certs]),
        }
        block["kappa"] = {"by_table": {k: _quantiles(v) for k, v in kap.items()}, "fraction_above_1": {k: float(np.mean(np.asarray(v) > 1.0)) for k, v in kap.items()},
                          "relative_spread_across_arms": _quantiles(spread)}
        block["structural"] = {"support_extra_null_counts": _counts([r["structural"]["support_extra_null"] for r in sr]),
                               "unclassified": [r["index"] for r in sr if r["structural"]["unclassified"]]}
        out[stratum] = block
    decision: dict[str, Any] = {"stratum": "3a"}
    if out["3a"]["n"]:
        c = out["3a"]["comparisons_primary_class"]
        decision |= {"P1": {**c["E"]["P1"], "passes": c["E"]["P1"]["lower"] > 0.0},
                     "P2": {**c["E"]["P2"], "outcome": p2_outcome(c["E"]["P2"])},
                     "m1_statement": m1_statement(c)}
    else:
        decision |= {"P1": None, "P2": None, "m1_statement": "no_3a_records"}
    return {"strata": out, "decision": decision}


def _counts(values: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_m1_added_measurement(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started_run = time.perf_counter()
    config_dir = Path(config_dir)
    run = _guard(config_dir, output_dir)
    values, declaration = run["values"], run["declaration"]
    ctx = _prepare_light(config_dir)
    if hashlib.sha256(json.dumps([list(m) for m in ctx["flagged"]]).encode("ascii")).hexdigest()[:16] != declaration["bank"]["flagged_mask_hash"]:
        raise RuntimeError("the flagged masks differ from the declaration")
    accepted, ledger = open_bank(ctx)
    for r in accepted:
        r["n_shot"] = ctx["n_shot"]
        attach_samples(ctx, r)
    if progress is not None:
        progress(f"bank opened: generated {ledger['generated']}, accepted {ledger['accepted']}, quota filled {ledger['quota_filled']}")
    tol = float(values["added_context"]["tie_relative_tolerance"])
    bank_id = values["bank"]["bank_id"]

    pool = None
    saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
    try:
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(run["root"]), str(config_dir.resolve())))
        else:
            _WORKER.clear()
            _WORKER.update(ctx)
        truths = _execute(pool, _worker_job, [{"kind": "truth", "index": r["index"], "mask": r["mask"], "R": r["R"], "candidates": r["candidates"]} for r in accepted], "truth points", progress)
        for r in accepted:
            r["Q_o"], r["Q_o_tied"] = argmax_with_ties({q: truths[("truth", r["index"])]["candidates"][q]["smin_rel"] for q in r["candidates"]}, r["candidates"], tol)
        wave1 = []
        for r in accepted:
            for v in dict.fromkeys(["support", "control", f"add:{r['Q_f']}", f"add:{r['Q_o']}"]):
                tables, n_shots = variant_tables(r, v)
                wave1.append({"kind": "fit", "index": r["index"], "variant": v, "tables": tables, "n_shots": n_shots})
        fits = _execute(pool, _worker_job, wave1, "fits (A0, C, A_f, A_o)", progress)
        wave2 = []
        for r in accepted:
            V_hat = _branch_vectors(fits[(r["index"], "support")]["fit"])
            r["s_Q_fit"] = {q: mac.orbit_invariant_sensitivity(V_hat, q, r["R"]) for q in r["candidates"]}
            r["Q_s"], r["Q_s_tied"] = argmax_with_ties(r["s_Q_fit"], r["candidates"], tol)
            if (r["index"], f"add:{r['Q_s']}") not in fits:
                tables, n_shots = variant_tables(r, f"add:{r['Q_s']}")
                wave2.append({"kind": "fit", "index": r["index"], "variant": f"add:{r['Q_s']}", "tables": tables, "n_shots": n_shots})
            wave2.append({"kind": "cert", "index": r["index"], "table": fits[(r["index"], "support")]["table"], "support_query_hash": f"{bank_id}:{values['deficit']}:{r['index']}"})
        second = _execute(pool, _worker_job, wave2, "fits (A_s) and A0 certificates", progress)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    certs = {k: v for k, v in second.items() if k[0] == "cert"}
    fits.update({k: v for k, v in second.items() if k[0] != "cert"})
    if any(x.get("instance_hash") != ctx["instance_hash"] for x in [*truths.values(), *fits.values(), *certs.values()]):
        raise RuntimeError("worker-rebuilt constructions differ from the parent constructions")
    rows = [record_row(ctx, r, truths[("truth", r["index"])], fits, certs.get(("cert", r["index"]))) for r in accepted]
    analysis = analyse(values, rows)
    artifact = {
        "artifact_type": "m1_added_measurement",
        "status": "development_only",
        "protocol_document": values["protocol_document"],
        "decision": analysis["decision"],
        "bank_ledger": ledger,
        "strata": analysis["strata"],
        "settings": {"bank": values["bank"], "added_context": values["added_context"], "control": values["control"], "evaluation": values["evaluation"], "endpoints": values["endpoints"],
                     "bootstrap_index_seed_rule": "default_rng(sha256('{seed}:{stratum}')[:16])", "frozen_radii_low": ctx["bodies"]["radii"], "rdn_version": RDN_VERSION, "screen_version": SCREEN_VERSION,
                     "checkpoint_hash": ctx["sbp"].checkpoint_hash, "flagged_mask_count": len(ctx["flagged"]), "candidate_pool": ctx["pool"]},
        "records": rows,
        "execution": {"workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial", "fits": len(fits), "certificates": len(certs),
                      "fit_seconds": _summ([f["elapsed_seconds"] for f in fits.values()]), "wall_seconds": time.perf_counter() - started_run},
        "claim_boundary": {"development_only": True, "sealed_bank_opened": False, "screen_v1_changed": False,
                           "exact_states_used_for": "oracle class (3a/3b), the oracle context Q_o, truth s_Q for rank stability, structural checks, errors, coverage, the constraint registry and the energy-pinched RDN candidate"},
    }
    artifact = m1._jsonable(artifact)
    post = import_closure_snapshot(run["root"], extra_files=run["extra_files"])
    artifact["declaration_sha256"] = file_sha256(Path(run["folder"]) / DECLARATION)
    artifact["config_sha256"] = run["config_sha"]
    artifact["source_snapshot_hash_declared"] = declaration["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(declaration["source_snapshot"], post)
    (Path(run["folder"]) / ARTIFACT).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
