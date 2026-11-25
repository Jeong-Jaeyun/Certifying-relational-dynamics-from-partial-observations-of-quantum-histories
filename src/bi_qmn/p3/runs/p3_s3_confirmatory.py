"""P3 S3 confirmatory run: the F_RDN relational-dynamics rate claim on the sealed evaluation bank.

Preregistered on 2026-11-21 (user decision). The declaration freezes the design read off the predeclared
D1-v7 rule, the frozen S3 body calibration (cited by SHA-256 and calibration hash), and the statistical
protocol v2 at a recorded SHA-256. The claim:

    Under the declared generative distribution, the RDN certification rate exceeds 0.80 in both the
    medium and the low stratum.

A block succeeds iff it is screened in, its ILGC BI existence claim is ``primal_verified`` and its RDN
claim is ``certified``. Every evaluation block stays in its stratum's denominator; a screened-out block
abstains and counts as a failure. Each stratum uses a one-sided exact binomial test at alpha 0.05
(160 blocks, pass count 137); both must pass (intersection-union, claim level 0.05). One final analysis.

Certificates follow D1-v7 for its primary set: BI with the constraint registry and the 80-digit
verifier; rdn_v1 with the energy-pinched refutation candidates, refined with the optimal classical
state when unresolved. The sealed bank is built and screened only by the run, after the declaration.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from bi_qmn.core import identifiability as ident
from bi_qmn.core.config import load_p1_config
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.ilgc.certificate import _compute_existence_claim  # existence claim only: no selector, no atlas
from bi_qmn.ilgc.model import calibration_bodies_from_radii, measurement_model_from_p1, p1_constraint_registry
from bi_qmn.ilgc.relational_dynamics import RDN_VERSION, adjudicate_relational_dynamics, energy_block_candidate, optimal_classical_candidate
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import full_p1_context_list, p1_effect_map
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairEstimator
from bi_qmn.p3.p3_ilgc_facing import load_ilgc_targets
from bi_qmn.p3.p3_support_screen import SCREEN_VERSION, SupportIdentifiabilityScreen
from bi_qmn.p3.runs.p3_d1_v6_development import TRUE_MASK_THRESHOLD, _execute, _json_default, _keys, _record_metrics, _summ, _table_dict, _true_branch_vectors
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint
from bi_qmn.p3.runs.p3_model_c_development_v5 import build_compact_dev_records, compact_target_index_manifest
from bi_qmn.stats.rdn_rate_test import critical_count, decide_rate, intersection_union, plan


CONFIG_FILE = "p3_s3_confirmatory.yaml"
RUNNER = "runners/p3/run_p3_s3_confirmatory.py"
DECLARATION = "p3_s3_confirmatory_declaration.json"
ARTIFACT = "p3_s3_confirmatory.json"
EVALUATION = "evaluation"
CLASSES = ("diag", "coh")
GENERATIVE_KEYS = ("dirichlet_concentration", "minimum_mode_probability", "phase_range")
CLAIM_KEYS = ("family", "statement", "unit", "block_success", "denominator", "strata", "p0", "test", "one_sided_alpha_per_stratum", "combination", "family_p_value")
DESIGN_KEYS = ("blocks_per_stratum", "pass_count", "planning_rate")
_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Configuration, bodies, declaration, guard
# ---------------------------------------------------------------------------


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load(config_dir: Path) -> tuple[dict[str, Any], str]:
    path = config_dir / CONFIG_FILE
    values = _yaml(path)
    if values.get("status") != "preregistered_confirmatory" or float(values["coherent_core_gamma"]) != 1.0:
        raise ValueError("S3 is a preregistered gamma=1 confirmatory run")
    src = values["design_source"]
    dev = _yaml(config_dir / src["development_config"])
    if values["estimator"] != dev["estimator"]:
        raise ValueError("estimator settings differ from the development config that set the S3 design")
    if values["screen"] != dev["screen"] or values["screen"]["method"] != SCREEN_VERSION or not values["screen"]["pauli_commutant"]:
        raise ValueError(f"the screen must be the frozen {SCREEN_VERSION} with the commutant check")
    dev_cert = dev["reporting"]["certificate"]
    cert = values["certificate"]
    if (cert["existence"], cert["precision_dps"], cert["proposal_backends"]) != (dev_cert["existence"], dev_cert["precision_dps"], dev_cert["proposal_backends"]):
        raise ValueError("certificate settings differ from the D1-v7 existence route")
    if (cert["rdn_version"], cert["rdn_refutation_candidates"]) != (dev["reporting"]["rdn_version"], dev["reporting"]["rdn_refutation_candidates"]) or cert["rdn_version"] != RDN_VERSION:
        raise ValueError("RDN settings differ from D1-v7")
    if cert["refine_unresolved_with_optimal_classical_state"] is not True:
        raise ValueError("S3 has one primary body set; D1-v7 refines unresolved RDN with the optimal classical state there")
    stat_path = config_dir / src["statistical_protocol"]
    if file_sha256(stat_path) != src["statistical_protocol_sha256"]:
        raise RuntimeError("the statistical protocol differs from the SHA-256 frozen by this preregistration")
    stat = _yaml(stat_path)["rdn_primary_claim"]
    claim = values["primary_claim"]
    for key in CLAIM_KEYS:
        if claim[key] != stat[key]:
            raise ValueError(f"primary_claim.{key} differs from the statistical protocol")
    design = stat["selected_design"]
    for key in DESIGN_KEYS:
        if claim[key] != design[key]:
            raise ValueError(f"primary_claim.{key} differs from the selected design")
    if claim["screened_out_block"] != design["screened_out_block"] or not design["support_screen"]["frozen_for_s3"] or design["calibration_draw"] != "fresh_draw":
        raise ValueError("the selected design (screen, abstention, fresh calibration) is not the one decided")
    if design["support_screen"]["disconnected_branch_null_tail_epsilon"] != values["screen"]["disconnected_branch_null_tail_epsilon"]:
        raise ValueError("the screen differs from the frozen S3 screen")
    frozen = design["frozen_calibration"]
    if (frozen["calibration_hash"], frozen["artifact"]) != (values["bodies"]["calibration_hash"], values["bodies"]["source_artifact"]):
        raise ValueError("the bodies are not the frozen S3 calibration named by the statistical protocol")
    n, k = int(claim["blocks_per_stratum"]), int(claim["pass_count"])
    if critical_count(n, float(claim["p0"]), float(claim["one_sided_alpha_per_stratum"])) != k:
        raise ValueError("the pass count is not the exact critical count")
    if claim["analysis"] != "single_final_analysis" or claim["strata"] != values["deficits"]:
        raise ValueError("S3 is one final analysis over the declared strata")
    scope = values["claim_scope"]
    if scope["confirmatory_families"] != ["F_RDN"] or scope["family_alpha"] != {"F_RDN": float(claim["one_sided_alpha_per_stratum"])}:
        raise ValueError("this S3 confirms F_RDN only, at its declared alpha")
    bank = values["evaluation_bank"]
    if int(bank["block_count"]) != n:
        raise ValueError("the evaluation bank must have blocks_per_stratum construction blocks")
    reference = dev["development_banks"]["evaluation"]
    if any(bank[key] != reference[key] for key in GENERATIVE_KEYS):
        raise ValueError("the evaluation bank must use the development generative specification")
    lo, hi = values["reserved_confirmatory_namespace"]["seed_range"]
    if not all(int(lo) <= int(bank[key]) <= int(hi) for key in ("seed", "observation_seed")):
        raise ValueError("the sealed evaluation bank must lie in the reserved confirmatory namespace")
    return values, file_sha256(path)


def _bodies(root: Path, values: Mapping[str, Any], config_dir: Path) -> dict[str, Any]:
    spec = values["bodies"]
    path = root / str(spec["source_artifact"])
    if file_sha256(path) != spec["source_artifact_sha256"]:
        raise RuntimeError("the frozen calibration artifact differs from its declared SHA-256")
    freeze = json.loads(path.read_text(encoding="utf-8"))["freeze"]
    if not freeze.get("frozen") or freeze["calibration_hash"] != spec["calibration_hash"]:
        raise RuntimeError("the body source is not the declared frozen calibration")
    radii = {d: {cls: float(freeze["radii"][d][cls]) for cls in CLASSES} for d in values["deficits"]}
    stated = _yaml(config_dir / values["design_source"]["statistical_protocol"])["rdn_primary_claim"]["selected_design"]["frozen_calibration"]["radii"]
    if radii != {d: {cls: float(stated[d][cls]) for cls in CLASSES} for d in values["deficits"]}:
        raise RuntimeError("the frozen radii differ from the radii recorded in the statistical protocol")
    return {"radii": radii, "calibration_hash": spec["calibration_hash"], "source_artifact": str(spec["source_artifact"]), "source_artifact_sha256": spec["source_artifact_sha256"]}


def _extra_files(config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    src = values["design_source"]
    names = [CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", str(values["ilgc_targets"]["operational_config"]), str(src["development_config"]), str(src["statistical_protocol"]), str(src["calibration_config"])]
    return [str(config_dir / n) for n in dict.fromkeys(names)] + [str(values["bodies"]["source_artifact"]), str(values["protocol_document"]), RUNNER]


def declare_p3_s3_confirmatory(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
    target = folder / DECLARATION
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    bodies = _bodies(root, values, config_dir)
    p1 = load_p1_config(config_dir)
    bank = values["evaluation_bank"]
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
        raise ValueError(f"the S3 evaluation bank is not disjoint: {sorted(overlaps)}")
    lo, hi = values["reserved_confirmatory_namespace"]["seed_range"]
    intruders = sorted(path for path, spec in harness.declared_bank_specs(config_dir, exclude=CONFIG_FILE) if any(k in spec and int(lo) <= int(spec[k]) <= int(hi) for k in ("seed", "observation_seed")))
    if intruders:
        raise ValueError(f"another declared bank uses the reserved confirmatory namespace: {intruders}")
    claim = values["primary_claim"]
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values))
    declaration = {
        "artifact_type": "p3_s3_confirmatory_declaration",
        "protocol_version": values["protocol_version"],
        "status": values["status"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha,
        "protocol_document": str(values["protocol_document"]),
        "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
        "statistical_protocol": {"path": str(values["design_source"]["statistical_protocol"]), "sha256": values["design_source"]["statistical_protocol_sha256"], "frozen_by_this_declaration": True},
        "primary_claim": claim,
        "claim_scope": values["claim_scope"],
        "planned_power": plan(int(claim["blocks_per_stratum"]), {d: float(claim["planning_rate"]) for d in claim["strata"]}, p0=float(claim["p0"]), alpha=float(claim["one_sided_alpha_per_stratum"])),
        "estimator": values["estimator"],
        "screen": values["screen"],
        "certificate": values["certificate"],
        "bodies": bodies,
        "ilgc_targets": load_ilgc_targets(config_dir, values["ilgc_targets"]["operational_config"]),
        "evaluation_bank": {**bank, "construction_fingerprint_hash": hashlib.sha256("".join(fingerprints).encode("ascii")).hexdigest()[:16], "observation_tables_opened": False},
        "deficits": values["deficits"],
        "independence": values["independence"],
        "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps, "reserved_namespace_other_users": intruders},
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
        raise FileExistsError("the S3 artifact already exists and is immutable: one final analysis")
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
# Context and worker jobs
# ---------------------------------------------------------------------------


def _prepare_light(config_dir: Path) -> dict[str, Any]:
    values, _ = _load(config_dir)
    p1 = load_p1_config(config_dir)
    instances = seeded_construction_bank_p1(p1.generator, values["evaluation_bank"])
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
    targets = load_ilgc_targets(config_dir, values["ilgc_targets"]["operational_config"])
    return {
        "values": values,
        "p1": p1,
        "K": K, "d_C": d_C, "d_S": d_S,
        "theta_con": float(p1.generator["thresholds"]["theta_con"]),
        "instances": instances,
        "instance_hash": hashlib.sha256("".join(physical_fingerprint(i) for i in instances).encode("ascii")).hexdigest()[:16],
        "contexts": contexts,
        "context_by_id": {c.context_id: c for c in contexts},
        "effects": {(cid, o): M for cid, row in effects_by_context.items() for o, M in row.items()},
        "sbp": sbp,
        "screen": SupportIdentifiabilityScreen(p1.generator["active_modes"], K, epsilon=float(values["screen"]["disconnected_branch_null_tail_epsilon"])),
        "bodies": _bodies(Path.cwd(), values, config_dir),
        "strata": {d: targets["strata"][d] for d in values["deficits"]},
    }


def _open_sealed_bank(ctx: dict[str, Any]) -> None:
    """Build and screen the sealed evaluation bank. Called only by the run, after the guard."""
    bank = ctx["values"]["evaluation_bank"]
    rows = build_compact_dev_records(ctx["instances"], ctx["p1"].measurement, role="s3_evaluation", bank_id=bank["bank_id"], observation_seed=int(bank["observation_seed"]), deficits=list(ctx["values"]["deficits"]))
    fw = ctx["sbp"].forward
    ctx["records"], ctx["screens"], ctx["oracle"] = rows, {}, {}
    for i, r in enumerate(rows):
        support = r.support.table.probabilities
        ctx["screens"][i] = ctx["screen"].evaluate(support, float(r.support.table.metadata["n_shots"]))
        mask = fw.pair_observability(_true_branch_vectors(ctx["instances"][r.instance_index]), [c for c in support if c.startswith("D,")], TRUE_MASK_THRESHOLD)
        ctx["oracle"][i] = {"true_mask_class": mask["mask_class"], "true_unidentifiable": bool(ctx["screens"][i]["pauli_commutant"]) or mask["mask_class"] == "branch_disconnected"}
    ctx["block_order"] = {d: [i for i, r in sorted(enumerate(rows), key=lambda item: item[1].instance_index) if r.deficit == d] for d in ctx["values"]["deficits"]}


def _fit_job(job: dict[str, Any]) -> dict[str, Any]:
    ctx = _WORKER
    started = time.perf_counter()
    support = MeasurementTable(probabilities=job["support"], contexts=[ctx["context_by_id"][c] for c in sorted(job["support"])], metadata={"n_shots": float(job["n_shots"])}, source_density=None)
    prediction = ctx["sbp"].predict_table(support, list(ctx["contexts"]))
    return {"key": (int(job["index"]),), "table": _table_dict(prediction), "fit": prediction.metadata["shared_branch_pair_fit"], "elapsed_seconds": time.perf_counter() - started, "instance_hash": ctx["instance_hash"]}


def _cert_job(job: dict[str, Any]) -> dict[str, Any]:
    """BI existence and RDN on the frozen bodies of the record's stratum (the D1-v7 primary-set route)."""
    ctx = _WORKER
    started = time.perf_counter()
    K, d_C, d_S, contexts = ctx["K"], ctx["d_C"], ctx["d_S"], ctx["contexts"]
    cert = ctx["values"]["certificate"]
    dps = int(cert["precision_dps"])
    instance = ctx["instances"][int(job["block"])]
    table = job["table"]
    deficit = job["deficit"]
    stratum = ctx["strata"][deficit]
    radii = ctx["bodies"]["radii"][deficit]
    calibration_hash = f"p3_s3_calibration:{ctx['bodies']['calibration_hash']}:{deficit}"
    model = measurement_model_from_p1(MeasurementTable(probabilities=table, contexts=list(contexts), metadata={"eta": 0.0}, source_density=None), K, stratum=stratum,
                                      estimator_checkpoint_hash=job["checkpoint_hash"], support_query_hash=job["support_query_hash"])
    bodies = calibration_bodies_from_radii(model, radii, calibration_hash=calibration_hash, stratum=stratum)
    registry = p1_constraint_registry(instance, ctx["theta_con"], calibration_hash=calibration_hash)
    claim, _ = _compute_existence_claim(model, bodies, registry, precision_dps=dps, proposal_backends=tuple(cert["proposal_backends"]))
    bi = {"status": claim.status, "verified_upper_bound": claim.verified_upper_bound, "verified_lower_bound": claim.verified_lower_bound, "reason": claim.decision_reason}
    rho_bi = np.asarray(claim.primal_evidence.density.data, dtype=complex) if claim.status == "primal_verified" else None
    bi_seconds = time.perf_counter() - started
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
                rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], candidates=[*candidates, ("optimal_classical_quantum_state", proposal["blocks"])], constraints=constraints, d_S=d_S) | {"optimal_classical_proposal_objective": proposal["objective"]}
    return {
        "key": (int(job["index"]),), "bi": bi,
        "rdn": {"status": rdn["status"], "kappa": rdn["kappa"], "tau_flat_distance": rdn["tau_flat_distance"], "refuting_candidate": rdn.get("refuting_candidate"), "optimal_classical_proposal_objective": rdn.get("optimal_classical_proposal_objective")},
        "seconds": {"bi": bi_seconds, "total": time.perf_counter() - started}, "instance_hash": ctx["instance_hash"],
    }


def _worker_init(root: str, config_dir: str) -> None:
    os.chdir(root)
    _WORKER.clear()
    _WORKER.update(_prepare_light(Path(config_dir)))


# ---------------------------------------------------------------------------
# Decision (pure)
# ---------------------------------------------------------------------------


def block_success(row: Mapping[str, Any]) -> bool:
    """Screened in, BI primal_verified and RDN certified. A screened-out block is a failure."""
    return bool(row["screen_passes"]) and row["bi"]["status"] == "primal_verified" and row["rdn"]["status"] == "certified"


def decide(rows_by_stratum: Mapping[str, list[Mapping[str, Any]]], claim: Mapping[str, Any]) -> dict[str, Any]:
    n = int(claim["blocks_per_stratum"])
    decisions = {}
    for stratum in claim["strata"]:
        rows = rows_by_stratum[stratum]
        if len(rows) != n:
            raise RuntimeError(f"stratum {stratum} has {len(rows)} blocks, not the declared {n}")
        decisions[stratum] = decide_rate(sum(block_success(r) for r in rows), n, p0=float(claim["p0"]), alpha=float(claim["one_sided_alpha_per_stratum"]))
        if decisions[stratum]["critical_count"] != int(claim["pass_count"]):
            raise RuntimeError("the exact critical count differs from the declared pass count")
    return intersection_union(decisions)


def _stratum_report(rows: list[Mapping[str, Any]], radii: Mapping[str, float]) -> dict[str, Any]:
    screened_in = [r for r in rows if r["screen_passes"]]
    return {
        "blocks": len(rows),
        "screened_out": len(rows) - len(screened_in),
        "bi_status_counts": dict(Counter(r["bi"]["status"] for r in screened_in)),
        "rdn_status_counts": dict(Counter(r["rdn"]["status"] for r in screened_in)),
        "failures": [{"block": r["block"], "screen_reasons": r.get("screen_reasons"), "bi": r["bi"]["status"], "rdn": r["rdn"]["status"]} for r in rows if not block_success(r)],
        "kappa": _summ([r["rdn"]["kappa"] for r in screened_in if r["rdn"]["kappa"] is not None]),
        "frozen_radii": dict(radii),
        "coverage": {cls: sum(r["covered"][cls] for r in screened_in) / len(screened_in) if screened_in else None for cls in CLASSES},
        "unobserved_D_coefficient_error_rms": _summ([r["coef_error_rms"] for r in screened_in if r["coef_error_rms"] is not None]),
        "screen_audit": {
            "identifiable_screened_in": sum(r["screen_passes"] and not r["true_unidentifiable"] for r in rows),
            "unidentifiable_screened_out": sum(not r["screen_passes"] and r["true_unidentifiable"] for r in rows),
            "discordant": sum(r["screen_passes"] == r["true_unidentifiable"] for r in rows),
        },
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_p3_s3_confirmatory(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started_run = time.perf_counter()
    config_dir = Path(config_dir)
    run = _guard(config_dir, output_dir)
    values, declaration = run["values"], run["declaration"]
    ctx = _prepare_light(config_dir)
    _open_sealed_bank(ctx)
    records, screens = ctx["records"], ctx["screens"]
    if progress is not None:
        progress(f"sealed bank opened: {len(records)} records; screened out {sum(not s['passes'] for s in screens.values())}")
    bank_id = values["evaluation_bank"]["bank_id"]
    keys = [i for d in values["deficits"] for i in ctx["block_order"][d] if screens[i]["passes"]]

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
        fit_jobs = [{"index": i, "support": {c: dict(row) for c, row in records[i].support.table.probabilities.items()}, "n_shots": float(records[i].support.table.metadata["n_shots"])} for i in keys]
        fits = _execute(pool, _fit_job, fit_jobs, "fits", progress)
        cert_jobs = [{"index": i, "block": int(records[i].instance_index), "deficit": records[i].deficit, "table": fits[(i,)]["table"], "checkpoint_hash": ctx["sbp"].checkpoint_hash,
                      "support_query_hash": f"{bank_id}:{records[i].deficit}:{records[i].instance_index}"} for i in keys]
        certs = _execute(pool, _cert_job, cert_jobs, "certificates", progress)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if any(r["instance_hash"] != ctx["instance_hash"] for r in [*fits.values(), *certs.values()]):
        raise RuntimeError("worker-rebuilt instances differ from the parent instances")

    rows_by_stratum: dict[str, list[dict]] = {}
    for d in values["deficits"]:
        radii = ctx["bodies"]["radii"][d]
        rows = []
        for i in ctx["block_order"][d]:
            r = records[i]
            s = screens[i]
            base = {"index": i, "block": int(r.instance_index), "deficit": d, "screen_passes": bool(s["passes"]), "screen_reasons": s["reasons"], **ctx["oracle"][i]}
            if not s["passes"]:
                rows.append(base | {"bi": {"status": "abstained"}, "rdn": {"status": "abstained", "kappa": None}, "success": False})
                continue
            m = _record_metrics(r, ctx["instances"][r.instance_index], fits[(i,)]["table"], ctx)
            row = base | {
                "unitary": m["unitary"], "diag_score": m["diag_score"], "coh_score": m["coh_score"], "covered": {cls: bool(m[f"{cls}_score"] <= radii[cls]) for cls in CLASSES},
                "coef_error_rms": m["coef_error_rms"], "fit_starts_within_near_optimal": fits[(i,)]["fit"]["starts_within_near_optimal"], "fit_seconds": fits[(i,)]["elapsed_seconds"],
                "bi": certs[(i,)]["bi"], "rdn": certs[(i,)]["rdn"], "seconds": certs[(i,)]["seconds"],
            }
            row["success"] = block_success(row)
            rows.append(row)
        rows_by_stratum[d] = rows
    claim = values["primary_claim"]
    decision = decide(rows_by_stratum, claim)
    artifact = {
        "artifact_type": "p3_s3_confirmatory",
        "protocol_version": values["protocol_version"],
        "family": claim["family"],
        "claim": claim["statement"],
        "status": "f_rdn_confirmed" if decision["claim_passes"] else "f_rdn_not_confirmed",
        "decision": decision,
        "claim_scope": values["claim_scope"],
        "rdn_version": RDN_VERSION,
        "screen_version": SCREEN_VERSION,
        "estimator": values["estimator"],
        "checkpoint_hash": ctx["sbp"].checkpoint_hash,
        "bodies": ctx["bodies"],
        "statistical_protocol": declaration["statistical_protocol"],
        "target_index_manifest": compact_target_index_manifest(records),
        "report_only": {d: _stratum_report(rows_by_stratum[d], ctx["bodies"]["radii"][d]) for d in values["deficits"]},
        "records": rows_by_stratum,
        "execution": {
            "workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial", "parallelism_changes_results": False,
            "certificates": len(keys), "certificate_seconds": _summ([c["seconds"]["total"] for c in certs.values()]), "wall_seconds": time.perf_counter() - started_run,
        },
        "claim_boundary": {
            "single_final_analysis": True, "sealed_bank_opened_once": True, "calibration_recomputed": False,
            "families_tested": ["F_RDN"], "atlas_evaluated": False, "selector_evaluated": False,
            "exact_states_used_for": "the report-only coverage scores and oracle screen audit, the constraint registry, and the energy-pinched RDN refutation candidate",
        },
    }
    post = import_closure_snapshot(run["root"], extra_files=run["extra_files"])
    artifact["declaration_sha256"] = file_sha256(Path(run["folder"]) / DECLARATION)
    artifact["config_sha256"] = run["config_sha"]
    artifact["source_snapshot_hash_declared"] = declaration["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(declaration["source_snapshot"], post)
    (Path(run["folder"]) / ARTIFACT).write_text(json.dumps(artifact, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return artifact
