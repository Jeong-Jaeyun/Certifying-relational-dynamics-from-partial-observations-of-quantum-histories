"""P3 S3 calibration freeze: one fresh calibration draw for the S3 confirmatory analysis.

User decisions of 2026-11-21 (``documents/P3-d1-v7-run.md`` §7):

* S3 freezes a **fresh** calibration draw, not one of the four D1-v7 draws already seen;
* the support-only screen ``support_identifiability_screen_v1`` is frozen as it is, and the residual
  support non-identifiability inside the screened-in population is a known limitation of the
  coverage claim.

The design is the one the predeclared D1-v7 rule selected
(``config/p3_statistical_protocol_v2.yaml`` ``rdn_primary_claim.selected_design``): per stratum, the
first 160 screened-in blocks of the bank in ascending block index, and a split-conformal radius per
context class (ILGC alpha, identity whitening). The estimator and the screen settings must equal
the D1-v7 settings, and the bank must use the same generative specification as the D1-v7 banks.

Nothing else is run: no evaluation bank is built or opened and no certificate is computed. The S3
preregistration cites the frozen radii through the artifact hash and ``calibration_hash``.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from bi_qmn.core.config import load_p1_config
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import full_p1_context_list, p1_effect_map
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairEstimator
from bi_qmn.p3.p3_ilgc_facing import class_dimensions, conformal_radius, load_ilgc_targets
from bi_qmn.p3.p3_support_screen import SCREEN_VERSION, SupportIdentifiabilityScreen, first_screened_in
from bi_qmn.p3.runs.p3_d1_v6_development import TRUE_MASK_THRESHOLD, _execute, _json_default, _record_metrics, _summ, _table_dict, _true_branch_vectors
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint
from bi_qmn.p3.runs.p3_model_c_development_v5 import build_compact_dev_records, compact_target_index_manifest


CONFIG_FILE = "p3_s3_calibration.yaml"
RUNNER = "runners/p3/run_p3_s3_calibration.py"
DECLARATION = "p3_s3_calibration_declaration.json"
ARTIFACT = "p3_s3_calibration.json"
STATUS = "s3_calibration_freeze"
ROLE = "s3_calibration"
CLASSES = ("diag", "coh")
GENERATIVE_KEYS = ("dirichlet_concentration", "minimum_mode_probability", "phase_range")
WIDE_TAIL_DIAG_SCORE = 0.15   # report-only, the selection threshold of the D1-v7 wide-tail diagnostic
_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Configuration, declaration, guard
# ---------------------------------------------------------------------------


def _load(config_dir: Path) -> tuple[dict[str, Any], str]:
    path = config_dir / CONFIG_FILE
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if values.get("status") != STATUS or float(values["coherent_core_gamma"]) != 1.0:
        raise ValueError("the S3 calibration freeze is restricted to its own status and gamma=1")
    source = values["design_source"]
    dev = yaml.safe_load((config_dir / source["development_config"]).read_text(encoding="utf-8"))
    if values["estimator"] != dev["estimator"]:
        raise ValueError("estimator settings differ from the development config that selected the design")
    if values["screen"] != dev["screen"] or values["screen"]["method"] != SCREEN_VERSION or not values["screen"]["pauli_commutant"]:
        raise ValueError(f"the screen must be the development {SCREEN_VERSION} with the commutant check")
    if list(values["deficits"]) != list(dev["development_banks"]["deficits"]):
        raise ValueError("deficits differ from the development config")
    bank = values["calibration_bank"]
    reference = dev["development_banks"]["evaluation"]
    if any(bank[k] != reference[k] for k in GENERATIVE_KEYS):
        raise ValueError("the calibration bank must use the development generative specification")
    stats = yaml.safe_load((config_dir / source["statistical_protocol"]).read_text(encoding="utf-8"))["rdn_primary_claim"]
    design = stats["selected_design"]
    cal = values["calibration"]
    if int(cal["blocks_per_stratum"]) != int(design["calibration_blocks_per_stratum"]):
        raise ValueError("blocks per stratum differ from the selected S3 design")
    if design["calibration_draw"] != "fresh_draw":
        raise ValueError("the selected S3 design does not name a fresh calibration draw")
    screen = design["support_screen"]
    if screen["version"] != values["screen"]["method"] or float(screen["disconnected_branch_null_tail_epsilon"]) != float(values["screen"]["disconnected_branch_null_tail_epsilon"]):
        raise ValueError("the screen differs from the selected S3 design")
    if list(stats["strata"]) != list(values["deficits"]):
        raise ValueError("deficits differ from the S3 strata")
    if cal["order"] != "block_index_ascending" or cal["population"] != "first_n_screened_in_blocks_per_stratum":
        raise ValueError("only the first-n-screened-in, ascending-block-index calibration is implemented")
    if int(bank["block_count"]) < int(cal["blocks_per_stratum"]):
        raise ValueError("the bank is smaller than the calibration size")
    return values, file_sha256(path)


def _extra_files(config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    source = values["design_source"]
    names = [CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", str(values["ilgc_targets"]["operational_config"]),
             str(source["development_config"]), str(source["statistical_protocol"])]
    return [str(config_dir / n) for n in names] + [str(values["protocol_document"]), RUNNER]


def declare_p3_s3_calibration(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
    target = folder / DECLARATION
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    p1 = load_p1_config(config_dir)
    bank = dict(values["calibration_bank"])
    reserved = values["reserved_future_confirmatory_namespace"]["seed_range"]
    for key in ("seed", "observation_seed"):
        if int(reserved[0]) <= int(bank[key]) <= int(reserved[1]):
            raise ValueError("the calibration bank intrudes on the reserved confirmatory namespace")
    fps = sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, bank))
    known = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked = {}
    for path, spec in harness.declared_bank_specs(config_dir, exclude=CONFIG_FILE):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    overlaps = {name: sorted(set(fps) & f) for name, f in known.items() if set(fps) & f}
    if overlaps or len(set(fps)) != len(fps):
        raise ValueError(f"the calibration bank is not disjoint: {sorted(overlaps)}")
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values))
    declaration = {
        "artifact_type": "p3_s3_calibration_declaration",
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha,
        "protocol_document": str(values["protocol_document"]),
        "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
        "design_source_sha256": {name: file_sha256(config_dir / str(values["design_source"][name])) for name in ("development_config", "statistical_protocol")},
        "estimator": values["estimator"],
        "screen": values["screen"],
        "calibration": values["calibration"],
        "ilgc_targets": load_ilgc_targets(config_dir, values["ilgc_targets"]["operational_config"]),
        "calibration_bank": {**bank, "construction_fingerprint_hash": hashlib.sha256("".join(fps).encode("ascii")).hexdigest()[:16], "observation_tables_opened": False},
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
        raise FileExistsError("the calibration artifact already exists and is immutable")
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
# Context
# ---------------------------------------------------------------------------


def _prepare_light(config_dir: Path) -> dict[str, Any]:
    values, _ = _load(config_dir)
    p1 = load_p1_config(config_dir)
    instances = seeded_construction_bank_p1(p1.generator, dict(values["calibration_bank"]))
    contexts = full_p1_context_list(p1.measurement)
    dims = p1.generator["dimensions"]
    K, d_S = int(dims["K"]), int(dims["d_S"])
    effects_by_context = {cid: {o: np.asarray(op.data, dtype=complex) for o, op in row.items()} for cid, row in p1_effect_map(contexts, K=K).items()}
    cfg = values["estimator"]
    sbp = SharedBranchPairEstimator(
        p1.generator["active_modes"], K, d_S, effects_by_context,
        starts=int(cfg["starts"]), seed=int(cfg["seed"]), probability_floor=float(cfg["probability_floor"]), likelihood_floor=float(cfg["likelihood_floor"]),
        near_optimal_nats=float(cfg["near_optimal_nats"]), max_iterations=int(cfg["max_iterations"]), pair_observability_threshold=float(cfg["pair_observability_threshold"]),
    )
    return {
        "values": values,
        "p1": p1,
        "K": K,
        "instances": instances,
        "instance_hash": hashlib.sha256("".join(physical_fingerprint(i) for i in instances).encode("ascii")).hexdigest()[:16],
        "contexts": contexts,
        "context_by_id": {c.context_id: c for c in contexts},
        "sbp": sbp,
        "screen": SupportIdentifiabilityScreen(p1.generator["active_modes"], K, epsilon=float(values["screen"]["disconnected_branch_null_tail_epsilon"])),
    }


def _prepare(config_dir: Path) -> dict[str, Any]:
    """Parent context: the bank records, the screen result of every record, and the oracle class."""
    ctx = _prepare_light(config_dir)
    values, p1 = ctx["values"], ctx["p1"]
    bank = values["calibration_bank"]
    deficits = list(values["deficits"])
    records = build_compact_dev_records(ctx["instances"], p1.measurement, role=ROLE, bank_id=bank["bank_id"], observation_seed=int(bank["observation_seed"]), deficits=deficits)
    fw = ctx["sbp"].forward
    screens: dict[int, dict] = {}
    oracle: dict[int, dict] = {}
    for i, r in enumerate(records):
        support = r.support.table.probabilities
        screens[i] = ctx["screen"].evaluate(support, float(r.support.table.metadata["n_shots"]))
        mask = fw.pair_observability(_true_branch_vectors(ctx["instances"][r.instance_index]), [c for c in support if c.startswith("D,")], TRUE_MASK_THRESHOLD)
        oracle[i] = {"true_mask_class": mask["mask_class"], "true_unidentifiable": bool(screens[i]["pauli_commutant"]) or mask["mask_class"] == "branch_disconnected"}
    order = {d: [i for i, r in sorted(enumerate(records), key=lambda item: item[1].instance_index) if r.deficit == d] for d in deficits}
    return {**ctx, "records": records, "screens": screens, "oracle": oracle, "block_order": order, "deficits": deficits}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _fit_job(job: dict[str, Any]) -> dict[str, Any]:
    ctx = _WORKER
    started = time.perf_counter()
    support = MeasurementTable(probabilities=job["support"], contexts=[ctx["context_by_id"][c] for c in sorted(job["support"])], metadata={"n_shots": float(job["n_shots"])}, source_density=None)
    prediction = ctx["sbp"].predict_table(support, list(ctx["contexts"]))
    return {"key": tuple(job["key"]), "table": _table_dict(prediction), "fit": prediction.metadata["shared_branch_pair_fit"], "elapsed_seconds": time.perf_counter() - started, "instance_hash": ctx["instance_hash"]}


def _worker_init(root: str, config_dir: str) -> None:
    os.chdir(root)
    _WORKER.clear()
    _WORKER.update(_prepare_light(Path(config_dir)))


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def calibration_hash(bank: Mapping[str, Any], estimator: Mapping[str, Any], screen: Mapping[str, Any], alpha: Mapping[str, float],
                     radii: Mapping[str, Mapping[str, float]], selected_blocks: Mapping[str, list[int]]) -> str:
    """Hash of everything an S3 body depends on; radii enter through their exact float repr."""
    payload = {
        "bank": {k: bank[k] for k in sorted(bank)},
        "estimator": dict(estimator),
        "screen": dict(screen),
        "alpha": {k: repr(float(v)) for k, v in alpha.items()},
        "radii": {d: {cls: repr(float(r)) for cls, r in by_class.items()} for d, by_class in radii.items()},
        "selected_blocks": {d: [int(b) for b in blocks] for d, blocks in selected_blocks.items()},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_p3_s3_calibration(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started_run = time.perf_counter()
    config_dir = Path(config_dir)
    run = _guard(config_dir, output_dir)
    values, declaration = run["values"], run["declaration"]
    ctx = _prepare(config_dir)
    records, screens, deficits = ctx["records"], ctx["screens"], ctx["deficits"]
    targets = declaration["ilgc_targets"]
    strata, alpha, ceilings = targets["strata"], targets["alpha"], targets["normalized_radius_ceiling"]
    dims = class_dimensions(ctx["contexts"])
    size = int(values["calibration"]["blocks_per_stratum"])
    if progress is not None:
        progress("records built; screened out: " + ", ".join(f"{d}={sum(not screens[i]['passes'] for i in ctx['block_order'][d])}" for d in deficits))

    selection = {d: first_screened_in([screens[i]["passes"] for i in ctx["block_order"][d]], size) for d in deficits}
    fit_keys = [ctx["block_order"][d][p] for d in deficits if selection[d]["available"] for p in selection[d]["positions"]]

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
        jobs = [{"key": (i,), "support": {c: dict(row) for c, row in records[i].support.table.probabilities.items()}, "n_shots": float(records[i].support.table.metadata["n_shots"])} for i in fit_keys]
        results = _execute(pool, _fit_job, jobs, "shared_branch_pair fits", progress)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if any(r["instance_hash"] != ctx["instance_hash"] for r in results.values()):
        raise RuntimeError("worker-rebuilt instances differ from the parent instances")

    sbp = ctx["sbp"]
    metrics: dict[int, dict] = {}
    for i in fit_keys:
        r = records[i]
        instance = ctx["instances"][r.instance_index]
        result = results[(i,)]
        m = _record_metrics(r, instance, result["table"], ctx)
        ids, A, counts = sbp._support_arrays(r.support.table.probabilities, float(r.support.table.metadata["n_shots"]))
        x = sbp.forward.vector(_true_branch_vectors(instance))
        q = np.einsum("i,oij,j->o", x.conj(), A, x).real
        m.update({
            "index": i,
            "fit_minus_true_nll": result["fit"]["objective_nats"] - (-float(counts @ np.log(q + sbp.likelihood_floor))),
            "starts_within_near_optimal": result["fit"]["starts_within_near_optimal"],
            "fit_seconds": result["elapsed_seconds"],
            "screen_pair_power": screens[i]["pair_power"],
        })
        metrics[i] = m

    calibration: dict[str, Any] = {}
    for d in deficits:
        sel = selection[d]
        order = ctx["block_order"][d]
        base = {"stratum": strata[d], "available": sel["available"], "screened_out_before_last_selected": sel["screened_out_before"], "scanned_through": sel["scanned_through"]}
        if not sel["available"]:
            calibration[d] = {**base, "within_ceiling": False}
            continue
        rows = [metrics[order[p]] for p in sel["positions"]]
        classes = {}
        for cls in CLASSES:
            scores = [m[f"{cls}_score"] for m in rows]
            rad = conformal_radius(scores, float(alpha[cls]))
            ranked = np.argsort(scores, kind="stable")
            normalized = rad["radius"] / math.sqrt(dims[cls])
            ceiling = float(ceilings[strata[d]][cls])
            classes[cls] = {
                **rad, "dimension": dims[cls], "normalized_radius": normalized, "anti_vacuity_ceiling": ceiling, "within_ceiling": bool(normalized <= ceiling),
                "radius_block": int(rows[int(ranked[rad["order_index"] - 1])]["block"]),
                "largest_scores": [{"block": int(rows[int(j)]["block"]), "score": float(scores[int(j)]), "unitary": rows[int(j)]["unitary"]} for j in ranked[::-1][:5]],
            }
        calibration[d] = {
            **base,
            "radius": {cls: classes[cls]["radius"] for cls in CLASSES},
            "classes": classes,
            "within_ceiling": all(classes[cls]["within_ceiling"] for cls in CLASSES),
            "selected_blocks": [int(m["block"]) for m in rows],
            "report_only": {
                "wide_tail_records": [{"block": int(m["block"]), "diag_score": m["diag_score"], "unitary": m["unitary"]} for m in rows if m["diag_score"] > WIDE_TAIL_DIAG_SCORE],
                "unobserved_D_complex_error_rms": _summ([m["coef_error_rms"] for m in rows]),
                "true_mask_class_counts": dict(Counter(m["true_mask_class"] for m in rows)),
            },
        }

    frozen = all(calibration[d]["available"] and calibration[d]["within_ceiling"] for d in deficits)
    if frozen:
        radii = {d: calibration[d]["radius"] for d in deficits}
        freeze = {
            "frozen": True,
            "radii": radii,
            "calibration_hash": calibration_hash(values["calibration_bank"], values["estimator"], values["screen"], alpha, radii, {d: calibration[d]["selected_blocks"] for d in deficits}),
        }
    else:
        reasons = [f"{d}:fewer_screened_in_blocks_than_required" for d in deficits if not calibration[d]["available"]]
        reasons += [f"{d}:anti_vacuity_ceiling" for d in deficits if calibration[d]["available"] and not calibration[d]["within_ceiling"]]
        freeze = {"frozen": False, "reasons": reasons, "outcome": "no_freeze_user_decides"}

    all_metrics = list(metrics.values())
    artifact = {
        "artifact_type": "p3_s3_calibration",
        "status": STATUS,
        "screen_version": SCREEN_VERSION,
        "estimator": values["estimator"],
        "screen": values["screen"],
        "checkpoint_hash": sbp.checkpoint_hash,
        "target_index_manifest": compact_target_index_manifest(records),
        "execution": {
            "workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial", "parallelism_changes_results": False,
            "fits": len(jobs), "wall_seconds": time.perf_counter() - started_run,
        },
        "freeze": freeze,
        "calibration": calibration,
        "screen_audit": {
            "records": len(records),
            "screen_vs_oracle": dict(Counter(f"{records[i].deficit}:{'unidentifiable' if ctx['oracle'][i]['true_unidentifiable'] else 'identifiable'}->{'out' if not s['passes'] else 'in'}" for i, s in screens.items())),
            "screened_out": [{"block": int(records[i].instance_index), "deficit": records[i].deficit, "reasons": s["reasons"], "pauli_commutant": s["pauli_commutant"], "true_mask_class": ctx["oracle"][i]["true_mask_class"]} for i, s in sorted(screens.items()) if not s["passes"]],
        },
        "adequacy": {
            "fit_minus_true_nll": _summ([m["fit_minus_true_nll"] for m in all_metrics]),
            "records_with_fit_worse_than_truth_by_1_nat": int(sum(m["fit_minus_true_nll"] > 1.0 for m in all_metrics)),
            "starts_within_near_optimal": dict(sorted(Counter(int(m["starts_within_near_optimal"]) for m in all_metrics).items())),
            "fit_seconds": _summ([m["fit_seconds"] for m in all_metrics]),
        },
        "record_metrics": {str(i): m for i, m in sorted(metrics.items())},
        "claim_boundary": {
            "calibration_only": True,
            "evaluation_bank_built": False,
            "certificates_run": False,
            "reserved_namespace_touched": False,
            "exact_tables_used_for": "calibration scores (as in every ILGC body calibration) and the report-only oracle screen audit",
            "screen_inputs": "support context list, support D tables, shot count",
            "estimator_inputs": "support table, K, active modes, d_S, measurement effects",
        },
    }
    if int(workers) > 1:
        artifact["execution"]["note"] = "elapsed times are wall-clock per job and are excluded from parallel/serial comparisons"
    post = import_closure_snapshot(run["root"], extra_files=run["extra_files"])
    artifact["declaration_sha256"] = file_sha256(Path(run["folder"]) / DECLARATION)
    artifact["config_sha256"] = run["config_sha"]
    artifact["source_snapshot_hash_declared"] = declaration["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(declaration["source_snapshot"], post)
    (Path(run["folder"]) / ARTIFACT).write_text(json.dumps(artifact, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return artifact
