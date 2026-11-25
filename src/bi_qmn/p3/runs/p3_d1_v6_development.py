"""P3 D1-v6 development run 1: shared branch-pair estimator vs Model A (report-only).

Major revision 2026-11-20 (formulation patch v3). D1-RDN found that Model A loses about half of the
``tau``-modulation of unobserved D contexts, while the declared-family information ceiling loses
none. D1-v6 tests an estimator that does not know the declared family but uses the fixed clock
frequencies and shares branch-pair coefficients across contexts
(``bi_qmn.p3.models.p3_shared_branch_pair``).

Three estimators:

* ``model_a``: the R4 certification estimator;
* ``model_a_spectrum_projected``: Model A projected onto the declared clock spectrum context by
  context. This isolates the frequency constraint itself;
* ``shared_branch_pair``: the D1-v6 candidate, which adds transfer to unobserved contexts.

For each estimator it measures (i) the complex Fourier-coefficient error of unobserved D contexts
per branch pair, with amplitude error reported separately, and (ii) the block success
``BI primal_verified AND RDN certified`` over all evaluation blocks. Radii are self-calibrated by
the ILGC protocol on four independent calibration draws, plus their 192-block pool. Results are
broken down by the evaluation-side support mask class (all pairs observed / a pair hidden but
connected / a branch disconnected). Every quantity is report-only and declared in
``documents/P3-d1-v6-protocol.md`` before the run.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from bi_qmn.core import identifiability as ident
from bi_qmn.core.config import load_p1_config
from bi_qmn.core.noise import outcome_key
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.ilgc.certificate import _compute_existence_claim  # existence claim only: the selector is not part of this run
from bi_qmn.ilgc.model import calibration_bodies_from_radii, measurement_model_from_p1, p1_constraint_registry
from bi_qmn.ilgc.relational_dynamics import RDN_VERSION, adjudicate_relational_dynamics, energy_block_candidate, optimal_classical_candidate
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import contexts_from_p1_measurement_config, full_p1_context_list, p1_effect_map
from bi_qmn.p1.p1_estimator import P1KernelFreeTableEstimator
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.models.p3_shared_branch_pair import DeclaredSpectrumProjection, SharedBranchPairEstimator
from bi_qmn.p3.p3_ilgc_facing import class_dimensions, conformal_radius, load_ilgc_targets, p1_record_from_p3, predict_full_table
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint
from bi_qmn.p3.runs.p3_model_c_development_v5 import build_compact_dev_records, compact_target_index_manifest
from bi_qmn.stats.rdn_rate_test import clopper_pearson_lower


CONFIG_FILE = "p3_d1_v6.yaml"
RUNNER = "runners/p3/run_p3_d1_v6_development.py"
DECLARATION = "p3_d1_v6_declaration.json"
ARTIFACT = "p3_d1_v6_development.json"
ESTIMATORS = ("model_a", "model_a_spectrum_projected", "shared_branch_pair")
CLASSES = ("diag", "coh")
POOLED = "pooled"
TRUE_MASK_THRESHOLD = 1.0e-9
_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Configuration, declaration, guard
# ---------------------------------------------------------------------------


def _load(config_dir: Path) -> tuple[dict[str, Any], str]:
    path = config_dir / CONFIG_FILE
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if values.get("status") != "development_only" or float(values["coherent_core_gamma"]) != 1.0:
        raise ValueError("D1-v6 is restricted to development-only gamma=1 work")
    draws = list(values["reporting"]["calibration_draws"])
    if values["reporting"]["primary_calibration"] != draws[0]:
        raise ValueError("the primary calibration must be the first declared draw")
    return values, file_sha256(path)


def _banks(values: Mapping[str, Any]) -> dict[str, dict]:
    return {role: dict(spec) for role, spec in values["development_banks"].items() if isinstance(spec, dict) and harness.BANK_KEYS <= set(spec)}


def _extra_files(config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    names = [CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", str(values["ilgc_targets"]["operational_config"])]
    return [str(config_dir / n) for n in names] + [str(values["protocol_document"]), RUNNER]


def declare_p3_d1_v6(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
    target = folder / DECLARATION
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    p1 = load_p1_config(config_dir)
    banks = _banks(values)
    dev = {role: sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)) for role, spec in banks.items()}
    known = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked = {}
    for path, bank in harness.declared_bank_specs(config_dir, exclude=CONFIG_FILE):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, bank)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    union = set().union(*map(set, dev.values()))
    overlaps = {name: sorted(union & fps) for name, fps in known.items() if union & fps}
    roles = list(dev)
    internal = any(set(dev[a]) & set(dev[b]) for i, a in enumerate(roles) for b in roles[i + 1 :])
    if overlaps or internal:
        raise ValueError(f"development banks are not disjoint: {sorted(overlaps)} internal={internal}")
    reserved = values["reserved_future_confirmatory_namespace"]["seed_range"]
    for spec in banks.values():
        for key in ("seed", "observation_seed"):
            if int(reserved[0]) <= int(spec[key]) <= int(reserved[1]):
                raise ValueError("development seeds intrude on the reserved confirmatory namespace")
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values))
    declaration = {
        "artifact_type": "p3_d1_v6_declaration",
        "status": "development_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha,
        "protocol_document": str(values["protocol_document"]),
        "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
        "estimators": values["estimators"],
        "reporting": values["reporting"],
        "ilgc_targets": load_ilgc_targets(config_dir, values["ilgc_targets"]["operational_config"]),
        "development_banks": {role: {**banks[role], "construction_fingerprint_hash": hashlib.sha256("".join(fps).encode("ascii")).hexdigest()[:16], "observation_tables_opened": False} for role, fps in dev.items()},
        "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps, "development_banks_mutually_disjoint": True},
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
        raise FileExistsError("development artifact already exists and is immutable")
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
# Context (rebuilt identically in every worker)
# ---------------------------------------------------------------------------


def _prepare(config_dir: Path) -> dict[str, Any]:
    values, _ = _load(config_dir)
    p1 = load_p1_config(config_dir)
    banks = _banks(values)
    deficits = values["development_banks"]
    instances = {role: seeded_construction_bank_p1(p1.generator, spec) for role, spec in banks.items()}
    records = {
        role: build_compact_dev_records(instances[role], p1.measurement, role=f"v6_{role}", bank_id=banks[role]["bank_id"], observation_seed=int(banks[role]["observation_seed"]),
                                        deficits=list(deficits["train_deficits"] if role == "train" else deficits["eval_deficits"]))
        for role in banks
    }
    contexts = full_p1_context_list(p1.measurement)
    dims = p1.generator["dimensions"]
    K, d_C, d_S = int(dims["K"]), int(dims["d_C"]), int(dims["d_S"])
    effect_map = p1_effect_map(contexts, K=K)
    effects_by_context = {cid: {o: np.asarray(op.data, dtype=complex) for o, op in row.items()} for cid, row in effect_map.items()}
    cfg = values["estimators"]["shared_branch_pair"]
    sbp = SharedBranchPairEstimator(
        p1.generator["active_modes"], K, d_S, effects_by_context,
        starts=int(cfg["starts"]), seed=int(cfg["seed"]), probability_floor=float(cfg["probability_floor"]), likelihood_floor=float(cfg["likelihood_floor"]),
        near_optimal_nats=float(cfg["near_optimal_nats"]), max_iterations=int(cfg["max_iterations"]), pair_observability_threshold=float(cfg["pair_observability_threshold"]),
    )
    return {
        "values": values,
        "p1": p1,
        "K": K, "d_C": d_C, "d_S": d_S,
        "theta_con": float(p1.generator["thresholds"]["theta_con"]),
        "instances": instances,
        "records": records,
        "contexts": contexts,
        "input_contexts": contexts_from_p1_measurement_config(p1.measurement)["input"],
        "effects": {(cid, o): M for cid, row in effects_by_context.items() for o, M in row.items()},
        "sbp": sbp,
        "record_hashes": {role: compact_target_index_manifest(records[role])["target_index_hash"] for role in records},
    }


def _table_dict(table) -> dict[str, dict[str, float]]:
    return {cid: {o: float(v) for o, v in row.items()} for cid, row in table.probabilities.items()}


def _keys(contexts, cls: str, K: int) -> list[tuple[str, str]]:
    return [(c.context_id, outcome_key(l, b)) for c in contexts if c.context_class == cls for l in range(K) for b in (0, 1)]


def _true_branch_vectors(instance) -> np.ndarray:
    U = instance.system_unitary
    U = np.asarray(getattr(U, "data", U), dtype=complex)
    c = np.asarray(instance.coefficients, dtype=complex)
    return U[:, : len(c)] * c[None, :]


# ---------------------------------------------------------------------------
# Worker jobs
# ---------------------------------------------------------------------------


def _sbp_job(job: tuple[str, int]) -> dict[str, Any]:
    role, index = job
    ctx = _WORKER
    record = ctx["records"][role][index]
    started = time.perf_counter()
    prediction = predict_full_table(ctx["sbp"], record.support.table, ctx["contexts"])
    return {"key": (role, index), "table": _table_dict(prediction), "fit": prediction.metadata["shared_branch_pair_fit"], "elapsed_seconds": time.perf_counter() - started, "record_hashes": ctx["record_hashes"]}


def _cert_job(job: dict[str, Any]) -> dict[str, Any]:
    ctx = _WORKER
    K, d_C, d_S, contexts = ctx["K"], ctx["d_C"], ctx["d_S"], ctx["contexts"]
    record = ctx["records"]["evaluation"][job["index"]]
    instance = ctx["instances"]["evaluation"][record.instance_index]
    reporting = ctx["values"]["reporting"]
    table = job["table"]
    radii = {cls: float(job["radii"][cls]) for cls in CLASSES}
    class_keys = {cls: _keys(contexts, cls, K) for cls in CLASSES}
    targets = {cls: ident.class_vector(table, contexts, cls, K) for cls in CLASSES}
    H = np.asarray(instance.H_tot.data, dtype=complex)
    G = H @ H / (instance.h_scale ** 2)
    constraints = [(G, None, ctx["theta_con"])]
    started = time.perf_counter()
    rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, d_S=d_S, precision_dps=int(reporting["certificate"]["precision_dps"]))
    certified = rdn["status"] == "certified"
    bi: dict[str, Any] = {"status": "not_run"}
    rho_bi = None
    if job["bi"] == "always" or (job["bi"] == "if_certified" and certified):
        measurement_table = MeasurementTable(probabilities=table, contexts=list(contexts), metadata={"eta": 0.0}, source_density=None)
        model = measurement_model_from_p1(measurement_table, K, stratum=job["stratum"], estimator_checkpoint_hash=job["checkpoint_hash"], support_query_hash=f"{record.metadata['bank_id']}:{record.deficit}:{record.instance_index}")
        calibration_hash = f"{job['estimator']}:{job['draw']}:{record.deficit}"
        bodies = calibration_bodies_from_radii(model, radii, calibration_hash=calibration_hash, stratum=job["stratum"])
        registry = p1_constraint_registry(instance, ctx["theta_con"], calibration_hash=calibration_hash)
        claim, _ = _compute_existence_claim(model, bodies, registry, precision_dps=int(reporting["certificate"]["precision_dps"]), proposal_backends=tuple(reporting["certificate"]["proposal_backends"]))
        bi = {"status": claim.status, "verified_upper_bound": claim.verified_upper_bound, "verified_lower_bound": claim.verified_lower_bound, "reason": claim.decision_reason}
        if claim.status == "primal_verified":
            rho_bi = np.asarray(claim.primal_evidence.density.data, dtype=complex)
    if not certified:
        candidates = []
        if rho_bi is not None:
            candidates.append(("energy_pinched_bi_primal_state", energy_block_candidate(rho_bi, d_C, d_S)))
        candidates.append(("energy_pinched_exact_state", energy_block_candidate(np.asarray(instance.density.data), d_C, d_S)))
        rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], candidates=candidates, constraints=constraints, d_S=d_S)
        if rdn["status"] == "unresolved" and job["refine"]:
            proposal = optimal_classical_candidate(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], constraints=constraints, d_C=d_C, d_S=d_S)
            if proposal["blocks"] is not None:
                rdn = adjudicate_relational_dynamics(class_keys=class_keys, targets=targets, radii=radii, effects=ctx["effects"], candidates=[*candidates, ("optimal_classical_quantum_state", proposal["blocks"])], constraints=constraints, d_S=d_S) | {"optimal_classical_proposal_objective": proposal["objective"]}
    return {
        "key": (job["estimator"], job["draw"], job["index"]),
        "bi": bi,
        "rdn_status": rdn["status"],
        "kappa": rdn["kappa"],
        "tau_flat_distance": rdn["tau_flat_distance"],
        "refuting_candidate": rdn.get("refuting_candidate"),
        "optimal_classical_proposal_objective": rdn.get("optimal_classical_proposal_objective"),
        "elapsed_seconds": time.perf_counter() - started,
        "record_hashes": ctx["record_hashes"],
    }


def _worker_init(root: str, config_dir: str) -> None:
    os.chdir(root)
    _WORKER.clear()
    _WORKER.update(_prepare(Path(config_dir)))


def _execute(pool, function: Callable[[Any], dict], jobs: list, label: str, progress: Callable[[str], None] | None) -> dict:
    out: dict = {}
    started = time.perf_counter()
    step = max(1, len(jobs) // 20)
    if pool is None:
        iterator = (function(job) for job in jobs)
    else:
        futures = [pool.submit(function, job) for job in jobs]
        iterator = (future.result() for future in as_completed(futures))
    for count, result in enumerate(iterator, start=1):
        out[tuple(result["key"])] = result
        if progress is not None and (count % step == 0 or count == len(jobs)):
            progress(f"{label}: {count}/{len(jobs)} ({time.perf_counter() - started:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _pair_coefficients(table: Mapping[str, Mapping[str, float]], cid: str, K: int, bins: list[int]) -> np.ndarray:
    q = np.asarray([[float(table[cid][outcome_key(l, b)]) for b in (0, 1)] for l in range(K)])
    F = np.fft.fft(K * (q[:, 0] - q[:, 1])) / K
    return F[bins]


def _record_metrics(record, instance, table: Mapping[str, Mapping[str, float]], ctx: Mapping[str, Any]) -> dict[str, Any]:
    contexts, K = ctx["contexts"], ctx["K"]
    forward = ctx["sbp"].forward
    exact = record.exact_table.probabilities
    observed = set(record.support.table.probabilities)
    out: dict[str, Any] = {"deficit": record.deficit, "block": int(record.instance_index), "unitary": instance.system_unitary_label}
    for cls in CLASSES:
        q = ident.class_vector(table, contexts, cls, K)
        e = ident.class_vector(exact, contexts, cls, K)
        unobs = [c for c in contexts if c.context_class == cls and c.context_id not in observed]
        out[f"{cls}_score"] = float(np.linalg.norm(e - q))
        out[f"{cls}_score_unobs"] = float(np.linalg.norm(ident.class_vector(exact, unobs, cls, K) - ident.class_vector(table, unobs, cls, K))) if unobs else 0.0
    mod_hat = ident.tau_modulation(table, contexts, K)
    mod_true = ident.tau_modulation(exact, contexts, K)
    unobs_d = [c.context_id for c in contexts if c.clock == "D" and c.context_id not in observed]
    out.update({
        "tau_modulation_hat": mod_hat["diag_class"],
        "tau_modulation_true": mod_true["diag_class"],
        "unobserved_D_contexts": len(unobs_d),
        "unobserved_D_modulation_hat": math.sqrt(sum(mod_hat["per_context"][c] ** 2 for c in unobs_d)),
        "unobserved_D_modulation_true": math.sqrt(sum(mod_true["per_context"][c] ** 2 for c in unobs_d)),
    })
    # Branch-pair Fourier coefficients of unobserved D contexts (complex and amplitude-only errors).
    labels = [f"{forward.modes[j]}-{forward.modes[k]}" for j, k in forward.pairs]
    bins = [forward.pair_bin(p) for p in forward.pairs]
    mask = forward.pair_observability(_true_branch_vectors(instance), [c for c in observed if c.startswith("D,")], TRUE_MASK_THRESHOLD)
    errors, amplitude, hidden_err, observed_err, true_mag = [], [], [], [], []
    for cid in unobs_d:
        c_hat, c_true = _pair_coefficients(table, cid, K, bins), _pair_coefficients(exact, cid, K, bins)
        for label, a, b in zip(labels, c_hat, c_true, strict=True):
            err = float(abs(a - b))
            errors.append(err)
            amplitude.append(float(abs(abs(a) - abs(b))))
            true_mag.append(float(abs(b)))
            (hidden_err if label in mask["hidden_pairs"] else observed_err).append(err)
    rms = lambda v: float(math.sqrt(np.mean(np.square(v)))) if v else None  # noqa: E731
    out.update({
        "true_mask_class": mask["mask_class"],
        "true_hidden_pairs": mask["hidden_pairs"],
        "coef_error_rms": rms(errors),
        "coef_error_max": float(max(errors)) if errors else None,
        "coef_amplitude_error_rms": rms(amplitude),
        "coef_amplitude_error_max": float(max(amplitude)) if amplitude else None,
        "coef_true_magnitude_rms": rms(true_mag),
        "coef_error_rms_observed_pairs": rms(observed_err),
        "coef_error_rms_hidden_pairs": rms(hidden_err),
    })
    for channel, batch in (("shot", record.query_shot), ("comp", record.query_comp), ("op", record.query_op)):
        total, count = 0.0, 0.0
        for cid, counts in batch.counts.items():
            for o, n in counts.items():
                total -= float(n) * math.log(max(float(table[cid][o]), 1.0e-15))
                count += float(n)
        out[f"nll_{channel}"] = total / count
    return out


def _radius_cell(rows: list[dict], cls: str, alpha: float) -> dict[str, Any]:
    scores = [r[f"{cls}_score"] for r in rows]
    cell = conformal_radius(scores, alpha)
    if scores:
        order = np.argsort(scores)[::-1]
        top = rows[int(order[0])]
        cell.update({
            "largest_score": float(scores[int(order[0])]),
            "largest_score_true_mask_class": top["true_mask_class"],
            "largest_score_unitary": top["unitary"],
            "second_largest_score": float(scores[int(order[1])]) if len(scores) > 1 else None,
        })
    return cell


def _rate(successes: int, n: int) -> dict[str, Any]:
    return {"successes": int(successes), "n": int(n), "rate": successes / n if n else None, "clopper_pearson_lower_95": clopper_pearson_lower(int(successes), int(n), 0.95) if n else None}


def _summ(values: list[float]) -> dict[str, float] | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)), "max": float(np.max(v)), "n": len(v)}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_p3_d1_v6_development(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started_run = time.perf_counter()
    config_dir = Path(config_dir)
    run = _guard(config_dir, output_dir)
    values, declaration = run["values"], run["declaration"]
    context = _prepare(config_dir)
    K, contexts = context["K"], context["contexts"]
    reporting = values["reporting"]
    draws = list(reporting["calibration_draws"])
    primary = str(reporting["primary_calibration"])
    radius_sets = draws + ([POOLED] if reporting.get("pooled_calibration") else [])
    roles = draws + ["evaluation"]
    eval_deficits = list(values["development_banks"]["eval_deficits"])
    claim_strata = list(reporting["s2_screen_readout"]["strata"])

    # Model A and its spectrum projection: fitted on the train bank, predicted in the parent process.
    cfg_a = values["estimators"]["model_a"]
    model_a = P1KernelFreeTableEstimator(context["input_contexts"], neighbor_count=int(cfg_a["neighbor_count"]), distance_power=float(cfg_a["distance_power"]), probability_floor=float(cfg_a["probability_floor"])).fit_records([p1_record_from_p3(r) for r in context["records"]["train"]])
    projector = DeclaredSpectrumProjection(model_a, context["p1"].generator["active_modes"], K, probability_floor=float(values["estimators"]["model_a_spectrum_projected"]["probability_floor"]))
    tables: dict[str, dict[tuple[str, int], dict]] = {name: {} for name in ESTIMATORS}
    clipped: dict[tuple[str, int], int] = {}
    for role in roles:
        for i, record in enumerate(context["records"][role]):
            prediction = predict_full_table(model_a, record.support.table, contexts)
            tables["model_a"][(role, i)] = _table_dict(prediction)
            projected, n_clipped = projector.project(prediction)
            tables["model_a_spectrum_projected"][(role, i)] = _table_dict(projected)
            clipped[(role, i)] = n_clipped
    checkpoints = {"model_a": model_a.checkpoint_hash, "model_a_spectrum_projected": projector.checkpoint_hash, "shared_branch_pair": context["sbp"].checkpoint_hash}

    pool = None
    saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
    try:
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(run["root"]), str(config_dir.resolve())))
        else:
            _WORKER.clear()
            _WORKER.update(context)

        # Phase 1: shared branch-pair fits.
        fit_jobs = [(role, i) for role in roles for i in range(len(context["records"][role]))]
        fit_results = _execute(pool, _sbp_job, fit_jobs, "shared_branch_pair fits", progress)
        if any(r["record_hashes"] != context["record_hashes"] for r in fit_results.values()):
            raise RuntimeError("worker-rebuilt development records differ from the parent records")
        fits = {}
        for key, r in fit_results.items():
            tables["shared_branch_pair"][key] = r["table"]
            fits[key] = r["fit"]

        # Phase 2 (parent): metrics and radii.
        targets = declaration["ilgc_targets"]
        strata, alpha, ceilings = targets["strata"], targets["alpha"], targets["normalized_radius_ceiling"]
        dims = class_dimensions(contexts)
        metrics = {
            name: {role: [_record_metrics(r, context["instances"][role][r.instance_index], tables[name][(role, i)], context) for i, r in enumerate(context["records"][role])] for role in roles}
            for name in ESTIMATORS
        }
        for role in roles:  # optimization adequacy: fit NLL minus NLL at the true parameters (evaluation side)
            for i, record in enumerate(context["records"][role]):
                sbp = context["sbp"]
                ids, A, counts = sbp._support_arrays(record.support.table.probabilities, float(record.support.table.metadata["n_shots"]))
                x = sbp.forward.vector(_true_branch_vectors(context["instances"][role][record.instance_index]))
                q = np.einsum("i,oij,j->o", x.conj(), A, x).real
                true_nll = -float(counts @ np.log(q + sbp.likelihood_floor))
                fit = fits[(role, i)]
                metrics["shared_branch_pair"][role][i].update({
                    "fit_minus_true_nll": fit["objective_nats"] - true_nll,
                    "starts_within_near_optimal": fit["starts_within_near_optimal"],
                    "estimated_mask_class": fit["pair_observability"]["mask_class"],
                    "estimated_hidden_pairs": fit["pair_observability"]["hidden_pairs"],
                })
        for name in ("model_a_spectrum_projected",):
            for role in roles:
                for i in range(len(context["records"][role])):
                    metrics[name][role][i]["clipped_entries"] = clipped[(role, i)]
        radii: dict[str, dict[str, dict[str, dict]]] = {}
        for name in ESTIMATORS:
            radii[name] = {}
            for draw in radius_sets:
                pool_roles = draws if draw == POOLED else [draw]
                radii[name][draw] = {}
                for deficit in eval_deficits:
                    rows = [m for role in pool_roles for m in metrics[name][role] if m["deficit"] == deficit]
                    radii[name][draw][deficit] = {cls: {**_radius_cell(rows, cls, float(alpha[cls])), "stratum": strata[deficit]} for cls in CLASSES}
        missing = [(n, d, f, c) for n in ESTIMATORS for d in radius_sets for f in eval_deficits for c in CLASSES if radii[n][d][f][c]["radius"] is None]
        if missing:
            raise RuntimeError(f"too few calibration blocks for the declared conformal level: {missing[:4]}")

        # Phase 3: BI existence and RDN on the evaluation bank.
        cert_jobs = []
        for name in ESTIMATORS:
            for i, record in enumerate(context["records"]["evaluation"]):
                for draw in radius_sets:
                    if draw != primary and record.deficit not in claim_strata:
                        continue
                    cell = radii[name][draw][record.deficit]
                    cert_jobs.append({
                        "estimator": name, "draw": draw, "index": i, "stratum": strata[record.deficit], "checkpoint_hash": checkpoints[name],
                        "table": tables[name][("evaluation", i)], "radii": {cls: cell[cls]["radius"] for cls in CLASSES},
                        "bi": "always" if draw == primary else "if_certified", "refine": draw == primary,
                    })
        cert_jobs.sort(key=lambda job: (job["draw"] != primary, job["estimator"], job["index"]))
        cert_results = _execute(pool, _cert_job, cert_jobs, "certificates", progress)
        if any(r["record_hashes"] != context["record_hashes"] for r in cert_results.values()):
            raise RuntimeError("worker-rebuilt development records differ from the parent records")
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    # Summaries.
    eval_records = context["records"]["evaluation"]
    rows_by = {name: {draw: [] for draw in radius_sets} for name in ESTIMATORS}
    for (name, draw, i), r in sorted(cert_results.items()):
        m = metrics[name]["evaluation"][i]
        rows_by[name][draw].append({
            "index": i, "deficit": eval_records[i].deficit, "block": int(eval_records[i].instance_index), "true_mask_class": m["true_mask_class"], "unitary": m["unitary"],
            "bi_status": r["bi"]["status"], "bi_verified_upper_bound": r["bi"].get("verified_upper_bound"), "rdn_status": r["rdn_status"], "kappa": r["kappa"],
            "refuting_candidate": r["refuting_candidate"], "optimal_classical_proposal_objective": r["optimal_classical_proposal_objective"],
            "success": bool(r["bi"]["status"] == "primal_verified" and r["rdn_status"] == "certified"),
        })
    summary: dict[str, Any] = {}
    for name in ESTIMATORS:
        summary[name] = {}
        for deficit in eval_deficits:
            ev = [m for m in metrics[name]["evaluation"] if m["deficit"] == deficit]
            n_blocks = len(ev)
            cell: dict[str, Any] = {
                "stratum": strata[deficit],
                "coefficients": {
                    "complex_error_rms": _summ([m["coef_error_rms"] for m in ev]),
                    "complex_error_max": _summ([m["coef_error_max"] for m in ev]),
                    "amplitude_error_rms": _summ([m["coef_amplitude_error_rms"] for m in ev]),
                    "true_magnitude_rms": _summ([m["coef_true_magnitude_rms"] for m in ev]),
                    "complex_error_rms_observed_pairs": _summ([m["coef_error_rms_observed_pairs"] for m in ev]),
                    "complex_error_rms_hidden_pairs": _summ([m["coef_error_rms_hidden_pairs"] for m in ev]),
                    "by_true_mask_class": {cls: _summ([m["coef_error_rms"] for m in ev if m["true_mask_class"] == cls]) for cls in ("all_pairs_observed", "pair_hidden_connected", "branch_disconnected")},
                },
                "true_mask_class_counts": dict(Counter(m["true_mask_class"] for m in ev)),
                "unobserved_D_modulation_recovery": float(sum(m["unobserved_D_modulation_hat"] for m in ev) / max(1e-15, sum(m["unobserved_D_modulation_true"] for m in ev))),
                "mean_block_score": {cls: float(np.mean([m[f"{cls}_score"] for m in ev])) for cls in CLASSES},
                "nll_mean": {ch: float(np.mean([m[f"nll_{ch}"] for m in ev])) for ch in ("shot", "comp", "op")},
                "by_calibration": {},
            }
            for draw in radius_sets:
                rad = radii[name][draw][deficit]
                rows = [x for x in rows_by[name][draw] if x["deficit"] == deficit]
                entry: dict[str, Any] = {
                    "radius": {cls: rad[cls]["radius"] for cls in CLASSES},
                    "normalized_radius": {cls: rad[cls]["radius"] / math.sqrt(dims[cls]) for cls in CLASSES},
                    "anti_vacuity_ceiling": ceilings[strata[deficit]],
                    "radius_determining_record": {cls: {k: rad[cls].get(k) for k in ("largest_score", "largest_score_true_mask_class", "largest_score_unitary", "second_largest_score", "order_index", "n_blocks")} for cls in CLASSES},
                    "evaluation_coverage": {cls: float(np.mean([m[f"{cls}_score"] <= rad[cls]["radius"] for m in ev])) for cls in CLASSES},
                    "oracle_centre_certifiable": float(np.mean([m["tau_modulation_true"] > rad["diag"]["radius"] for m in ev])),
                }
                if rows:
                    entry.update({
                        "rdn": dict(Counter(x["rdn_status"] for x in rows)),
                        "kappa_min_median_max": [float(np.min([x["kappa"] for x in rows])), float(np.median([x["kappa"] for x in rows])), float(np.max([x["kappa"] for x in rows]))],
                        "bi": dict(Counter(x["bi_status"] for x in rows)),
                        "block_success": _rate(sum(x["success"] for x in rows), n_blocks),
                        "block_success_by_true_mask_class": {cls: _rate(sum(x["success"] for x in rows if x["true_mask_class"] == cls), sum(1 for x in rows if x["true_mask_class"] == cls)) for cls in ("all_pairs_observed", "pair_hidden_connected", "branch_disconnected")},
                    })
                cell["by_calibration"][draw] = entry
            summary[name][deficit] = cell

    def success_vector(name: str, draw: str, deficit: str) -> dict[int, bool]:
        return {x["index"]: x["success"] for x in rows_by[name][draw] if x["deficit"] == deficit}

    paired = {}
    for other in ("model_a", "model_a_spectrum_projected"):
        paired[f"shared_branch_pair_vs_{other}"] = {}
        for deficit in eval_deficits:
            s, o = success_vector("shared_branch_pair", primary, deficit), success_vector(other, primary, deficit)
            paired[f"shared_branch_pair_vs_{other}"][deficit] = {"both": sum(s[i] and o[i] for i in s), "shared_branch_pair_only": sum(s[i] and not o[i] for i in s), f"{other}_only": sum(o[i] and not s[i] for i in s), "neither": sum(not s[i] and not o[i] for i in s)}
    threshold = float(reporting["s2_screen_readout"]["observed_rate_at_least"])
    s2 = {name: {draw: {d: summary[name][d]["by_calibration"][draw]["block_success"]["rate"] for d in claim_strata} | {"screen_passes": all(summary[name][d]["by_calibration"][draw]["block_success"]["rate"] >= threshold for d in claim_strata)} for draw in radius_sets} for name in ESTIMATORS}
    planning = {draw: summary["shared_branch_pair"][reporting["planning_readout"]["stratum"]]["by_calibration"][draw]["block_success"] for draw in radius_sets}
    sbp_rows = [m for role in roles for m in metrics["shared_branch_pair"][role]]
    adequacy = {
        "fit_minus_true_nll": _summ([m["fit_minus_true_nll"] for m in sbp_rows]),
        "records_with_fit_worse_than_truth_by_1_nat": int(sum(m["fit_minus_true_nll"] > 1.0 for m in sbp_rows)),
        "records": len(sbp_rows),
        "estimated_vs_true_mask_class": dict(Counter(f"{m['true_mask_class']}->{m['estimated_mask_class']}" for m in sbp_rows)),
        "fit_seconds": _summ([r["elapsed_seconds"] for r in fit_results.values()]),
    }

    artifact = {
        "artifact_type": "p3_d1_v6_development",
        "status": "development_only",
        "rdn_version": RDN_VERSION,
        "estimators": values["estimators"],
        "checkpoint_hashes": checkpoints,
        "target_index_manifest": {role: compact_target_index_manifest(context["records"][role]) for role in context["records"]},
        "execution": {
            "workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial", "parallelism_changes_results": False,
            "certificate_jobs": len(cert_jobs), "bi_solves": sum(r["bi"]["status"] != "not_run" for r in cert_results.values()),
            "certificate_seconds": _summ([r["elapsed_seconds"] for r in cert_results.values()]),
            "wall_seconds": time.perf_counter() - started_run,
        },
        "radii": radii,
        "summary": summary,
        "readouts": {
            "s2_entry_screen_observed_rate": s2,
            "planning_low_stratum_shared_branch_pair": planning,
            "paired_block_success_primary_calibration": paired,
            "shared_branch_pair_optimization_adequacy": adequacy,
        },
        "record_metrics": {name: {role: metrics[name][role] for role in roles} for name in ESTIMATORS},
        "certificate_records": rows_by,
        "shared_branch_pair_fits": {f"{role}:{i}": {k: v for k, v in fit.items() if k != "branch_vectors_real_imag"} for (role, i), fit in sorted(fits.items())},
        "claim_boundary": {
            "development_only": True,
            "decision_rule": None,
            "s2_entry_screen_adjudicated": False,
            "selector_and_atlas_run": False,
            "quantum_existence_run": False,
            "confirmatory_bank_opened": False,
            "reserved_namespace_touched": False,
            "exact_tables_used_for_training": False,
            "estimator_inputs": "support table, K, active modes, d_S, measurement effects",
            "exact_states_used_for": "evaluation scores, mask classification, the energy-pinched refutation candidate, and the constraint registry (as in R4)",
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


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)}")
