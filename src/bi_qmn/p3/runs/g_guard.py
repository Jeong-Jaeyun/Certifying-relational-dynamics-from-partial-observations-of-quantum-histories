"""G guard: a support goodness-of-fit abstention guard for the certification route (development).

Protocol: documents/G-guard-protocol.md (frozen by the declaration). Two declared stages on two new
development banks:

1. ``calibrate``: the support misfit ``G`` of pure histories (320 screened-in records per stratum, S3 mask
   rule) gives the split-conformal threshold ``g_alpha(stratum)``, alpha = 0.05.
2. ``evaluate``: the frozen guard on independent controlled truths (pure, dephased nu = 0.1/0.25/0.5, readout
   eta = 0.02/0.05, static, registry-violating), 64 constructions x 2 strata. The full S3 certificates run
   where the guarded route would speak.

The route functions (truth families, the S3 certification route, the misfit) come from
``bi_qmn.p3.runs.e5_rdn_specificity``.
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
import yaml

from bi_qmn.core import identifiability as ident
from bi_qmn.core.config import load_p1_config
from bi_qmn.core.observations import sample_shot_batch
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.ilgc.relational_dynamics import tau_flat_distance
from bi_qmn.p1.model.p1_banks import support_contexts_for_deficit
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import p1_measurement_table
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.runs import e5_rdn_specificity as e5
from bi_qmn.p3.runs import m1_added_measurement as am
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _summ, _table_dict
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint


CONFIG_FILE = "g_guard.yaml"
RUNNER = "runners/p3/run_g_guard.py"
DECLARATION = "g_guard_declaration.json"
CALIBRATION = "g_guard_calibration.json"
EVALUATION = "g_guard_evaluation.json"
STAGES = ("calibration", "evaluation")
ALLOWED_FAMILIES = ("pure", "dephased_0.1", "dephased_0.25", "dephased_0.5", "readout_0.02", "readout_0.05", "tau_flat", "registry_violating")
CLASSES = ("diag", "coh")
_W: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:16], 16)


def _load(config_dir: Path) -> tuple[dict[str, Any], str]:
    path = Path(config_dir) / CONFIG_FILE
    values = _yaml(path)
    if values.get("status") != "development_only":
        raise ValueError("the G guard is a development protocol")
    alpha = float(values["statistic"]["alpha"])
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    if list(values["strata"]) != ["medium", "low"]:
        raise ValueError("the guard is calibrated and evaluated on the S3 strata")
    fams = list(values["evaluation"]["families"])
    if any(f not in ALLOWED_FAMILIES for f in fams) or "pure" not in fams or not set(values["evaluation"]["misspecified"]) <= set(fams):
        raise ValueError("unknown or missing evaluation families")
    lo, hi = values["reserved_future_confirmatory_namespace"]["seed_range"]
    seeds = [int(values[s]["bank"][k]) for s in STAGES for k in ("seed", "observation_seed", "haar_seed")]
    if len(set(seeds)) != len(seeds) or any(int(lo) <= s <= int(hi) for s in seeds):
        raise ValueError("guard seeds must be distinct and outside the reserved confirmatory namespace")
    if int(values["calibration"]["per_stratum"]) > int(values["calibration"]["bank"]["block_count"]):
        raise ValueError("the calibration cap must allow the declared number of records")
    return values, file_sha256(path)


def threshold_index(n: int, alpha: float) -> int:
    """Split-conformal order statistic: the k-th smallest of n scores, k = ceil((n + 1)(1 - alpha)) (k > n means no finite threshold)."""
    return int(math.ceil(round((n + 1) * (1.0 - float(alpha)), 9)))


def conformal_threshold(scores: Sequence[float], alpha: float) -> float:
    k = threshold_index(len(scores), alpha)
    return math.inf if k > len(scores) else float(np.sort(np.asarray(scores, dtype=float))[k - 1])


def _ctx(config_dir: Path, stage: str) -> dict[str, Any]:
    values, _ = _load(config_dir)
    ctx = e5.prepare(config_dir, dict(values[stage]["bank"]))
    ctx["guard_values"] = values
    return ctx


# ---------------------------------------------------------------------------
# One record
# ---------------------------------------------------------------------------


def family_truth(ctx: Mapping[str, Any], i: int, family: str) -> tuple[np.ndarray, float]:
    """The family's density operator and its readout rate (the estimator always assumes eta = 0)."""
    if family.startswith("readout_"):
        return e5.family_density(ctx, i, "pure"), float(family.split("_", 1)[1])
    return e5.family_density(ctx, i, family), 0.0


def record_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    i, deficit, family = int(job["index"]), str(job["deficit"]), str(job["family"])
    bank = ctx["bank"]
    inst = ctx["instances"][i]
    rho, eta = family_truth(ctx, i, family)
    meta = {"instance_index": i, "deficit": deficit, "family": family, "bank_id": bank["bank_id"]}
    effective = p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=inst.K, eta=eta, metadata=meta).copy_without_source()
    ideal = effective if eta == 0.0 else p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=inst.K, eta=0.0, metadata=meta).copy_without_source()
    ideal_tables = {c: dict(row) for c, row in ideal.probabilities.items()}
    effective_tables = {c: dict(row) for c, row in effective.probabilities.items()}
    n_shot = int(ctx["p1"].measurement["observation_deficit"][deficit]["n_shot"])
    support_contexts = support_contexts_for_deficit(ctx["p1"].measurement, deficit, seed_key=f"p3v2:{bank['bank_id']}:{deficit}:{i}")
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bank['bank_id']}:{deficit}:{i}"))       # common random numbers across families
    support = sample_shot_batch(effective, support_contexts, n_shots=n_shot, rng=rng, shot_start=(e5.DEFICITS.index(deficit) * 1000 + i) * 100_000_000, role="g_guard_support").table.probabilities
    support = {c: dict(row) for c, row in support.items()}
    screen = ctx["screen"].evaluate(support, float(n_shot))
    radii = ctx["bodies"]["radii"][deficit]
    true_dist = tau_flat_distance(ident.class_vector(ideal_tables, ctx["contexts"], "diag", ctx["K"]), ctx["diag_keys"])
    row: dict[str, Any] = {"index": i, "deficit": deficit, "family": family, "eta": eta, "observed": sorted(support),
                           "screen_passes": bool(screen["passes"]), "screen_reasons": list(screen["reasons"]),
                           "truth": {"kappa": true_dist / radii["diag"], "registered_energy": e5.registered_energy(ctx, i, rho)}}
    if row["screen_passes"]:
        prediction = ctx["sbp"].predict_table(MeasurementTable(probabilities=support, contexts=[ctx["context_by_id"][c] for c in sorted(support)], metadata={"n_shots": float(n_shot)}, source_density=None), list(ctx["contexts"]))
        table = _table_dict(prediction)
        G = e5.support_misfit(support, table, float(n_shot))["G"]
        row["G"] = G
        row["fit_kappa"] = tau_flat_distance(ident.class_vector(table, ctx["contexts"], "diag", ctx["K"]), ctx["diag_keys"]) / radii["diag"]
        cov = {}
        for name, ref in (("ideal", ideal_tables), ("effective", effective_tables)):
            s = {cls: float(np.linalg.norm(ident.class_vector(table, ctx["contexts"], cls, ctx["K"]) - ident.class_vector(ref, ctx["contexts"], cls, ctx["K"]))) for cls in CLASSES}
            cov[name] = {**{cls: bool(s[cls] <= radii[cls]) for cls in CLASSES}, "joint": all(s[cls] <= radii[cls] for cls in CLASSES), "scores": s}
        row["coverage"] = cov
        threshold = job.get("threshold")
        row["guard_passes"] = None if threshold is None else bool(G <= float(threshold))
        if job.get("certify") and row["guard_passes"]:
            row["certificate"] = e5.certify(ctx, i, deficit, table, rho, f"{bank['bank_id']}:{deficit}:{i}:{family}")
    row["seconds"] = time.perf_counter() - started
    return {"key": (i, deficit, family), "row": row}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _worker_init(root: str, config_dir: str, stage: str) -> None:
    os.chdir(root)
    _W.clear()
    _W.update(_ctx(Path(config_dir), stage))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    out = record_job(_W, job)
    out["instance_hash"] = _W["instance_hash"]
    return out


def _run_jobs(ctx: Mapping[str, Any], config_dir: Path, stage: str, jobs: list[dict], workers: int, label: str, progress) -> dict:
    pool = None
    saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
    try:
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(Path.cwd()), str(config_dir.resolve()), stage))
        else:
            _W.clear()
            _W.update(ctx)
        results = _execute(pool, _worker_job, jobs, label, progress)
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
    return results


# ---------------------------------------------------------------------------
# Declaration and guard
# ---------------------------------------------------------------------------


def _folder(values: Mapping[str, Any], output_dir: str | Path) -> Path:
    return Path(output_dir) / Path(str(values["output_root"])).name


def _extra_files(config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    names = [CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", "p1_ilgc_operational_r4.yaml", str(values["design_source"]["development_config"])]
    return [str(Path(config_dir) / n) for n in dict.fromkeys(names)] + [*values["design_source"]["dev_inputs"], str(values["protocol_document"]), RUNNER]


def declare_g_guard(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = _folder(values, output_dir)
    target = folder / DECLARATION
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    dev_inputs = am._dev_inputs(root, values)
    p1 = load_p1_config(config_dir)
    prints = {s: sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, values[s]["bank"])) for s in STAGES}
    if set(prints["calibration"]) & set(prints["evaluation"]):
        raise ValueError("the calibration and evaluation banks share constructions")
    known = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked = {}
    for path, spec in harness.declared_bank_specs(config_dir, exclude=CONFIG_FILE):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    overlaps = {f"{s}~{name}": sorted(set(prints[s]) & f) for s in STAGES for name, f in known.items() if set(prints[s]) & f}
    if overlaps:
        raise ValueError(f"the guard banks are not disjoint from declared banks: {sorted(overlaps)}")
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values))
    declaration = {
        "artifact_type": "g_guard_declaration", "status": values["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha, "protocol_document": str(values["protocol_document"]), "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
        "dev_inputs": dev_inputs, "statistic": values["statistic"], "endpoints": values["endpoints"],
        "banks": {s: {**values[s]["bank"], "construction_fingerprint_hash": hashlib.sha256("".join(prints[s]).encode("ascii")).hexdigest()[:16], "observation_tables_opened": False} for s in STAGES},
        "families": values["evaluation"]["families"],
        "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps, "calibration_evaluation_disjoint": True},
        "source_snapshot": snapshot, "source_snapshot_hash": snapshot_hash(snapshot),
    }
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return declaration


def _guard(config_dir: Path, output_dir: str | Path, artifact: str) -> dict[str, Any]:
    root = Path.cwd()
    values, config_sha = _load(config_dir)
    folder = _folder(values, output_dir)
    declaration = json.loads((folder / DECLARATION).read_text(encoding="utf-8"))
    if (folder / artifact).exists():
        raise FileExistsError(f"{artifact} already exists and is immutable")
    if declaration["config_sha256"] != config_sha:
        raise RuntimeError("config drifted after declaration")
    if declaration["protocol_document_sha256"] != file_sha256(root / str(values["protocol_document"])):
        raise RuntimeError("protocol document drifted after declaration")
    extra = [p for p in declaration["source_snapshot"] if not p.startswith("src/")]
    drift = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    if drift["changed"] or drift["missing"]:
        raise RuntimeError(f"source drift after declaration: {drift}")
    return {"root": root, "values": values, "config_sha": config_sha, "declaration": declaration, "folder": folder, "extra_files": extra}


def _finish(run: Mapping[str, Any], artifact: dict[str, Any], name: str) -> dict[str, Any]:
    artifact = m1._jsonable(artifact)
    post = import_closure_snapshot(run["root"], extra_files=run["extra_files"])
    artifact["declaration_sha256"] = file_sha256(Path(run["folder"]) / DECLARATION)
    artifact["config_sha256"] = run["config_sha"]
    artifact["source_snapshot_hash_declared"] = run["declaration"]["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(run["declaration"]["source_snapshot"], post)
    (Path(run["folder"]) / name).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


# ---------------------------------------------------------------------------
# Stage 1: calibration
# ---------------------------------------------------------------------------


def calibrate(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started = time.perf_counter()
    config_dir = Path(config_dir)
    run = _guard(config_dir, output_dir, CALIBRATION)
    values = run["values"]
    ctx = _ctx(config_dir, "calibration")
    cap, per = int(values["calibration"]["bank"]["block_count"]), int(values["calibration"]["per_stratum"])
    jobs = [{"index": i, "deficit": d, "family": "pure", "threshold": None, "certify": False} for d in values["strata"] for i in range(cap)]
    results = _run_jobs(ctx, config_dir, "calibration", jobs, workers, "calibration records", progress)
    alpha = float(values["statistic"]["alpha"])
    strata = {}
    for d in values["strata"]:
        rows = [results[(i, d, "pure")]["row"] for i in range(cap)]
        kept = [r for r in rows if r["screen_passes"]][:per]
        scores = [r["G"] for r in kept]
        strata[d] = {"kept": len(kept), "shortfall": max(0, per - len(kept)), "screened_out_before_last_kept": sum(not r["screen_passes"] for r in rows[: (kept[-1]["index"] + 1) if kept else 0]),
                     "k": threshold_index(len(kept), alpha), "threshold": conformal_threshold(scores, alpha),
                     "G_quantiles": {q: float(np.quantile(scores, float(q))) for q in ("0.0", "0.5", "0.9", "0.95", "1.0")} if scores else None,
                     "coverage_ideal_joint": sum(r["coverage"]["ideal"]["joint"] for r in kept), "kept_indices": [r["index"] for r in kept]}
    artifact = {"artifact_type": "g_guard_calibration", "status": "development_only", "alpha": alpha, "strata": strata,
                "thresholds": {d: strata[d]["threshold"] for d in values["strata"]},
                "records": [results[(i, d, "pure")]["row"] for d in values["strata"] for i in range(cap)],
                "execution": {"workers": int(workers), "records": len(jobs), "record_seconds": _summ([r["row"]["seconds"] for r in results.values()]), "wall_seconds": time.perf_counter() - started}}
    return _finish(run, artifact, CALIBRATION)


# ---------------------------------------------------------------------------
# Stage 2: evaluation
# ---------------------------------------------------------------------------


def clopper_pearson(k: int, n: int) -> tuple[float, float] | None:
    return e5.clopper_pearson(k, n) if n else None


def analyse_evaluation(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    alpha = float(values["statistic"]["alpha"])
    ep = values["endpoints"]
    fams = list(values["evaluation"]["families"])
    out: dict[str, Any] = {"by_family": {}}
    for fam in fams:
        block = {}
        for scope in ("medium", "low", "pooled"):
            sr = [r for r in rows if r["family"] == fam and (scope == "pooled" or r["deficit"] == scope)]
            inn = [r for r in sr if r["screen_passes"]]
            passed = [r for r in inn if r["guard_passes"]]
            abst = len(inn) - len(passed)
            certs = [r for r in passed if "certificate" in r]
            rdn_cert = [r for r in certs if r["certificate"]["rdn"]["status"] == "certified"]
            success = [r for r in rdn_cert if r["certificate"]["bi"]["status"] == "primal_verified"]
            block[scope] = {
                "n": len(sr), "screened_in": len(inn), "guard_abstentions": abst, "guard_abstention_rate": abst / len(inn) if inn else None,
                "guard_abstention_ci95": clopper_pearson(abst, len(inn)),
                "unguarded_distance_stage_certified": sum(r["fit_kappa"] > 1.0 for r in inn),
                "unguarded_coverage_ideal_joint": sum(r["coverage"]["ideal"]["joint"] for r in inn),
                "guarded_rdn_certified": len(rdn_cert), "guarded_s3_success": len(success),
                "bi_status_counts": e5._counts([r["certificate"]["bi"]["status"] for r in certs]),
                "rdn_status_counts": e5._counts([r["certificate"]["rdn"]["status"] for r in certs]),
                "coverage_ideal_joint_among_guarded_certified": sum(r["coverage"]["ideal"]["joint"] for r in rdn_cert),
                "coverage_rate_among_guarded_certified": (sum(r["coverage"]["ideal"]["joint"] for r in rdn_cert) / len(rdn_cert)) if rdn_cert else None,
                "coverage_ci95_among_guarded_certified": clopper_pearson(sum(r["coverage"]["ideal"]["joint"] for r in rdn_cert), len(rdn_cert)),
                "G": e5._q([r["G"] for r in inn]), "truth_kappa": e5._q([r["truth"]["kappa"] for r in sr]),
            }
        out["by_family"][fam] = block
    pure = out["by_family"]["pure"]["pooled"]
    g1_ci = pure["guard_abstention_ci95"]
    out["endpoints"] = {
        "G1_false_abstention_pure": {"abstentions": pure["guard_abstentions"], "screened_in": pure["screened_in"], "rate": pure["guard_abstention_rate"], "ci95": g1_ci,
                                     "consistent_with_alpha": bool(g1_ci is not None and g1_ci[0] <= alpha)},
        "G2_detection": {f: {k: out["by_family"][f]["pooled"][k] for k in ("guard_abstentions", "screened_in", "guard_abstention_rate", "guard_abstention_ci95")} for f in values["evaluation"]["misspecified"]},
        "G3_validity": {},
        "G4_static": {"guarded_rdn_certified": out["by_family"].get("tau_flat", {}).get("pooled", {}).get("guarded_rdn_certified")},
        "G5_registry": {"bi_status_counts": out["by_family"].get("registry_violating", {}).get("pooled", {}).get("bi_status_counts")},
        "G6_cost": {"guarded_s3_success_pure": pure["guarded_s3_success"], "n_pure": pure["n"]},
    }
    target, min_records = float(ep["G3_validity"]["target"]), int(ep["G3_validity"]["min_records"])
    for f in fams:
        p = out["by_family"][f]["pooled"]
        evaluable = p["guarded_rdn_certified"] >= min_records
        out["endpoints"]["G3_validity"][f] = {"guarded_rdn_certified": p["guarded_rdn_certified"], "covered": p["coverage_ideal_joint_among_guarded_certified"],
                                              "rate": p["coverage_rate_among_guarded_certified"], "evaluable": evaluable,
                                              "meets_target": (p["coverage_rate_among_guarded_certified"] >= target) if evaluable else None}
    mis = values["evaluation"]["misspecified"]
    g3_mis = [out["endpoints"]["G3_validity"][f] for f in mis if out["endpoints"]["G3_validity"][f]["evaluable"]]
    out["decision"] = {"G1_consistent": out["endpoints"]["G1_false_abstention_pure"]["consistent_with_alpha"],
                       "G3_misspecified_evaluable": len(g3_mis), "G3_all_evaluable_meet_target": all(x["meets_target"] for x in g3_mis) if g3_mis else None,
                       "adopt_guard": bool(out["endpoints"]["G1_false_abstention_pure"]["consistent_with_alpha"] and all(x["meets_target"] for x in g3_mis))}
    return out


def evaluate(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started = time.perf_counter()
    config_dir = Path(config_dir)
    run = _guard(config_dir, output_dir, EVALUATION)
    values = run["values"]
    cal_path = Path(run["folder"]) / CALIBRATION
    if not cal_path.exists():
        raise RuntimeError("the calibration stage has not been run")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if cal["source_snapshot_hash_declared"] != run["declaration"]["source_snapshot_hash"] or cal["source_snapshot_post_run_drift"]["changed"]:
        raise RuntimeError("the calibration artifact does not belong to this declaration")
    thresholds = {d: float(cal["thresholds"][d]) for d in values["strata"]}
    ctx = _ctx(config_dir, "evaluation")
    n = int(values["evaluation"]["bank"]["block_count"])
    jobs = [{"index": i, "deficit": d, "family": f, "threshold": thresholds[d], "certify": True} for d in values["strata"] for i in range(n) for f in values["evaluation"]["families"]]
    results = _run_jobs(ctx, config_dir, "evaluation", jobs, workers, "evaluation records", progress)
    rows = [results[(j["index"], j["deficit"], j["family"])]["row"] for j in jobs]
    artifact = {"artifact_type": "g_guard_evaluation", "status": "development_only", "calibration_artifact_sha256": file_sha256(cal_path), "thresholds": thresholds,
                **analyse_evaluation(values, rows), "records": rows,
                "execution": {"workers": int(workers), "records": len(rows), "certificates": sum("certificate" in r for r in rows),
                              "record_seconds": _summ([r["seconds"] for r in rows]), "wall_seconds": time.perf_counter() - started}}
    return _finish(run, artifact, EVALUATION)
