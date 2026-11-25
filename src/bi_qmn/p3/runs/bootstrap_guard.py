"""Record-specific parametric-bootstrap goodness-of-fit guard for the model-based certification route.

Protocol: documents/bootstrap-guard-protocol.md (frozen by the declaration).

The conformal G guard (documents/G-guard-run.md) used one threshold per stratum, calibrated on pure records, and
its false abstention did not transfer (11.8% against 5%): the null distribution of the support deviance ``G``
depends on the construction. This guard calibrates each record against its own fitted pure-history model:

* fit the support with the frozen estimator (32 starts), giving ``theta_hat`` and the fitted support table;
* draw ``B`` supports from the fitted table (independent multinomial shots per observed context);
* refit each bootstrap support from ``theta_hat`` plus ``extra_starts`` random starts and record its ``G*``;
* ``p = (1 + #{G* >= G_obs}) / (B + 1)``; abstain when ``p <= alpha``.

The run evaluates the frozen rule on a fresh development bank of controlled truths and compares it, record by
record, with the frozen conformal thresholds. Certification of relational dynamics is the distance stage of
``rdn_v1`` (``kappa_fit > 1``), the only stage that certifies.
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
from scipy.linalg import logm
from scipy.optimize import minimize
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
from bi_qmn.p3.runs import g_guard as gg
from bi_qmn.p3.runs import m1_added_measurement as am
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _table_dict
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint


CONFIG_FILE = "bootstrap_guard.yaml"
RUNNER = "runners/p3/run_bootstrap_guard.py"
DECLARATION = "bootstrap_guard_declaration.json"
ARTIFACT = "bootstrap_guard.json"
CLASSES = ("diag", "coh")
DEFICITS = ("medium", "low")
_W: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Bootstrap machinery (pure given the estimator)
# ---------------------------------------------------------------------------


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:16], 16)


def theta_from_fit(sbp, fit: Mapping[str, Any]) -> np.ndarray:
    """Parameters whose branch vectors equal the fit's ``V`` (unitary completion and principal logarithm)."""
    d, n = sbp.forward.d_S, sbp.forward.n
    V = np.asarray(fit["branch_vectors_real_imag"][0]) + 1j * np.asarray(fit["branch_vectors_real_imag"][1])
    p = np.asarray(fit["populations"], dtype=float)
    phases = np.asarray(fit["relative_phases"], dtype=float)
    c = np.sqrt(p) * np.exp(1j * phases)
    U = V / c[None, :]
    rng = np.random.default_rng(0)
    M = np.concatenate([U, rng.normal(size=(d, d - n)) + 1j * rng.normal(size=(d, d - n))], axis=1)
    Q, R = np.linalg.qr(M)
    Q = Q * (np.diag(R) / np.abs(np.diag(R)))[None, :]
    W = np.concatenate([U, Q[:, n:]], axis=1)
    X = logm(W)
    X = 0.5 * (X - X.conj().T)
    basis = sbp.basis
    coords = np.asarray([np.real(np.vdot(B, X)) / np.real(np.vdot(B, B)) for B in basis])
    z = np.log(np.maximum(p, 1e-300)) - math.log(max(p[0], 1e-300))
    return np.concatenate([coords, z[1:], phases[1:]])


def output_table(sbp, V: np.ndarray, ids: Sequence[str]) -> dict[str, dict[str, float]]:
    """The estimator's output convention: clip at zero, add the probability floor, renormalize per context."""
    T = sbp.forward.tables(V, list(ids))
    out = {}
    for cid in ids:
        v = np.clip(T[cid], 0.0, None) + sbp.probability_floor
        v = v / v.sum()
        out[cid] = dict(zip(sbp.forward.outcomes, v.tolist(), strict=True))
    return out


def refit(sbp, A: np.ndarray, counts: np.ndarray, theta0: np.ndarray, rng: np.random.Generator, extra_starts: int) -> tuple[np.ndarray, float]:
    d, n = sbp.forward.d_S, sbp.forward.n
    starts = [np.asarray(theta0, dtype=float)]
    for _ in range(int(extra_starts)):
        starts.append(np.concatenate([rng.normal(0.0, 1.0, d * d), np.zeros(n - 1), rng.uniform(0.0, 2.0 * math.pi, n - 1)]))
    best = (math.inf, None)
    for t0 in starts:
        res = minimize(sbp._objective, t0, args=(A, counts), jac=True, method="L-BFGS-B",
                       options={"maxiter": sbp.max_iterations, "ftol": 1.0e-13, "gtol": 1.0e-9})
        if float(res.fun) < best[0]:
            best = (float(res.fun), res.x.copy())
    return best[1], best[0]


def bootstrap_test(sbp, support: Mapping[str, Mapping[str, float]], n_shot: int, fit: Mapping[str, Any], B: int, extra_starts: int, seed_text: str) -> dict[str, Any]:
    """Parametric bootstrap of the support deviance ``G`` under the record's own fitted pure-history model."""
    ids = sorted(support)
    theta_hat = theta_from_fit(sbp, fit)
    V_hat = sbp.branch_vectors(theta_hat)
    fitted = output_table(sbp, V_hat, ids)
    G_obs = e5.support_misfit(support, fitted, float(n_shot))["G"]
    rng = np.random.default_rng(_seed(seed_text))
    A = np.concatenate([sbp.forward.A[cid] for cid in ids])
    outcomes = sbp.forward.outcomes
    G_star = []
    for _ in range(int(B)):
        boot = {}
        for cid in ids:
            p = np.asarray([fitted[cid][o] for o in outcomes])
            draw = rng.multinomial(int(n_shot), p / p.sum())
            boot[cid] = dict(zip(outcomes, (draw / n_shot).tolist(), strict=True))
        counts = np.concatenate([[boot[cid][o] * n_shot for o in outcomes] for cid in ids])
        theta_b, _ = refit(sbp, A, counts, theta_hat, rng, extra_starts)
        G_star.append(e5.support_misfit(boot, output_table(sbp, sbp.branch_vectors(theta_b), ids), float(n_shot))["G"])
    G_star = np.asarray(G_star)
    tol = 1.0e-9 * max(1.0, abs(G_obs))
    return {"G": float(G_obs), "p_bootstrap": float((1 + np.sum(G_star >= G_obs - tol)) / (int(B) + 1)), "G_star_median": float(np.median(G_star)),
            "G_star_p95": float(np.quantile(G_star, 0.95)), "theta_reconstruction_error": float(np.max(np.abs(V_hat - (np.asarray(fit["branch_vectors_real_imag"][0]) + 1j * np.asarray(fit["branch_vectors_real_imag"][1])))))}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load(config_dir: Path) -> tuple[dict[str, Any], str]:
    path = Path(config_dir) / CONFIG_FILE
    values = _yaml(path)
    if values.get("status") not in ("development_only", "test_only"):
        raise ValueError("the bootstrap guard is a development protocol")
    st = values["statistic"]
    if not 0.0 < float(st["alpha"]) < 0.5 or int(st["bootstrap_resamples"]) < 19 or int(st["extra_starts"]) < 0:
        raise ValueError("alpha in (0, 0.5), at least 19 resamples, non-negative extra starts")
    bank = values["bank"]
    lo, hi = values["reserved_future_confirmatory_namespace"]["seed_range"]
    seeds = [int(bank["seed"]), int(bank["observation_seed"]), int(bank["haar_seed"]), int(st["bootstrap_seed"])]
    if len(set(seeds)) != len(seeds) or any(int(lo) <= s <= int(hi) for s in seeds):
        raise ValueError("bootstrap-guard seeds must be distinct and outside the reserved confirmatory namespace")
    if not set(values["misspecified"]) <= set(values["families"]) or "pure" not in values["families"]:
        raise ValueError("pure and the misspecified families must be evaluated")
    return values, file_sha256(path)


def _ctx(config_dir: Path) -> dict[str, Any]:
    values, _ = _load(config_dir)
    ctx = e5.prepare(config_dir, dict(values["bank"]))
    cal = json.loads(Path(values["conformal_comparator"]["artifact"]).read_text(encoding="utf-8"))
    ctx["conformal_thresholds"] = {d: float(cal["thresholds"][d]["threshold"]) if isinstance(cal["thresholds"][d], dict) else float(cal["thresholds"][d]) for d in DEFICITS}
    ctx["guard_values"] = values
    return ctx


# ---------------------------------------------------------------------------
# One record
# ---------------------------------------------------------------------------


def record_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    values = ctx["guard_values"]
    i, deficit, family = int(job["index"]), str(job["deficit"]), str(job["family"])
    bank = ctx["bank"]
    inst = ctx["instances"][i]
    rho, eta = gg.family_truth(ctx, i, family)
    meta = {"instance_index": i, "deficit": deficit, "family": family, "bank_id": bank["bank_id"]}
    effective = p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=inst.K, eta=eta, metadata=meta).copy_without_source()
    ideal = effective if eta == 0.0 else p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=inst.K, eta=0.0, metadata=meta).copy_without_source()
    ideal_tables = {c: dict(row) for c, row in ideal.probabilities.items()}
    n_shot = int(ctx["p1"].measurement["observation_deficit"][deficit]["n_shot"])
    support_contexts = support_contexts_for_deficit(ctx["p1"].measurement, deficit, seed_key=f"p3v2:{bank['bank_id']}:{deficit}:{i}")
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bank['bank_id']}:{deficit}:{i}"))       # common random numbers across families
    support = sample_shot_batch(effective, support_contexts, n_shots=n_shot, rng=rng, shot_start=(e5.DEFICITS.index(deficit) * 1000 + i) * 100_000_000, role="bootstrap_guard_support").table.probabilities
    support = {c: dict(row) for c, row in support.items()}
    screen = ctx["screen"].evaluate(support, float(n_shot))
    radii = ctx["bodies"]["radii"][deficit]
    row: dict[str, Any] = {"index": i, "deficit": deficit, "family": family, "eta": eta, "observed": sorted(support), "screen_passes": bool(screen["passes"]),
                           "truth": {"kappa": tau_flat_distance(ident.class_vector(ideal_tables, ctx["contexts"], "diag", ctx["K"]), ctx["diag_keys"]) / radii["diag"]}}
    if row["screen_passes"]:
        prediction = ctx["sbp"].predict_table(MeasurementTable(probabilities=support, contexts=[ctx["context_by_id"][c] for c in sorted(support)], metadata={"n_shots": float(n_shot)}, source_density=None), list(ctx["contexts"]))
        table = _table_dict(prediction)
        fit = prediction.metadata["shared_branch_pair_fit"]
        st = values["statistic"]
        boot = bootstrap_test(ctx["sbp"], support, n_shot, fit, int(st["bootstrap_resamples"]), int(st["extra_starts"]), f"{st['bootstrap_seed']}:{bank['bank_id']}:{deficit}:{i}:{family}")
        row["bootstrap"] = boot
        row["G"] = e5.support_misfit(support, table, float(n_shot))["G"]
        row["fit_kappa"] = tau_flat_distance(ident.class_vector(table, ctx["contexts"], "diag", ctx["K"]), ctx["diag_keys"]) / radii["diag"]
        s = {cls: float(np.linalg.norm(ident.class_vector(table, ctx["contexts"], cls, ctx["K"]) - ident.class_vector(ideal_tables, ctx["contexts"], cls, ctx["K"]))) for cls in CLASSES}
        row["coverage"] = {**{cls: bool(s[cls] <= radii[cls]) for cls in CLASSES}, "joint": all(s[cls] <= radii[cls] for cls in CLASSES), "scores": s}
        row["bootstrap_guard_passes"] = bool(boot["p_bootstrap"] > float(st["alpha"]))
        row["conformal_guard_passes"] = bool(row["G"] <= ctx["conformal_thresholds"][deficit])
    row["seconds"] = time.perf_counter() - started
    return {"key": (i, deficit, family), "row": row}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _worker_init(root: str, config_dir: str) -> None:
    os.chdir(root)
    _W.clear()
    _W.update(_ctx(Path(config_dir)))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    out = record_job(_W, job)
    out["instance_hash"] = _W["instance_hash"]
    return out


def _run_jobs(ctx: Mapping[str, Any], config_dir: Path, jobs: list[dict], workers: int, progress) -> dict:
    pool = None
    saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
    try:
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(Path.cwd()), str(Path(config_dir).resolve())))
        else:
            _W.clear()
            _W.update(ctx)
        results = _execute(pool, _worker_job, jobs, "bootstrap guard", progress)
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
# Analysis (pure)
# ---------------------------------------------------------------------------


def _cp(k: int, n: int):
    return e5.clopper_pearson(int(k), int(n)) if n else None


def analyse(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    alpha = float(values["statistic"]["alpha"])
    ep = values["endpoints"]
    out: dict[str, Any] = {"by_family": {}}
    for fam in values["families"]:
        block = {}
        for scope in ("medium", "low", "pooled"):
            sr = [r for r in rows if r["family"] == fam and (scope == "pooled" or r["deficit"] == scope)]
            inn = [r for r in sr if r["screen_passes"]]
            b = {"n": len(sr), "screened_in": len(inn), "truth_kappa": e5._q([r["truth"]["kappa"] for r in sr]),
                 "unguarded_certified": sum(r["fit_kappa"] > 1.0 for r in inn), "unguarded_covered": sum(r["coverage"]["joint"] for r in inn),
                 "p_bootstrap": e5._q([r["bootstrap"]["p_bootstrap"] for r in inn]), "G": e5._q([r["G"] for r in inn])}
            for guard in ("bootstrap", "conformal"):
                passed = [r for r in inn if r[f"{guard}_guard_passes"]]
                cert = [r for r in passed if r["fit_kappa"] > 1.0]
                cov = sum(r["coverage"]["joint"] for r in cert)
                b[guard] = {"abstentions": len(inn) - len(passed), "abstention_rate": (len(inn) - len(passed)) / len(inn) if inn else None,
                            "abstention_ci95": _cp(len(inn) - len(passed), len(inn)), "guarded_certified": len(cert), "covered_among_guarded_certified": cov,
                            "coverage_rate": cov / len(cert) if cert else None, "coverage_ci95": _cp(cov, len(cert))}
            b["paired_abstentions"] = {f"boot_{'pass' if x else 'abst'}|conf_{'pass' if y else 'abst'}": sum(r["bootstrap_guard_passes"] == x and r["conformal_guard_passes"] == y for r in inn)
                                       for x in (True, False) for y in (True, False)}
            block[scope] = b
        out["by_family"][fam] = block
    pure = out["by_family"]["pure"]["pooled"]["bootstrap"]
    b1 = {"abstentions": pure["abstentions"], "screened_in": out["by_family"]["pure"]["pooled"]["screened_in"], "rate": pure["abstention_rate"],
          "ci95": pure["abstention_ci95"], "consistent_with_alpha": bool(pure["abstention_ci95"] is not None and pure["abstention_ci95"][0] <= alpha)}
    target, min_records = float(ep["B3_validity"]["target"]), int(ep["B3_validity"]["min_records"])
    b3 = {}
    for fam in values["families"]:
        p = out["by_family"][fam]["pooled"]["bootstrap"]
        evaluable = p["guarded_certified"] >= min_records
        b3[fam] = {"guarded_certified": p["guarded_certified"], "covered": p["covered_among_guarded_certified"], "rate": p["coverage_rate"],
                   "evaluable": evaluable, "meets_target": (p["coverage_rate"] >= target) if evaluable else None}
    mis = [b3[f] for f in values["misspecified"] if b3[f]["evaluable"]]
    out["endpoints"] = {"B1_false_abstention_pure": b1,
                        "B2_detection": {f: {k: out["by_family"][f]["pooled"]["bootstrap"][k] for k in ("abstentions", "abstention_rate", "abstention_ci95")} for f in values["misspecified"]},
                        "B3_validity": b3}
    out["decision"] = {"B1_consistent": b1["consistent_with_alpha"], "B3_misspecified_evaluable": len(mis),
                       "B3_all_evaluable_meet_target": all(x["meets_target"] for x in mis) if mis else None,
                       "adopt_guard": bool(b1["consistent_with_alpha"] and all(x["meets_target"] for x in mis))}
    return out


# ---------------------------------------------------------------------------
# Declaration, guard, run
# ---------------------------------------------------------------------------


def _folder(values: Mapping[str, Any], output_dir: str | Path) -> Path:
    return Path(output_dir) / Path(str(values["output_root"])).name


def _extra_files(config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    names = [CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", "p1_ilgc_operational_r4.yaml", "p3_d1_v7.yaml", "g_guard.yaml"]
    return [str(Path(config_dir) / n) for n in dict.fromkeys(names)] + [*values["design_source"]["dev_inputs"], str(values["protocol_document"]), RUNNER]


def declare_bootstrap_guard(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = _folder(values, output_dir)
    target = folder / DECLARATION
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    dev_inputs = am._dev_inputs(root, values)
    p1 = load_p1_config(config_dir)
    prints = sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, values["bank"]))
    known = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked = {}
    for path, spec in harness.declared_bank_specs(config_dir, exclude=CONFIG_FILE):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    overlaps = {name: sorted(set(prints) & f) for name, f in known.items() if set(prints) & f}
    if overlaps:
        raise ValueError(f"the bootstrap-guard bank is not disjoint: {sorted(overlaps)}")
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values))
    declaration = {"artifact_type": "bootstrap_guard_declaration", "status": values["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "config_sha256": config_sha, "protocol_document": str(values["protocol_document"]), "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
                   "dev_inputs": dev_inputs, "statistic": values["statistic"], "endpoints": values["endpoints"],
                   "bank": {**values["bank"], "construction_fingerprint_hash": hashlib.sha256("".join(prints).encode("ascii")).hexdigest()[:16]},
                   "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps},
                   "source_snapshot": snapshot, "source_snapshot_hash": snapshot_hash(snapshot)}
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return declaration


def run_bootstrap_guard(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started = time.perf_counter()
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = _folder(values, output_dir)
    declaration = json.loads((folder / DECLARATION).read_text(encoding="utf-8"))
    if (folder / ARTIFACT).exists():
        raise FileExistsError("the bootstrap-guard artifact already exists and is immutable")
    if declaration["config_sha256"] != config_sha or declaration["protocol_document_sha256"] != file_sha256(root / str(values["protocol_document"])):
        raise RuntimeError("config or protocol drifted after declaration")
    extra = [p for p in declaration["source_snapshot"] if not p.startswith("src/")]
    drift = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    if drift["changed"] or drift["missing"]:
        raise RuntimeError(f"source drift after declaration: {drift}")
    ctx = _ctx(config_dir)
    jobs = [{"index": i, "deficit": d, "family": f} for i in range(int(values["bank"]["block_count"])) for d in DEFICITS for f in values["families"]]
    results = _run_jobs(ctx, config_dir, jobs, workers, progress)
    rows = [results[k]["row"] for k in sorted(results)]
    artifact: dict[str, Any] = {"artifact_type": "bootstrap_guard", "status": values["status"], "analysis": analyse(values, rows), "records": rows,
                                "settings": {"statistic": values["statistic"], "bank": values["bank"], "families": values["families"], "conformal_thresholds": ctx["conformal_thresholds"]},
                                "claim_boundary": {"development_only": True, "sealed_bank_opened": False,
                                                   "exact_states_used_for": "truth families, true kappa and coverage; the guard reads the support and its own fit"},
                                "execution": {"workers": int(workers), "records": len(rows), "wall_seconds": time.perf_counter() - started,
                                              "record_seconds": e5._q([r["seconds"] for r in rows])}}
    artifact = m1._jsonable(artifact)
    artifact["declaration_sha256"] = file_sha256(folder / DECLARATION)
    artifact["config_sha256"] = config_sha
    artifact["source_snapshot_hash_declared"] = declaration["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    (folder / ARTIFACT).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
