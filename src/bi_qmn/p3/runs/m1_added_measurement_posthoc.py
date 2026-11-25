"""M1 Q1 post-hoc analyses on the declared development bank (report only; not part of the declared protocol).

Source: the declared run ``outputs/m1_added_measurement/m1_added_measurement.json`` (SHA-256 checked),
protocol documents/M1-added-measurement-design.md (frozen, read only). Requested by the user on
2026-11-22 after the run report. Everything here is post hoc on a development bank, so nothing below
changes P1 or P2, and any rule suggested by it needs its own declaration on a new bank.

Action ``posthoc`` (items 1-3):

1. **Fiber coverage (3a).** The RDN certificate's chain to the truth is "the body covers the truth"
   plus "no tau-flat state in the body". In 3a the body around the A0 table covered the truth jointly in
   5/128 records, but physical tau-dependence is invariant along the support fiber ``exp(itR)``. So the
   chain holds when the body covers *some* point of the truth's orbit. For every 3a record: the class
   scores of the A0 (and A_s) table against the exact table at each orbit point, the smallest joint
   ratio ``max(diag / r_diag, coh / r_coh)`` over the orbit and its angle.
2. **Orbit coefficient matrices (3a).** Along the exact orbit the table of an added D context is
   ``T(t) = c + a cos 2t + b sin 2t``. With ``M = [a b]``,
   ``||T(t) - T(0)|| = 2 |sin t| ||M (-sin t, cos t)||``, so ``sigma_min(M)`` is the tight
   worst-direction separation along the orbit; it is orbit-invariant and computable at a fit. For every
   candidate: ``sigma_min`` and ``sigma_max`` at the truth and at the A0 fit, the local tangent
   ``2 ||b||``, the weakest angle, the shot-noise scale, and the separation (Euclidean and
   chi-square) at the A0 fit's nearest orbit angle. Deterministic refits of A0, A_s and A_f locate each
   fit on the orbit (record 771 included).
3. **Re-certification after the added context.** BI existence (constraint registry, 80-digit verifier)
   and rdn_v1 for the refitted A_s table of every record, on the frozen S3 low bodies, with the coverage
   of the truth by the A_s body.

Action ``h_known`` (item 4, the registered-Hamiltonian comparator; code ready, run on request): the
support-only fit of :class:`ZeroEnergyHistoryEstimator`, its errors on the declared sets, its paired
CVaR comparisons with the declared arms (same bootstrap index rule), and optionally its certificates.

Refits are deterministic (restart seeds from a hash of the support); the run checks that the refitted
errors reproduce the declared artifact.
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
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.models.p3_zero_energy_history import (
    METHOD as ZEH_METHOD,
    ZeroEnergyHistoryEstimator,
    feasible_half_width,
    mode_embedding,
    orbit_energy_coefficient,
)
from bi_qmn.p3.runs import m1_added_measurement as am
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _summ, _table_dict, _true_branch_vectors


METHOD = "m1_added_measurement_posthoc_v1"
SOURCE = Path("outputs") / "m1_added_measurement" / "m1_added_measurement.json"
SOURCE_SHA256 = "bb61f9c6f6fc20d20cb3a783a890d514a8f8f71f6b648d9b4857fb7070d3b776"
OUTPUT_FOLDER = "m1_added_measurement_posthoc"
ARTIFACT = "m1_added_measurement_posthoc.json"
H_ARTIFACT = "m1_added_measurement_h_known.json"
RUNNER = "runners/p3/run_m1_added_measurement_posthoc.py"
ORBIT_POINTS = 96
REPRODUCTION_RELATIVE_TOLERANCE = 1.0e-6
CLASSES = am.CLASSES
_W: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Source and bank
# ---------------------------------------------------------------------------


def load_source(path: str | Path = SOURCE, sha256: str | None = SOURCE_SHA256) -> tuple[dict[str, Any], str]:
    path = Path(path)
    digest = file_sha256(path)
    if sha256 is not None and digest != sha256:
        raise RuntimeError(f"the declared M1 artifact differs from its recorded SHA-256: {digest}")
    return json.loads(path.read_text(encoding="utf-8")), digest


def rebuild_records(ctx: Mapping[str, Any], source: Mapping[str, Any], *, strata: Sequence[str] = am.STRATA, indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Rebuild the accepted constructions and their samples, and check them against the declared artifact."""
    keep = None if indices is None else {int(i) for i in indices}
    out = []
    for row in source["records"]:
        if row["class"] not in strata or (keep is not None and int(row["index"]) not in keep):
            continue
        r = am.build_construction(ctx, int(row["index"]))
        if (r["mask"], r["R"], r["class"]) != (row["mask"], row["R"], row["class"]) or not r["screen_passes"]:
            raise RuntimeError(f"rebuilt construction {row['index']} differs from the declared artifact")
        r["n_shot"] = ctx["n_shot"]
        am.attach_samples(ctx, r)
        if r["candidates"] != row["candidates"] or r["Q_f"] != row["Q_f"]:
            raise RuntimeError(f"rebuilt candidates of construction {row['index']} differ from the declared artifact")
        r.update(Q_s=row["Q_s"], Q_o=row["Q_o"], source=row)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Orbit geometry (pure)
# ---------------------------------------------------------------------------


def orbit_table(fw, V: np.ndarray, G: np.ndarray, t: float, ids: Sequence[str]) -> dict[str, dict[str, float]]:
    T = fw.tables(expm(1j * float(t) * G) @ V, ids)
    return {cid: dict(zip(fw.outcomes, T[cid].tolist(), strict=True)) for cid in ids}


def _refine(f: Callable[[float], float], grid: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    k = int(np.argmin(values))
    h = math.pi / grid.size
    res = minimize_scalar(f, bounds=(grid[k] - h, grid[k] + h), method="bounded", options={"xatol": 1.0e-7})
    if res.success and float(res.fun) < float(values[k]):
        return float(res.x) % math.pi, float(res.fun)
    return float(grid[k]), float(values[k])


def angle_from_truth(t: float) -> float:
    t = float(t) % math.pi
    return min(t, math.pi - t)


def fiber_coverage(ctx: Mapping[str, Any], table: Mapping[str, Mapping[str, float]], V: np.ndarray, R: str, points: int = ORBIT_POINTS) -> dict[str, Any]:
    """Class-score ratios of ``table`` against the exact tables along the truth's orbit ``exp(itR) V``."""
    fw, ids, radii = ctx["fw"], ctx["ids"], ctx["bodies"]["radii"]
    G = m1.pauli_matrix(R)

    def ratios(t: float) -> tuple[float, float]:
        s = am.class_errors(ctx, table, orbit_table(fw, V, G, t, ids), ids)
        return s["diag"] / radii["diag"], s["coh"] / radii["coh"]

    grid = np.linspace(0.0, math.pi, int(points), endpoint=False)
    vals = np.asarray([ratios(t) for t in grid])
    t_joint, joint = _refine(lambda t: max(ratios(t)), grid, vals.max(axis=1))
    t_diag, diag = _refine(lambda t: ratios(t)[0], grid, vals[:, 0])
    t_coh, coh = _refine(lambda t: ratios(t)[1], grid, vals[:, 1])
    return {
        "truth_ratio": {"diag": float(vals[0, 0]), "coh": float(vals[0, 1])},
        "fiber_min_ratio": {"joint": joint, "diag": diag, "coh": coh},
        "fiber_argmin_t": {"joint": t_joint, "diag": t_diag, "coh": t_coh},
        "fiber_argmin_angle_from_truth": angle_from_truth(t_joint),
        "covered": {"truth_joint": bool(vals[0].max() <= 1.0), "fiber_diag": bool(diag <= 1.0), "fiber_coh": bool(coh <= 1.0), "fiber_joint": bool(joint <= 1.0)},
    }


def coefficient_matrix(fw, V: np.ndarray, G: np.ndarray, q: str, n_shot: float) -> dict[str, Any]:
    """``T_q(t) = c + a cos 2t + b sin 2t`` along ``exp(itG) V``; singular values of ``M = [a b]``."""
    T = lambda t: fw.tables(expm(1j * t * G) @ V, [q])[q]   # noqa: E731
    T0, Tq, Th = T(0.0), T(math.pi / 4), T(math.pi / 2)
    c, a = (T0 + Th) / 2.0, (T0 - Th) / 2.0
    b = Tq - c
    residual = max(float(np.max(np.abs(T(t) - (c + a * math.cos(2 * t) + b * math.sin(2 * t))))) for t in (math.pi / 8, 3 * math.pi / 8, 0.3, 2.1))
    M = np.column_stack([a, b])
    _, s, vh = np.linalg.svd(M, full_matrices=False)
    v_min = vh[-1]
    return {
        "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
        "t_weak": float(math.atan2(-v_min[0], v_min[1]) % math.pi),
        "local_tangent": float(2.0 * np.linalg.norm(b)),
        "noise_l2": float(math.sqrt(max(0.0, 1.0 - float(T0 @ T0)) / float(n_shot))),
        "sinusoid_residual": residual,
        "a": a, "b": b, "T0": T0,
    }


def separation(cm: Mapping[str, Any], t: float, n_shot: float) -> dict[str, float]:
    """Euclidean and chi-square separation of the added table between orbit angle ``t`` and the truth."""
    d = cm["a"] * (math.cos(2 * t) - 1.0) + cm["b"] * math.sin(2 * t)
    l2 = float(np.linalg.norm(d))
    chi2 = float(n_shot * np.sum(d * d / np.maximum(cm["T0"], 1.0e-12)))
    return {"l2": l2, "l2_over_noise": l2 / cm["noise_l2"] if cm["noise_l2"] > 0 else math.inf, "chi2": chi2,
            "lower_bound": float(2.0 * abs(math.sin(t)) * cm["sigma_min"])}


def nearest_orbit_point(fw, V_hat: np.ndarray, V: np.ndarray, G: np.ndarray, points: int = ORBIT_POINTS) -> dict[str, float]:
    xh = fw.vector(V_hat)
    f = lambda t: m1._state_distance(xh, fw.vector(expm(1j * t * G) @ V))   # noqa: E731
    grid = np.linspace(0.0, math.pi, int(points), endpoint=False)
    t, dist = _refine(f, grid, np.asarray([f(t) for t in grid]))
    return {"t": t, "angle_from_truth": angle_from_truth(t), "distance": dist}


def _branch_vectors(fit: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(fit["branch_vectors_real_imag"][0]) + 1j * np.asarray(fit["branch_vectors_real_imag"][1])


# ---------------------------------------------------------------------------
# Worker jobs
# ---------------------------------------------------------------------------


def zero_energy_estimator(ctx: Mapping[str, Any], instance) -> ZeroEnergyHistoryEstimator:
    cfg = ctx["values"]["estimator"]
    effects = {}
    for (cid, o), M in ctx["effects"].items():
        effects.setdefault(cid, {})[o] = M
    return ZeroEnergyHistoryEstimator(
        ctx["p1"].generator["active_modes"], ctx["K"], ctx["d_S"], effects, np.asarray(instance.H_tot.data, dtype=complex), float(instance.h_scale),
        starts=int(cfg["starts"]), seed=int(cfg["seed"]), probability_floor=float(cfg["probability_floor"]), likelihood_floor=float(cfg["likelihood_floor"]),
        near_optimal_nats=float(cfg["near_optimal_nats"]), max_iterations=int(cfg["max_iterations"]),
    )


def _fit_h_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    est = zero_energy_estimator(ctx, ctx["instances"][int(job["index"])])
    support = MeasurementTable(probabilities=job["tables"], contexts=[ctx["context_by_id"][c] for c in sorted(job["tables"])], metadata={"n_shots": float(job["n_shots"])}, source_density=None)
    prediction = est.predict_table(support, list(ctx["contexts"]))
    fit = prediction.metadata["zero_energy_history_fit"]
    keep = ("objective_nats", "starts_converged", "starts_within_near_optimal", "kernel_dimension", "registered_energy", "populations", "branch_vectors_real_imag", "n_shots", "support_context_ids")
    return {"key": (int(job["index"]), str(job["variant"])), "table": _table_dict(prediction), "fit": {k: fit[k] for k in keep},
            "checkpoint_hash": est.checkpoint_hash, "elapsed_seconds": time.perf_counter() - started}


def _cert_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    out = am._cert_job(ctx, job)
    out["key"] = ("cert", int(job["index"]), str(job["variant"]))
    return out


_JOBS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]] = {"fit": am._fit_job, "fit_h": _fit_h_job, "cert": _cert_job}


def _worker_init(root: str, config_dir: str) -> None:
    os.chdir(root)
    _W.clear()
    _W.update(am._prepare_light(Path(config_dir)))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    out = _JOBS[job["kind"]](_W, job)
    out["instance_hash"] = _W["instance_hash"]
    return out


def _run_jobs(ctx: Mapping[str, Any], config_dir: Path, waves: Sequence[tuple[str, Callable[[dict], list[dict]]]], workers: int, progress) -> dict:
    """Run job waves in order; each wave builder sees the results so far."""
    results: dict = {}
    pool = None
    saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
    try:
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(Path.cwd()), str(config_dir.resolve())))
        else:
            _W.clear()
            _W.update(ctx)
        for label, build in waves:
            jobs = build(results)
            if jobs:
                results.update(_execute(pool, _worker_job, jobs, label, progress))
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
# Per-record analyses (parent)
# ---------------------------------------------------------------------------


def _sets(ctx: Mapping[str, Any], r: Mapping[str, Any]) -> dict[str, list[str]]:
    rest = [c for c in ctx["ids"] if c not in set(r["mask"])]
    E = [c for c in rest if c not in {r["Q_s"], r["Q_f"]}]
    return {"E": E, "E_o": [c for c in E if c != r["Q_o"]], "own_support_only": rest}


def reproduction(ctx: Mapping[str, Any], r: Mapping[str, Any], arm: str, table: Mapping[str, Mapping[str, float]]) -> float:
    """Largest relative deviation of the refitted errors on E from the declared artifact."""
    now = am.class_errors(ctx, table, r["exact"], _sets(ctx, r)["E"])
    then = r["source"]["errors"][arm]["E"]
    return max(abs(now[k] - then[k]) / max(abs(then[k]), 1.0e-12) for k in (*CLASSES, "max_abs"))


def certificate_summary(ctx: Mapping[str, Any], r: Mapping[str, Any], cert: Mapping[str, Any], table: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    full = am.class_errors(ctx, table, r["exact"], ctx["ids"])
    radii = ctx["bodies"]["radii"]
    covered = {cls: bool(full[cls] <= radii[cls]) for cls in CLASSES}
    return {"bi": cert["bi"], "rdn": cert["rdn"], "seconds": cert["seconds"],
            "success": cert["bi"]["status"] == "primal_verified" and cert["rdn"]["status"] == "certified",
            "truth_scores": full, "truth_covered": {**covered, "joint": all(covered.values())}}


def posthoc_row_3a(ctx: Mapping[str, Any], r: Mapping[str, Any], fits: Mapping[tuple, Mapping[str, Any]], points: int = ORBIT_POINTS) -> dict[str, Any]:
    i, fw, n_shot = int(r["index"]), ctx["fw"], float(ctx["n_shot"])
    V = _true_branch_vectors(ctx["instances"][i])
    G = m1.pauli_matrix(r["R"])
    variants = {"A0": "support", "A_s": f"add:{r['Q_s']}", "A_f": f"add:{r['Q_f']}"}
    t0_table = orbit_table(fw, V, G, 0.0, ctx["ids"])
    exact_gap = max(abs(t0_table[c][o] - float(r["exact"][c][o])) for c in ctx["ids"] for o in t0_table[c])
    angles = {arm: nearest_orbit_point(fw, _branch_vectors(fits[(i, v)]["fit"]), V, G, points) for arm, v in variants.items()}
    V_hat = _branch_vectors(fits[(i, "support")]["fit"])
    G_hat = G   # the fitted orbit uses the same R (the mask's commutant)
    struct = r["source"]["structural"]["truth_smin_rel_with_candidate"]
    rank = r["source"]["rank_stability"]
    cands = {}
    for q in r["candidates"]:
        cm = coefficient_matrix(fw, V, G, q, n_shot)
        cm_fit = coefficient_matrix(fw, V_hat, G_hat, q, n_shot)
        cands[q] = {
            "sigma_min_truth": cm["sigma_min"], "sigma_max_truth": cm["sigma_max"], "t_weak_truth": cm["t_weak"],
            "local_tangent_truth": cm["local_tangent"], "noise_l2": cm["noise_l2"], "sinusoid_residual": cm["sinusoid_residual"],
            "sigma_min_fit": cm_fit["sigma_min"], "sigma_max_fit": cm_fit["sigma_max"],
            "smin_rel_truth": float(struct[q]), "s_Q_truth": float(rank["s_Q_truth"][q]), "s_Q_fit": float(rank["s_Q_fit"][q]),
            "separation_at_A0_angle": separation(cm, angles["A0"]["t"], n_shot),
        }
    best = {k: max(c[k] for c in cands.values()) for k in ("sigma_min_truth", "sigma_min_fit", "smin_rel_truth", "s_Q_truth", "s_Q_fit")}
    for c in cands.values():
        c["ratio_to_best"] = {k: (c[k] / best[k] if best[k] > 0 else None) for k in best}
    tol = float(ctx["values"]["added_context"]["tie_relative_tolerance"])
    sigma_pick_fit, sigma_tied_fit = am.argmax_with_ties({q: cands[q]["sigma_min_fit"] for q in r["candidates"]}, r["candidates"], tol)
    sigma_pick_truth, _ = am.argmax_with_ties({q: cands[q]["sigma_min_truth"] for q in r["candidates"]}, r["candidates"], tol)
    H = np.asarray(ctx["instances"][i].H_tot.data, dtype=complex)
    h = float(ctx["instances"][i].h_scale)
    B_coef = orbit_energy_coefficient(V, G, H, h, mode_embedding(ctx["p1"].generator["active_modes"], ctx["K"], ctx["d_S"]))
    src_err = r["source"]["errors"]
    return {
        "index": i, "unitary": r["unitary"], "R": r["R"], "mask": r["mask"], "candidates": r["candidates"],
        "Q_s": r["Q_s"], "Q_f": r["Q_f"], "Q_o": r["Q_o"],
        "exact_table_gap_at_t0": exact_gap,
        "fiber_A0": fiber_coverage(ctx, fits[(i, "support")]["table"], V, r["R"], points),
        "fiber_A_s": fiber_coverage(ctx, fits[(i, f"add:{r['Q_s']}")]["table"], V, r["R"], points),
        "fitted_orbit_point": angles,
        "candidates_orbit": cands,
        "sigma_min_rule": {"pick_at_fit": sigma_pick_fit, "tied_at_fit": sigma_tied_fit, "pick_at_truth": sigma_pick_truth},
        "energy": {"B": B_coef, "feasible_half_width": feasible_half_width(B_coef, float(ctx["theta_con"]))},
        "errors_E_o_diag": {arm: float(src_err[arm]["E_o"]["diag"]) for arm in am.ARMS},
        "errors_E_diag": {arm: float(src_err[arm]["E"]["diag"]) for arm in am.ARMS},
        "a0_bi_status": r["source"]["certificate"]["bi"]["status"], "a0_rdn_status": r["source"]["certificate"]["rdn"]["status"],
    }


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------


def _q(v: Sequence[float]) -> dict[str, float] | None:
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return {k: float(np.quantile(v, float(k))) for k in ("0.0", "0.1", "0.5", "0.9", "1.0")} if v else None


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rho = spearmanr(x, y)[0]
    return float(rho) if np.isfinite(rho) else None


def summarize_3a(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    by_status: dict[str, dict[str, int]] = {}
    for r in rows:
        s = by_status.setdefault(r["a0_bi_status"], {"n": 0, "fiber_joint": 0, "truth_joint": 0})
        s["n"] += 1
        s["fiber_joint"] += int(r["fiber_A0"]["covered"]["fiber_joint"])
        s["truth_joint"] += int(r["fiber_A0"]["covered"]["truth_joint"])
    fiber = {
        "n": n,
        "A0": {k: sum(r["fiber_A0"]["covered"][k] for r in rows) for k in ("truth_joint", "fiber_diag", "fiber_coh", "fiber_joint")},
        "A0_fiber_min_joint_ratio": _q([r["fiber_A0"]["fiber_min_ratio"]["joint"] for r in rows]),
        "A0_truth_joint_ratio": _q([max(r["fiber_A0"]["truth_ratio"].values()) for r in rows]),
        "A0_fiber_argmin_angle_from_truth": _q([r["fiber_A0"]["fiber_argmin_angle_from_truth"] for r in rows]),
        "A0_by_bi_status": by_status,
        "rdn_certified_and_fiber_covered": sum(r["a0_rdn_status"] == "certified" and r["fiber_A0"]["covered"]["fiber_joint"] for r in rows),
        "A_s": {k: sum(r["fiber_A_s"]["covered"][k] for r in rows) for k in ("truth_joint", "fiber_diag", "fiber_coh", "fiber_joint")},
        "A_s_fiber_argmin_angle_from_truth": _q([r["fiber_A_s"]["fiber_argmin_angle_from_truth"] for r in rows]),
    }
    # pick-level predictors of the error on the oracle set E_o (common to A_s, A_f and A_o)
    pairs = []
    for r in rows:
        for arm, key in (("A_s", "Q_s"), ("A_f", "Q_f"), ("A_o", "Q_o")):
            c = r["candidates_orbit"][r[key]]
            pairs.append({"index": r["index"], "arm": arm, "context": r[key], "error": r["errors_E_o_diag"][arm],
                          "sigma_min_ratio": c["ratio_to_best"]["sigma_min_truth"], "smin_rel_ratio": c["ratio_to_best"]["smin_rel_truth"],
                          "s_Q_ratio": c["ratio_to_best"]["s_Q_truth"], "sigma_min": c["sigma_min_truth"], "local_tangent": c["local_tangent_truth"],
                          "chi2_at_A0_angle": c["separation_at_A0_angle"]["chi2"], "l2_over_noise_at_A0_angle": c["separation_at_A0_angle"]["l2_over_noise"]})
    predictors = ("sigma_min_ratio", "smin_rel_ratio", "s_Q_ratio", "sigma_min", "local_tangent", "chi2_at_A0_angle", "l2_over_noise_at_A0_angle")
    correlation = {}
    for arm in ("A_s", "A_f", "A_o", "pooled"):
        sel = [p for p in pairs if arm == "pooled" or p["arm"] == arm]
        correlation[arm] = {k: _spearman([p[k] for p in sel], [p["error"] for p in sel]) for k in predictors}
    tail = sorted([p for p in pairs if p["arm"] == "A_f"], key=lambda p: -p["error"])[: am.tail_count(n, 0.9)] if n else []
    sigma_fit_vs_truth = [abs(c["sigma_min_fit"] - c["sigma_min_truth"]) / c["sigma_min_truth"] for r in rows for c in r["candidates_orbit"].values() if c["sigma_min_truth"] > 0]
    picks = {
        "sigma_rule_at_fit_equals_Q_s": sum(r["sigma_min_rule"]["pick_at_fit"] == r["Q_s"] for r in rows),
        "sigma_rule_at_fit_equals_Q_f": sum(r["sigma_min_rule"]["pick_at_fit"] == r["Q_f"] for r in rows),
        "sigma_rule_at_fit_equals_Q_o": sum(r["sigma_min_rule"]["pick_at_fit"] == r["Q_o"] for r in rows),
        "sigma_rule_at_fit_equals_truth": sum(r["sigma_min_rule"]["pick_at_fit"] == r["sigma_min_rule"]["pick_at_truth"] for r in rows),
        "below_half_sigma_min": {arm: sum(1 for p in pairs if p["arm"] == arm and p["sigma_min_ratio"] is not None and p["sigma_min_ratio"] < 0.5) for arm in ("A_s", "A_f", "A_o")},
    }
    return {
        "fiber_coverage": fiber,
        "orbit_coefficients": {
            "sinusoid_residual_max": max(c["sinusoid_residual"] for r in rows for c in r["candidates_orbit"].values()) if rows else None,
            "sigma_min_truth": _q([c["sigma_min_truth"] for r in rows for c in r["candidates_orbit"].values()]),
            "sigma_min_fit_relative_error": _q(sigma_fit_vs_truth),
            "spearman_error_vs_predictor_E_o_diag": correlation,
            "A_f_tail_records": tail,
            "picks": picks,
        },
        "fitted_orbit_point": {arm: {"angle_from_truth": _q([r["fitted_orbit_point"][arm]["angle_from_truth"] for r in rows]),
                                     "distance": _q([r["fitted_orbit_point"][arm]["distance"] for r in rows])} for arm in ("A0", "A_s", "A_f")},
        "energy": {"B": _q([r["energy"]["B"] for r in rows]), "feasible_half_width": _q([r["energy"]["feasible_half_width"] for r in rows])},
        "exact_table_gap_at_t0_max": max((r["exact_table_gap_at_t0"] for r in rows), default=None),
    }


def summarize_recertification(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for stratum in am.STRATA:
        sr = [r for r in rows if r["class"] == stratum]
        if not sr:
            continue
        transitions: dict[str, int] = {}
        for r in sr:
            key = f"{r['a0_bi_status']}->{r['A_s']['bi']['status']}"
            transitions[key] = transitions.get(key, 0) + 1
        out[stratum] = {
            "n": len(sr),
            "A_s_bi_status_counts": am._counts([r["A_s"]["bi"]["status"] for r in sr]),
            "A_s_rdn_status_counts": am._counts([r["A_s"]["rdn"]["status"] for r in sr]),
            "A_s_success": sum(r["A_s"]["success"] for r in sr),
            "A0_success_declared": sum(r["a0_success"] for r in sr),
            "bi_transitions_A0_to_A_s": dict(sorted(transitions.items())),
            "A_s_truth_coverage": {**{cls: sum(r["A_s"]["truth_covered"][cls] for r in sr) for cls in CLASSES}, "joint": sum(r["A_s"]["truth_covered"]["joint"] for r in sr)},
            "A_s_kappa": _q([r["A_s"]["rdn"]["kappa"] for r in sr]),
            "certificate_seconds": _summ([r["A_s"]["seconds"] for r in sr]),
        }
    return out


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _folder(output_dir: str | Path) -> Path:
    return Path(output_dir) / OUTPUT_FOLDER


def _provenance(config_dir: Path, source_path: Path, source_sha: str) -> dict[str, Any]:
    root = Path.cwd()
    extra = [str(Path(config_dir) / am.CONFIG_FILE)]
    if (root / RUNNER).exists():
        extra.append(RUNNER)
    snapshot = import_closure_snapshot(root, extra_files=extra)
    return {"source_artifact": source_path.as_posix(), "source_artifact_sha256": source_sha, "config_sha256": file_sha256(Path(config_dir) / am.CONFIG_FILE),
            "source_snapshot_hash": snapshot_hash(snapshot), "source_snapshot": snapshot, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def run_posthoc(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, source: str | Path = SOURCE, source_sha256: str | None = SOURCE_SHA256,
                workers: int = 1, progress: Callable[[str], None] | None = print, strata: Sequence[str] = am.STRATA, indices: Sequence[int] | None = None,
                points: int = ORBIT_POINTS, certify: bool = True, write: bool = True) -> dict[str, Any]:
    """Items 1-3: fiber coverage and orbit coefficients (3a), re-certification of A_s (both strata)."""
    started = time.perf_counter()
    config_dir = Path(config_dir)
    target = _folder(output_dir) / ARTIFACT
    if write and target.exists():
        raise FileExistsError(f"post-hoc artifact already exists: {target}")
    src, src_sha = load_source(source, source_sha256)
    if src["config_sha256"] != file_sha256(config_dir / am.CONFIG_FILE):
        raise RuntimeError("the config differs from the one the declared artifact was run with")
    ctx = am._prepare_light(config_dir)
    records = rebuild_records(ctx, src, strata=strata, indices=indices)
    if progress is not None:
        progress(f"rebuilt {len(records)} records: {dict((s, sum(r['class'] == s for r in records)) for s in am.STRATA)}")
    by_index = {int(r["index"]): r for r in records}

    def fit_wave(_: dict) -> list[dict]:
        jobs = []
        for r in records:
            wanted = [f"add:{r['Q_s']}"] + (["support", f"add:{r['Q_f']}"] if r["class"] == "3a" else [])
            for v in dict.fromkeys(wanted):
                tables, n_shots = am.variant_tables(r, v)
                jobs.append({"kind": "fit", "index": r["index"], "variant": v, "tables": tables, "n_shots": n_shots})
        return jobs

    def cert_wave(results: dict) -> list[dict]:
        if not certify:
            return []
        bank_id, deficit = ctx["values"]["bank"]["bank_id"], ctx["values"]["deficit"]
        return [{"kind": "cert", "index": r["index"], "variant": f"add:{r['Q_s']}", "table": results[(r["index"], f"add:{r['Q_s']}")]["table"],
                 "support_query_hash": f"{bank_id}:{deficit}:{r['index']}:add:{r['Q_s']}"} for r in records]

    results = _run_jobs(ctx, config_dir, [("refits (A0, A_s, A_f)", fit_wave), ("A_s certificates", cert_wave)], workers, progress)
    fits = {k: v for k, v in results.items() if k[0] != "cert"}
    certs = {k: v for k, v in results.items() if k[0] == "cert"}

    # reproduction of the declared fits
    deviation = {}
    for (i, v), f in fits.items():
        r = by_index[i]
        arm = "A0" if v == "support" else "A_s" if v == f"add:{r['Q_s']}" else "A_f"
        deviation[f"{i}:{arm}"] = reproduction(ctx, r, arm, f["table"])
    worst = max(deviation.values(), default=0.0)
    if worst > REPRODUCTION_RELATIVE_TOLERANCE:
        raise RuntimeError(f"refitted errors do not reproduce the declared artifact (max relative deviation {worst:.3e})")

    rows_3a = [posthoc_row_3a(ctx, r, fits, points) for r in records if r["class"] == "3a"]
    if progress is not None:
        progress(f"3a orbit analyses: {len(rows_3a)}")
    recert_rows = []
    for r in records:
        key = ("cert", int(r["index"]), f"add:{r['Q_s']}")
        if key not in certs:
            continue
        recert_rows.append({"index": int(r["index"]), "class": r["class"], "Q_s": r["Q_s"], "a0_bi_status": r["source"]["certificate"]["bi"]["status"],
                            "a0_success": bool(r["source"]["certificate"]["success"]),
                            "A_s": certificate_summary(ctx, r, certs[key], fits[(int(r["index"]), f"add:{r['Q_s']}")]["table"])})
    artifact = {
        "artifact_type": "m1_added_measurement_posthoc",
        "method": METHOD,
        "status": "development_only_post_hoc",
        "claim_boundary": {"post_hoc": True, "changes_declared_endpoints": False, "sealed_bank_opened": False,
                           "note": "post-hoc analyses requested after the declared run; any rule suggested here needs its own declaration on a new bank",
                           "exact_states_used_for": "orbit tables, coefficient matrices, fitted orbit angles, coverage, the constraint registry and the energy coefficient"},
        "settings": {"orbit_points": int(points), "frozen_radii_low": ctx["bodies"]["radii"], "n_shot": ctx["n_shot"], "theta_con": ctx["theta_con"], "certify": bool(certify),
                     "strata": list(strata), "indices": None if indices is None else [int(i) for i in indices]},
        "reproduction": {"max_relative_deviation": worst, "fits": len(fits)},
        "summary_3a": summarize_3a(rows_3a) if rows_3a else None,
        "recertification": summarize_recertification(recert_rows),
        "records_3a": rows_3a,
        "records_recertification": recert_rows,
        "execution": {"workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial", "fits": len(fits), "certificates": len(certs),
                      "fit_seconds": _summ([f["elapsed_seconds"] for f in fits.values()]), "wall_seconds": time.perf_counter() - started},
        **_provenance(config_dir, Path(source), src_sha),
    }
    artifact = m1._jsonable(artifact)
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run_h_known(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, source: str | Path = SOURCE, source_sha256: str | None = SOURCE_SHA256,
                workers: int = 1, progress: Callable[[str], None] | None = print, strata: Sequence[str] = am.STRATA, indices: Sequence[int] | None = None,
                certify: bool = False, points: int = ORBIT_POINTS, write: bool = True) -> dict[str, Any]:
    """Item 4: the registered-Hamiltonian comparator A0_H (support only, zero-energy kernel)."""
    started = time.perf_counter()
    config_dir = Path(config_dir)
    target = _folder(output_dir) / H_ARTIFACT
    if write and target.exists():
        raise FileExistsError(f"H-known artifact already exists: {target}")
    src, src_sha = load_source(source, source_sha256)
    if src["config_sha256"] != file_sha256(config_dir / am.CONFIG_FILE):
        raise RuntimeError("the config differs from the one the declared artifact was run with")
    ctx = am._prepare_light(config_dir)
    records = rebuild_records(ctx, src, strata=strata, indices=indices)
    bank_id, deficit = ctx["values"]["bank"]["bank_id"], ctx["values"]["deficit"]

    def fit_wave(_: dict) -> list[dict]:
        return [{"kind": "fit_h", "index": r["index"], "variant": "h_known:support", "tables": r["support"], "n_shots": float(r["n_shot"])} for r in records]

    def cert_wave(results: dict) -> list[dict]:
        if not certify:
            return []
        return [{"kind": "cert", "index": r["index"], "variant": "h_known:support", "table": results[(r["index"], "h_known:support")]["table"],
                 "support_query_hash": f"{bank_id}:{deficit}:{r['index']}:h_known"} for r in records]

    results = _run_jobs(ctx, config_dir, [("A0_H fits", fit_wave), ("A0_H certificates", cert_wave)], workers, progress)
    rows = []
    for r in records:
        i = int(r["index"])
        f = results[(i, "h_known:support")]
        sets = _sets(ctx, r)
        V = _true_branch_vectors(ctx["instances"][i])
        row = {"index": i, "class": r["class"], "unitary": r["unitary"], "R": r["R"],
               "errors": {name: am.class_errors(ctx, f["table"], r["exact"], ids) for name, ids in sets.items()},
               "fit": {k: f["fit"][k] for k in ("objective_nats", "starts_within_near_optimal", "kernel_dimension", "registered_energy")},
               "fitted_orbit_point": nearest_orbit_point(ctx["fw"], _branch_vectors(f["fit"]), V, m1.pauli_matrix(r["R"]), points),
               "checkpoint_hash": f["checkpoint_hash"], "seconds": f["elapsed_seconds"]}
        key = ("cert", i, "h_known:support")
        if key in results:
            row["certificate"] = certificate_summary(ctx, r, results[key], f["table"])
        rows.append(row)
    src_rows = {int(r["index"]): r for r in src["records"]}
    ev = ctx["values"]["evaluation"]
    level, boot = float(ev["statistic"]["level"]), ev["bootstrap"]
    primary = str(ev["primary_class"])
    strata_out = {}
    for stratum in am.STRATA:
        sr = [r for r in rows if r["class"] == stratum]
        if not sr:
            continue
        n_full = sum(1 for r in src["records"] if r["class"] == stratum)
        # the declared bootstrap index rule; identical indices to the declared analysis when the whole stratum is used
        rng = np.random.default_rng(am._seed(f"{boot['seed']}:{stratum}"))
        index = rng.integers(0, len(sr), size=(int(boot["resamples"]), len(sr)))
        h = {s: [r["errors"][s][primary] for r in sr] for s in ("E", "E_o")}
        arm = {(a, s): [src_rows[r["index"]]["errors"][a][s][primary] for r in sr] for a in am.ARMS for s in ("E", "E_o")}
        comps = {
            "A0_minus_A0_H": am.paired_cvar_difference(arm[("A0", "E")], h["E"], index, level, boot["interval"]),
            "C_minus_A0_H": am.paired_cvar_difference(arm[("C", "E")], h["E"], index, level, boot["interval"]),
            "A0_H_minus_A_s": am.paired_cvar_difference(h["E"], arm[("A_s", "E")], index, level, boot["interval"]),
            "A0_H_minus_A_o_on_E_o": am.paired_cvar_difference(h["E_o"], arm[("A_o", "E_o")], index, level, boot["interval"]),
        }
        block = {"n": len(sr), "whole_stratum": len(sr) == n_full, "comparisons_primary_class": comps,
                 "A0_H_errors_E_diag": _q(h["E"]), "registered_energy": _q([r["fit"]["registered_energy"] for r in sr]),
                 "kernel_dimension_counts": am._counts([r["fit"]["kernel_dimension"] for r in sr]),
                 "fitted_angle_from_truth": _q([r["fitted_orbit_point"]["angle_from_truth"] for r in sr])}
        certs = [r["certificate"] for r in sr if "certificate" in r]
        if certs:
            block["certification"] = {"bi_status_counts": am._counts([c["bi"]["status"] for c in certs]), "rdn_status_counts": am._counts([c["rdn"]["status"] for c in certs]),
                                      "success": sum(c["success"] for c in certs), "n": len(certs),
                                      "truth_coverage": {**{cls: sum(c["truth_covered"][cls] for c in certs) for cls in CLASSES}, "joint": sum(c["truth_covered"]["joint"] for c in certs)}}
        strata_out[stratum] = block
    artifact = {
        "artifact_type": "m1_added_measurement_h_known",
        "method": METHOD,
        "estimator": ZEH_METHOD,
        "status": "development_only_post_hoc",
        "claim_boundary": {"post_hoc": True, "changes_declared_endpoints": False, "sealed_bank_opened": False,
                           "information": "the estimator is told the registered instance Hamiltonian H_tot (the same matrix as the BI constraint registry); the declared arms are not"},
        "settings": {"certify": bool(certify), "strata": list(strata), "indices": None if indices is None else [int(i) for i in indices], "orbit_points": int(points)},
        "strata": strata_out,
        "records": rows,
        "execution": {"workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial",
                      "fits": sum(1 for k in results if k[0] != "cert"), "certificates": sum(1 for k in results if k[0] == "cert"), "wall_seconds": time.perf_counter() - started},
        **_provenance(config_dir, Path(source), src_sha),
    }
    artifact = m1._jsonable(artifact)
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
