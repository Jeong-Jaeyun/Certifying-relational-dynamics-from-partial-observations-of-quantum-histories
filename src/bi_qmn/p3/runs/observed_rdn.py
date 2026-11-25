"""Observed RDN: a model-free companion certificate of relational dynamics (development).

Protocol: documents/observed-RDN-protocol.md (frozen by the declaration).

For each observed time-basis (D) context the counts ``N(tau, s)`` are tested for tau-flatness. Given the
outcome totals ``N_s``, a tau-flat table makes the clock readings within each outcome uniform, so the
conditional null has no nuisance parameter and the Monte Carlo p-value is exact. Every energy-pinched (static)
history has tau-flat tables in every D context, so a rejection rejects all static histories at once, at level
alpha, with no model, calibration bank or coverage; symmetric readout noise preserves tau-flatness, so the
guarantee survives it.

The run evaluates the test on a fresh development bank of controlled truths (no fits needed) and on the
supports of the E5 and G-guard development banks, regenerated deterministically and joined with the
model-based route's stored outcomes.
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
from scipy.stats import chi2
import yaml

from bi_qmn.core import identifiability as ident
from bi_qmn.core.config import load_p1_config
from bi_qmn.core.observations import sample_shot_batch
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.ilgc.relational_dynamics import tau_flat_distance
from bi_qmn.p1.model.p1_banks import support_contexts_for_deficit
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import p1_measurement_table
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.runs import e5_rdn_specificity as e5
from bi_qmn.p3.runs import m1_added_measurement as am
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _summ
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint


CONFIG_FILE = "observed_rdn.yaml"
RUNNER = "runners/p3/run_observed_rdn.py"
DECLARATION = "observed_rdn_declaration.json"
ARTIFACT = "observed_rdn.json"
_W: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# The test (pure)
# ---------------------------------------------------------------------------


def d_counts(support: Mapping[str, Mapping[str, float]], n_shots: int, K: int) -> dict[str, np.ndarray]:
    """Integer counts ``N[tau, s]`` of every observed D context (the support stores counts / n)."""
    out = {}
    for cid, row in support.items():
        if not cid.startswith("D,"):
            continue
        N = np.zeros((K, 2), dtype=np.int64)
        for l in range(K):
            for s in (0, 1):
                N[l, s] = int(round(float(row.get(f"c{l}_s{s}", 0.0)) * n_shots))
        out[cid] = N
    return out


def _g_columns(N: np.ndarray, K: int) -> float:
    total = 0.0
    for s in range(N.shape[1]):
        col = N[:, s].astype(float)
        Ns = col.sum()
        if Ns <= 0:
            continue
        nz = col > 0
        total += 2.0 * float(np.sum(col[nz] * np.log(K * col[nz] / Ns)))
    return total


def g_statistic(counts: Mapping[str, np.ndarray], K: int) -> float:
    return float(sum(_g_columns(N, K) for N in counts.values()))


def null_draws(counts: Mapping[str, np.ndarray], K: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """``G`` under the conditional null: within each outcome, the N_s clock readings are uniform over K."""
    G = np.zeros(int(B))
    p = np.full(K, 1.0 / K)
    for cid in sorted(counts):
        N = counts[cid]
        for s in range(N.shape[1]):
            Ns = int(N[:, s].sum())
            if Ns <= 0:
                continue
            draws = rng.multinomial(Ns, p, size=int(B)).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(draws > 0, draws * np.log(K * draws / Ns), 0.0)
            G += 2.0 * terms.sum(axis=1)
    return G


def observed_rdn_test(counts: Mapping[str, np.ndarray], K: int, B: int, rng: np.random.Generator) -> dict[str, float]:
    G_obs = g_statistic(counts, K)
    G_null = null_draws(counts, K, B, rng)
    df = sum((K - 1) for N in counts.values() for s in range(N.shape[1]) if N[:, s].sum() > 0)
    return {"G": G_obs, "p_monte_carlo": float((1 + np.sum(G_null >= G_obs - 1e-9)) / (B + 1)), "p_chi2": float(chi2.sf(G_obs, df)) if df else 1.0,
            "df": int(df), "null_mean": float(np.mean(G_null)), "contexts": len(counts)}


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
        raise ValueError("observed RDN is a development protocol")
    if not 0.0 < float(values["test"]["alpha_primary"]) < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    lo, hi = values["reserved_future_confirmatory_namespace"]["seed_range"]
    fb = values["fresh_bank"]
    seeds = [fb["seed"], fb["observation_seed"], fb["haar_seed"], values["test"]["monte_carlo_seed"]]
    if len(set(seeds)) != len(seeds) or any(int(lo) <= int(s) <= int(hi) for s in seeds):
        raise ValueError("observed-RDN seeds must be distinct and outside the reserved confirmatory namespace")
    if not set(fb["null_families"]) <= set(fb["families"]):
        raise ValueError("null families must be evaluated")
    return values, file_sha256(path)


def family_truth(ctx: Mapping[str, Any], i: int, family: str) -> tuple[np.ndarray, float]:
    if "_readout_" in family:
        base, eta = family.split("_readout_")
        return e5.family_density(ctx, i, base), float(eta)
    if family.startswith("readout_"):
        return e5.family_density(ctx, i, "pure"), float(family.split("_", 1)[1])
    return e5.family_density(ctx, i, family), 0.0


def record(ctx: Mapping[str, Any], i: int, deficit: str, family: str, role: str, B: int, mc_seed: int) -> dict[str, Any]:
    """Regenerate the support exactly as the E5 / G-guard modules do and run the screen and the observed test."""
    bank = ctx["bank"]
    inst = ctx["instances"][int(i)]
    rho, eta = family_truth(ctx, i, family)
    meta = {"instance_index": int(i), "deficit": deficit, "family": family, "bank_id": bank["bank_id"]}
    effective = p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=inst.K, eta=eta, metadata=meta).copy_without_source()
    ideal = effective if eta == 0.0 else p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=inst.K, eta=0.0, metadata=meta).copy_without_source()
    n_shot = int(ctx["p1"].measurement["observation_deficit"][deficit]["n_shot"])
    support_contexts = support_contexts_for_deficit(ctx["p1"].measurement, deficit, seed_key=f"p3v2:{bank['bank_id']}:{deficit}:{int(i)}")
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bank['bank_id']}:{deficit}:{int(i)}"))
    support = sample_shot_batch(effective, support_contexts, n_shots=n_shot, rng=rng, shot_start=(e5.DEFICITS.index(deficit) * 1000 + int(i)) * 100_000_000, role=role).table.probabilities
    support = {c: dict(row) for c, row in support.items()}
    screen = ctx["screen"].evaluate(support, float(n_shot))
    counts = d_counts(support, n_shot, ctx["K"])
    test = observed_rdn_test(counts, ctx["K"], B, np.random.default_rng(_seed(f"{mc_seed}:{bank['bank_id']}:{deficit}:{int(i)}:{family}")))
    ideal_tables = {c: dict(row) for c, row in ideal.probabilities.items()}
    radius = ctx["bodies"]["radii"][deficit]["diag"]
    return {"index": int(i), "deficit": deficit, "family": family, "eta": eta, "observed": sorted(support), "screen_passes": bool(screen["passes"]),
            "test": test, "truth": {"kappa": tau_flat_distance(ident.class_vector(ideal_tables, ctx["contexts"], "diag", ctx["K"]), ctx["diag_keys"]) / radius,
                                    "conditional_tau_spread": e5.conditional_tau_spread(rho, ctx["K"], ctx["d_S"])}}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _stage_bank(values: Mapping[str, Any], config_dir: Path, stage: str) -> dict[str, Any]:
    if stage == "fresh":
        return dict(values["fresh_bank"])
    if stage == "e5":
        return dict(json.loads(Path(values["existing_banks"]["e5"]["artifact"]).read_text(encoding="utf-8"))["settings"]["bank"])
    if stage == "g_guard":
        return dict(_yaml(Path(config_dir) / values["existing_banks"]["g_guard"]["config"])["evaluation"]["bank"])
    raise ValueError(stage)


def _ctx(config_dir: Path, stage: str) -> dict[str, Any]:
    values, _ = _load(config_dir)
    return e5.prepare(config_dir, _stage_bank(values, config_dir, stage))


def _job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    rows = [record(ctx, int(job["index"]), d, f, job["role"], int(job["B"]), int(job["mc_seed"])) for d, f in job["cells"]]
    return {"key": (str(job["stage"]), int(job["index"])), "rows": rows}


def _worker_init(root: str, config_dir: str, stage: str) -> None:
    os.chdir(root)
    _W.clear()
    _W.update(_ctx(Path(config_dir), stage))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    out = _job(_W, job)
    out["instance_hash"] = _W["instance_hash"]
    return out


def _run_stage(config_dir: Path, stage: str, jobs: list[dict], workers: int, progress) -> list[dict]:
    ctx = _ctx(config_dir, stage)
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
        results = _execute(pool, _worker_job, jobs, f"observed RDN [{stage}]", progress)
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
    return [row for key in sorted(results) for row in results[key]["rows"]]


# ---------------------------------------------------------------------------
# Analysis (pure)
# ---------------------------------------------------------------------------


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "ci95": e5.clopper_pearson(k, n) if n else None}


def analyse_fresh(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    alphas = [float(values["test"]["alpha_primary"]), *[float(a) for a in values["test"]["alpha_reported"]]]
    fams = values["fresh_bank"]["families"]
    out: dict[str, Any] = {"by_family": {}}
    for fam in fams:
        block = {}
        for scope in ("medium", "low", "pooled"):
            sr = [r for r in rows if r["family"] == fam and (scope == "pooled" or r["deficit"] == scope)]
            block[scope] = {f"certified_at_{a:g}": _rate(sum(r["test"]["p_monte_carlo"] <= a for r in sr), len(sr)) for a in alphas}
            block[scope]["screen_pass"] = _rate(sum(r["screen_passes"] for r in sr), len(sr))
            block[scope]["truth_kappa"] = e5._q([r["truth"]["kappa"] for r in sr])
            block[scope]["chi2_vs_mc_max_abs_difference_of_decision"] = int(sum((r["test"]["p_chi2"] <= alphas[0]) != (r["test"]["p_monte_carlo"] <= alphas[0]) for r in sr))
        out["by_family"][fam] = block
    a0 = alphas[0]
    out["O1_type_one"] = {fam: out["by_family"][fam]["pooled"][f"certified_at_{a0:g}"] for fam in values["fresh_bank"]["null_families"]}
    bins = [float(b) for b in values["endpoints"]["O3_dose_response"]["bins"]]
    nonnull = [r for r in rows if r["family"] not in values["fresh_bank"]["null_families"]]
    dose = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        for d in ("medium", "low"):
            sr = [r for r in nonnull if r["deficit"] == d and lo <= r["truth"]["kappa"] < hi]
            dose[f"{d}:[{lo:g},{hi:g})"] = _rate(sum(r["test"]["p_monte_carlo"] <= a0 for r in sr), len(sr))
    out["O3_dose_response"] = dose
    cross = {}
    for fam in fams:
        sr = [r for r in rows if r["family"] == fam]
        cross[fam] = {f"screen_{'in' if si else 'out'}|ordn_{'cert' if oc else 'no'}": sum(r["screen_passes"] == si and (r["test"]["p_monte_carlo"] <= a0) == oc for r in sr)
                      for si in (True, False) for oc in (True, False)}
    out["O5_screen_agreement"] = cross
    return out


def analyse_existing(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], stored: Mapping[tuple, Mapping[str, Any]], bank: str) -> dict[str, Any]:
    a0 = float(values["test"]["alpha_primary"])
    mismatched = [(r["index"], r["deficit"], r["family"]) for r in rows
                  if stored[(r["index"], r["deficit"], r["family"])]["observed"] != r["observed"] or stored[(r["index"], r["deficit"], r["family"])]["screen_passes"] != r["screen_passes"]]
    out = {"records": len(rows), "regeneration_mismatches": mismatched, "by_family": {}}
    for fam in sorted({r["family"] for r in rows}):
        sr = [r for r in rows if r["family"] == fam]
        cells: dict[str, int] = {}
        for r in sr:
            s = stored[(r["index"], r["deficit"], r["family"])]
            ordn = "ordn_cert" if r["test"]["p_monte_carlo"] <= a0 else "ordn_no"
            cert = s.get("certificate")
            if bank == "g_guard":
                cov = s.get("coverage", {}).get("ideal", {}).get("joint") if s["screen_passes"] else None
                guard = "screen_out" if not s["screen_passes"] else ("guard_pass" if s.get("guard_passes") else "guard_abstain")
                model = "model_rdn_cert" if (cert and cert["rdn"]["status"] == "certified") else "model_no_claim"
                key = f"{guard}|{model}|cov_{cov}|{ordn}"
            else:
                cov = s["coverage"]["joint"]
                model = "screen_out" if not s["screen_passes"] else ("model_rdn_cert" if cert and cert["rdn"]["status"] == "certified" else "model_no_claim")
                key = f"{model}|cov_{cov}|{ordn}"
            cells[key] = cells.get(key, 0) + 1
        out["by_family"][fam] = {"n": len(sr), "observed_rdn_certified": _rate(sum(r["test"]["p_monte_carlo"] <= a0 for r in sr), len(sr)), "joint_cells": dict(sorted(cells.items()))}
    return out


# ---------------------------------------------------------------------------
# Declaration, guard, run
# ---------------------------------------------------------------------------


def _extra_files(config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    names = [CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", "p1_ilgc_operational_r4.yaml", str(values["design_source"]["development_config"]), str(values["existing_banks"]["g_guard"]["config"])]
    return [str(Path(config_dir) / n) for n in dict.fromkeys(names)] + [*values["design_source"]["dev_inputs"], str(values["protocol_document"]), RUNNER]


def declare_observed_rdn(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
    target = folder / DECLARATION
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    dev_inputs = am._dev_inputs(root, values)
    p1 = load_p1_config(config_dir)
    prints = sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, values["fresh_bank"]))
    known = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked = {}
    for path, spec in harness.declared_bank_specs(config_dir, exclude=CONFIG_FILE):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    overlaps = {name: sorted(set(prints) & f) for name, f in known.items() if set(prints) & f}
    if overlaps:
        raise ValueError(f"the observed-RDN fresh bank is not disjoint: {sorted(overlaps)}")
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values))
    declaration = {"artifact_type": "observed_rdn_declaration", "status": values["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "config_sha256": config_sha, "protocol_document": str(values["protocol_document"]), "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
                   "dev_inputs": dev_inputs, "test": values["test"], "endpoints": values["endpoints"],
                   "fresh_bank": {**values["fresh_bank"], "construction_fingerprint_hash": hashlib.sha256("".join(prints).encode("ascii")).hexdigest()[:16]},
                   "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps},
                   "source_snapshot": snapshot, "source_snapshot_hash": snapshot_hash(snapshot)}
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return declaration


def run_observed_rdn(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print,
                     existing: bool = True) -> dict:
    started = time.perf_counter()
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
    declaration = json.loads((folder / DECLARATION).read_text(encoding="utf-8"))
    if (folder / ARTIFACT).exists():
        raise FileExistsError("the observed-RDN artifact already exists and is immutable")
    if declaration["config_sha256"] != config_sha or declaration["protocol_document_sha256"] != file_sha256(root / str(values["protocol_document"])):
        raise RuntimeError("config or protocol drifted after declaration")
    extra = [p for p in declaration["source_snapshot"] if not p.startswith("src/")]
    drift = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    if drift["changed"] or drift["missing"]:
        raise RuntimeError(f"source drift after declaration: {drift}")
    B, mc = int(values["test"]["monte_carlo_draws"]), int(values["test"]["monte_carlo_seed"])
    fb = values["fresh_bank"]
    cells = [(d, f) for d in values["strata"] for f in fb["families"]]
    fresh_rows = _run_stage(config_dir, "fresh", [{"stage": "fresh", "index": i, "cells": cells, "role": "observed_rdn_support", "B": B, "mc_seed": mc} for i in range(int(fb["block_count"]))], workers, progress)
    artifact: dict[str, Any] = {"artifact_type": "observed_rdn", "status": "development_only", "fresh": analyse_fresh(values, fresh_rows), "fresh_records": fresh_rows}
    if existing:
        for name in ("e5", "g_guard"):
            spec = values["existing_banks"][name]
            stored_art = json.loads(Path(spec["artifact"]).read_text(encoding="utf-8"))
            stored = {(int(r["index"]), r["deficit"], r["family"]): r for r in stored_art["records"]}
            by_index: dict[int, list] = {}
            for (i, d, f) in stored:
                by_index.setdefault(i, []).append((d, f))
            jobs = [{"stage": name, "index": i, "cells": sorted(v), "role": spec["role"], "B": B, "mc_seed": mc} for i, v in sorted(by_index.items())]
            rows = _run_stage(config_dir, name, jobs, workers, progress)
            artifact[f"existing_{name}"] = analyse_existing(values, rows, stored, name)
            artifact[f"existing_{name}_records"] = rows
    artifact["settings"] = {"test": values["test"], "fresh_bank": fb, "strata": values["strata"]}
    artifact["claim_boundary"] = {"development_only": True, "sealed_bank_opened": False, "model_free": True,
                                  "exact_states_used_for": "truth families and true kappa only; the test uses the observed counts"}
    artifact["execution"] = {"workers": int(workers), "fresh_records": len(fresh_rows), "wall_seconds": time.perf_counter() - started}
    artifact = m1._jsonable(artifact)
    artifact["declaration_sha256"] = file_sha256(folder / DECLARATION)
    artifact["config_sha256"] = config_sha
    artifact["source_snapshot_hash_declared"] = declaration["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    (folder / ARTIFACT).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
