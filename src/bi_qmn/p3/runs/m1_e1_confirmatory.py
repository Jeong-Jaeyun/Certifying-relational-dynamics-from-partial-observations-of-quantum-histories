"""E1 sealed confirmation: same-budget recovery by one added D context (M1).

Preregistration: documents/E1-preregistration.md (frozen by the declaration). User decisions of 2026-11-22.

* **Banks.** Two independent sealed banks under the D1-v7 generative law, each construction with a low mask
  drawn uniformly from the 110 flagged masks. H1 accepts screened-in constructions of oracle class 3a until
  256; H2 accepts every screened-in construction until 256 (class recorded for description only).
* **Arms.** A0 support; C the same 2,048 extra shots spread over the support (256 each); C_F all 2,048 extra
  shots on the observed F contexts that anticommute with ``R``; A_s support + the max-``s_Q`` candidate at the
  A0 fit; A_f support + the fixed-order candidate; A_o support + the truth-``smin_rel`` oracle; A0_H the
  support with the registered Hamiltonian (zero-energy kernel; information level ii).
* **Primary.** ``CVaR_0.9(C) - CVaR_0.9(A_s)`` of the diag error on ``E = all - support - {Q_s, Q_f}``; each
  hypothesis passes when the 2.5% paired-bootstrap bound exceeds one frozen low diag radius; E1 is
  confirmed when H1 and H2 both pass (intersection-union).
* **Secondary (descriptive).** Other contrasts, the oracle block, class breakdown, leave-one-record-out,
  rank stability, A0 and A_s certificates and coverage, kappa, fiber coverage, the orbit energy and the
  ``sigma_min`` of the picks.
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
from scipy.optimize import minimize
from scipy.stats import spearmanr
import yaml

from bi_qmn.core.config import load_p1_config
from bi_qmn.core.observations import sample_shot_batch
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.ilgc.relational_dynamics import RDN_VERSION
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import full_p1_context_list, p1_effect_map
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairEstimator
from bi_qmn.p3.models.p3_zero_energy_history import mode_embedding, orbit_energy_coefficient
from bi_qmn.p3.p3_ilgc_facing import load_ilgc_targets
from bi_qmn.p3.p3_support_screen import SCREEN_VERSION, SupportIdentifiabilityScreen
from bi_qmn.p3.runs import m1_added_measurement as am
from bi_qmn.p3.runs import m1_added_measurement_posthoc as ph
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _keys, _summ, _table_dict, _true_branch_vectors
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint


CONFIG_FILE = "m1_e1_confirmatory.yaml"
RUNNER = "runners/p3/run_m1_e1_confirmatory.py"
DECLARATION = "m1_e1_confirmatory_declaration.json"
ARTIFACT = "m1_e1_confirmatory.json"
BANKS = ("H1", "H2")
ACCEPTANCE = {"H1": "enriched_3a", "H2": "natural"}
ARMS = ("A0", "C", "C_F", "A_s", "A_f", "A_o", "A0_H")
CLASSES = ("diag", "coh")
SECONDARY_E = (("A0_minus_A_s", "A0", "A_s"), ("C_F_minus_A_s", "C_F", "A_s"), ("A_f_minus_A_s", "A_f", "A_s"), ("C_minus_C_F", "C", "C_F"),
               ("A0_minus_C", "A0", "C"), ("A0_H_minus_A_s", "A0_H", "A_s"))
ORACLE_E_O = (("A0_minus_A_o", "A0", "A_o"), ("A_s_minus_A_o", "A_s", "A_o"), ("A_f_minus_A_o", "A_f", "A_o"), ("C_minus_A_s", "C", "A_s"))
GENERATIVE_KEYS = ("dirichlet_concentration", "minimum_mode_probability", "phase_range")
STATUSES = ("sealed_confirmatory", "test_only")
_W: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:16], 16)


def _all_seeds(values: Mapping[str, Any]) -> list[int]:
    seeds = [int(values["banks"][b][k]) for b in BANKS for k in ("seed", "observation_seed", "mask_seed")]
    return seeds + [int(values["evaluation"]["bootstrap"]["seed"])]


def _load(config_dir: Path) -> tuple[dict[str, Any], str]:
    path = Path(config_dir) / CONFIG_FILE
    values = _yaml(path)
    if values.get("status") not in STATUSES or float(values["coherent_core_gamma"]) != 1.0:
        raise ValueError("E1 is a gamma=1 sealed confirmatory (or test-only) run")
    dev = _yaml(Path(config_dir) / values["design_source"]["development_config"])
    if values["estimator"] != dev["estimator"]:
        raise ValueError("estimator settings differ from the D1-v7 development config")
    if values["screen"] != dev["screen"] or values["screen"]["method"] != SCREEN_VERSION or not values["screen"]["pauli_commutant"]:
        raise ValueError(f"the screen must be the frozen {SCREEN_VERSION} with the commutant check")
    if values["deficit"] != "low":
        raise ValueError("the flagged masks are low masks")
    reference = dev["development_banks"]["evaluation"]
    if set(values["banks"]) != set(BANKS):
        raise ValueError("E1 declares exactly the banks H1 and H2")
    for b in BANKS:
        spec = values["banks"][b]
        if any(spec[k] != reference[k] for k in GENERATIVE_KEYS):
            raise ValueError(f"bank {b} must use the D1-v7 generative specification")
        if spec["acceptance"] != ACCEPTANCE[b] or int(spec["quota"]) <= 0 or int(spec["block_count"]) < int(spec["quota"]):
            raise ValueError(f"bank {b}: acceptance must be {ACCEPTANCE[b]} with a positive quota below the cap")
    seeds = _all_seeds(values)
    if len(set(seeds)) != len(seeds):
        raise ValueError("every bank, observation, mask and bootstrap seed must be distinct")
    ns = values["reserved_future_confirmatory_namespace"]
    lo, hi = (int(v) for v in ns["seed_range"])
    inside = [lo <= s <= hi for s in seeds]
    if values["status"] == "sealed_confirmatory":
        if not all(inside):
            raise ValueError("sealed confirmatory seeds must lie in the reserved namespace")
        if set(seeds) & {int(v) for v in ns.get("previously_allocated", [])}:
            raise ValueError("sealed confirmatory seeds reuse a previously allocated seed")
        if sorted(int(v) for v in ns.get("allocated_here", [])) != sorted(seeds):
            raise ValueError("allocated_here must list exactly the seeds of this protocol")
    elif any(inside):
        raise ValueError("test-only seeds intrude on the reserved confirmatory namespace")
    cert, dev_cert = values["certificate"], dev["reporting"]["certificate"]
    if (cert["existence"], cert["proposal_backends"]) != (dev_cert["existence"], dev_cert["proposal_backends"]) or cert["rdn_version"] != RDN_VERSION:
        raise ValueError("certificate settings differ from the D1-v7/S3 route")
    ep = values["endpoints"]
    if ep["contrast"] != ["C", "A_s"] or ep["delta_min"] != "frozen_low_diag_radius" or ep["combination"] != "intersection_union":
        raise ValueError("the declared E1 primary contrast, margin or combination was changed")
    if int(values["control"]["extra_shots_per_support_context"]) * 8 != int(values["added_context"]["n_shot"]) or int(values["control_f"]["extra_shots_total"]) != int(values["added_context"]["n_shot"]):
        raise ValueError("C, C_F and the added context must spend the same extra-shot budget")
    return values, file_sha256(path)


def _prepare(config_dir: Path, bank_key: str) -> dict[str, Any]:
    """A context compatible with the M1 protocol functions, for one bank."""
    values, _ = _load(config_dir)
    p1 = load_p1_config(config_dir)
    bank = dict(values["banks"][bank_key])
    instances = seeded_construction_bank_p1(p1.generator, bank)
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
        raise ValueError("the added context gets the support's per-context shots")
    targets = load_ilgc_targets(config_dir, values["ilgc_targets"]["operational_config"])
    return {
        "values": {**values, "bank": bank}, "bank_key": bank_key, "p1": p1,
        "K": K, "d_C": d_C, "d_S": d_S, "n": len(p1.generator["active_modes"]),
        "eta": float(p1.measurement["noise"]["readout_eta"]), "theta_con": float(p1.generator["thresholds"]["theta_con"]),
        "instances": instances,
        "instance_hash": hashlib.sha256("".join(physical_fingerprint(i) for i in instances).encode("ascii")).hexdigest()[:16],
        "contexts": contexts, "context_by_id": {c.context_id: c for c in contexts}, "ids": [c.context_id for c in contexts],
        "effects": {(cid, o): M for cid, row in effects_by_context.items() for o, M in row.items()},
        "sbp": sbp, "fw": sbp.forward,
        "screen": SupportIdentifiabilityScreen(p1.generator["active_modes"], K, epsilon=float(values["screen"]["disconnected_branch_null_tail_epsilon"])),
        "bodies": am._bodies(Path.cwd(), values), "stratum": targets["strata"][values["deficit"]],
        "flagged": am.flagged_masks(p1.measurement),
        "pool": sorted({cid for m in m1.deficit_masks(p1.measurement, "low") for cid in m if cid.startswith("D,")}),
        "n_shot": n_shot, "diag_keys": _keys(contexts, "diag", K),
    }


# ---------------------------------------------------------------------------
# Banks and samples
# ---------------------------------------------------------------------------


def open_bank(ctx: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate in index order; H1 accepts screened-in 3a, H2 every screened-in construction, until the quota or the cap."""
    bank = ctx["values"]["bank"]
    quota, mode = int(bank["quota"]), str(bank["acceptance"])
    accepted: list[dict[str, Any]] = []
    ledger = {"bank": ctx["bank_key"], "acceptance": mode, "cap": int(bank["block_count"]), "quota": quota, "generated": 0,
              "screened_out": dict.fromkeys(am.STRATA, 0), "screened_in": dict.fromkeys(am.STRATA, 0), "skipped_not_3a": 0,
              "accepted": dict.fromkeys(am.STRATA, 0), "screen_out_reasons": {}}
    for i in range(int(bank["block_count"])):
        if len(accepted) >= quota:
            break
        r = am.build_construction(ctx, i)
        ledger["generated"] += 1
        if not r["screen_passes"]:
            ledger["screened_out"][r["class"]] += 1
            for reason in r["screen_reasons"]:
                ledger["screen_out_reasons"][reason] = ledger["screen_out_reasons"].get(reason, 0) + 1
            continue
        ledger["screened_in"][r["class"]] += 1
        if mode == "enriched_3a" and r["class"] != "3a":
            ledger["skipped_not_3a"] += 1
            continue
        ledger["accepted"][r["class"]] += 1
        accepted.append(r)
    ledger["quota_filled"] = len(accepted) >= quota
    ledger["shortfall"] = max(0, quota - len(accepted))
    ledger["last_index_generated"] = ledger["generated"] - 1
    return accepted, ledger


def anticommuting_f(mask: Sequence[str], R: str) -> list[str]:
    G = m1.pauli_matrix(R)
    out = []
    for cid in mask:
        if cid.startswith("F,"):
            P = m1.pauli_matrix(cid.split(",")[1])
            if np.linalg.norm(P @ G + G @ P) < 1e-12:
                out.append(cid)
    return sorted(out)


def attach_control_f(ctx: Mapping[str, Any], r: dict[str, Any]) -> None:
    """C_F: the whole extra budget on the observed F contexts anticommuting with R, split equally."""
    bank = ctx["values"]["bank"]
    bid, i, n_shot = bank["bank_id"], int(r["index"]), int(ctx["n_shot"])
    targets = anticommuting_f(r["mask"], r["R"])
    if not targets:
        raise RuntimeError(f"construction {i}: no observed F context anticommutes with R (a type-1 mask should have been screened out)")
    total = int(ctx["values"]["control_f"]["extra_shots_total"])
    extra_n = {c: total // len(targets) + (1 if k < total % len(targets) else 0) for k, c in enumerate(targets)}
    exact = MeasurementTable(probabilities=r["exact"], contexts=list(ctx["contexts"]), metadata={}, source_density=None)
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bid}:control_f:{i}"))
    tables = {c: dict(row) for c, row in r["support"].items()}
    n_by = {c: n_shot for c in r["mask"]}
    for k, c in enumerate(targets):
        extra = sample_shot_batch(exact, [ctx["context_by_id"][c]], n_shots=extra_n[c], rng=rng, shot_start=i * 100_000_000 + 60_000_000 + k * total, role="m1e1_control_f").table.probabilities[c]
        n = n_shot + extra_n[c]
        tables[c] = {o: (n_shot * float(r["support"][c][o]) + extra_n[c] * float(extra.get(o, 0.0))) / n for o in r["support"][c]}
        n_by[c] = n
    r["control_f"] = {"tables": tables, "n_by_context": n_by, "targets": targets, "extra_by_context": extra_n}


# ---------------------------------------------------------------------------
# Heterogeneous-shot SBP fit (C_F)
# ---------------------------------------------------------------------------


def fit_heterogeneous(sbp: SharedBranchPairEstimator, support: Mapping[str, Mapping[str, float]], n_by_context: Mapping[str, float]) -> dict[str, Any]:
    """``SharedBranchPairEstimator.fit_support`` with per-context shot counts (same likelihood, starts and seeding)."""
    ids = sorted(support)
    A = np.concatenate([sbp.forward.A[cid] for cid in ids])
    counts = np.concatenate([[float(support[cid].get(o, 0.0)) * float(n_by_context[cid]) for o in sbp.forward.outcomes] for cid in ids])
    rng = np.random.default_rng(sbp._seed_for(ids, counts))
    d, n = sbp.forward.d_S, sbp.forward.n
    runs = []
    for _ in range(sbp.starts):
        theta0 = np.concatenate([rng.normal(0.0, 1.0, d * d), np.zeros(n - 1), rng.uniform(0.0, 2.0 * math.pi, n - 1)])
        res = minimize(sbp._objective, theta0, args=(A, counts), jac=True, method="L-BFGS-B", options={"maxiter": sbp.max_iterations, "ftol": 1.0e-13, "gtol": 1.0e-9})
        runs.append((float(res.fun), res.x.copy(), bool(res.success)))
    runs.sort(key=lambda item: item[0])
    best_value, best_theta = runs[0][0], runs[0][1]
    V = sbp.branch_vectors(best_theta)
    _, p, phases, _, _ = sbp._unpack(best_theta)
    return {"method": sbp.method, "objective_nats": best_value, "starts": sbp.starts, "starts_converged": int(sum(r[2] for r in runs)),
            "starts_within_near_optimal": sum(1 for r in runs if r[0] <= best_value + sbp.near_optimal_nats),
            "populations": p.tolist(), "branch_vectors_real_imag": [V.real.tolist(), V.imag.tolist()],
            "n_by_context": {c: float(v) for c, v in n_by_context.items()}, "support_context_ids": ids}


def predicted_table(ctx: Mapping[str, Any], V: np.ndarray) -> dict[str, dict[str, float]]:
    """The SBP output convention: clip, add the probability floor, renormalize per context."""
    sbp = ctx["sbp"]
    T = sbp.forward.tables(V, ctx["ids"])
    out = {}
    for cid in ctx["ids"]:
        v = np.clip(T[cid], 0.0, None) + sbp.probability_floor
        out[cid] = dict(zip(sbp.forward.outcomes, (v / v.sum()).tolist(), strict=True))
    return out


def _fit_het_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    fit = fit_heterogeneous(ctx["sbp"], job["tables"], job["n_by_context"])
    V = np.asarray(fit["branch_vectors_real_imag"][0]) + 1j * np.asarray(fit["branch_vectors_real_imag"][1])
    keep = ("objective_nats", "starts_converged", "starts_within_near_optimal", "populations", "branch_vectors_real_imag", "support_context_ids")
    return {"key": (int(job["index"]), str(job["variant"])), "table": predicted_table(ctx, V), "fit": {k: fit[k] for k in keep}, "elapsed_seconds": time.perf_counter() - started}


def _cert_job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    out = am._cert_job(ctx, job)
    out["key"] = ("cert", int(job["index"]), str(job["variant"]))
    return out


_JOBS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]] = {
    "truth": am._truth_job, "fit": am._fit_job, "fit_het": _fit_het_job, "fit_h": ph._fit_h_job, "cert": _cert_job}


def _worker_init(root: str, config_dir: str, bank_key: str) -> None:
    os.chdir(root)
    _W.clear()
    _W.update(_prepare(Path(config_dir), bank_key))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    out = _JOBS[job["kind"]](_W, job)
    out["instance_hash"] = _W["instance_hash"]
    return out


# ---------------------------------------------------------------------------
# Declaration and guard
# ---------------------------------------------------------------------------


def _extra_files(config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    names = [CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", str(values["ilgc_targets"]["operational_config"]), str(values["design_source"]["development_config"])]
    return [str(Path(config_dir) / n) for n in dict.fromkeys(names)] + [str(values["bodies"]["source_artifact"]), *values["design_source"]["dev_inputs"], str(values["protocol_document"]), RUNNER]


def declare_m1_e1(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
    target = folder / DECLARATION
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    bodies = am._bodies(root, values)
    dev_inputs = am._dev_inputs(root, values)
    p1 = load_p1_config(config_dir)
    prints = {b: sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, values["banks"][b])) for b in BANKS}
    if set(prints["H1"]) & set(prints["H2"]):
        raise ValueError("the H1 and H2 banks share constructions")
    known = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked = {}
    for path, spec in harness.declared_bank_specs(config_dir, exclude=CONFIG_FILE):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    overlaps = {f"{b}~{name}": sorted(set(prints[b]) & f) for b in BANKS for name, f in known.items() if set(prints[b]) & f}
    if overlaps:
        raise ValueError(f"the E1 banks are not disjoint from declared banks: {sorted(overlaps)}")
    flagged = am.flagged_masks(p1.measurement)
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values))
    declaration = {
        "artifact_type": "m1_e1_confirmatory_declaration", "status": values["status"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha,
        "protocol_document": str(values["protocol_document"]), "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
        "dev_inputs": dev_inputs, "bodies": bodies, "delta_min": float(bodies["radii"]["diag"]),
        "estimator": values["estimator"], "screen": values["screen"],
        "banks": {b: {**values["banks"][b], "construction_fingerprint_hash": hashlib.sha256("".join(prints[b]).encode("ascii")).hexdigest()[:16], "observation_tables_opened": False} for b in BANKS},
        "flagged_mask_count": len(flagged), "flagged_mask_hash": hashlib.sha256(json.dumps([list(m) for m in flagged]).encode("ascii")).hexdigest()[:16],
        "added_context": values["added_context"], "control": values["control"], "control_f": values["control_f"],
        "evaluation": values["evaluation"], "endpoints": values["endpoints"], "certificate": values["certificate"],
        "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps, "h1_h2_disjoint": True},
        "reserved_future_confirmatory_namespace": values["reserved_future_confirmatory_namespace"],
        "source_snapshot": snapshot, "source_snapshot_hash": snapshot_hash(snapshot),
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
        raise FileExistsError("the E1 artifact already exists and is immutable: the sealed banks are opened once")
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
# Per-record row
# ---------------------------------------------------------------------------


VARIANTS = {"A0": "support", "C": "control", "C_F": "control_f", "A0_H": "h_known:support"}


def _variant(arm: str, r: Mapping[str, Any]) -> str:
    return VARIANTS.get(arm) or f"add:{r[{'A_s': 'Q_s', 'A_f': 'Q_f', 'A_o': 'Q_o'}[arm]]}"


def record_row(ctx: Mapping[str, Any], r: Mapping[str, Any], truth: Mapping[str, Any], fits: Mapping[tuple, Mapping[str, Any]], certs: Mapping[tuple, Mapping[str, Any]]) -> dict[str, Any]:
    i, ev = int(r["index"]), ctx["values"]["evaluation"]
    tables = {arm: fits[(i, _variant(arm, r))]["table"] for arm in ARMS}
    rest = [c for c in ctx["ids"] if c not in set(r["mask"])]
    E = [c for c in rest if c not in {r["Q_s"], r["Q_f"]}]
    E_o = [c for c in E if c != r["Q_o"]]
    own = {arm: rest for arm in ("A0", "C", "C_F", "A0_H")} | {arm: [c for c in rest if c != r[q]] for arm, q in (("A_s", "Q_s"), ("A_f", "Q_f"), ("A_o", "Q_o"))}
    errors = {arm: {"E": am.class_errors(ctx, tables[arm], r["exact"], E), "E_o": am.class_errors(ctx, tables[arm], r["exact"], E_o), "own": am.class_errors(ctx, tables[arm], r["exact"], own[arm])} for arm in ARMS}
    radii = ctx["bodies"]["radii"]
    V_hat = am._branch_vectors(fits[(i, "support")]["fit"])
    V = _true_branch_vectors(ctx["instances"][i])
    tol = float(ctx["values"]["added_context"]["tie_relative_tolerance"])
    s_fit = {q: float(r["s_Q_fit"][q]) for q in r["candidates"]}
    s_true = {q: float(truth["candidates"][q]["s_Q"]) for q in r["candidates"]}
    _, truth_tied = am.argmax_with_ties(s_true, r["candidates"], tol)
    rho = spearmanr([s_fit[q] for q in r["candidates"]], [s_true[q] for q in r["candidates"]])[0] if len(set(s_true.values())) > 1 and len(set(s_fit.values())) > 1 else None
    G = m1.pauli_matrix(r["R"])
    sigma = {q: ph.coefficient_matrix(ctx["fw"], V, G, q, float(ctx["n_shot"]))["sigma_min"] for q in r["candidates"]}
    best_sigma = max(sigma.values())
    row = {
        "index": i, "class": r["class"], "unitary": r["unitary"], "mask": r["mask"], "R": r["R"], "branch_condition": r["branch_condition"],
        "candidates": r["candidates"], "Q_s": r["Q_s"], "Q_f": r["Q_f"], "Q_o": r["Q_o"], "Q_s_tied_at_fit": r["Q_s_tied"], "Q_o_tied": r["Q_o_tied"],
        "control_f": {"targets": r["control_f"]["targets"], "extra_by_context": r["control_f"]["extra_by_context"]},
        "evaluation_sets": {"E": E, "E_o": E_o},
        "errors": errors,
        "kappa": {"exact": am._kappa(ctx, r["exact"]), **{arm: am._kappa(ctx, tables[arm]) for arm in ARMS}},
        "rank_stability": {"s_Q_fit": s_fit, "s_Q_truth": s_true, "truth_tied_argmax": truth_tied, "agrees_with_truth": r["Q_s"] in truth_tied,
                           "spearman": None if rho is None or not np.isfinite(rho) else float(rho),
                           "max_relative_error": max(abs(s_fit[q] - s_true[q]) / s_true[q] for q in r["candidates"]),
                           "orbit_distance": am._orbit_distance(ctx, V_hat, V, r["R"], int(ev["orbit_distance_t_points"]))},
        "sigma_min": {"by_candidate": sigma, "ratio_of_picks": {k: (sigma[r[k]] / best_sigma if best_sigma > 0 else None) for k in ("Q_s", "Q_f", "Q_o")}},
        "structural": {"support_extra_null": truth["support_extra_null"], "gauge_dim": truth["gauge_dim"],
                       "candidate_extra_null": {q: truth["candidates"][q]["extra_null"] for q in r["candidates"]},
                       "truth_smin_rel_with_candidate": {q: truth["candidates"][q]["smin_rel"] for q in r["candidates"]}},
        "fits": {arm: {k: fits[(i, _variant(arm, r))]["fit"].get(k) for k in ("objective_nats", "starts_within_near_optimal")} | {"seconds": fits[(i, _variant(arm, r))]["elapsed_seconds"]} for arm in ARMS},
        "a0_h": {k: fits[(i, "h_known:support")]["fit"][k] for k in ("kernel_dimension", "registered_energy")},
    }
    row["structural"]["unclassified"] = (r["class"] == "3a" and row["structural"]["support_extra_null"] != 1) or any(v != 0 for q, v in row["structural"]["candidate_extra_null"].items() if q in {r["Q_s"], r["Q_f"], r["Q_o"]})
    row["certificates"] = {arm: ph.certificate_summary(ctx, r, certs[("cert", i, _variant(arm, r))], tables[arm]) for arm in ("A0", "A_s") if ("cert", i, _variant(arm, r)) in certs}
    if r["class"] == "3a":
        H = np.asarray(ctx["instances"][i].H_tot.data, dtype=complex)
        row["fiber_A0"] = ph.fiber_coverage(ctx, tables["A0"], V, r["R"], int(ev["orbit_distance_t_points"]))
        row["energy_B"] = orbit_energy_coefficient(V, G, H, float(ctx["instances"][i].h_scale), mode_embedding(ctx["p1"].generator["active_modes"], ctx["K"], ctx["d_S"]))
    return row


# ---------------------------------------------------------------------------
# Analysis (pure)
# ---------------------------------------------------------------------------


def _q(v: Sequence[float]) -> dict[str, float] | None:
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return {k: float(np.quantile(v, float(k))) for k in ("0.0", "0.1", "0.5", "0.9", "1.0")} if v else None


def _arm_summary(rows: Sequence[Mapping[str, Any]], set_name: str, level: float) -> dict[str, Any]:
    out = {}
    for arm in ARMS:
        for key in (*CLASSES, "max_abs"):
            v = [r["errors"][arm][set_name][key] for r in rows]
            out[f"{arm}:{key}"] = {"cvar": am.cvar(v, level), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "max": float(np.max(v))} if v else None
    return out


def leave_one_out(a: Sequence[float], b: Sequence[float], level: float, indices: Sequence[int]) -> dict[str, Any]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    full = am.cvar(a, level) - am.cvar(b, level)
    vals = [am.cvar(np.delete(a, k), level) - am.cvar(np.delete(b, k), level) for k in range(a.size)] if a.size > 1 else [full]
    k = int(np.argmax(np.abs(np.asarray(vals) - full)))
    return {"full": full, "min": float(min(vals)), "max": float(max(vals)), "most_influential_index": int(indices[k]), "most_influential_change": float(vals[k] - full)}


def analyse_bank(values: Mapping[str, Any], bank_key: str, rows: Sequence[Mapping[str, Any]], delta_min: float) -> dict[str, Any]:
    ev = values["evaluation"]
    level, boot, primary = float(ev["statistic"]["level"]), ev["bootstrap"], str(ev["primary_class"])
    n = len(rows)
    block: dict[str, Any] = {"n": n, "delta_min": float(delta_min)}
    if n == 0:
        block["primary"] = None
        return block
    rng = np.random.default_rng(_seed(f"{boot['seed']}:{bank_key}"))
    index = rng.integers(0, n, size=(int(boot["resamples"]), n))
    err = lambda arm, s: [r["errors"][arm][s][primary] for r in rows]   # noqa: E731
    prim = am.paired_cvar_difference(err("C", "E"), err("A_s", "E"), index, level, boot["interval"])
    block["primary"] = {**prim, "passes": bool(prim["lower"] > delta_min), "exceeds_zero": bool(prim["lower"] > 0.0)}
    block["secondary_E"] = {name: am.paired_cvar_difference(err(a, "E"), err(b, "E"), index, level, boot["interval"]) for name, a, b in SECONDARY_E}
    block["oracle_E_o"] = {name: am.paired_cvar_difference(err(a, "E_o"), err(b, "E_o"), index, level, boot["interval"]) for name, a, b in ORACLE_E_O}
    block["primary_coh"] = am.paired_cvar_difference([r["errors"]["C"]["E"]["coh"] for r in rows], [r["errors"]["A_s"]["E"]["coh"] for r in rows], index, level, boot["interval"])
    block["leave_one_out_primary"] = leave_one_out(err("C", "E"), err("A_s", "E"), level, [r["index"] for r in rows])
    block["arm_summaries"] = {s: _arm_summary(rows, s, level) for s in ("E", "E_o", "own")}
    by_class = {}
    for cls in am.STRATA:
        sr = [r for r in rows if r["class"] == cls]
        if not sr:
            continue
        cert = {arm: [r["certificates"][arm] for r in sr if arm in r["certificates"]] for arm in ("A0", "A_s")}
        by_class[cls] = {
            "n": len(sr),
            "cvar_E_diag": {arm: am.cvar([r["errors"][arm]["E"][primary] for r in sr], level) for arm in ARMS},
            "certification": {arm: {"bi_status_counts": am._counts([c["bi"]["status"] for c in cs]), "rdn_status_counts": am._counts([c["rdn"]["status"] for c in cs]),
                                    "success": sum(c["success"] for c in cs), "n": len(cs),
                                    "truth_coverage": {**{k: sum(c["truth_covered"][k] for c in cs) for k in CLASSES}, "joint": sum(c["truth_covered"]["joint"] for c in cs)}}
                              for arm, cs in cert.items() if cs},
            "kappa": {k: _q([r["kappa"][k] for r in sr]) for k in ("exact", "A0", "A_s")},
            "rank_agreement": float(np.mean([r["rank_stability"]["agrees_with_truth"] for r in sr])),
            "sigma_min_pick_below_half": {k: sum(1 for r in sr if (r["sigma_min"]["ratio_of_picks"][k] or 1.0) < 0.5) for k in ("Q_s", "Q_f", "Q_o")},
            "support_extra_null_counts": am._counts([r["structural"]["support_extra_null"] for r in sr]),
        }
        if cls == "3a":
            fib = [r["fiber_A0"] for r in sr]
            by_class[cls]["fiber_coverage_A0"] = {k: sum(f["covered"][k] for f in fib) for k in ("truth_joint", "fiber_diag", "fiber_coh", "fiber_joint")}
            by_class[cls]["energy_B"] = _q([r["energy_B"] for r in sr])
    block["by_class"] = by_class
    block["picks"] = {"Q_s_equals_Q_f": sum(r["Q_s"] == r["Q_f"] for r in rows), "Q_o_equals_Q_s": sum(r["Q_o"] == r["Q_s"] for r in rows),
                      "Q_s_tied_at_fit": sum(len(r["Q_s_tied_at_fit"]) > 1 for r in rows)}
    block["rank_stability"] = {"agreement_rate": float(np.mean([r["rank_stability"]["agrees_with_truth"] for r in rows])),
                               "orbit_distance": _q([r["rank_stability"]["orbit_distance"] for r in rows])}
    block["unclassified"] = [r["index"] for r in rows if r["structural"]["unclassified"]]
    block["a0_h"] = {"registered_energy_max": max(r["a0_h"]["registered_energy"] for r in rows), "kernel_dimension_counts": am._counts([r["a0_h"]["kernel_dimension"] for r in rows])}
    return block


def e1_statement(h1: Mapping[str, Any] | None, h2: Mapping[str, Any] | None) -> str:
    p1 = bool(h1 and h1.get("primary") and h1["primary"]["passes"])
    p2 = bool(h2 and h2.get("primary") and h2["primary"]["passes"])
    if p1 and p2:
        return "e1_confirmed"
    if p1:
        return "mechanism_confirmed_operational_not_shown"
    if p2:
        return "operational_shown_mechanism_not_shown"
    return "not_confirmed"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _run_bank(config_dir: Path, bank_key: str, workers: int, progress) -> dict[str, Any]:
    started = time.perf_counter()
    ctx = _prepare(config_dir, bank_key)
    accepted, ledger = open_bank(ctx)
    for r in accepted:
        r["n_shot"] = ctx["n_shot"]
        am.attach_samples(ctx, r)
        attach_control_f(ctx, r)
    if progress is not None:
        progress(f"[{bank_key}] bank opened: generated {ledger['generated']}, accepted {ledger['accepted']}, shortfall {ledger['shortfall']}")
    values = ctx["values"]
    tol = float(values["added_context"]["tie_relative_tolerance"])
    bank_id, deficit = values["bank"]["bank_id"], values["deficit"]
    pool = None
    saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
    try:
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(Path.cwd()), str(config_dir.resolve()), bank_key))
        else:
            _W.clear()
            _W.update(ctx)
        truths = _execute(pool, _worker_job, [{"kind": "truth", "index": r["index"], "mask": r["mask"], "R": r["R"], "candidates": r["candidates"]} for r in accepted], f"[{bank_key}] truth points", progress)
        for r in accepted:
            r["Q_o"], r["Q_o_tied"] = am.argmax_with_ties({q: truths[("truth", r["index"])]["candidates"][q]["smin_rel"] for q in r["candidates"]}, r["candidates"], tol)
        wave1 = []
        for r in accepted:
            for v in dict.fromkeys(["support", "control", f"add:{r['Q_f']}", f"add:{r['Q_o']}"]):
                tables, n_shots = am.variant_tables(r, v)
                wave1.append({"kind": "fit", "index": r["index"], "variant": v, "tables": tables, "n_shots": n_shots})
            wave1.append({"kind": "fit_het", "index": r["index"], "variant": "control_f", "tables": r["control_f"]["tables"], "n_by_context": r["control_f"]["n_by_context"]})
            wave1.append({"kind": "fit_h", "index": r["index"], "variant": "h_known:support", "tables": r["support"], "n_shots": float(r["n_shot"])})
        fits = _execute(pool, _worker_job, wave1, f"[{bank_key}] fits (A0, C, C_F, A_f, A_o, A0_H)", progress)
        wave2 = []
        for r in accepted:
            V_hat = am._branch_vectors(fits[(r["index"], "support")]["fit"])
            r["s_Q_fit"] = {q: am.mac.orbit_invariant_sensitivity(V_hat, q, r["R"]) for q in r["candidates"]}
            r["Q_s"], r["Q_s_tied"] = am.argmax_with_ties(r["s_Q_fit"], r["candidates"], tol)
            if (r["index"], f"add:{r['Q_s']}") not in fits:
                tables, n_shots = am.variant_tables(r, f"add:{r['Q_s']}")
                wave2.append({"kind": "fit", "index": r["index"], "variant": f"add:{r['Q_s']}", "tables": tables, "n_shots": n_shots})
            wave2.append({"kind": "cert", "index": r["index"], "variant": "support", "table": fits[(r["index"], "support")]["table"], "support_query_hash": f"{bank_id}:{deficit}:{r['index']}"})
        second = _execute(pool, _worker_job, wave2, f"[{bank_key}] fits (A_s) and A0 certificates", progress)
        fits.update({k: v for k, v in second.items() if k[0] != "cert"})
        certs = {k: v for k, v in second.items() if k[0] == "cert"}
        wave3 = [{"kind": "cert", "index": r["index"], "variant": f"add:{r['Q_s']}", "table": fits[(r["index"], f"add:{r['Q_s']}")]["table"],
                  "support_query_hash": f"{bank_id}:{deficit}:{r['index']}:add:{r['Q_s']}"} for r in accepted]
        certs.update(_execute(pool, _worker_job, wave3, f"[{bank_key}] A_s certificates", progress))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if any(x.get("instance_hash") != ctx["instance_hash"] for x in [*truths.values(), *fits.values(), *certs.values()]):
        raise RuntimeError("worker-rebuilt constructions differ from the parent constructions")
    rows = [record_row(ctx, r, truths[("truth", r["index"])], fits, certs) for r in accepted]
    return {"ledger": ledger, "rows": rows, "delta_min": float(ctx["bodies"]["radii"]["diag"]),
            "execution": {"fits": len(fits), "certificates": len(certs), "fit_seconds": _summ([f["elapsed_seconds"] for f in fits.values()]),
                          "certificate_seconds": _summ([c["seconds"] for c in certs.values()]), "wall_seconds": time.perf_counter() - started}}


def run_m1_e1(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started = time.perf_counter()
    config_dir = Path(config_dir)
    run = _guard(config_dir, output_dir)
    values, declaration = run["values"], run["declaration"]
    p1 = load_p1_config(config_dir)
    if hashlib.sha256(json.dumps([list(m) for m in am.flagged_masks(p1.measurement)]).encode("ascii")).hexdigest()[:16] != declaration["flagged_mask_hash"]:
        raise RuntimeError("the flagged masks differ from the declaration")
    results = {b: _run_bank(config_dir, b, workers, progress) for b in BANKS}
    delta_min = float(declaration["delta_min"])
    if any(abs(res["delta_min"] - delta_min) > 0.0 for res in results.values()):
        raise RuntimeError("the frozen radius differs from the declared margin")
    banks = {b: analyse_bank(values, b, results[b]["rows"], delta_min) for b in BANKS}
    decision = {"delta_min": delta_min, "H1": banks["H1"].get("primary"), "H2": banks["H2"].get("primary"), "combination": "intersection_union",
                "statement": e1_statement(banks["H1"], banks["H2"])}
    artifact = {
        "artifact_type": "m1_e1_confirmatory", "status": values["status"], "protocol_document": values["protocol_document"],
        "decision": decision,
        "banks": {b: {"ledger": results[b]["ledger"], **banks[b], "execution": results[b]["execution"]} for b in BANKS},
        "records": {b: results[b]["rows"] for b in BANKS},
        "settings": {"banks": values["banks"], "added_context": values["added_context"], "control": values["control"], "control_f": values["control_f"],
                     "evaluation": values["evaluation"], "endpoints": values["endpoints"], "frozen_radii_low": am._bodies(Path.cwd(), values)["radii"],
                     "rdn_version": RDN_VERSION, "screen_version": SCREEN_VERSION, "bootstrap_index_seed_rule": "default_rng(sha256('{seed}:{bank}')[:16])"},
        "claim_boundary": {"sealed_banks_opened_once": values["status"] == "sealed_confirmatory", "screen_v1_changed": False,
                           "exact_states_used_for": "oracle class (H1 acceptance and H2 description), Q_o, truth s_Q, structural checks, errors, coverage, the constraint registry, the A0_H Hamiltonian, fiber coverage, orbit energy and sigma_min",
                           "information_levels": "A0, C, C_F, A_s, A_f are level i (no Hamiltonian); A0_H is level ii (registered H_tot); A_o is a truth oracle"},
        "execution": {"workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial", "wall_seconds": time.perf_counter() - started},
    }
    artifact = m1._jsonable(artifact)
    post = import_closure_snapshot(run["root"], extra_files=run["extra_files"])
    artifact["declaration_sha256"] = file_sha256(Path(run["folder"]) / DECLARATION)
    artifact["config_sha256"] = run["config_sha"]
    artifact["source_snapshot_hash_declared"] = declaration["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(declaration["source_snapshot"], post)
    (Path(run["folder"]) / ARTIFACT).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
