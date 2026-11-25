"""E3: support-only rules for the single added D context, compared with every candidate fitted (development).

Protocol: documents/M1-E3-rule-comparison-protocol.md (frozen by the declaration). A new development bank
(enriched 512 type-3a + 128 type-3b records). Every candidate context is fitted, so every rule's choice is
evaluated on the same records. Rules: ``s_Q`` (reference), ``sigma_min``, ``sigma_min_chi`` (Fisher-whitened),
``local`` (tangent at the fit), ``fixed``, ``random``; ``oracle`` and ``hindsight`` are truth-based references.
The declared carry-forward rule decides which rule the undeclared E1 preregistration uses.
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
import yaml

from bi_qmn.core.config import load_p1_config
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import full_p1_context_list, p1_effect_map
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairEstimator
from bi_qmn.p3.p3_support_screen import SCREEN_VERSION, SupportIdentifiabilityScreen
from bi_qmn.p3.runs import m1_added_measurement as am
from bi_qmn.p3.runs import m1_added_measurement_posthoc as ph
from bi_qmn.p3.runs import m1_evidence_package as ep
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _keys, _summ, _true_branch_vectors
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint


CONFIG_FILE = "m1_e3_rules.yaml"
RUNNER = "runners/p3/run_m1_e3_rules.py"
DECLARATION = "m1_e3_rules_declaration.json"
ARTIFACT = "m1_e3_rules.json"
SUPPORT_RULES = ("s_Q", "sigma_min", "sigma_min_chi", "local", "fixed", "random")
RULES = (*SUPPORT_RULES, "oracle", "hindsight")
COMPARISONS = (("fixed", "s_Q"), ("sigma_min", "s_Q"), ("sigma_min_chi", "s_Q"), ("random", "s_Q"), ("local", "s_Q"), ("s_Q", "oracle"), ("sigma_min_chi", "sigma_min"))
CHALLENGERS = ("sigma_min", "sigma_min_chi")
CLASSES = ("diag", "coh")
GENERATIVE_KEYS = ("dirichlet_concentration", "minimum_mode_probability", "phase_range")
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
    if values.get("status") != "development_only" or float(values["coherent_core_gamma"]) != 1.0:
        raise ValueError("E3 is a development-only gamma=1 run")
    dev = _yaml(Path(config_dir) / values["design_source"]["development_config"])
    if values["estimator"] != dev["estimator"]:
        raise ValueError("estimator settings differ from the D1-v7 development config")
    if values["screen"] != dev["screen"] or values["screen"]["method"] != SCREEN_VERSION or not values["screen"]["pauli_commutant"]:
        raise ValueError(f"the screen must be the frozen {SCREEN_VERSION} with the commutant check")
    if values["deficit"] != "low":
        raise ValueError("the flagged masks are low masks")
    bank, reference = values["bank"], dev["development_banks"]["evaluation"]
    if any(bank[k] != reference[k] for k in GENERATIVE_KEYS):
        raise ValueError("the bank must use the D1-v7 generative specification")
    if set(bank["quotas"]) != set(am.STRATA) or any(int(v) < 0 for v in bank["quotas"].values()):
        raise ValueError("quotas are declared for 3a and 3b")
    lo, hi = values["reserved_future_confirmatory_namespace"]["seed_range"]
    seeds = [bank["seed"], bank["observation_seed"], bank["mask_seed"], values["evaluation"]["bootstrap"]["seed"], values["added_context"]["random_rule_seed"]]
    if any(int(lo) <= int(s) <= int(hi) for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("E3 seeds must be distinct and outside the reserved confirmatory namespace")
    if [tuple(c) for c in values["comparisons"]] != list(COMPARISONS) or set(values["rules"]) != set(RULES):
        raise ValueError("the declared rules or comparisons were changed")
    if tuple(values["carry_forward"]["challengers"]) != CHALLENGERS or values["carry_forward"]["default"] != "s_Q":
        raise ValueError("the declared carry-forward rule was changed")
    return values, file_sha256(path)


def _prepare(config_dir: Path) -> dict[str, Any]:
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
        raise ValueError("the added context gets the support's per-context shots")
    return {
        "values": values, "p1": p1, "K": K, "d_C": d_C, "d_S": d_S, "n": len(p1.generator["active_modes"]),
        "eta": float(p1.measurement["noise"]["readout_eta"]), "theta_con": float(p1.generator["thresholds"]["theta_con"]),
        "instances": instances, "instance_hash": hashlib.sha256("".join(physical_fingerprint(i) for i in instances).encode("ascii")).hexdigest()[:16],
        "contexts": contexts, "context_by_id": {c.context_id: c for c in contexts}, "ids": [c.context_id for c in contexts],
        "effects": {(cid, o): M for cid, row in effects_by_context.items() for o, M in row.items()},
        "sbp": sbp, "fw": sbp.forward,
        "screen": SupportIdentifiabilityScreen(p1.generator["active_modes"], K, epsilon=float(values["screen"]["disconnected_branch_null_tail_epsilon"])),
        "flagged": am.flagged_masks(p1.measurement),
        "pool": sorted({cid for m in m1.deficit_masks(p1.measurement, "low") for cid in m if cid.startswith("D,")}),
        "n_shot": n_shot, "diag_keys": _keys(contexts, "diag", K),
    }


def _extra_files(config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    names = [CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", str(values["ilgc_targets"]["operational_config"]), str(values["design_source"]["development_config"])]
    return [str(Path(config_dir) / n) for n in dict.fromkeys(names)] + [*values["design_source"]["dev_inputs"], str(values["protocol_document"]), RUNNER]


def declare_m1_e3(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
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
        raise ValueError(f"the E3 bank is not disjoint: {sorted(overlaps)}")
    flagged = am.flagged_masks(p1.measurement)
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values))
    declaration = {
        "artifact_type": "m1_e3_rules_declaration", "status": values["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha, "protocol_document": str(values["protocol_document"]), "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
        "dev_inputs": dev_inputs, "estimator": values["estimator"], "screen": values["screen"],
        "bank": {**values["bank"], "construction_fingerprint_hash": hashlib.sha256("".join(prints).encode("ascii")).hexdigest()[:16], "observation_tables_opened": False},
        "flagged_mask_count": len(flagged), "flagged_mask_hash": hashlib.sha256(json.dumps([list(m) for m in flagged]).encode("ascii")).hexdigest()[:16],
        "rules": values["rules"], "comparisons": values["comparisons"], "carry_forward": values["carry_forward"], "evaluation": values["evaluation"],
        "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps},
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
        raise FileExistsError("the E3 artifact already exists and is immutable")
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
# Rule criteria (pure)
# ---------------------------------------------------------------------------


def criteria(ctx: Mapping[str, Any], V: np.ndarray, R: str, candidates: Sequence[str]) -> dict[str, dict[str, float]]:
    """Every support-computable criterion of every candidate at branch vectors ``V`` (fitted or true)."""
    G = m1.pauli_matrix(R)
    n = float(ctx["n_shot"])
    out: dict[str, dict[str, float]] = {k: {} for k in ("s_Q", "sigma_min", "sigma_min_chi", "local", "noise")}
    for q in candidates:
        cm = ph.coefficient_matrix(ctx["fw"], V, G, q, n)
        W = 1.0 / np.sqrt(np.maximum(cm["T0"], 1.0e-6))
        Mw = np.column_stack([W * cm["a"], W * cm["b"]])
        out["s_Q"][q] = float(am.mac.orbit_invariant_sensitivity(V, q, R))
        out["sigma_min"][q] = cm["sigma_min"]
        out["sigma_min_chi"][q] = float(np.linalg.svd(Mw, compute_uv=False)[-1] * math.sqrt(n))
        out["local"][q] = cm["local_tangent"]
        out["noise"][q] = cm["noise_l2"]
    return out


def detector_score(V: np.ndarray, mask: Sequence[str], R: str) -> float | None:
    """Support-only 3a score: largest branch component of an observed F Pauli anticommuting with R (small for 3a)."""
    comp = ep.branch_components(np.asarray(V, dtype=complex), mask, R)
    vals = [math.hypot(x["P"], x["iPR"]) for rows in comp.values() for x in rows]
    return max(vals) if vals else None


def picks_for(ctx: Mapping[str, Any], r: Mapping[str, Any], crit_fit: Mapping[str, Mapping[str, float]], oracle_scores: Mapping[str, float]) -> dict[str, Any]:
    tol = float(ctx["values"]["added_context"]["tie_relative_tolerance"])
    cands = list(r["candidates"])
    picks, ties = {}, {}
    for rule in ("s_Q", "sigma_min", "sigma_min_chi", "local"):
        picks[rule], ties[rule] = am.argmax_with_ties(crit_fit[rule], cands, tol)
    picks["fixed"], ties["fixed"] = cands[0], [cands[0]]
    rng = np.random.default_rng(_seed(f"{ctx['values']['added_context']['random_rule_seed']}:{ctx['values']['bank']['bank_id']}:{int(r['index'])}"))
    picks["random"] = cands[int(rng.integers(0, len(cands)))]
    ties["random"] = [picks["random"]]
    picks["oracle"], ties["oracle"] = am.argmax_with_ties(dict(oracle_scores), cands, tol)
    return {"picks": picks, "ties": ties}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


_JOBS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]] = {"truth": am._truth_job, "fit": am._fit_job}


def _worker_init(root: str, config_dir: str) -> None:
    os.chdir(root)
    _W.clear()
    _W.update(_prepare(Path(config_dir)))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    out = _JOBS[job["kind"]](_W, job)
    out["instance_hash"] = _W["instance_hash"]
    return out


# ---------------------------------------------------------------------------
# Rows and analysis
# ---------------------------------------------------------------------------


def record_row(ctx: Mapping[str, Any], r: Mapping[str, Any], truth: Mapping[str, Any], fits: Mapping[tuple, Mapping[str, Any]]) -> dict[str, Any]:
    i = int(r["index"])
    cands = list(r["candidates"])
    ev = ctx["values"]["evaluation"]
    exact = r["exact"]
    tables = {q: fits[(i, f"add:{q}")]["table"] for q in cands}
    rest = [c for c in ctx["ids"] if c not in set(r["mask"])]
    E_all = [c for c in rest if c not in set(cands)]
    V_hat = am._branch_vectors(fits[(i, "support")]["fit"])
    V = _true_branch_vectors(ctx["instances"][i])
    crit_fit = criteria(ctx, V_hat, r["R"], cands)
    crit_true = criteria(ctx, V, r["R"], cands)
    oracle_scores = {q: float(truth["candidates"][q]["smin_rel"]) for q in cands}
    sel = picks_for(ctx, r, crit_fit, oracle_scores)
    err_all = {q: am.class_errors(ctx, tables[q], exact, E_all) for q in cands}
    err_own = {q: am.class_errors(ctx, tables[q], exact, [c for c in rest if c != q]) for q in cands}
    sel["picks"]["hindsight"] = min(cands, key=lambda q: (err_all[q]["diag"], cands.index(q)))
    sel["ties"]["hindsight"] = [sel["picks"]["hindsight"]]
    picks = sel["picks"]
    pairwise = {}
    for a, b in COMPARISONS:
        E_ab = [c for c in rest if c not in {picks[a], picks[b]}]
        pairwise[f"{a}|{b}"] = {a: am.class_errors(ctx, tables[picks[a]], exact, E_ab), b: am.class_errors(ctx, tables[picks[b]], exact, E_ab), "set_size": len(E_ab)}
    tol = float(ctx["values"]["added_context"]["tie_relative_tolerance"])
    agreement = {}
    for rule in ("s_Q", "sigma_min", "sigma_min_chi", "local"):
        _, tied_truth = am.argmax_with_ties(crit_true[rule], cands, tol)
        agreement[rule] = picks[rule] in tied_truth
    near = {q: bool(2.0 * crit_true["sigma_min"][q] < crit_true["noise"][q]) for q in cands}
    thr = float(ev["large_failure_threshold_own_diag"])
    return {
        "index": i, "class": r["class"], "unitary": r["unitary"], "mask": r["mask"], "R": r["R"], "candidates": cands,
        "picks": picks, "ties": sel["ties"], "criteria_fit": crit_fit, "criteria_truth": crit_true, "oracle_scores": oracle_scores,
        "errors_E_all": err_all, "errors_own": err_own, "pairwise": pairwise, "E_all_size": len(E_all),
        "agreement_with_truth": agreement, "near_degenerate": near,
        "large_failure_by_rule": {rule: bool(err_own[picks[rule]]["diag"] > thr) for rule in RULES},
        "detector_score": detector_score(V_hat, r["mask"], r["R"]),
        "a0_error_E_all": am.class_errors(ctx, fits[(i, "support")]["table"], exact, E_all),
        "structural": {"support_extra_null": truth["support_extra_null"], "candidate_extra_null": {q: truth["candidates"][q]["extra_null"] for q in cands}},
    }


def _q(v: Sequence[float]) -> dict[str, float] | None:
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return {k: float(np.quantile(v, float(k))) for k in ("0.0", "0.1", "0.5", "0.9", "1.0")} if v else None


def auc(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (pos.size * neg.size))


def carry_forward(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Declared rule: a challenger replaces s_Q only if CVaR(s_Q) - CVaR(challenger) has its interval above 0."""
    qualified = {}
    for ch in CHALLENGERS:
        d = comparisons[f"{ch}|s_Q"]                     # CVaR(ch) - CVaR(s_Q)
        if d["upper"] < 0.0:                               # i.e. CVaR(s_Q) - CVaR(ch) interval above 0
            qualified[ch] = -d["point"]
    if not qualified:
        return {"rule": "s_Q", "qualified_challengers": {}}
    return {"rule": max(qualified, key=lambda k: qualified[k]), "qualified_challengers": qualified}


def analyse(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ev = values["evaluation"]
    level, boot, primary = float(ev["statistic"]["level"]), ev["bootstrap"], str(ev["primary_class"])
    out: dict[str, Any] = {}
    for stratum in am.STRATA:
        sr = [r for r in rows if r["class"] == stratum]
        n = len(sr)
        block: dict[str, Any] = {"n": n}
        if n == 0:
            out[stratum] = block
            continue
        rng = np.random.default_rng(_seed(f"{boot['seed']}:{stratum}"))
        index = rng.integers(0, n, size=(int(boot["resamples"]), n))
        comps = {}
        for a, b in COMPARISONS:
            key = f"{a}|{b}"
            comps[key] = am.paired_cvar_difference([r["pairwise"][key][a][primary] for r in sr], [r["pairwise"][key][b][primary] for r in sr], index, level, boot["interval"])
        block["comparisons_pairwise"] = comps
        table = {}
        for rule in RULES:
            for cls in CLASSES:
                v = [r["errors_E_all"][r["picks"][rule]][cls] for r in sr]
                table[f"{rule}:{cls}"] = {"cvar": am.cvar(v, level), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "max": float(np.max(v))}
            vo = [r["errors_own"][r["picks"][rule]]["diag"] for r in sr]
            table[f"{rule}:own_diag"] = {"cvar": am.cvar(vo, level), "median": float(np.median(vo)), "max": float(np.max(vo))}
        a0 = [r["a0_error_E_all"]["diag"] for r in sr]
        table["A0:diag"] = {"cvar": am.cvar(a0, level), "median": float(np.median(a0)), "p90": float(np.quantile(a0, 0.9)), "max": float(np.max(a0))}
        block["all_rule_table_E_all"] = table
        block["large_failures"] = {rule: sum(r["large_failure_by_rule"][rule] for r in sr) for rule in RULES}
        nd_records = [r for r in sr if any(r["near_degenerate"].values())]
        block["near_degenerate"] = {
            "candidates": sum(sum(r["near_degenerate"].values()) for r in sr), "candidates_total": sum(len(r["candidates"]) for r in sr), "records_with_any": len(nd_records),
            "picked": {rule: sum(r["near_degenerate"][r["picks"][rule]] for r in sr) for rule in RULES},
            "picked_and_failed": {rule: sum(r["near_degenerate"][r["picks"][rule]] and r["large_failure_by_rule"][rule] for r in sr) for rule in RULES},
        }
        block["agreement_with_truth"] = {rule: float(np.mean([r["agreement_with_truth"][rule] for r in sr])) for rule in ("s_Q", "sigma_min", "sigma_min_chi", "local")}
        block["tie_rates"] = {rule: float(np.mean([len(r["ties"][rule]) > 1 for r in sr])) for rule in ("s_Q", "sigma_min", "sigma_min_chi", "local", "oracle")}
        block["pick_agreement"] = {f"{a}=={b}": sum(r["picks"][a] == r["picks"][b] for r in sr) for a, b in (("s_Q", "sigma_min"), ("s_Q", "sigma_min_chi"), ("sigma_min", "sigma_min_chi"), ("s_Q", "oracle"), ("s_Q", "hindsight"), ("sigma_min_chi", "hindsight"))}
        block["structural"] = {"support_extra_null_counts": am._counts([r["structural"]["support_extra_null"] for r in sr])}
        out[stratum] = block
    scores = {s: [r["detector_score"] for r in rows if r["class"] == s and r["detector_score"] is not None] for s in am.STRATA}
    decision = {"carry_forward": carry_forward(out["3a"]["comparisons_pairwise"]) if out["3a"].get("n") else {"rule": "s_Q", "qualified_challengers": {}}}
    return {"strata": out, "decision": decision,
            "detector_3a_vs_3b": {"auc_3b_scores_above_3a": auc(scores["3b"], scores["3a"]), "score_3a": _q(scores["3a"]), "score_3b": _q(scores["3b"]),
                                  "note": "enriched bank: not a natural-population precision"}}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_m1_e3(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print) -> dict:
    started = time.perf_counter()
    config_dir = Path(config_dir)
    run = _guard(config_dir, output_dir)
    values, declaration = run["values"], run["declaration"]
    ctx = _prepare(config_dir)
    if hashlib.sha256(json.dumps([list(m) for m in ctx["flagged"]]).encode("ascii")).hexdigest()[:16] != declaration["flagged_mask_hash"]:
        raise RuntimeError("the flagged masks differ from the declaration")
    accepted, ledger = am.open_bank(ctx)
    for r in accepted:
        r["n_shot"] = ctx["n_shot"]
        am.attach_samples(ctx, r)
    if progress is not None:
        progress(f"bank opened: generated {ledger['generated']}, accepted {ledger['accepted']}, quota filled {ledger['quota_filled']}")
    pool = None
    saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
    try:
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(run["root"]), str(config_dir.resolve())))
        else:
            _W.clear()
            _W.update(ctx)
        jobs = [{"kind": "truth", "index": r["index"], "mask": r["mask"], "R": r["R"], "candidates": r["candidates"]} for r in accepted]
        for r in accepted:
            for v in ["support", *[f"add:{q}" for q in r["candidates"]]]:
                tables, n_shots = am.variant_tables(r, v)
                jobs.append({"kind": "fit", "index": r["index"], "variant": v, "tables": tables, "n_shots": n_shots})
        results = _execute(pool, _worker_job, jobs, "truth points and fits (support + every candidate)", progress)
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
    truths = {k: v for k, v in results.items() if k[0] == "truth"}
    fits = {k: v for k, v in results.items() if k[0] != "truth"}
    rows = [record_row(ctx, r, truths[("truth", r["index"])], fits) for r in accepted]
    analysis = analyse(values, rows)
    artifact = {
        "artifact_type": "m1_e3_rules", "status": "development_only", "protocol_document": values["protocol_document"],
        "decision": analysis["decision"], "bank_ledger": ledger, "strata": analysis["strata"], "detector_3a_vs_3b": analysis["detector_3a_vs_3b"],
        "settings": {"bank": values["bank"], "added_context": values["added_context"], "rules": values["rules"], "comparisons": values["comparisons"],
                     "carry_forward": values["carry_forward"], "evaluation": values["evaluation"], "checkpoint_hash": ctx["sbp"].checkpoint_hash, "candidate_pool": ctx["pool"]},
        "records": rows,
        "execution": {"workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial", "fits": len(fits), "truth_points": len(truths),
                      "fit_seconds": _summ([f["elapsed_seconds"] for f in fits.values()]), "wall_seconds": time.perf_counter() - started},
        "claim_boundary": {"development_only": True, "sealed_bank_opened": False,
                           "exact_states_used_for": "class label, oracle, hindsight, errors, near-degeneracy, criterion agreement; never a support-only rule"},
    }
    artifact = m1._jsonable(artifact)
    post = import_closure_snapshot(run["root"], extra_files=run["extra_files"])
    artifact["declaration_sha256"] = file_sha256(Path(run["folder"]) / DECLARATION)
    artifact["config_sha256"] = run["config_sha"]
    artifact["source_snapshot_hash_declared"] = declaration["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(declaration["source_snapshot"], post)
    (Path(run["folder"]) / ARTIFACT).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
