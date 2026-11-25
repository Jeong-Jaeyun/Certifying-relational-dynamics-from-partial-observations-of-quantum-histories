"""Model C v5 development run (P3 D1): ILGC-facing endpoints on fresh banks, no SDP.

Endpoints are the quantities ILGC calibration actually uses: per-class (diag/coh) block scores
||q_exact - q_hat||_2 over the full 20-context table, conformal radii and the anti-vacuity
sharpness gate. Model C v4 architecture is unchanged; its predictions pass observed contexts
through as smoothed support (the P1 kernel estimators' rule). Model A/B are fitted on the same
development fit bank as reference baselines.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.stats import p3_statistics as engine
from bi_qmn.core.config import load_p1_config
from bi_qmn.p1.p1_estimator import P1KernelFreeTableEstimator, P1MatchedGenericComparator
from bi_qmn.p1.model.p1_generator import seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import contexts_from_p1_measurement_config, full_p1_context_list
from bi_qmn.p3.p3_clock_graph import CONTROL_BUILDERS, ClockCoordinateGraph, degree_preserving_rewire_control
from bi_qmn.p3.p3_ilgc_facing import (
    CONTEXT_CLASSES,
    SupportPassthroughPredictor,
    body_report,
    class_dimensions,
    class_residuals,
    load_ilgc_targets,
    non_inferiority_margin,
    p1_record_from_p3,
    predict_full_table,
)
from bi_qmn.core.observations import ShotBatch, sample_shot_batch
from bi_qmn.p1.model.p1_banks import support_contexts_for_deficit
from bi_qmn.p1.model.p1_measurement import p1_measurement_table
from bi_qmn.p3.runs.p3_model_c_development_v2 import _coordinate_map, _decide, _derangement_support, _endpoint_summary
from bi_qmn.p3.p3_records import P3ObservationRecord
from bi_qmn.p3.models.p3_relational_dirac_v2 import null_support_table
from bi_qmn.p3.models.p3_relational_dirac_v4 import P3BispectralClockDiracEstimator


SPEC = harness.DevSpec(version="v5", config_file="p3_model_c_v5.yaml", runner="runners/p3/run_p3_model_c_development_v5.py")
MODEL_C_CONDITIONS = ("primary", "relation_permutation", "coordinate_permutation", "degree_preserving_rewire")
BASELINES = ("model_a", "model_b")
_WORKER_CONTEXT: dict[str, Any] = {}


def declare_p3_model_c_v5_development(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    values, _ = harness.load_values(SPEC, Path(config_dir))
    targets = load_ilgc_targets(config_dir, values["ilgc_targets"]["operational_config"])
    contexts = full_p1_context_list(load_p1_config(config_dir).measurement)
    dims = class_dimensions(contexts)
    margin = non_inferiority_margin(targets, dims, fraction=float(values["decision_rule"]["non_inferiority"]["margin_fraction_of_ceiling"]), reference_stratum=str(values["decision_rule"]["non_inferiority"]["reference_stratum"]))
    return harness.declare(SPEC, config_dir, output_dir, extra={"ilgc_targets": targets, "class_dimensions": dims, "non_inferiority_margin": margin})


# ---------------------------------------------------------------------------
# Compact development records (attempt 2)
#
# The v2 builder keeps every shot id as a Python int in a tuple (~10 MB per record). v5 attempt 1
# ran out of memory with 352 records in 7 processes. Shot ids are contiguous by construction
# (``sample_shot_batch`` assigns range(start, start + n)), so they are stored as ``range`` objects and
# support/target disjointness is checked by interval arithmetic. The RNG call sequence, tables,
# counts and probabilities are identical to ``p3_model_c_development_v2.build_p3_dev_records``.
# ---------------------------------------------------------------------------


def _compact(batch: ShotBatch) -> ShotBatch:
    ranges: dict[str, range] = {}
    for context_id, ids in batch.shot_ids.items():
        start, stop = int(ids[0]), int(ids[-1]) + 1
        if stop - start != len(ids):
            raise AssertionError("shot ids are not contiguous; compaction would lose information")
        ranges[context_id] = range(start, stop)
    return ShotBatch(table=batch.table, shot_ids=ranges, counts=batch.counts)


def compact_split_valid(record: P3ObservationRecord) -> bool:
    batches = (record.support, record.query_shot, record.query_comp, record.query_op)
    intervals: dict[str, list[tuple[int, int]]] = {}
    for batch in batches:
        for context_id, ids in batch.shot_ids.items():
            intervals.setdefault(context_id, []).append((ids.start, ids.stop))
    for spans in intervals.values():
        spans.sort()
        if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
            return False
    support_ids = set(record.support.table.probabilities)
    if set(record.query_shot.table.probabilities) != support_ids:
        return False
    if support_ids & set(record.query_comp.table.probabilities) or support_ids & set(record.query_op.table.probabilities):
        return False
    return not set(record.query_comp.table.probabilities) & set(record.query_op.table.probabilities)


def compact_target_index_manifest(records: list[P3ObservationRecord]) -> dict[str, Any]:
    digest = hashlib.sha256()
    channels = {"support": 0, "shot": 0, "comp": 0, "op": 0}
    for record in records:
        digest.update(f"{record.split}:{record.deficit}:{record.gamma}:{record.instance_index}:{record.replicate}".encode("ascii"))
        for name, batch in (("support", record.support), ("shot", record.query_shot), ("comp", record.query_comp), ("op", record.query_op)):
            digest.update(name.encode("ascii"))
            for context_id in sorted(batch.shot_ids):
                ids = batch.shot_ids[context_id]
                channels[name] += len(ids)
                digest.update(f"{context_id}:{ids.start}:{ids.stop}".encode("ascii"))
    return {
        "record_count": len(records),
        "shot_index_counts": channels,
        "target_index_hash": digest.hexdigest()[:16],
        "encoding": "contiguous_range_v1",
        "all_records_disjoint": all(compact_split_valid(r) for r in records),
    }


def build_compact_dev_records(instances, measurement_config: dict, *, role: str, bank_id: str, observation_seed: int, deficits: list[str]) -> list[P3ObservationRecord]:
    grouped = contexts_from_p1_measurement_config(measurement_config)
    contexts = full_p1_context_list(measurement_config)
    eta = float(measurement_config["noise"]["readout_eta"])
    records: list[P3ObservationRecord] = []
    record_index = 0
    for deficit in deficits:
        n_shot = int(measurement_config["observation_deficit"][deficit]["n_shot"])
        for instance_index, instance in enumerate(instances):
            exact = p1_measurement_table(
                instance.density,
                contexts,
                K=instance.K,
                eta=eta,
                metadata={"instance": instance.coefficient_label, "instance_index": instance_index, "gamma": 1.0, "deficit": deficit, "eta": eta, "bank_id": bank_id},
            ).copy_without_source()
            support_contexts = support_contexts_for_deficit(measurement_config, deficit, seed_key=f"p3v2:{bank_id}:{deficit}:{instance_index}")
            seed_material = f"{observation_seed}:{bank_id}:{deficit}:{instance_index}".encode("ascii")
            rng = np.random.default_rng(int(hashlib.sha256(seed_material).hexdigest()[:16], 16))
            start = record_index * 100_000_000
            record = P3ObservationRecord(
                split=role,
                deficit=deficit,
                gamma=1.0,
                instance_index=instance_index,
                replicate=0,
                exact_table=exact,
                support=_compact(sample_shot_batch(exact, support_contexts, n_shots=n_shot, rng=rng, shot_start=start, role="p3v2_support")),
                query_shot=_compact(sample_shot_batch(exact, support_contexts, n_shots=n_shot, rng=rng, shot_start=start + 20_000_000, role="p3v2_query_shot")),
                query_comp=_compact(sample_shot_batch(exact, grouped["compositional_heldout"], n_shots=n_shot, rng=rng, shot_start=start + 40_000_000, role="p3v2_query_comp")),
                query_op=_compact(sample_shot_batch(exact, grouped["operator_heldout"], n_shots=n_shot, rng=rng, shot_start=start + 60_000_000, role="p3v2_query_op")),
                metadata={"record_index": record_index, "n_shot": n_shot, "bank_id": bank_id, "model_c_coherent_core": True},
            )
            if not compact_split_valid(record):
                raise AssertionError("P3 v5 support and target indices overlap")
            records.append(record)
            record_index += 1
    return records


# ---------------------------------------------------------------------------
# Context and jobs
# ---------------------------------------------------------------------------


def _prepare_context(root: Path, config_dir: Path) -> dict[str, Any]:
    values, _ = harness.load_values(SPEC, config_dir)
    p1 = load_p1_config(config_dir)
    graph = harness.declared_graph(values, p1, _coordinate_map(root, values, p1))
    banks = harness.development_bank_specs(values)
    instances = {role: seeded_construction_bank_p1(p1.generator, spec) for role, spec in banks.items()}
    fit_records = build_compact_dev_records(instances["fit"], p1.measurement, role="dev_fit", bank_id=banks["fit"]["bank_id"], observation_seed=int(banks["fit"]["observation_seed"]), deficits=list(values["development_banks"]["fit_deficits"]))
    eval_records = build_compact_dev_records(instances["eval"], p1.measurement, role="dev_eval", bank_id=banks["eval"]["bank_id"], observation_seed=int(banks["eval"]["observation_seed"]), deficits=list(values["development_banks"]["eval_deficits"]))
    grouped = contexts_from_p1_measurement_config(p1.measurement)
    return {
        "values": values,
        "graph": graph,
        "fit_records": fit_records,
        "eval_records": eval_records,
        "donors": _derangement_support(eval_records),
        "input_contexts": grouped["input"],
        "contexts": full_p1_context_list(p1.measurement),
        "record_hashes": (compact_target_index_manifest(fit_records)["target_index_hash"], compact_target_index_manifest(eval_records)["target_index_hash"]),
    }


def _condition_graph(kind: str, declared: ClockCoordinateGraph, seed: int, swaps: int) -> ClockCoordinateGraph:
    if kind == "declared":
        return declared
    rng = np.random.default_rng(int(hashlib.sha256(f"p3v5-control:{kind}:{seed}".encode("ascii")).hexdigest()[:16], 16))
    if kind == "degree_preserving_rewire":
        return degree_preserving_rewire_control(declared, rng, swaps_per_edge=swaps)
    return CONTROL_BUILDERS[kind](declared, rng)


def _scores(model: Any, record: P3ObservationRecord, contexts: list, *, support_table=None, restrict_to: frozenset[str] | None = None) -> dict[str, dict[str, float]]:
    """Class scores; ``unobs`` uses the true support's unobserved set unless ``restrict_to`` is given."""
    table = record.support.table if support_table is None else support_table
    prediction = predict_full_table(model, table, contexts)
    observed = set(record.support.table.probabilities) if restrict_to is None else {c.context_id for c in contexts} - set(restrict_to)
    residuals = class_residuals(prediction, record.exact_table, contexts, observed_ids=observed)
    return {cls: {kind: float(np.linalg.norm(v)) for kind, v in parts.items()} for cls, parts in residuals.items()}


def _nll(model: Any, record: P3ObservationRecord, channel: str) -> float:
    batch = {"shot": record.query_shot, "comp": record.query_comp, "op": record.query_op}[channel]
    contexts = list(batch.table.contexts)
    table = predict_full_table(model, record.support.table, contexts)
    total, count = 0.0, 0.0
    for context in contexts:
        for outcome, n in batch.counts[context.context_id].items():
            total -= float(n) * float(np.log(max(float(table.probabilities[context.context_id][outcome]), 1.0e-15)))
            count += float(n)
    return total / count


def _evaluate(model: Any, context: Mapping[str, Any], *, with_full: Any | None = None, retention: bool = False) -> dict[str, Any]:
    contexts = context["contexts"]
    out: dict[str, Any] = {"scores": {}, "nll": {ch: {} for ch in ("shot", "comp", "op")}}
    if with_full is not None:
        out["full_scores"] = {}
    if retention:
        out["retention"] = {"support_null": {}, "support_permutation": {}}
    null_table = null_support_table()
    for record in context["eval_records"]:
        key = (record.deficit, record.instance_index)
        out["scores"][key] = _scores(model, record, contexts)
        for channel in ("shot", "comp", "op"):
            out["nll"][channel][key] = _nll(model, record, channel)
        if with_full is not None:
            out["full_scores"][key] = _scores(with_full, record, contexts)
        if retention:
            unobserved = frozenset(c.context_id for c in contexts if c.context_id not in record.support.table.probabilities)
            donor = context["donors"][key].support.table
            for name, table in (("support_null", null_table), ("support_permutation", donor)):
                substituted = _scores(model, record, contexts, support_table=table, restrict_to=unobserved)
                out["retention"][name][key] = {cls: substituted[cls]["unobs"] - out["scores"][key][cls]["unobs"] for cls in substituted}
    return out


def _fit_and_evaluate(context: Mapping[str, Any], job: tuple[str, int | None]) -> dict[str, Any]:
    name, seed = job
    values = context["values"]
    if name in BASELINES:
        cfg = values["baselines"][name]
        cls = P1KernelFreeTableEstimator if name == "model_a" else P1MatchedGenericComparator
        model = cls(context["input_contexts"], neighbor_count=int(cfg["neighbor_count"]), distance_power=float(cfg["distance_power"]), probability_floor=float(cfg["probability_floor"])).fit_records([p1_record_from_p3(r) for r in context["fit_records"]])
        result = _evaluate(model, context)
        result.update({"name": name, "seed": None, "manifest": {"checkpoint_hash": model.checkpoint_hash, "method": cfg["method"]}})
    else:
        spec = values["conditions"][name]
        settings = values["estimator"]
        estimator = P3BispectralClockDiracEstimator(
            context["input_contexts"],
            _condition_graph(str(spec["graph"]), context["graph"], int(seed), int(values["graph"]["rewire_swaps_per_edge"])),
            readout=str(settings["readout"]),
            bispectrum_epsilon=float(settings["bispectrum_epsilon"]),
            latent_dim=int(settings["latent_dim"]),
            lambda_dirac=float(settings["lambda_dirac"]),
            lambda_mask=float(settings["lambda_mask"]),
            lambda_proxy=float(settings["lambda_proxy"]),
            l2=float(settings["l2"]),
            learning_rate=float(settings["learning_rate"]),
            epochs=int(settings["epochs"]),
            init_seed=int(seed),
            propagate=bool(settings["propagate"]),
        ).fit_records(context["fit_records"])
        primary = name == "primary"
        result = _evaluate(SupportPassthroughPredictor(estimator), context, with_full=estimator if primary else None, retention=primary)
        result.update({"name": name, "seed": int(seed), "manifest": estimator.manifest()})
    result["record_hashes"] = context["record_hashes"]
    return result


def _worker_init(root: str, config_dir: str) -> None:
    os.chdir(root)
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(_prepare_context(Path(root), Path(config_dir)))


def _worker_job(job):
    return _fit_and_evaluate(_WORKER_CONTEXT, job)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _block_scores(scores: Mapping[tuple[str, int], Mapping[str, Mapping[str, float]]], strata: Mapping[str, str], kind: str = "all") -> dict[str, dict[str, dict[int, float]]]:
    out: dict[str, dict[str, dict[int, float]]] = {}
    for (deficit, block), by_class in scores.items():
        for cls, parts in by_class.items():
            out.setdefault(strata[deficit], {}).setdefault(cls, {})[int(block)] = float(parts[kind])
    return out


def _sharpness_gate(reports: list[dict], required: list[str], strata: Mapping[str, str], fraction: float) -> dict[str, Any]:
    cells = {}
    for deficit in required:
        stratum = strata[deficit]
        for cls in CONTEXT_CLASSES:
            passed = [bool(r[stratum][cls]["sharpness"]["qualified"]) for r in reports]
            values = [v for v in (r[stratum][cls]["sharpness"]["normalized_radius"] for r in reports) if v is not None]
            cells[f"{stratum}:{cls}"] = {
                "seed_pass_fraction": float(np.mean(passed)),
                "normalized_radius_median": float(np.median(values)) if values else None,
                "ceiling": reports[0][stratum][cls]["sharpness"]["ceiling"],
                "qualified": float(np.mean(passed)) >= fraction,
            }
    return {"cells": cells, "qualified": all(c["qualified"] for c in cells.values())}


def run_p3_model_c_v5_development(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1) -> dict:
    config_dir = Path(config_dir)
    run = harness.guard_run(SPEC, config_dir, output_dir)
    values, declaration = run["values"], run["declaration"]
    context = _prepare_context(run["root"], config_dir)
    if context["graph"].graph_hash != declaration["graphs"]["declared"]["graph_hash"]:
        raise RuntimeError("declared graph hash mismatch")
    seeds = [int(s) for s in values["init_seeds"]]
    jobs = [(name, seed) for seed in seeds for name in MODEL_C_CONDITIONS] + [(name, None) for name in BASELINES]
    results = harness.execute(jobs, workers=int(workers), serial_context=context, job_function=_fit_and_evaluate, worker_initializer=_worker_init, worker_job=_worker_job, initargs=(str(run["root"]), str(config_dir.resolve())))
    if any(r["record_hashes"] != context["record_hashes"] for r in results):
        raise RuntimeError("worker-rebuilt development records differ from the parent records")
    by = {(r["name"], r["seed"]): r for r in results}

    rule = values["decision_rule"]
    targets = declaration["ilgc_targets"]
    strata = targets["strata"]
    margin = declaration["non_inferiority_margin"]
    dims = declaration["class_dimensions"]
    eval_deficits = list(values["development_banks"]["eval_deficits"])
    required = list(rule["required_strata"])
    blocks = sorted({r.instance_index for r in context["eval_records"]})
    keys = list(by[("primary", seeds[0])]["scores"])

    summaries: dict[str, dict] = {}
    for control in rule["relational_controls"]:
        for cls in CONTEXT_CLASSES:
            deltas = {s: {k: by[(control, s)]["scores"][k][cls]["unobs"] - by[("primary", s)]["scores"][k][cls]["unobs"] for k in keys} for s in seeds}
            summaries[f"{control}:{cls}"] = _endpoint_summary(deltas, required, blocks, eval_deficits)
    for cls in CONTEXT_CLASSES:
        deltas = {s: {k: by[("model_a", None)]["scores"][k][cls]["all"] - by[("primary", s)]["scores"][k][cls]["all"] + float(margin[cls]) for k in keys} for s in seeds}
        summaries[f"non_inferiority_vs_model_a:{cls}"] = _endpoint_summary(deltas, required, blocks, eval_deficits)
    for name in ("support_null", "support_permutation"):
        for cls in CONTEXT_CLASSES:
            deltas = {s: {k: by[("primary", s)]["retention"][name][k][cls] for k in keys} for s in seeds}
            summaries[f"{name}:{cls}"] = _endpoint_summary(deltas, required, blocks, eval_deficits)

    relational = _decide(summaries, rule, [f"{c}:{cls}" for c in rule["relational_controls"] for cls in CONTEXT_CLASSES])
    non_inferiority = _decide(summaries, rule, [f"non_inferiority_vs_model_a:{cls}" for cls in CONTEXT_CLASSES])
    retention = _decide(summaries, rule, [f"{n}:{cls}" for n in ("support_null", "support_permutation") for cls in CONTEXT_CLASSES])
    stable_relational = [n for n, v in relational.items() if v["stable_direction"]]
    relational_status = "stable_direction_all" if len(stable_relational) == len(relational) else ("partial" if stable_relational else "not_established")

    split_seed = int(rule["coverage_split_seed"])
    body = {
        "model_c_passthrough": {str(s): body_report(_block_scores(by[("primary", s)]["scores"], strata), dims, targets, split_seed=split_seed) for s in seeds},
        "model_c_full": {str(s): body_report(_block_scores(by[("primary", s)]["full_scores"], strata), dims, targets, split_seed=split_seed) for s in seeds},
        "model_a": body_report(_block_scores(by[("model_a", None)]["scores"], strata), dims, targets, split_seed=split_seed),
        "model_b": body_report(_block_scores(by[("model_b", None)]["scores"], strata), dims, targets, split_seed=split_seed),
        **{f"{c}_passthrough": {str(s): body_report(_block_scores(by[(c, s)]["scores"], strata), dims, targets, split_seed=split_seed) for s in seeds} for c in rule["relational_controls"]},
    }
    sharp = _sharpness_gate(list(body["model_c_passthrough"].values()), required, strata, float(rule["sharpness_seed_pass_fraction_min"]))
    sharp_full = _sharpness_gate(list(body["model_c_full"].values()), required, strata, float(rule["sharpness_seed_pass_fraction_min"]))
    sharp_a = _sharpness_gate([body["model_a"]], required, strata, 1.0)
    gate = {
        "sharpness_model_c_passthrough": sharp["qualified"],
        "non_inferior_vs_model_a_both_classes": all(v["stable_direction"] for v in non_inferiority.values()),
        "retention_stable": all(v["stable_direction"] for v in retention.values()),
        "relational_at_least_partial": relational_status in {"partial", "stable_direction_all"},
    }

    def mean_over(result_key, name, seed_list):
        return {
            ch: {d: float(np.mean([by[(name, s)]["nll"][ch][(d, b)] for s in seed_list for b in blocks])) for d in eval_deficits}
            for ch in ("shot", "comp", "op")
        }

    nll_means = {name: mean_over("nll", name, seeds) for name in MODEL_C_CONDITIONS}
    nll_means.update({name: mean_over("nll", name, [None]) for name in BASELINES})
    artifact = {
        "artifact_type": "p3_model_c_v5_development",
        "status": "development_only",
        "model_family": values["model_family"],
        "coordinate_map": declaration["coordinate_map"],
        "graphs": declaration["graphs"],
        "ilgc_targets": targets,
        "class_dimensions": dims,
        "non_inferiority_margin": margin,
        "target_index_manifest": {"fit": compact_target_index_manifest(context["fit_records"]), "eval": compact_target_index_manifest(context["eval_records"])},
        "seeds": seeds,
        "execution": {"workers": int(workers), "mode": "process_pool_single_blas_thread" if int(workers) > 1 else "serial", "parallelism_changes_results": False},
        "model_manifests": {f"{r['name']}:{r['seed']}": r["manifest"] for r in results},
        "endpoint_summaries": summaries,
        "body_reports": body,
        "sharpness_gates": {"model_c_passthrough": sharp, "model_c_full_report_only": sharp_full, "model_a_report_only": sharp_a},
        "nll_means_report_only": nll_means,
        "decision": {
            "rule": rule,
            "relational_controls": relational,
            "non_inferiority_vs_model_a": non_inferiority,
            "retention": retention,
            "relational_status": relational_status,
            "d2_gate": gate,
            "ready_for_d2": all(gate.values()),
        },
        "claim_boundary": {
            "development_only": True,
            "sdp_or_certificate_run": False,
            "confirmatory_bank_opened": False,
            "reserved_namespace_touched": False,
            "exact_tables_used_for_training": False,
            "exact_tables_used_for_evaluation_only": True,
            "latent_graph_or_residual_is_physical": False,
        },
    }
    return harness.finalize(SPEC, run, artifact)
