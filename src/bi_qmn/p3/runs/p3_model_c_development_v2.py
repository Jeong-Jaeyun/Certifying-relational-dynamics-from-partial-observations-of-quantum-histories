"""Model C v2 development run: Stage-A clock graph, capacity-matched relation controls, fresh dev bank.

Two steps, mirroring the project's freeze discipline:

1. ``declare_p3_model_c_v2_development`` freezes config, protocol document, decision rule, graph,
   bank construction fingerprints (no observation tables) and the import-closure source snapshot.
2. ``run_p3_model_c_v2_development`` refuses to run on any drift, opens only the declared
   development banks, fits all seeds x conditions with common random numbers, and evaluates the
   predeclared decision rule with the v2 statistical engine.

Everything here is development-only (P3 D1). No body, selector, atlas/path, oracle or confirmatory
object is created.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bi_qmn.stats import p3_statistics as engine
from bi_qmn.core.config import load_p1_config
from bi_qmn.core.observations import sample_shot_batch
from bi_qmn.p1.model.p1_banks import support_contexts_for_deficit
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import contexts_from_p1_measurement_config, full_p1_context_list, p1_measurement_table
from bi_qmn.p3.p3_clock_graph import (
    CONTROL_BUILDERS,
    ClockCoordinateGraph,
    build_clock_coordinate_graph,
    clock_coordinate_map_from_stage_a,
    degree_preserving_rewire_control,
)
from bi_qmn.p3.p3_records import P3ObservationRecord, p3_target_index_manifest, p3_target_split_valid
from bi_qmn.p3.models.p3_relational_dirac import finite_relation_generator
from bi_qmn.p3.models.p3_relational_dirac_v2 import P3ClockRelationalDiracEstimator, null_support_table
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash


CONFIG_FILE = "p3_model_c_v2.yaml"
DECLARATION = "p3_model_c_v2_development_declaration.json"
ARTIFACT = "p3_model_c_v2_development.json"
RUNNER = "runners/p3/run_p3_model_c_development_v2.py"
_BANK_KEYS = {"bank_id", "seed", "block_count", "dirichlet_concentration"}


def _load(config_dir: Path) -> tuple[dict[str, Any], str]:
    path = config_dir / CONFIG_FILE
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if values.get("status") != "development_only" or float(values["coherent_core_gamma"]) != 1.0:
        raise ValueError("Model C v2 runner is restricted to development-only gamma=1 work")
    return values, file_sha256(path)


def physical_fingerprint(instance) -> str:
    unitary = np.asarray(instance.system_unitary, dtype=complex)
    payload = {
        "active_modes": [int(v) for v in instance.active_modes],
        "coefficients": [[round(float(v.real), 12), round(float(v.imag), 12)] for v in instance.coefficients],
        "system_unitary": [[[round(float(v.real), 12), round(float(v.imag), 12)] for v in row] for row in unitary],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _declared_bank_specs(config_dir: Path) -> list[tuple[str, dict]]:
    specs: list[tuple[str, dict]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if _BANK_KEYS <= set(node):
                specs.append((path, node))
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    for filename in sorted(glob.glob(str(config_dir / "*.yaml"))):
        if Path(filename).name == CONFIG_FILE:
            continue
        walk(yaml.safe_load(Path(filename).read_text(encoding="utf-8")), Path(filename).name)
    return specs


def _graph(values: dict, p1, coordinate_map) -> ClockCoordinateGraph:
    graph_config = values["graph"]
    scale = p1.measurement["scale"]
    return build_clock_coordinate_graph(
        coordinate_map,
        anchors=[int(v) for v in graph_config["anchors"]],
        local_window=[int(v) for v in scale["local_window"]],
        h_max=int(scale["h_max"]),
    )


def _coordinate_map(root: Path, values: dict, p1):
    stage_a_path = root / str(values["stage_a_artifact"])
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    return clock_coordinate_map_from_stage_a(
        stage_a,
        stage_a_path=str(values["stage_a_artifact"]),
        stage_a_sha256=file_sha256(stage_a_path),
        generator_config=p1.generator,
    )


def _dev_bank_specs(values: dict) -> dict[str, dict]:
    return {role: dict(spec) for role, spec in values["development_banks"].items() if role in {"fit", "eval"}}


def declare_p3_model_c_v2_development(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    destination = Path(output_dir) / Path(values["output_root"]).name
    path = destination / DECLARATION
    if path.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {path}")
    p1 = load_p1_config(config_dir)
    coordinate_map = _coordinate_map(root, values, p1)
    graph = _graph(values, p1, coordinate_map)

    dev_specs = _dev_bank_specs(values)
    dev_banks = {role: seeded_construction_bank_p1(p1.generator, spec) for role, spec in dev_specs.items()}
    dev_fingerprints = {role: sorted(physical_fingerprint(i) for i in bank) for role, bank in dev_banks.items()}
    known: dict[str, set[str]] = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked: dict[str, str] = {}
    for spec_path, spec in _declared_bank_specs(config_dir):
        try:
            known[spec_path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[spec_path] = repr(error)
    dev_all = set(dev_fingerprints["fit"]) | set(dev_fingerprints["eval"])
    overlaps = {name: sorted(dev_all & fps) for name, fps in known.items() if dev_all & fps}
    if overlaps or set(dev_fingerprints["fit"]) & set(dev_fingerprints["eval"]):
        raise ValueError(f"development banks are not disjoint from declared banks: {sorted(overlaps)}")
    reserved = values["reserved_future_confirmatory_namespace"]["seed_range"]
    for spec in dev_specs.values():
        for key in ("seed", "observation_seed"):
            if int(reserved[0]) <= int(spec[key]) <= int(reserved[1]):
                raise ValueError("development seeds intrude on the reserved confirmatory namespace")

    protocol_path = root / str(values["protocol_document"])
    config_files = [str(config_dir / name) for name in (CONFIG_FILE, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml")]
    snapshot = import_closure_snapshot(root, extra_files=(*config_files, str(values["protocol_document"]), RUNNER))
    declaration = {
        "artifact_type": "p3_model_c_v2_development_declaration",
        "status": "development_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha,
        "protocol_document": str(values["protocol_document"]),
        "protocol_document_sha256": file_sha256(protocol_path),
        "decision_rule": values["decision_rule"],
        "conditions": values["conditions"],
        "init_seeds": values["init_seeds"],
        "coordinate_map": {
            "selected_clock": coordinate_map.selected_clock,
            "clock_qubits": list(coordinate_map.clock_qubits),
            "system_qubits": list(coordinate_map.system_qubits),
            "K": coordinate_map.K,
            "stage_a_sha256": coordinate_map.stage_a_sha256,
            "map_hash": coordinate_map.map_hash,
        },
        "declared_graph": {"graph_hash": graph.graph_hash, **graph.capacity_signature()},
        "development_banks": {
            role: {
                **dev_specs[role],
                "construction_fingerprint_hash": hashlib.sha256("".join(fps).encode("ascii")).hexdigest()[:16],
                "observation_tables_opened": False,
            }
            for role, fps in dev_fingerprints.items()
        },
        "disjointness": {
            "checked_bank_count": len(known),
            "unchecked_bank_specs": unchecked,
            "overlaps": overlaps,
            "fit_eval_disjoint": True,
            "seed_collision_note": "53xxx seed namespace unused by any other config at declaration time",
        },
        "reserved_future_confirmatory_namespace": values["reserved_future_confirmatory_namespace"],
        "source_snapshot": snapshot,
        "source_snapshot_hash": snapshot_hash(snapshot),
        "v1_generator_audit": {
            "v1_forward_equals_backward_at_latent_dim_2": bool(
                np.allclose(finite_relation_generator("clock_forward", 2), finite_relation_generator("clock_backward", 2))
            ),
            "consequence": "v1 edge_relation_scramble differed from primary only through the graph-hash-derived "
            "initialisation seed",
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return declaration


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def build_p3_dev_records(instances, measurement_config: dict, *, role: str, bank_id: str, observation_seed: int, deficits: list[str]) -> list[P3ObservationRecord]:
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
            )
            exact = exact.copy_without_source()
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
                support=sample_shot_batch(exact, support_contexts, n_shots=n_shot, rng=rng, shot_start=start, role="p3v2_support"),
                query_shot=sample_shot_batch(exact, support_contexts, n_shots=n_shot, rng=rng, shot_start=start + 20_000_000, role="p3v2_query_shot"),
                query_comp=sample_shot_batch(exact, grouped["compositional_heldout"], n_shots=n_shot, rng=rng, shot_start=start + 40_000_000, role="p3v2_query_comp"),
                query_op=sample_shot_batch(exact, grouped["operator_heldout"], n_shots=n_shot, rng=rng, shot_start=start + 60_000_000, role="p3v2_query_op"),
                metadata={"record_index": record_index, "n_shot": n_shot, "bank_id": bank_id, "model_c_coherent_core": True},
            )
            if not p3_target_split_valid(record):
                raise AssertionError("P3 v2 support and target indices overlap")
            records.append(record)
            record_index += 1
    return records


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def _condition_graph(name: str, spec: dict, declared: ClockCoordinateGraph, seed: int, swaps_per_edge: int) -> ClockCoordinateGraph:
    kind = spec["graph"]
    if kind == "declared":
        return declared
    rng = np.random.default_rng(int(hashlib.sha256(f"p3v2-control:{kind}:{seed}".encode("ascii")).hexdigest()[:16], 16))
    if kind == "degree_preserving_rewire":
        return degree_preserving_rewire_control(declared, rng, swaps_per_edge=swaps_per_edge)
    return CONTROL_BUILDERS[kind](declared, rng)


def _derangement_support(records: list[P3ObservationRecord]) -> dict[tuple[str, int], P3ObservationRecord]:
    """Cyclic shift within each deficit stratum; the donor is always a different construction."""
    mapping: dict[tuple[str, int], P3ObservationRecord] = {}
    for deficit in sorted({record.deficit for record in records}):
        stratum = sorted((r for r in records if r.deficit == deficit), key=lambda r: r.instance_index)
        for position, record in enumerate(stratum):
            donor = stratum[(position + 1) % len(stratum)]
            if donor.instance_index == record.instance_index:
                raise AssertionError("support permutation must cross constructions")
            mapping[(deficit, record.instance_index)] = donor
    return mapping


def _endpoint_summary(values: dict[int, dict[tuple[str, int], float]], strata: list[str], blocks: list[int], all_strata: list[str]) -> dict:
    """values[seed][(deficit, block)] -> delta; positive favours the primary/true-support model."""
    seeds = sorted(values)
    block_deltas = [float(np.mean([values[s][(d, b)] for s in seeds for d in strata])) for b in blocks]
    seed_deltas = {str(s): float(np.mean([values[s][(d, b)] for b in blocks for d in strata])) for s in seeds}
    stratum_deltas = {d: float(np.mean([values[s][(d, b)] for s in seeds for b in blocks])) for d in all_strata}
    return {"block_deltas": block_deltas, "seed_deltas": seed_deltas, "stratum_deltas": stratum_deltas}


def _decide(summaries: dict[str, dict], rule: dict, family: list[str]) -> dict:
    tests = {
        name: engine.sign_flip_test(summaries[name]["block_deltas"], alternative="greater", mc_draws=int(rule["mc_draws"]), seed=int(rule["test_seed"]))
        for name in family
    }
    rejected = engine.holm({name: tests[name]["p_value"] for name in family}, alpha=float(rule["family_alpha"]))
    verdicts = {}
    for name in family:
        seed_fraction = float(np.mean([value > 0.0 for value in summaries[name]["seed_deltas"].values()]))
        strata_ok = all(summaries[name]["stratum_deltas"][d] > 0.0 for d in rule["required_strata"])
        verdicts[name] = {
            "test": tests[name],
            "holm_rejected": bool(rejected[name]),
            "seed_positive_fraction": seed_fraction,
            "required_strata_positive": bool(strata_ok),
            "stable_direction": bool(rejected[name] and seed_fraction >= float(rule["seed_positive_fraction_min"]) and strata_ok),
        }
    return verdicts


def run_p3_model_c_v2_development(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1) -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    destination = Path(output_dir) / Path(values["output_root"]).name
    declaration = json.loads((destination / DECLARATION).read_text(encoding="utf-8"))
    if (destination / ARTIFACT).exists():
        raise FileExistsError("development artifact already exists and is immutable")
    if declaration["config_sha256"] != config_sha:
        raise RuntimeError("config drifted after declaration")
    if declaration["protocol_document_sha256"] != file_sha256(root / str(values["protocol_document"])):
        raise RuntimeError("protocol document drifted after declaration")
    pre_run = import_closure_snapshot(root, extra_files=[path for path in declaration["source_snapshot"] if not path.startswith("src/")])
    drift = snapshot_drift(declaration["source_snapshot"], pre_run)
    if drift["changed"] or drift["missing"]:
        raise RuntimeError(f"source drift after declaration: {drift}")

    p1 = load_p1_config(config_dir)
    coordinate_map = _coordinate_map(root, values, p1)
    declared_graph = _graph(values, p1, coordinate_map)
    if declared_graph.graph_hash != declaration["declared_graph"]["graph_hash"]:
        raise RuntimeError("declared graph hash mismatch")
    specs = _dev_bank_specs(values)
    banks = {role: seeded_construction_bank_p1(p1.generator, spec) for role, spec in specs.items()}
    fit_records = build_p3_dev_records(banks["fit"], p1.measurement, role="dev_fit", bank_id=specs["fit"]["bank_id"], observation_seed=int(specs["fit"]["observation_seed"]), deficits=list(values["development_banks"]["fit_deficits"]))
    eval_deficits = list(values["development_banks"]["eval_deficits"])
    eval_records = build_p3_dev_records(banks["eval"], p1.measurement, role="dev_eval", bank_id=specs["eval"]["bank_id"], observation_seed=int(specs["eval"]["observation_seed"]), deficits=eval_deficits)
    input_contexts = contexts_from_p1_measurement_config(p1.measurement)["input"]
    settings = values["estimator"]
    swaps = int(values["graph"]["rewire_swaps_per_edge"])

    jobs = []
    for seed in values["init_seeds"]:
        for name, spec in values["conditions"].items():
            jobs.append((int(seed), name, spec))

    def fit(job):
        seed, name, spec = job
        graph = _condition_graph(name, spec, declared_graph, seed, swaps)
        estimator = P3ClockRelationalDiracEstimator(
            input_contexts,
            graph,
            latent_dim=int(settings["latent_dim"]),
            lambda_dirac=float(spec["lambda_dirac"]),
            lambda_mask=float(settings["lambda_mask"]),
            lambda_proxy=float(settings["lambda_proxy"]),
            l2=float(settings["l2"]),
            learning_rate=float(settings["learning_rate"]),
            epochs=int(settings["epochs"]),
            init_seed=seed,
        ).fit_records(fit_records)
        return job, estimator

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fitted = list(executor.map(fit, jobs))
    else:
        fitted = [fit(job) for job in jobs]

    donors = _derangement_support(eval_records)
    null_table = null_support_table()
    channels = ("comp", "shot", "op")
    nll: dict[str, dict[int, dict[str, dict[tuple[str, int], float]]]] = {}
    retention: dict[str, dict[int, dict[tuple[str, int], float]]] = {"support_null": {}, "support_permutation": {}}
    manifests: dict[str, dict[str, Any]] = {}
    for (seed, name, _), estimator in fitted:
        per_channel = {channel: {(r.deficit, r.instance_index): estimator.channel_nll(r, channel) for r in eval_records} for channel in channels}
        nll.setdefault(name, {})[seed] = per_channel
        manifests.setdefault(name, {})[str(seed)] = estimator.manifest()
        if name == "primary":
            for key, table_for in (("support_null", lambda r: null_table), ("support_permutation", lambda r: donors[(r.deficit, r.instance_index)].support.table)):
                retention[key][seed] = {
                    (r.deficit, r.instance_index): estimator.channel_nll(r, "comp", support_table=table_for(r)) - per_channel["comp"][(r.deficit, r.instance_index)]
                    for r in eval_records
                }

    rule = values["decision_rule"]
    blocks = sorted({r.instance_index for r in eval_records})
    required = list(rule["required_strata"])
    primary_comp = {seed: nll["primary"][seed]["comp"] for seed in nll["primary"]}
    summaries: dict[str, dict] = {}
    for name in values["conditions"]:
        if name == "primary":
            continue
        deltas = {seed: {key: nll[name][seed]["comp"][key] - primary_comp[seed][key] for key in primary_comp[seed]} for seed in nll[name]}
        summaries[name] = _endpoint_summary(deltas, required, blocks, eval_deficits)
    for name, deltas in retention.items():
        summaries[name] = _endpoint_summary(deltas, required, blocks, eval_deficits)

    relational = _decide(summaries, rule, list(rule["relational_controls"]))
    retained = _decide(summaries, rule, list(rule["retention_diagnostics"]))
    report_only = {
        name: {
            "test": engine.sign_flip_test(summaries[name]["block_deltas"], alternative="greater", mc_draws=int(rule["mc_draws"]), seed=int(rule["test_seed"])),
            "seed_positive_fraction": float(np.mean([v > 0.0 for v in summaries[name]["seed_deltas"].values()])),
            "stratum_deltas": summaries[name]["stratum_deltas"],
            "interpretation": "positive delta means the condition has higher comp NLL than the primary",
        }
        for name in rule["report_only_conditions"]
    }
    stable_relational = [name for name, verdict in relational.items() if verdict["stable_direction"]]
    relational_status = (
        "stable_direction_all_relational_controls"
        if len(stable_relational) == len(relational)
        else ("partial" if stable_relational else "not_established")
    )
    retention_status = "stable" if all(v["stable_direction"] for v in retained.values()) else ("partial" if any(v["stable_direction"] for v in retained.values()) else "not_established")

    channel_means = {
        name: {
            channel: {
                deficit: float(np.mean([nll[name][s][channel][(deficit, b)] for s in nll[name] for b in blocks]))
                for deficit in eval_deficits
            }
            for channel in channels
        }
        for name in nll
    }
    post_run = import_closure_snapshot(root, extra_files=[path for path in declaration["source_snapshot"] if not path.startswith("src/")])
    artifact = {
        "artifact_type": "p3_model_c_v2_development",
        "status": "development_only",
        "declaration_sha256": file_sha256(destination / DECLARATION),
        "config_sha256": config_sha,
        "model_family": values["model_family"],
        "coordinate_map": declaration["coordinate_map"],
        "declared_graph": declaration["declared_graph"],
        "target_index_manifest": {"fit": p3_target_index_manifest(fit_records), "eval": p3_target_index_manifest(eval_records)},
        "seeds": [int(s) for s in values["init_seeds"]],
        "conditions": values["conditions"],
        "model_manifests": manifests,
        "channel_mean_nll": channel_means,
        "endpoint_summaries": summaries,
        "decision": {
            "rule": rule,
            "relational_controls": relational,
            "retention_diagnostics": retained,
            "report_only_conditions": report_only,
            "relational_status": relational_status,
            "retention_status": retention_status,
            "ready_to_draft_fresh_p3_preregistration": relational_status == "stable_direction_all_relational_controls" and retention_status == "stable",
        },
        "source_snapshot_hash_declared": declaration["source_snapshot_hash"],
        "source_snapshot_post_run_drift": snapshot_drift(declaration["source_snapshot"], post_run),
        "claim_boundary": {
            "development_only": True,
            "confirmatory_bank_opened": False,
            "reserved_namespace_touched": False,
            "certificate_or_selector_optimized": False,
            "atlas_or_path_optimized": False,
            "gamma_lt_1_excluded_from_coherent_core": True,
            "op_target_report_only": True,
            "latent_graph_or_residual_is_physical": False,
        },
    }
    (destination / ARTIFACT).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
