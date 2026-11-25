"""Shared freeze/execute harness for P3 Model C development protocols (v5 onward).

Generalises the v4 pattern so each development version only defines its conditions and endpoints:

* declaration: config/protocol hashes, Stage-A coordinate map, declared graphs, fresh development
  banks with physical-fingerprint disjointness against every seeded bank in ``config/*.yaml``,
  reserved confirmatory namespace check, and an import-closure source snapshot;
* run guard: refuses on config, protocol, or loaded-source drift;
* execution: up to N worker processes with one BLAS thread each (spawned workers inherit the
  environment before importing numpy); a serial path is kept for tests and must give identical
  results.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from bi_qmn.core.config import load_p1_config
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p3.p3_clock_graph import ClockCoordinateGraph, build_clock_coordinate_graph
from bi_qmn.p3.runs.p3_model_c_development_v2 import _coordinate_map, physical_fingerprint
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash


BANK_KEYS = {"bank_id", "seed", "block_count", "dirichlet_concentration"}
BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class DevSpec:
    version: str
    config_file: str
    runner: str

    @property
    def declaration(self) -> str:
        return f"p3_model_c_{self.version}_development_declaration.json"

    @property
    def artifact(self) -> str:
        return f"p3_model_c_{self.version}_development.json"


def load_values(spec: DevSpec, config_dir: Path) -> tuple[dict[str, Any], str]:
    path = Path(config_dir) / spec.config_file
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if values.get("status") != "development_only" or float(values["coherent_core_gamma"]) != 1.0:
        raise ValueError(f"Model C {spec.version} runner is restricted to development-only gamma=1 work")
    return values, file_sha256(path)


def destination(values: Mapping[str, Any], output_dir: str | Path) -> Path:
    return Path(output_dir) / Path(str(values["output_root"])).name


def declared_bank_specs(config_dir: Path, *, exclude: str) -> list[tuple[str, dict]]:
    specs: list[tuple[str, dict]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if BANK_KEYS <= set(node):
                specs.append((path, node))
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    for filename in sorted(glob.glob(str(Path(config_dir) / "*.yaml"))):
        if Path(filename).name != exclude:
            walk(yaml.safe_load(Path(filename).read_text(encoding="utf-8")), Path(filename).name)
    return specs


def development_bank_specs(values: Mapping[str, Any]) -> dict[str, dict]:
    return {role: dict(spec) for role, spec in values["development_banks"].items() if isinstance(spec, dict) and BANK_KEYS <= set(spec)}


def declared_graph(values: Mapping[str, Any], p1, coordinate_map) -> ClockCoordinateGraph:
    scale = p1.measurement["scale"]
    anchors = list(range(coordinate_map.K)) if values["graph"]["anchors"] == "all" else [int(v) for v in values["graph"]["anchors"]]
    return build_clock_coordinate_graph(coordinate_map, anchors=anchors, local_window=[int(v) for v in scale["local_window"]], h_max=int(scale["h_max"]))


def _config_files(spec: DevSpec, config_dir: Path, values: Mapping[str, Any]) -> list[str]:
    names = [spec.config_file, "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", *values.get("additional_config_files", [])]
    return [str(Path(config_dir) / name) for name in names]


def declare(spec: DevSpec, config_dir: str | Path, output_dir: str | Path, *, extra: Mapping[str, Any] | None = None) -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = load_values(spec, config_dir)
    target = destination(values, output_dir) / spec.declaration
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    p1 = load_p1_config(config_dir)
    coordinate_map = _coordinate_map(root, values, p1)
    graph = declared_graph(values, p1, coordinate_map)
    banks = development_bank_specs(values)
    dev = {role: sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, bank)) for role, bank in banks.items()}
    known: dict[str, set[str]] = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked: dict[str, str] = {}
    for path, bank in declared_bank_specs(config_dir, exclude=spec.config_file):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, bank)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    union: set[str] = set().union(*dev.values()) if dev else set()
    overlaps = {name: sorted(union & fps) for name, fps in known.items() if union & fps}
    roles = list(dev)
    internal = any(set(dev[a]) & set(dev[b]) for i, a in enumerate(roles) for b in roles[i + 1 :])
    if overlaps or internal:
        raise ValueError(f"development banks are not disjoint: {sorted(overlaps)} internal={internal}")
    reserved = values["reserved_future_confirmatory_namespace"]["seed_range"]
    for bank in banks.values():
        for key in ("seed", "observation_seed"):
            if int(reserved[0]) <= int(bank[key]) <= int(reserved[1]):
                raise ValueError("development seeds intrude on the reserved confirmatory namespace")
    snapshot = import_closure_snapshot(root, extra_files=(*_config_files(spec, config_dir, values), str(values["protocol_document"]), spec.runner))
    declaration = {
        "artifact_type": f"p3_model_c_{spec.version}_development_declaration",
        "status": "development_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_sha,
        "protocol_document": str(values["protocol_document"]),
        "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
        "decision_rule": values["decision_rule"],
        "conditions": values["conditions"],
        "init_seeds": values["init_seeds"],
        "coordinate_map": {"selected_clock": coordinate_map.selected_clock, "K": coordinate_map.K, "stage_a_sha256": coordinate_map.stage_a_sha256, "map_hash": coordinate_map.map_hash},
        "graphs": {"declared": {"graph_hash": graph.graph_hash, **graph.capacity_signature()}},
        "development_banks": {
            role: {**banks[role], "construction_fingerprint_hash": hashlib.sha256("".join(fps).encode("ascii")).hexdigest()[:16], "observation_tables_opened": False}
            for role, fps in dev.items()
        },
        "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps, "development_banks_mutually_disjoint": True},
        "reserved_future_confirmatory_namespace": values["reserved_future_confirmatory_namespace"],
        "source_snapshot": snapshot,
        "source_snapshot_hash": snapshot_hash(snapshot),
        **dict(extra or {}),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return declaration


def guard_run(spec: DevSpec, config_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Load the frozen declaration and refuse to run on any drift."""
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = load_values(spec, config_dir)
    folder = destination(values, output_dir)
    declaration = json.loads((folder / spec.declaration).read_text(encoding="utf-8"))
    if (folder / spec.artifact).exists():
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


def execute(
    jobs: list,
    *,
    workers: int,
    serial_context: Mapping[str, Any],
    job_function: Callable[[Mapping[str, Any], Any], Any],
    worker_initializer: Callable[..., None],
    worker_job: Callable[[Any], Any],
    initargs: tuple,
) -> list:
    if workers <= 1:
        return [job_function(serial_context, job) for job in jobs]
    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    try:
        for name in BLAS_THREAD_VARIABLES:
            os.environ[name] = "1"
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_initializer, initargs=initargs) as executor:
            return list(executor.map(worker_job, jobs))
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def finalize(spec: DevSpec, run: Mapping[str, Any], artifact: dict) -> dict:
    post = import_closure_snapshot(run["root"], extra_files=run["extra_files"])
    artifact["declaration_sha256"] = file_sha256(Path(run["folder"]) / spec.declaration)
    artifact["config_sha256"] = run["config_sha"]
    artifact["source_snapshot_hash_declared"] = run["declaration"]["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(run["declaration"]["source_snapshot"], post)
    (Path(run["folder"]) / spec.artifact).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
