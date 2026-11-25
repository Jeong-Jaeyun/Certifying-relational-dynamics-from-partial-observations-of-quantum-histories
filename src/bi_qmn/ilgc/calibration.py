from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bi_qmn.p0.adequacy import adequacy_gates
from bi_qmn.core.config import (
    P0Config,
    P0ILGCOperationalConfig,
    load_p0_config,
    load_p0_ilgc_operational_config,
)
from bi_qmn.p0.estimator import FreeMeasurementTableEstimator
from bi_qmn.core.generator import construction_bank, construction_fingerprint
from bi_qmn.core.measurement import contexts_from_measurement_config, full_context_list
from bi_qmn.core.observations import (
    ObservationRecord,
    build_observation_records,
    records_by_split,
)
from bi_qmn.core.types import MeasurementContext, to_jsonable
from .model import (
    bind_measurement_observation,
    effective_measurement_map_from_p0,
    observation_table_from_p0,
)
from .types import (
    CalibrationBody,
    calibration_bodies_hash,
    stable_hash,
)


CALIBRATION_ARTIFACT = "p0_ilgc_calibration.json"


def _cell_key(stratum: str, context_class: str) -> str:
    return f"{stratum}:{context_class}"


@dataclass(frozen=True)
class CalibrationCell:
    stratum: str
    context_class: str
    body: CalibrationBody
    block_scores: tuple[tuple[int, float], ...]
    alpha: float
    conformal_order_index: int
    conformal_radius: float | None
    finite_sample_coverage_ceiling: float
    qualified: bool
    qualification_reasons: tuple[str, ...]
    measurement_map_hash: str

    def to_manifest(self) -> dict[str, Any]:
        body_manifest = self.body.to_manifest()
        body_manifest["whitening"] = np.asarray(
            self.body.whitening,
            dtype=float,
        ).tolist()
        return {
            "stratum": self.stratum,
            "context_class": self.context_class,
            "body": body_manifest,
            "block_scores": [
                {"construction_block": block, "score": score}
                for block, score in self.block_scores
            ],
            "block_score_hash": stable_hash(self.block_scores),
            "construction_block_count": len(self.block_scores),
            "alpha": self.alpha,
            "requested_coverage": 1.0 - self.alpha,
            "finite_sample_coverage_ceiling": self.finite_sample_coverage_ceiling,
            "minimum_blocks_for_requested_coverage": int(
                math.ceil((1.0 - self.alpha) / self.alpha - 1.0e-12)
            ),
            "conformal_order_index": self.conformal_order_index,
            "conformal_radius": self.conformal_radius,
            "operational_radius": self.body.radius,
            "qualified": self.qualified,
            "qualification_reasons": list(self.qualification_reasons),
            "measurement_map_hash": self.measurement_map_hash,
            "whitening_condition_number": float(
                np.linalg.cond(np.asarray(self.body.whitening, dtype=float))
            ),
        }


@dataclass(frozen=True)
class ILGCCalibrationBank:
    cells: Mapping[str, CalibrationCell]
    estimator_checkpoint_hash: str
    calibration_config_hash: str
    p0_config_hash: str
    split_manifest: Mapping[str, Any]
    simultaneous_global_alpha: float
    coverage_allocation: str

    @property
    def cells_qualified(self) -> bool:
        return bool(self.cells) and all(cell.qualified for cell in self.cells.values())

    @property
    def split_qualified(self) -> bool:
        return bool(self.split_manifest.get("four_way_disjoint", False))

    @property
    def qualified(self) -> bool:
        return bool(
            self.cells_qualified
            and self.split_qualified
            and self.simultaneous_coverage_qualified
        )

    @property
    def simultaneous_coverage_qualified(self) -> bool:
        class_alphas = {
            cell.context_class: cell.alpha for cell in self.cells.values()
        }
        return bool(
            self.coverage_allocation == "bonferroni_context_classes_v1"
            and 0.0 < self.simultaneous_global_alpha < 1.0
            and sum(class_alphas.values()) <= self.simultaneous_global_alpha + 1.0e-15
        )

    def bodies_for_stratum(
        self,
        stratum: str,
        *,
        require_qualified: bool = True,
    ) -> dict[str, CalibrationBody]:
        selected = {
            cell.context_class: cell.body
            for cell in self.cells.values()
            if cell.stratum == stratum
        }
        if not selected:
            raise KeyError(f"no ILGC calibration cells for stratum {stratum}")
        if require_qualified and not self.split_qualified:
            raise ValueError("ILGC calibration bank violates four-way split disjointness")
        if require_qualified:
            unqualified = [
                cell.context_class
                for cell in self.cells.values()
                if cell.stratum == stratum and not cell.qualified
            ]
            if unqualified:
                raise ValueError(
                    f"ILGC calibration stratum {stratum} is unqualified: {sorted(unqualified)}"
                )
        return selected

    def body_set_hash(self, stratum: str) -> str:
        return calibration_bodies_hash(
            self.bodies_for_stratum(stratum, require_qualified=False)
        )

    @property
    def manifest_hash(self) -> str:
        return stable_hash(
            {
                "checkpoint_hash": self.estimator_checkpoint_hash,
                "calibration_config_hash": self.calibration_config_hash,
                "p0_config_hash": self.p0_config_hash,
                "split_manifest": dict(self.split_manifest),
                "simultaneous_global_alpha": self.simultaneous_global_alpha,
                "coverage_allocation": self.coverage_allocation,
                "cells": {
                    key: cell.to_manifest() for key, cell in sorted(self.cells.items())
                },
            }
        )

    def to_manifest(self) -> dict[str, Any]:
        strata = sorted({cell.stratum for cell in self.cells.values()})
        return {
            "calibration_manifest_hash": self.manifest_hash,
            "qualified": self.qualified,
            "cells_qualified": self.cells_qualified,
            "split_qualified": self.split_qualified,
            "simultaneous_coverage": {
                "qualified": self.simultaneous_coverage_qualified,
                "allocation": self.coverage_allocation,
                "global_alpha": self.simultaneous_global_alpha,
                "guaranteed_coverage": 1.0 - self.simultaneous_global_alpha,
                "allocated_alpha_sum": sum(
                    {
                        cell.context_class: cell.alpha for cell in self.cells.values()
                    }.values()
                ),
            },
            "estimator_checkpoint_hash": self.estimator_checkpoint_hash,
            "calibration_config_hash": self.calibration_config_hash,
            "p0_config_hash": self.p0_config_hash,
            "split_manifest": dict(self.split_manifest),
            "body_set_hashes": {
                stratum: self.body_set_hash(stratum) for stratum in strata
            },
            "cells": {
                key: cell.to_manifest() for key, cell in sorted(self.cells.items())
            },
        }


@dataclass(frozen=True)
class P0ILGCCalibrationContext:
    p0: P0Config
    operational: P0ILGCOperationalConfig
    estimator: FreeMeasurementTableEstimator
    records: tuple[ObservationRecord, ...]
    calibration_bank: ILGCCalibrationBank
    contexts: tuple[MeasurementContext, ...]
    calibration_generator_config: Mapping[str, Any]
    reference_construction_instances: tuple[Any, ...]


def _record_split_manifest(
    records: list[ObservationRecord],
    instances: list[Any],
    *,
    source_bank_id: str,
) -> tuple[dict[str, Any], set[str]]:
    indices = {record.instance_index for record in records}
    fingerprints = {
        construction_fingerprint(instances[record.instance_index]) for record in records
    }
    payload = {
        "source_bank_id": source_bank_id,
        "construction_indices": sorted(indices),
        "construction_fingerprints": sorted(fingerprints),
        "record_count": len(records),
        "record_ids": sorted(int(record.metadata["record_index"]) for record in records),
        "seeds": sorted(int(record.metadata["seed"]) for record in records),
    }
    payload["split_hash"] = stable_hash(payload)
    return payload, fingerprints


def _split_manifest(
    observation_records: list[ObservationRecord],
    calibration_records: list[ObservationRecord],
    observation_instances: list[Any],
    calibration_instances: list[Any],
    reference_instances: list[Any],
    *,
    calibration_bank_id: str,
    reference_bank_id: str,
) -> dict[str, Any]:
    train, train_ids = _record_split_manifest(
        records_by_split(observation_records, "train"),
        observation_instances,
        source_bank_id="declared_observation_bank",
    )
    calibration, calibration_ids = _record_split_manifest(
        calibration_records,
        calibration_instances,
        source_bank_id=calibration_bank_id,
    )
    test, test_ids = _record_split_manifest(
        records_by_split(observation_records, "test"),
        observation_instances,
        source_bank_id="declared_observation_bank",
    )
    reference_ids = {construction_fingerprint(instance) for instance in reference_instances}
    pairwise_disjoint = (
        train_ids.isdisjoint(calibration_ids)
        and train_ids.isdisjoint(test_ids)
        and calibration_ids.isdisjoint(test_ids)
    )
    construction_disjoint = reference_ids.isdisjoint(
        train_ids | calibration_ids | test_ids
    )
    return {
        "construction_bank": {
            "source_bank_id": reference_bank_id,
            "construction_indices": list(range(len(reference_instances))),
            "construction_fingerprints": sorted(reference_ids),
            "exact_simulator_access": True,
            "construction_bank_hash": stable_hash(sorted(reference_ids)),
        },
        "splits": {"train": train, "calibration": calibration, "test": test},
        "construction_indices_pairwise_disjoint": pairwise_disjoint,
        "construction_bank_disjoint_from_observation_splits": construction_disjoint,
        "four_way_disjoint": bool(pairwise_disjoint and construction_disjoint),
        "no_overlap_assertions": {
            "train_calibration": train_ids.isdisjoint(calibration_ids),
            "train_test": train_ids.isdisjoint(test_ids),
            "calibration_test": calibration_ids.isdisjoint(test_ids),
            "construction_observations": construction_disjoint,
        },
        "threshold_source_split": "calibration",
        "test_records_used_for_calibration": 0,
        "legacy_observation_calibration_records_used": 0,
    }


def _record_residuals_by_class(
    record: ObservationRecord,
    estimator: FreeMeasurementTableEstimator,
    contexts: list[MeasurementContext],
    K: int,
) -> tuple[dict[str, np.ndarray], dict[str, int], str]:
    qhat = estimator.predict_table(
        record.support.table,
        contexts,
        metadata={
            "eta": record.eta,
            "regime": record.regime,
            "instance_index": record.instance_index,
            "replicate": record.replicate,
            "record_index": record.metadata["record_index"],
            "estimated": True,
        },
    )
    measurement_map = effective_measurement_map_from_p0(
        qhat.contexts,
        K=K,
        eta=float(record.eta),
    )
    observation = observation_table_from_p0(
        qhat,
        measurement_map,
        stratum=record.regime,
        estimator_checkpoint_hash=estimator.checkpoint_hash,
        support_query_hash=stable_hash(
            {
                "record_index": record.metadata["record_index"],
                "support_table": record.support.table.metadata,
            }
        ),
    )
    model = bind_measurement_observation(
        measurement_map,
        observation,
        binding_mode="frozen_map_estimator_observation",
    )
    residuals: dict[str, list[float]] = defaultdict(list)
    for entry in model.entries:
        truth = float(
            record.exact_table.probabilities[entry.context_id][entry.outcome]
        )
        residuals[entry.context_class].append(truth - entry.target)
    arrays = {key: np.asarray(value, dtype=float) for key, value in residuals.items()}
    sizes = {key: int(value.size) for key, value in arrays.items()}
    return arrays, sizes, model.effective_map_hash


def fit_calibration_bank(
    calibration_records: list[ObservationRecord],
    estimator: FreeMeasurementTableEstimator,
    contexts: list[MeasurementContext],
    generator_config: dict[str, Any],
    operational_config: P0ILGCOperationalConfig,
    *,
    p0_config_hash: str,
    split_manifest: Mapping[str, Any],
) -> ILGCCalibrationBank:
    if not calibration_records:
        raise ValueError("ILGC calibration requires calibration records")
    if any(record.split != "calibration" for record in calibration_records):
        raise ValueError("ILGC calibration received a non-calibration record")

    cfg = operational_config.values["calibration_body"]
    if cfg["whitening"]["method"] != "identity_weighting_v1":
        raise ValueError("ILGC v1.1 currently supports identity_weighting_v1")
    min_blocks = int(cfg["minimum_construction_blocks"])
    radius_floor = float(cfg["positive_radius_floor"])
    inflation = float(cfg["descriptive_radius_inflation"])
    simultaneous_global_alpha = float(cfg["simultaneous_global_alpha"])
    coverage_allocation = str(cfg["coverage_allocation"])
    if min_blocks <= 0 or radius_floor <= 0.0 or inflation < 1.0:
        raise ValueError("invalid ILGC calibration-body configuration")
    allocated_alpha = sum(float(value) for value in cfg["alpha"].values())
    if (
        coverage_allocation != "bonferroni_context_classes_v1"
        or not 0.0 < simultaneous_global_alpha < 1.0
        or allocated_alpha > simultaneous_global_alpha + 1.0e-15
    ):
        raise ValueError("invalid simultaneous coverage allocation")

    instances = construction_bank(generator_config)
    per_block_vectors: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    sizes_by_cell: dict[tuple[str, str], int] = {}
    map_hashes_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in calibration_records:
        residuals, sizes, map_hash = _record_residuals_by_class(
            record,
            estimator,
            contexts,
            instances[record.instance_index].K,
        )
        for context_class, residual in residuals.items():
            cell = (record.regime, context_class)
            prior_size = sizes_by_cell.setdefault(cell, sizes[context_class])
            if prior_size != sizes[context_class]:
                raise ValueError(f"calibration cell {cell} changed dimension")
            map_hashes_by_cell[cell].add(map_hash)
            per_block_vectors[(record.regime, context_class, record.instance_index)].append(
                residual
            )

    block_scores: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (stratum, context_class, block), vectors in per_block_vectors.items():
        block_scores[(stratum, context_class)][block] = max(
            float(np.linalg.norm(vector, ord=2)) for vector in vectors
        )

    cells: dict[str, CalibrationCell] = {}
    for (stratum, context_class), scores_by_block in sorted(block_scores.items()):
        alpha = float(cfg["alpha"][context_class])
        if not 0.0 < alpha < 1.0:
            raise ValueError("calibration alpha must be in (0, 1)")
        ordered_scores = sorted(scores_by_block.items())
        score_values = sorted(score for _, score in ordered_scores)
        n_blocks = len(score_values)
        order_index = int(math.ceil((n_blocks + 1) * (1.0 - alpha)))
        finite_sample_ceiling = n_blocks / (n_blocks + 1.0)
        reasons: list[str] = []
        if n_blocks < min_blocks:
            reasons.append("insufficient_construction_blocks")
        if order_index > n_blocks:
            reasons.append("requested_coverage_exceeds_finite_sample_ceiling")
        conformal_radius = (
            None if order_index > n_blocks else float(score_values[order_index - 1])
        )
        if conformal_radius is not None and conformal_radius <= 0.0:
            reasons.append("zero_radius_not_supported_by_ilgc_v1_1")
        if len(map_hashes_by_cell[(stratum, context_class)]) != 1:
            reasons.append("nonexchangeable_measurement_maps_within_cell")
        qualified = not reasons
        descriptive_radius = max(radius_floor, max(score_values) * inflation)
        operational_radius = (
            float(conformal_radius) if qualified and conformal_radius is not None
            else descriptive_radius
        )
        calibration_hash = stable_hash(
            {
                "method": cfg["method"],
                "checkpoint_hash": estimator.checkpoint_hash,
                "stratum": stratum,
                "context_class": context_class,
                "alpha": alpha,
                "minimum_construction_blocks": min_blocks,
                "block_scores": ordered_scores,
                "qualified": qualified,
            }
        )
        size = sizes_by_cell[(stratum, context_class)]
        body = CalibrationBody(
            context_class=context_class,
            radius=operational_radius,
            whitening=np.eye(size, dtype=float),
            calibration_hash=calibration_hash,
            stratum=stratum,
            role=(
                "split_conformal_calibrated_body"
                if qualified
                else "descriptive_unqualified_body"
            ),
        )
        body.validate(size)
        cells[_cell_key(stratum, context_class)] = CalibrationCell(
            stratum=stratum,
            context_class=context_class,
            body=body,
            block_scores=tuple(ordered_scores),
            alpha=alpha,
            conformal_order_index=order_index,
            conformal_radius=conformal_radius,
            finite_sample_coverage_ceiling=finite_sample_ceiling,
            qualified=qualified,
            qualification_reasons=tuple(reasons),
            measurement_map_hash=next(iter(map_hashes_by_cell[(stratum, context_class)])),
        )

    return ILGCCalibrationBank(
        cells=cells,
        estimator_checkpoint_hash=estimator.checkpoint_hash,
        calibration_config_hash=operational_config.config_hash,
        p0_config_hash=p0_config_hash,
        split_manifest=split_manifest,
        simultaneous_global_alpha=simultaneous_global_alpha,
        coverage_allocation=coverage_allocation,
    )


def _seeded_bank_generator_config(
    generator_config: Mapping[str, Any],
    bank_spec: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    required = {"bank_id", "seed", "block_count", "alpha_range", "phi_range"}
    missing = required - set(bank_spec)
    if missing:
        raise ValueError(f"seeded bank specification is missing {sorted(missing)}")
    configured = copy.deepcopy(dict(generator_config))
    configured["seeded_construction_bank"] = {
        "method": method,
        **dict(bank_spec),
    }
    return configured


def _dedicated_bank_records(
    p0: P0Config,
    generator_config: Mapping[str, Any],
    bank_spec: Mapping[str, Any],
    *,
    split: str,
    record_offset: int,
) -> list[ObservationRecord]:
    measurement = copy.deepcopy(p0.measurement)
    observation = measurement["observation"]
    block_count = int(bank_spec["block_count"])
    observation["train_indices"] = []
    observation["calibration_indices"] = []
    observation["test_indices"] = []
    observation[f"{split}_indices"] = list(range(block_count))
    observation["train_replicates"] = 0
    observation["calibration_replicates"] = 0
    observation["test_replicates"] = 0
    observation[f"{split}_replicates"] = int(bank_spec["replicates"])
    records = build_observation_records(
        dict(generator_config),
        measurement,
        seed=int(bank_spec["observation_seed"]),
    )
    return [
        replace(
            record,
            metadata={
                **record.metadata,
                "record_index": record_offset + int(record.metadata["record_index"]),
                "construction_bank_id": str(bank_spec["bank_id"]),
            },
        )
        for record in records
    ]


def prepare_p0_ilgc_calibration(
    config_dir: str | Path = "config",
) -> P0ILGCCalibrationContext:
    p0 = load_p0_config(config_dir)
    operational = load_p0_ilgc_operational_config(config_dir)
    values = operational.values
    if values.get("formulation_version") != "ilgc_v1_1":
        raise ValueError("p0_ilgc_operational.yaml must use ilgc_v1_1")
    if values.get("role") != "estimator_facing_calibration_candidate":
        raise ValueError("unexpected p0_ilgc_operational.yaml role")

    contexts = full_context_list(p0.measurement)
    grouped = contexts_from_measurement_config(p0.measurement)
    disjoint_banks = values.get("disjoint_banks")
    if not isinstance(disjoint_banks, dict):
        raise ValueError("ILGC operational config requires disjoint_banks")
    method = str(disjoint_banks.get("method", ""))
    construction_spec = disjoint_banks.get("construction")
    calibration_spec = disjoint_banks.get("calibration")
    if method != "seeded_uniform_alpha_phi_v1":
        raise ValueError("unsupported ILGC disjoint-bank method")
    if not isinstance(construction_spec, dict) or not isinstance(calibration_spec, dict):
        raise ValueError("ILGC disjoint banks require construction and calibration specs")

    observation_records = build_observation_records(
        p0.generator,
        p0.measurement,
        seed=int(p0.seeds.get("shot_seed", 0)),
    )
    train_records = records_by_split(observation_records, "train")
    estimator = FreeMeasurementTableEstimator(grouped["input"]).fit_records(train_records)
    calibration_generator_config = _seeded_bank_generator_config(
        p0.generator,
        calibration_spec,
        method=method,
    )
    calibration_instances = construction_bank(calibration_generator_config)
    if len(calibration_instances) != int(calibration_spec["block_count"]):
        raise RuntimeError("calibration construction bank count changed during generation")
    calibration_records = _dedicated_bank_records(
        p0,
        calibration_generator_config,
        calibration_spec,
        split="calibration",
        record_offset=1_000_000,
    )
    reference_generator_config = _seeded_bank_generator_config(
        p0.generator,
        construction_spec,
        method=method,
    )
    reference_instances = construction_bank(reference_generator_config)
    observation_instances = construction_bank(p0.generator)
    split_manifest = _split_manifest(
        observation_records,
        calibration_records,
        observation_instances,
        calibration_instances,
        reference_instances,
        calibration_bank_id=str(calibration_spec["bank_id"]),
        reference_bank_id=str(construction_spec["bank_id"]),
    )
    bank = fit_calibration_bank(
        calibration_records,
        estimator,
        contexts,
        calibration_generator_config,
        operational,
        p0_config_hash=p0.config_hash,
        split_manifest=split_manifest,
    )
    return P0ILGCCalibrationContext(
        p0=p0,
        operational=operational,
        estimator=estimator,
        records=tuple(
            records_by_split(observation_records, "train")
            + calibration_records
            + records_by_split(observation_records, "test")
        ),
        calibration_bank=bank,
        contexts=tuple(contexts),
        calibration_generator_config=calibration_generator_config,
        reference_construction_instances=tuple(reference_instances),
    )


def run_p0_ilgc_calibration(
    config_dir: str | Path = "config",
    output_dir: str | Path = "outputs/ilgc",
) -> dict[str, Any]:
    context = prepare_p0_ilgc_calibration(config_dir)
    p0 = context.p0
    bank = context.calibration_bank
    instances = list(context.reference_construction_instances)
    from .bootstrap import build_clean_embedding_metric

    bootstrap_cfg = context.operational.values["selector_bootstrap"]
    metric_cfg = bootstrap_cfg["embedding_metric"]
    embedding_metric = build_clean_embedding_metric(
        instances,
        shrinkage=float(metric_cfg["shrinkage"]),
        eigenvalue_floor=float(metric_cfg["eigenvalue_floor"]),
    )
    input_contexts = contexts_from_measurement_config(p0.measurement)["input"]
    adequacy: dict[str, Any] = {}
    for stratum, regime in p0.measurement["observation"]["regimes"].items():
        adequacy[stratum] = adequacy_gates(
            input_contexts,
            p0.measurement["stage_a_thresholds"],
            eta=float(regime["eta"]),
        )
    adequacy_qualified = all(
        bool(record["pass"])
        for stratum in adequacy.values()
        for record in stratum.values()
    )
    split_qualified = bool(bank.split_manifest["four_way_disjoint"])
    scientific_qualified = bool(bank.qualified and adequacy_qualified and split_qualified)
    reasons: list[str] = []
    if not bank.cells_qualified:
        reasons.append("calibration_body_unqualified")
    if not adequacy_qualified:
        reasons.append("measurement_adequacy_unqualified")
    if not split_qualified:
        if not bank.split_manifest["construction_indices_pairwise_disjoint"]:
            reasons.append("train_calibration_test_construction_overlap")
        if not bank.split_manifest[
            "construction_bank_disjoint_from_observation_splits"
        ]:
            reasons.append("construction_bank_overlaps_observation_splits")

    artifact = {
        "run_id": "p0-ilgc-calibration-v1.1",
        "stage": "p0_ilgc_calibration",
        "formulation_version": "ilgc_v1_1",
        "status": "qualified" if scientific_qualified else "unqualified",
        "claim_scope": "estimator_facing_calibration_candidate",
        "scientific_calibration_qualified": scientific_qualified,
        "qualification_reasons": reasons,
        "p0_config_hash": p0.config_hash,
        "ilgc_operational_config_hash": context.operational.config_hash,
        "estimator": {
            "model_family": "FreeMeasurementTableEstimator",
            "checkpoint_hash": context.estimator.checkpoint_hash,
            "frozen_before_calibration": True,
        },
        "calibration_bank": bank.to_manifest(),
        "adequacy": adequacy,
        "selector_bootstrap_protocol": {
            "status": "ready" if scientific_qualified else "blocked_by_calibration",
            "intended_claim_scope": "scientific",
            "B_boot": int(bootstrap_cfg["B_boot"]),
            "alpha_wit": float(bootstrap_cfg["alpha_wit"]),
            "emission_lcb_alpha": float(bootstrap_cfg["emission_lcb_alpha"]),
            "pi_sel_min": float(bootstrap_cfg["pi_sel_min"]),
            "B_sel_min": int(bootstrap_cfg["B_sel_min"]),
            "max_unresolved_rate": float(bootstrap_cfg["max_unresolved_rate"]),
            "embedding_metric": embedding_metric.to_manifest(),
            "semantic_fallback": False,
        },
        "downstream_status": {
            "selector_bootstrap": (
                "not_evaluated" if scientific_qualified else "blocked_by_calibration"
            ),
            "atlas": "not_evaluated" if scientific_qualified else "blocked_by_calibration",
            "test_bank": "sealed_not_evaluated",
        },
        "discipline": {
            "construction_seed_blocked": True,
            "threshold_source_split": "calibration",
            "test_records_used_for_calibration": 0,
            "test_certificate_evaluations": 0,
            "test_atlas_evaluations": 0,
            "legacy_projection_thresholds_used": False,
            "semantic_fallback": False,
        },
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / CALIBRATION_ARTIFACT
    target.write_text(
        json.dumps(to_jsonable(artifact), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return artifact
