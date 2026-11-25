from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bi_qmn.core.config import load_p0_config, load_p0_ilgc_operational_config
from bi_qmn.p0.estimator import FreeMeasurementTableEstimator
from bi_qmn.core.generator import construction_bank, construction_fingerprint
from bi_qmn.core.measurement import contexts_from_measurement_config, full_context_list
from bi_qmn.core.observations import ObservationRecord, records_by_regime, records_by_split
from bi_qmn.core.types import to_jsonable
from .atlas import build_ilgc_atlas
from .bootstrap import (
    build_clean_embedding_metric,
    estimate_selector_bootstrap,
    wilson_lcb,
)
from .atlas_calibration import ATLAS_METRICS
from .calibration import (
    _dedicated_bank_records,
    _seeded_bank_generator_config,
    prepare_p0_ilgc_calibration,
)
from .certificate import compute_certificate
from .model import (
    bind_measurement_observation,
    effective_measurement_map_from_p0,
    observation_table_from_p0,
    p0_constraint_registry,
    p0_fiber_signature,
)
from .types import stable_hash


SCIENTIFIC_EVALUATION_ARTIFACT = "p0_ilgc_scientific_evaluation_v1_2.json"


def _load_atlas_threshold_bank(
    path: Path,
    *,
    calibration_manifest_hash: str,
) -> tuple[dict[str, dict[str, float]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "qualified":
        raise ValueError("atlas threshold artifact is not qualified")
    bank = payload.get("atlas_threshold_bank", {})
    if bank.get("calibration_manifest_hash") != calibration_manifest_hash:
        raise ValueError("atlas threshold artifact has a different calibration manifest")
    if not bank.get("qualified", False):
        raise ValueError("atlas threshold bank is unqualified")
    thresholds: dict[str, dict[str, float]] = {}
    for stratum, cell in bank.get("cells", {}).items():
        if not cell.get("qualified", False):
            raise ValueError(f"atlas threshold cell is unqualified for {stratum}")
        values = cell.get("thresholds", {})
        if set(values) != set(ATLAS_METRICS):
            raise ValueError(f"atlas threshold cell has an invalid metric set for {stratum}")
        thresholds[str(stratum)] = {name: float(values[name]) for name in ATLAS_METRICS}
    if not thresholds:
        raise ValueError("atlas threshold artifact has no stratum cells")
    return thresholds, stable_hash(payload)


def _evaluation_bank_manifest(context, instances, records) -> dict[str, Any]:
    fingerprints = {construction_fingerprint(instance) for instance in instances}
    prior = context.calibration_bank.split_manifest
    prior_fingerprints = set(
        prior["construction_bank"]["construction_fingerprints"]
    )
    for split in prior["splits"].values():
        prior_fingerprints.update(split["construction_fingerprints"])
    return {
        "bank_id": records[0].metadata["construction_bank_id"] if records else None,
        "construction_fingerprints": sorted(fingerprints),
        "record_ids": sorted(int(record.metadata["record_index"]) for record in records),
        "record_count": len(records),
        "disjoint_from_prior_reference_train_calibration_and_v1_1_test": fingerprints.isdisjoint(prior_fingerprints),
    }


def _estimator_table(
    record: ObservationRecord,
    estimator: FreeMeasurementTableEstimator,
    contexts,
):
    return estimator.predict_table(
        record.support.table,
        contexts,
        metadata={
            "eta": record.eta,
            "regime": record.regime,
            "instance_index": record.instance_index,
            "record_index": record.metadata["record_index"],
            "estimated": True,
            "evaluation_role": "sealed_test",
        },
    )


def _certificate_for_record(
    record: ObservationRecord,
    instance,
    estimator: FreeMeasurementTableEstimator,
    contexts,
    bodies,
    registry,
    *,
    selector_error_budget: float,
    precision_dps: int,
):
    table = _estimator_table(record, estimator, contexts)
    measurement_map = effective_measurement_map_from_p0(
        contexts,
        K=instance.K,
        eta=float(record.eta),
    )
    observation = observation_table_from_p0(
        table,
        measurement_map,
        stratum=record.regime,
        estimator_checkpoint_hash=estimator.checkpoint_hash,
        support_query_hash=stable_hash(
            {
                "record_index": record.metadata["record_index"],
                "role": "sealed_test_certificate",
            }
        ),
    )
    model = bind_measurement_observation(
        measurement_map,
        observation,
        binding_mode="frozen_map_estimator_observation",
    )
    fiber = p0_fiber_signature(
        instance,
        model,
        checkpoint_hash=estimator.checkpoint_hash,
        stratum=record.regime,
        observation_id="p0_ilgc_sealed_test_v1",
    )
    certificate = compute_certificate(
        model,
        bodies,
        registry,
        fiber,
        selector_error_budget=selector_error_budget,
        precision_dps=precision_dps,
        precondition_status="scientific_calibration_verified",
    )
    return table, certificate


def run_p0_ilgc_scientific_evaluation(
    config_dir: str | Path = "config",
    output_dir: str | Path = "outputs/ilgc",
    *,
    B_boot_override: int | None = None,
    max_test_records_per_regime: int | None = None,
    atlas_calibration_path: str | Path = "outputs/ilgc/p0_ilgc_atlas_calibration.json",
) -> dict[str, Any]:
    """Run calibrated v1.2 atlas evaluation on a fresh sealed construction bank."""
    context = prepare_p0_ilgc_calibration(config_dir)
    operational = load_p0_ilgc_operational_config(config_dir)
    p0 = load_p0_config(config_dir)
    bank = context.calibration_bank
    bootstrap_cfg = operational.values["selector_bootstrap"]
    contexts = full_context_list(p0.measurement)
    grouped = contexts_from_measurement_config(p0.measurement)
    estimator = context.estimator
    selector_error_budget = float(operational.values["selector_error_budget"])
    precision_dps = int(operational.values["precision_dps"])
    B_boot = int(B_boot_override or bootstrap_cfg["B_boot"])
    calibration_instances = construction_bank(context.calibration_generator_config)
    disjoint_banks = operational.values["disjoint_banks"]
    evaluation_spec = disjoint_banks.get("evaluation")
    if not isinstance(evaluation_spec, dict):
        raise ValueError("v1.2 scientific evaluation requires an evaluation bank")
    evaluation_generator_config = _seeded_bank_generator_config(
        p0.generator,
        evaluation_spec,
        method=str(disjoint_banks["method"]),
    )
    evaluation_instances = construction_bank(evaluation_generator_config)
    evaluation_records = _dedicated_bank_records(
        p0,
        evaluation_generator_config,
        evaluation_spec,
        split="test",
        record_offset=2_000_000,
    )
    atlas_thresholds, atlas_calibration_hash = _load_atlas_threshold_bank(
        Path(atlas_calibration_path),
        calibration_manifest_hash=bank.manifest_hash,
    )
    evaluation_manifest = _evaluation_bank_manifest(
        context, evaluation_instances, evaluation_records
    )
    if not evaluation_manifest["disjoint_from_prior_reference_train_calibration_and_v1_1_test"]:
        raise ValueError("v1.2 evaluation bank overlaps an earlier protocol bank")
    embedding_cfg = bootstrap_cfg["embedding_metric"]
    embedding_metric = build_clean_embedding_metric(
        list(context.reference_construction_instances),
        shrinkage=float(embedding_cfg["shrinkage"]),
        eigenvalue_floor=float(embedding_cfg["eigenvalue_floor"]),
    )

    artifact: dict[str, Any] = {
        "run_id": "p0-ilgc-scientific-evaluation-v1.2",
        "stage": "p0_ilgc_scientific_evaluation",
        "formulation_version": "ilgc_v1_1",
        "p0_config_hash": p0.config_hash,
        "ilgc_operational_config_hash": operational.config_hash,
        "calibration_manifest_hash": bank.manifest_hash,
        "atlas_calibration_artifact_hash": atlas_calibration_hash,
        "atlas_thresholds_by_stratum": atlas_thresholds,
        "evaluation_bank": evaluation_manifest,
        "claim_scope": "scientific",
        "discipline": {
            "threshold_source_split": "dedicated_calibration",
            "test_records_used_for_calibration": 0,
            "test_records_inspected_before_bootstrap_gate": 0,
            "test_bank_evaluated_once": False,
            "semantic_fallback": False,
            "static_atlas_thresholds_used": False,
        },
    }
    if not bank.qualified:
        artifact.update(
            {
                "status": "blocked_by_calibration",
                "scientific_p0_status": "not_established",
                "calibration": bank.to_manifest(),
                "selector_bootstrap": {"status": "not_evaluated"},
                "test_bank": {"status": "sealed_not_evaluated"},
            }
        )
    else:
        calibration_records = records_by_split(list(context.records), "calibration")
        stability_by_regime: dict[str, Any] = {}
        pilot_certificates: dict[str, Any] = {}
        for regime in p0.measurement["observation"]["regimes"]:
            candidates = records_by_regime(calibration_records, regime)
            if not candidates:
                raise RuntimeError(f"missing dedicated calibration record for {regime}")
            pilot = min(candidates, key=lambda record: int(record.metadata["record_index"]))
            instance = calibration_instances[pilot.instance_index]
            bodies = bank.bodies_for_stratum(regime)
            registry = p0_constraint_registry(
                instance,
                float(p0.generator["thresholds"]["theta_con"]),
                calibration_hash=bank.body_set_hash(regime),
            )
            _, base_certificate = _certificate_for_record(
                pilot,
                instance,
                estimator,
                contexts,
                bodies,
                registry,
                selector_error_budget=selector_error_budget,
                precision_dps=precision_dps,
            )
            if base_certificate.selector_claim.witness is None:
                pilot_certificates[regime] = base_certificate.to_artifact()
                stability_by_regime[regime] = None
                continue
            certificate, stability = estimate_selector_bootstrap(
                pilot,
                estimator,
                contexts,
                instance,
                bodies,
                registry,
                embedding_metric,
                selector_error_budget=selector_error_budget,
                precision_dps=precision_dps,
                B_boot=B_boot,
                seed=(
                    int(pilot.metadata["seed"])
                    + int(p0.seeds.get("bootstrap_seed", 0))
                ),
                alpha_wit=float(bootstrap_cfg["alpha_wit"]),
                emission_lcb_alpha=float(bootstrap_cfg["emission_lcb_alpha"]),
                pi_sel_min=float(bootstrap_cfg["pi_sel_min"]),
                B_sel_min=int(bootstrap_cfg["B_sel_min"]),
                max_unresolved_rate=float(bootstrap_cfg["max_unresolved_rate"]),
                claim_scope="scientific",
            )
            pilot_certificates[regime] = certificate.to_artifact()
            stability_by_regime[regime] = stability

        artifact["selector_bootstrap"] = {
            "status": (
                "qualified"
                if all(
                    stability is not None and stability.qualified
                    for stability in stability_by_regime.values()
                )
                else "unqualified"
            ),
            "B_boot": B_boot,
            "by_regime": {
                regime: stability.to_manifest()
                for regime, stability in stability_by_regime.items()
                if stability is not None
            },
            "pilot_certificates": pilot_certificates,
            "base_selector_failures": {
                regime: {
                    "selector_status": pilot_certificates[regime]["selector_claim"][
                        "status"
                    ],
                    "reason": pilot_certificates[regime]["selector_claim"]["diagnostics"].get(
                        "reason"
                    ),
                    "representation_error_bound": pilot_certificates[regime][
                        "selector_claim"
                    ]["diagnostics"].get("representation_error_bound"),
                }
                for regime, stability in stability_by_regime.items()
                if stability is None
            },
        }
        if not all(
            stability is not None and stability.qualified
            for stability in stability_by_regime.values()
        ):
            artifact.update(
                {
                    "status": "blocked_by_selector_bootstrap",
                    "scientific_p0_status": "not_established",
                    "test_bank": {"status": "sealed_not_evaluated"},
                }
            )
        else:
            selected_test_records: list[ObservationRecord] = []
            for regime in p0.measurement["observation"]["regimes"]:
                regime_records = records_by_regime(evaluation_records, regime)
                if max_test_records_per_regime is not None:
                    regime_records = regime_records[:max_test_records_per_regime]
                selected_test_records.extend(regime_records)
            test_results: dict[str, list[dict[str, Any]]] = {
                regime: [] for regime in p0.measurement["observation"]["regimes"]
            }
            for record in selected_test_records:
                instance = evaluation_instances[record.instance_index]
                bodies = bank.bodies_for_stratum(record.regime)
                registry = p0_constraint_registry(
                    instance,
                    float(p0.generator["thresholds"]["theta_con"]),
                    calibration_hash=bank.body_set_hash(record.regime),
                )
                table, certificate = _certificate_for_record(
                    record,
                    instance,
                    estimator,
                    contexts,
                    bodies,
                    registry,
                    selector_error_budget=selector_error_budget,
                    precision_dps=precision_dps,
                )
                atlas = build_ilgc_atlas(
                    certificate,
                    table,
                    atlas_thresholds[record.regime],
                    p0.measurement["scale"],
                    stability_by_regime[record.regime],
                )
                test_results[record.regime].append(
                    {
                        "record_index": record.metadata["record_index"],
                        "instance_index": record.instance_index,
                        "replicate": record.replicate,
                        "certificate_status": certificate.bi_existence_claim.status,
                        "selector_status": certificate.selector_claim.status,
                        "atlas_status": atlas.status,
                        "certificate_accepted": certificate.selector_claim.witness is not None,
                        "atlas_accepted": atlas.status == "accepted",
                        "certificate": certificate.to_artifact(),
                        "atlas": atlas.to_manifest(),
                    }
                )
            coverage: dict[str, Any] = {}
            coverage_pass = True
            for regime, results in test_results.items():
                total = len(results)
                certificate_lcb = wilson_lcb(
                    sum(bool(result["certificate_accepted"]) for result in results),
                    total,
                    float(p0.measurement["observation"]["lcb_alpha"]),
                )
                atlas_lcb = wilson_lcb(
                    sum(bool(result["atlas_accepted"]) for result in results),
                    total,
                    float(p0.measurement["observation"]["lcb_alpha"]),
                )
                floors = p0.measurement["observation"]["coverage_floors"][regime]
                passed = (
                    certificate_lcb >= float(floors["cert"])
                    and atlas_lcb >= float(floors["atlas"])
                )
                coverage_pass = coverage_pass and passed
                coverage[regime] = {
                    "record_count": total,
                    "certificate_lcb": certificate_lcb,
                    "atlas_lcb": atlas_lcb,
                    "floors": dict(floors),
                    "pass": passed,
                }
            artifact.update(
                {
                    "status": "accepted" if coverage_pass else "rejected",
                    "scientific_p0_status": (
                        "accepted" if coverage_pass else "rejected"
                    ),
                    "test_bank": {
                        "status": "evaluated_once",
                        "coverage": coverage,
                        "records": test_results,
                    },
                }
            )
            artifact["discipline"]["test_bank_evaluated_once"] = True
            artifact["discipline"]["test_records_inspected_before_bootstrap_gate"] = 0

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / SCIENTIFIC_EVALUATION_ARTIFACT
    target.write_text(
        json.dumps(to_jsonable(artifact), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return artifact
