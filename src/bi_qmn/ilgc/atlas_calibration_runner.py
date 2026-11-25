from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bi_qmn.p0.atlas import conditional_slices
from bi_qmn.core.config import load_p0_ilgc_operational_config
from bi_qmn.core.generator import construction_bank
from bi_qmn.core.linalg import bures_distance
from bi_qmn.core.observations import records_by_regime, records_by_split
from bi_qmn.core.types import to_jsonable
from .atlas_calibration import AtlasCalibrationSample, fit_atlas_threshold_bank
from .calibration import prepare_p0_ilgc_calibration
from .model import p0_constraint_registry
from .scientific import _certificate_for_record
from .types import stable_hash


ATLAS_CALIBRATION_ARTIFACT = "p0_ilgc_atlas_calibration.json"


def _load_qualified_selector_stability(
    path: Path,
    *,
    p0_config_hash: str,
) -> dict[str, dict[int, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("claim_scope") != "scientific":
        raise ValueError("selector stability artifact is not scientific")
    if payload.get("p0_config_hash") != p0_config_hash:
        raise ValueError("selector stability artifact has a different P0 configuration")
    bootstrap = payload.get("selector_bootstrap", {})
    if bootstrap.get("status") != "qualified":
        raise ValueError("selector stability artifact is not qualified")
    output: dict[str, dict[int, float]] = {}
    for stratum, manifest in bootstrap.get("by_regime", {}).items():
        if not manifest.get("qualified", False):
            raise ValueError(f"selector stability is unqualified for {stratum}")
        distances = manifest.get("distance_by_tau", {})
        output[str(stratum)] = {int(tau): float(value) for tau, value in distances.items()}
    if not output:
        raise ValueError("selector stability artifact has no distance cells")
    return output


def _sample_from_certificate(
    certificate,
    table,
    *,
    stratum: str,
    construction_block: int,
    scale: dict[str, Any],
    witness_uncertainties: dict[int, float],
) -> AtlasCalibrationSample:
    witness = certificate.selector_claim.witness
    if witness is None:
        raise ValueError("atlas calibration requires a verified canonical witness")
    slices = conditional_slices(
        witness,
        table,
        K=4,
        witness_uncertainties=witness_uncertainties,
    )
    edge_transport: list[float] = []
    edge_direction: list[float] = []
    local_window = [int(value) for value in scale["local_window"]]
    for anchor in range(4):
        physical = [(anchor + local) % 4 for local in local_window]
        for left, right in zip(physical[:-1], physical[1:], strict=True):
            edge_transport.append(
                bures_distance(slices[left].rho.data, slices[right].rho.data)
            )
            edge_direction.append(abs(slices[left].probability - slices[right].probability))
    return AtlasCalibrationSample(
        stratum=stratum,
        construction_block=construction_block,
        values={
            "p_min": min(item.probability for item in slices.values()),
            "theta_align": max(item.align for item in slices.values()),
            "theta_wit": max(item.witness_uncertainty for item in slices.values()),
            "theta_tr": max(edge_transport),
            "theta_dir": max(edge_direction),
        },
    )


def run_p0_ilgc_atlas_calibration(
    config_dir: str | Path = "config",
    output_dir: str | Path = "outputs/ilgc",
    *,
    selector_stability_path: str | Path = "outputs/ilgc/p0_ilgc_scientific_evaluation.json",
) -> dict[str, Any]:
    """Fit atlas gates only from the dedicated calibration bank.

    This runner deliberately has no access to the test split.  It consumes the already
    qualified selector-stability calibration object solely to freeze the typed witness
    uncertainty component of the vertex gate.
    """
    context = prepare_p0_ilgc_calibration(config_dir)
    operational = load_p0_ilgc_operational_config(config_dir)
    selector_path = Path(selector_stability_path)
    stability_by_stratum = _load_qualified_selector_stability(
        selector_path,
        p0_config_hash=context.p0.config_hash,
    )
    instances = construction_bank(context.calibration_generator_config)
    samples: list[AtlasCalibrationSample] = []
    failures: list[dict[str, Any]] = []
    calibration_records = records_by_split(list(context.records), "calibration")
    for record in calibration_records:
        bodies = context.calibration_bank.bodies_for_stratum(record.regime)
        registry = p0_constraint_registry(
            instances[record.instance_index],
            float(context.p0.generator["thresholds"]["theta_con"]),
            calibration_hash=context.calibration_bank.body_set_hash(record.regime),
        )
        table, certificate = _certificate_for_record(
            record,
            instances[record.instance_index],
            context.estimator,
            list(context.contexts),
            bodies,
            registry,
            selector_error_budget=float(operational.values["selector_error_budget"]),
            precision_dps=int(operational.values["precision_dps"]),
        )
        try:
            samples.append(
                _sample_from_certificate(
                    certificate,
                    table,
                    stratum=record.regime,
                    construction_block=record.instance_index,
                    scale=context.p0.measurement["scale"],
                    witness_uncertainties=stability_by_stratum[record.regime],
                )
            )
        except ValueError as error:
            failures.append(
                {
                    "stratum": record.regime,
                    "construction_block": record.instance_index,
                    "record_index": record.metadata["record_index"],
                    "selector_status": certificate.selector_claim.status,
                    "reason": str(error),
                }
            )
    threshold_bank = fit_atlas_threshold_bank(
        samples,
        operational.values["atlas_calibration"],
        calibration_manifest_hash=context.calibration_bank.manifest_hash,
    )
    required_strata = set(context.p0.measurement["observation"]["regimes"])
    missing_strata = sorted(required_strata - set(threshold_bank.cells))
    qualified = (
        context.calibration_bank.qualified
        and not failures
        and not missing_strata
        and threshold_bank.qualified
    )
    artifact = {
        "run_id": "p0-ilgc-atlas-calibration-v1.2",
        "stage": "p0_ilgc_atlas_threshold_calibration",
        "formulation_version": "ilgc_v1_1",
        "claim_scope": "scientific_recalibration_candidate",
        "status": "qualified" if qualified else "unqualified",
        "scientific_p0_status": "not_evaluated",
        "calibration_manifest_hash": context.calibration_bank.manifest_hash,
        "operational_config_hash": operational.config_hash,
        "selector_stability_source": {
            "path": str(selector_path),
            "artifact_hash": stable_hash(json.loads(selector_path.read_text(encoding="utf-8"))),
        },
        "discipline": {
            "threshold_source_split": "dedicated_calibration",
            "test_records_used_for_calibration": 0,
            "test_records_evaluated": 0,
            "static_atlas_thresholds_used": False,
        },
        "certificate_failures": failures,
        "missing_strata": missing_strata,
        "atlas_threshold_bank": threshold_bank.to_manifest(),
    }
    target = Path(output_dir) / ATLAS_CALIBRATION_ARTIFACT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(to_jsonable(artifact), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return artifact
