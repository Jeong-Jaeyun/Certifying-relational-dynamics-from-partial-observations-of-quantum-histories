from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Mapping

import numpy as np
from qiskit.quantum_info import Operator

from bi_qmn.core.generator import P0Instance
from bi_qmn.core.measurement import effective_effect_map
from bi_qmn.p1.model.p1_generator import P1Instance
from bi_qmn.p1.model.p1_measurement import p1_effective_effect_map
from bi_qmn.core.types import MeasurementTable
from .types import (
    CalibrationBody,
    ConstraintInequality,
    ConstraintRegistry,
    EffectiveMeasurementMap,
    FiberSignature,
    MeasurementEntry,
    MeasurementEffectEntry,
    MeasurementModel,
    ObservationTable,
    ObservationTarget,
    measurement_model_from_entries,
    stable_hash,
)


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def effective_measurement_map_from_p0(
    contexts,
    *,
    K: int,
    eta: float,
) -> EffectiveMeasurementMap:
    contexts = sorted(contexts, key=lambda context: context.context_id)
    effects = effective_effect_map(contexts, K=K, eta=eta)
    entries: list[MeasurementEffectEntry] = []
    for context in contexts:
        context_id = context.context_id
        for outcome in sorted(effects[context_id]):
            entries.append(
                MeasurementEffectEntry(
                    context_id=context_id,
                    context_class=context.context_class,
                    outcome=outcome,
                    effect=effects[context_id][outcome],
                )
            )
    combined = tuple(
        MeasurementEntry(
            context_id=entry.context_id,
            context_class=entry.context_class,
            outcome=entry.outcome,
            effect=entry.effect,
            target=1.0 / len(effects[entry.context_id]),
        )
        for entry in entries
    )
    validated = measurement_model_from_entries(combined, eta=eta)
    if validated.measurement_map is None:
        raise RuntimeError("measurement-map construction failed")
    return validated.measurement_map


def effective_measurement_map_from_p1(
    contexts,
    *,
    K: int,
    eta: float,
) -> EffectiveMeasurementMap:
    contexts = sorted(contexts, key=lambda context: context.context_id)
    effects = p1_effective_effect_map(contexts, K=K, eta=eta)
    entries: list[MeasurementEffectEntry] = []
    for context in contexts:
        context_id = context.context_id
        for outcome in sorted(effects[context_id]):
            entries.append(
                MeasurementEffectEntry(
                    context_id=context_id,
                    context_class=context.context_class,
                    outcome=outcome,
                    effect=effects[context_id][outcome],
                )
            )
    combined = tuple(
        MeasurementEntry(
            context_id=entry.context_id,
            context_class=entry.context_class,
            outcome=entry.outcome,
            effect=entry.effect,
            target=1.0 / len(effects[entry.context_id]),
        )
        for entry in entries
    )
    validated = measurement_model_from_entries(combined, eta=eta)
    if validated.measurement_map is None:
        raise RuntimeError("P1 measurement-map construction failed")
    return validated.measurement_map


def observation_table_from_p0(
    table: MeasurementTable,
    measurement_map: EffectiveMeasurementMap,
    *,
    stratum: str,
    estimator_checkpoint_hash: str,
    support_query_hash: str,
) -> ObservationTable:
    expected_keys = {
        (entry.context_id, entry.outcome) for entry in measurement_map.entries
    }
    actual_keys = {
        (context_id, outcome)
        for context_id, outcomes in table.probabilities.items()
        for outcome in outcomes
    }
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"observation/frozen-map key mismatch: missing={missing}, extra={extra}"
        )
    targets: list[ObservationTarget] = []
    digest = hashlib.sha256()
    for entry in measurement_map.entries:
        if entry.context_id not in table.probabilities:
            raise ValueError(f"observation missing context {entry.context_id}")
        if entry.outcome not in table.probabilities[entry.context_id]:
            raise ValueError(
                f"observation missing outcome {entry.context_id}/{entry.outcome}"
            )
        target = float(table.probabilities[entry.context_id][entry.outcome])
        targets.append(
            ObservationTarget(
                context_id=entry.context_id,
                context_class=entry.context_class,
                outcome=entry.outcome,
                target=target,
            )
        )
        digest.update(entry.context_id.encode("utf-8"))
        digest.update(entry.context_class.encode("ascii"))
        digest.update(entry.outcome.encode("utf-8"))
        digest.update(target.hex().encode("ascii"))
    return ObservationTable(
        targets=tuple(targets),
        stratum=str(stratum),
        estimator_checkpoint_hash=str(estimator_checkpoint_hash),
        support_query_hash=str(support_query_hash),
        table_hash=digest.hexdigest()[:16],
    )


def bind_measurement_observation(
    measurement_map: EffectiveMeasurementMap,
    observation: ObservationTable,
    *,
    binding_mode: str,
) -> MeasurementModel:
    effect_keys = [
        (entry.context_id, entry.context_class, entry.outcome)
        for entry in measurement_map.entries
    ]
    target_keys = [
        (entry.context_id, entry.context_class, entry.outcome)
        for entry in observation.targets
    ]
    if effect_keys != target_keys:
        raise ValueError("observation entries do not match frozen measurement map")
    combined = tuple(
        MeasurementEntry(
            context_id=effect.context_id,
            context_class=effect.context_class,
            outcome=effect.outcome,
            effect=effect.effect,
            target=target.target,
        )
        for effect, target in zip(
            measurement_map.entries,
            observation.targets,
            strict=True,
        )
    )
    validated = measurement_model_from_entries(
        combined,
        eta=measurement_map.eta,
        stratum=observation.stratum,
        estimator_checkpoint_hash=observation.estimator_checkpoint_hash,
        support_query_hash=observation.support_query_hash,
        binding_mode=binding_mode,
    )
    if validated.effective_map_hash != measurement_map.map_hash:
        raise ValueError("frozen measurement-map hash changed during observation binding")
    if validated.observation_table_hash != observation.table_hash:
        raise ValueError("observation-table hash changed during binding")
    return replace(
        validated,
        measurement_map=measurement_map,
        observation=observation,
        binding_mode=binding_mode,
    )


def measurement_model_from_p0(table: MeasurementTable, K: int) -> MeasurementModel:
    eta = float(table.metadata.get("eta", 0.0))
    measurement_map = effective_measurement_map_from_p0(
        table.contexts,
        K=K,
        eta=eta,
    )
    observation = observation_table_from_p0(
        table,
        measurement_map,
        stratum=str(table.metadata.get("stratum", table.metadata.get("regime", "unspecified"))),
        estimator_checkpoint_hash=str(
            table.metadata.get("estimator_checkpoint_hash", "unspecified")
        ),
        support_query_hash=stable_hash(
            {
                "observation_id": table.metadata.get(
                    "observation_id",
                    "combined_p0_table",
                ),
                "contexts": [context.context_id for context in table.contexts],
            }
        ),
    )
    return bind_measurement_observation(
        measurement_map,
        observation,
        binding_mode="derived_together_mechanical",
    )


def measurement_model_from_p1(
    table: MeasurementTable,
    K: int,
    *,
    stratum: str,
    estimator_checkpoint_hash: str,
    support_query_hash: str,
) -> MeasurementModel:
    eta = float(table.metadata.get("eta", 0.0))
    measurement_map = effective_measurement_map_from_p1(
        table.contexts,
        K=K,
        eta=eta,
    )
    observation = observation_table_from_p0(
        table,
        measurement_map,
        stratum=stratum,
        estimator_checkpoint_hash=estimator_checkpoint_hash,
        support_query_hash=support_query_hash,
    )
    return bind_measurement_observation(
        measurement_map,
        observation,
        binding_mode="p1_frozen_map_estimator_observation",
    )


def p0_constraint_registry(
    instance: P0Instance,
    theta_con: float,
    *,
    calibration_hash: str,
) -> ConstraintRegistry:
    H = np.asarray(instance.H_tot.data, dtype=complex)
    normalized_H2 = (H @ H) / (float(instance.h_scale) ** 2)
    return ConstraintRegistry(
        records=(
            ConstraintInequality(
                constraint_id="global_constraint_H2",
                scope="ambient_witness",
                operator=Operator(normalized_H2),
                upper_bound=float(theta_con),
                calibration_hash=calibration_hash,
                provenance="P0Instance.H_tot_squared_over_h_scale_squared",
            ),
        )
    )


def p1_constraint_registry(
    instance: P1Instance,
    theta_con: float,
    *,
    calibration_hash: str,
) -> ConstraintRegistry:
    H = np.asarray(instance.H_tot.data, dtype=complex)
    normalized_H2 = (H @ H) / (float(instance.h_scale) ** 2)
    return ConstraintRegistry(
        records=(
            ConstraintInequality(
                constraint_id="global_constraint_H2",
                scope="ambient_witness",
                operator=Operator(normalized_H2),
                upper_bound=float(theta_con),
                calibration_hash=calibration_hash,
                provenance="P1Instance.H_tot_squared_over_h_scale_squared",
            ),
        )
    )


def p0_fiber_signature(
    instance: P0Instance,
    model: MeasurementModel,
    *,
    checkpoint_hash: str,
    stratum: str,
    clock_id: str = "declared_clock",
    observation_id: str = "full_declared_table",
) -> FiberSignature:
    H = np.ascontiguousarray(np.asarray(instance.H_tot.data, dtype=np.complex128))
    construction_digest = hashlib.sha256()
    construction_digest.update(instance.label.encode("utf-8"))
    construction_digest.update(H.view(np.float64).tobytes())
    readout_hash = _json_hash({"eta": model.eta, "model": "symmetric_clock_system_readout_v1"})
    observation_hash = _json_hash(
        {
            "observation_id": observation_id,
            "entries": [
                (entry.context_id, entry.context_class, entry.outcome) for entry in model.entries
            ],
        }
    )
    return FiberSignature(
        construction_id=construction_digest.hexdigest()[:16],
        clock_id=clock_id,
        measurement_map_hash=model.effective_map_hash,
        readout_map_hash=readout_hash,
        observation_hash=observation_hash,
        stratum=stratum,
        checkpoint_hash=checkpoint_hash,
        hilbert_dim=model.hilbert_dim,
    )


def p1_fiber_signature(
    instance: P1Instance,
    model: MeasurementModel,
    *,
    checkpoint_hash: str,
    stratum: str,
    clock_id: str,
    observation_id: str,
    support_context_ids: list[str],
    n_shot: int,
) -> FiberSignature:
    H = np.ascontiguousarray(np.asarray(instance.H_tot.data, dtype=np.complex128))
    construction_digest = hashlib.sha256()
    construction_digest.update(instance.label.encode("utf-8"))
    construction_digest.update(H.view(np.float64).tobytes())
    readout_hash = _json_hash(
        {"eta": model.eta, "model": "symmetric_clock_system_readout_v1"}
    )
    observation_hash = _json_hash(
        {
            "observation_id": observation_id,
            "stratum": stratum,
            "n_shot": int(n_shot),
            "support_context_ids": sorted(support_context_ids),
            "query_entries": [
                (entry.context_id, entry.context_class, entry.outcome)
                for entry in model.entries
            ],
        }
    )
    return FiberSignature(
        construction_id=construction_digest.hexdigest()[:16],
        clock_id=clock_id,
        measurement_map_hash=model.effective_map_hash,
        readout_map_hash=readout_hash,
        observation_hash=observation_hash,
        stratum=stratum,
        checkpoint_hash=checkpoint_hash,
        hilbert_dim=model.hilbert_dim,
    )


def calibration_bodies_from_radii(
    model: MeasurementModel,
    radii: Mapping[str, float],
    *,
    calibration_hash: str,
    stratum: str,
    role: str = "calibrated_acceptance_body",
) -> dict[str, CalibrationBody]:
    missing = set(model.context_classes) - set(radii)
    if missing:
        raise ValueError(f"missing calibration body radii for {sorted(missing)}")
    bodies: dict[str, CalibrationBody] = {}
    for context_class in model.context_classes:
        size = len(model.indices_for_class(context_class))
        body = CalibrationBody(
            context_class=context_class,
            radius=float(radii[context_class]),
            whitening=np.eye(size, dtype=float),
            calibration_hash=calibration_hash,
            stratum=stratum,
            role=role,
        )
        body.validate(size)
        bodies[context_class] = body
    return bodies
