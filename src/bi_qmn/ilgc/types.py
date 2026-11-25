from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal, Mapping

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator


FORMULATION_VERSION = "ilgc_v1_1"

SemanticStatus = Literal[
    "primal_verified",
    "dual_refuted",
    "unresolved",
]
EvaluationStatus = Literal[
    "evaluated",
    "unsupported",
    "blocked_upstream",
    "not_evaluated",
]
ExistenceStatus = SemanticStatus | EvaluationStatus
SelectorSemanticStatus = Literal["verified", "unresolved"]
SelectorEvaluationStatus = Literal[
    "evaluated",
    "blocked_upstream",
    "not_evaluated",
]
SelectorStatus = SelectorSemanticStatus | SelectorEvaluationStatus
PreconditionStatus = Literal[
    "verified",
    "mechanical_input_verified",
    "scientific_calibration_verified",
    "unsupported",
]


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def stable_hash(payload: Any) -> str:
    return _stable_hash(payload)


def calibration_bodies_hash(
    bodies: Mapping[str, "CalibrationBody"],
) -> str:
    return _stable_hash(
        {key: body.to_manifest() for key, body in sorted(bodies.items())}
    )


def operator_hash(operator: Operator) -> str:
    data = np.ascontiguousarray(np.asarray(operator.data, dtype=np.complex128))
    digest = hashlib.sha256()
    digest.update(str(data.shape).encode("ascii"))
    digest.update(data.view(np.float64).tobytes())
    return digest.hexdigest()[:16]


def complex_matrix_manifest(matrix: np.ndarray) -> dict[str, Any]:
    data = np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))
    return {
        "encoding": "split_real_imag_v1",
        "shape": list(data.shape),
        "real": data.real.tolist(),
        "imag": data.imag.tolist(),
        "sha256": hashlib.sha256(data.view(np.float64).tobytes()).hexdigest(),
    }


@dataclass(frozen=True)
class FiberSignature:
    construction_id: str
    clock_id: str
    measurement_map_hash: str
    readout_map_hash: str
    observation_hash: str
    stratum: str
    checkpoint_hash: str
    hilbert_dim: int
    formulation_version: str = FORMULATION_VERSION

    def to_manifest(self) -> dict[str, Any]:
        return {
            "construction_id": self.construction_id,
            "clock_id": self.clock_id,
            "measurement_map_hash": self.measurement_map_hash,
            "readout_map_hash": self.readout_map_hash,
            "observation_hash": self.observation_hash,
            "stratum": self.stratum,
            "checkpoint_hash": self.checkpoint_hash,
            "hilbert_dim": self.hilbert_dim,
            "formulation_version": self.formulation_version,
        }

    @property
    def fiber_hash(self) -> str:
        return _stable_hash(self.to_manifest())


@dataclass(frozen=True)
class MeasurementEntry:
    context_id: str
    context_class: str
    outcome: str
    effect: Operator
    target: float


@dataclass(frozen=True)
class MeasurementEffectEntry:
    context_id: str
    context_class: str
    outcome: str
    effect: Operator


@dataclass(frozen=True)
class ObservationTarget:
    context_id: str
    context_class: str
    outcome: str
    target: float


@dataclass(frozen=True)
class EffectiveMeasurementMap:
    entries: tuple[MeasurementEffectEntry, ...]
    hilbert_dim: int
    eta: float
    map_hash: str
    readout_hash: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "hilbert_dim": self.hilbert_dim,
            "eta": self.eta,
            "map_hash": self.map_hash,
            "readout_hash": self.readout_hash,
            "entries": [
                {
                    "context_id": entry.context_id,
                    "context_class": entry.context_class,
                    "outcome": entry.outcome,
                    "effect_hash": operator_hash(entry.effect),
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True)
class ObservationTable:
    targets: tuple[ObservationTarget, ...]
    stratum: str
    estimator_checkpoint_hash: str
    support_query_hash: str
    table_hash: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "estimator_checkpoint_hash": self.estimator_checkpoint_hash,
            "support_query_hash": self.support_query_hash,
            "table_hash": self.table_hash,
            "entry_count": len(self.targets),
        }


@dataclass(frozen=True)
class MeasurementModel:
    entries: tuple[MeasurementEntry, ...]
    hilbert_dim: int
    eta: float
    effective_map_hash: str
    measurement_map: EffectiveMeasurementMap | None = None
    observation: ObservationTable | None = None
    binding_mode: str = "legacy_combined"

    @property
    def context_classes(self) -> tuple[str, ...]:
        return tuple(sorted({entry.context_class for entry in self.entries}))

    def indices_for_class(self, context_class: str) -> tuple[int, ...]:
        return tuple(
            index for index, entry in enumerate(self.entries) if entry.context_class == context_class
        )

    def targets(self) -> np.ndarray:
        return np.asarray([entry.target for entry in self.entries], dtype=float)

    @property
    def observation_table_hash(self) -> str:
        if self.observation is not None:
            return self.observation.table_hash
        digest = hashlib.sha256()
        for entry in self.entries:
            digest.update(entry.context_id.encode("utf-8"))
            digest.update(entry.context_class.encode("ascii"))
            digest.update(entry.outcome.encode("utf-8"))
            digest.update(float(entry.target).hex().encode("ascii"))
        return digest.hexdigest()[:16]

    def conjugated(self, unitary: Operator) -> "MeasurementModel":
        W = np.asarray(unitary.data, dtype=complex)
        if W.shape != (self.hilbert_dim, self.hilbert_dim):
            raise ValueError("coordinate transform dimension does not match measurement model")
        transformed = tuple(
            MeasurementEntry(
                context_id=entry.context_id,
                context_class=entry.context_class,
                outcome=entry.outcome,
                effect=Operator(W @ np.asarray(entry.effect.data, dtype=complex) @ W.conj().T),
                target=entry.target,
            )
            for entry in self.entries
        )
        return measurement_model_from_entries(
            transformed,
            eta=self.eta,
            stratum=("unspecified" if self.observation is None else self.observation.stratum),
            estimator_checkpoint_hash=(
                "unspecified"
                if self.observation is None
                else self.observation.estimator_checkpoint_hash
            ),
            support_query_hash=(
                "unspecified"
                if self.observation is None
                else self.observation.support_query_hash
            ),
            binding_mode=self.binding_mode,
        )


def measurement_model_from_entries(
    entries: tuple[MeasurementEntry, ...],
    *,
    eta: float,
    stratum: str = "unspecified",
    estimator_checkpoint_hash: str = "unspecified",
    support_query_hash: str = "unspecified",
    binding_mode: str = "combined_mechanical",
) -> MeasurementModel:
    if not entries:
        raise ValueError("measurement model requires at least one entry")
    if not np.isfinite(float(eta)):
        raise ValueError("measurement eta must be finite")
    dim = int(np.asarray(entries[0].effect.data).shape[0])
    digest = hashlib.sha256()
    seen: set[tuple[str, str]] = set()
    context_effects: dict[str, list[np.ndarray]] = {}
    context_targets: dict[str, list[float]] = {}
    context_classes: dict[str, str] = {}
    for entry in entries:
        data = np.asarray(entry.effect.data, dtype=np.complex128)
        if data.shape != (dim, dim):
            raise ValueError("measurement effects must share one ambient Hilbert space")
        if not np.all(np.isfinite(data)):
            raise ValueError("measurement effects must be finite")
        if not np.allclose(data, data.conj().T, atol=1.0e-12, rtol=0.0):
            raise ValueError("measurement effects must be Hermitian")
        if float(np.min(np.linalg.eigvalsh(data))) < -1.0e-10:
            raise ValueError("measurement effects must be positive semidefinite")
        target = float(entry.target)
        if not np.isfinite(target) or target < -1.0e-10 or target > 1.0 + 1.0e-10:
            raise ValueError("measurement targets must be finite probabilities")
        key = (entry.context_id, entry.outcome)
        if key in seen:
            raise ValueError(f"duplicate measurement entry {key}")
        seen.add(key)
        prior_class = context_classes.setdefault(entry.context_id, entry.context_class)
        if prior_class != entry.context_class:
            raise ValueError("one context cannot belong to multiple context classes")
        context_effects.setdefault(entry.context_id, []).append(data)
        context_targets.setdefault(entry.context_id, []).append(target)
        digest.update(entry.context_id.encode("utf-8"))
        digest.update(entry.context_class.encode("ascii"))
        digest.update(entry.outcome.encode("utf-8"))
        digest.update(np.ascontiguousarray(data).view(np.float64).tobytes())
    identity = np.eye(dim, dtype=complex)
    for context_id, effects in context_effects.items():
        if not np.allclose(sum(effects), identity, atol=1.0e-10, rtol=0.0):
            raise ValueError(f"measurement effects for {context_id} do not sum to identity")
        if not np.isclose(
            sum(context_targets[context_id]),
            1.0,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError(f"measurement targets for {context_id} do not sum to one")
    digest.update(repr(float(eta)).encode("ascii"))
    map_hash = digest.hexdigest()[:16]
    observation_digest = hashlib.sha256()
    for entry in entries:
        observation_digest.update(entry.context_id.encode("utf-8"))
        observation_digest.update(entry.context_class.encode("ascii"))
        observation_digest.update(entry.outcome.encode("utf-8"))
        observation_digest.update(float(entry.target).hex().encode("ascii"))
    table_hash = observation_digest.hexdigest()[:16]
    readout_hash = stable_hash({"eta": float(eta), "model": "effective_readout_v1"})
    measurement_map = EffectiveMeasurementMap(
        entries=tuple(
            MeasurementEffectEntry(
                context_id=entry.context_id,
                context_class=entry.context_class,
                outcome=entry.outcome,
                effect=entry.effect,
            )
            for entry in entries
        ),
        hilbert_dim=dim,
        eta=float(eta),
        map_hash=map_hash,
        readout_hash=readout_hash,
    )
    observation = ObservationTable(
        targets=tuple(
            ObservationTarget(
                context_id=entry.context_id,
                context_class=entry.context_class,
                outcome=entry.outcome,
                target=float(entry.target),
            )
            for entry in entries
        ),
        stratum=str(stratum),
        estimator_checkpoint_hash=str(estimator_checkpoint_hash),
        support_query_hash=str(support_query_hash),
        table_hash=table_hash,
    )
    return MeasurementModel(
        entries=entries,
        hilbert_dim=dim,
        eta=float(eta),
        effective_map_hash=map_hash,
        measurement_map=measurement_map,
        observation=observation,
        binding_mode=binding_mode,
    )


@dataclass(frozen=True)
class ConstraintInequality:
    constraint_id: str
    scope: str
    operator: Operator
    upper_bound: float | None
    calibration_hash: str
    provenance: str
    lower_bound: float | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "scope": self.scope,
            "operator_hash": operator_hash(self.operator),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "calibration_hash": self.calibration_hash,
            "provenance": self.provenance,
        }

    def conjugated(self, unitary: Operator) -> "ConstraintInequality":
        W = np.asarray(unitary.data, dtype=complex)
        G = np.asarray(self.operator.data, dtype=complex)
        if W.shape != G.shape:
            raise ValueError("coordinate transform dimension does not match constraint")
        return ConstraintInequality(
            constraint_id=self.constraint_id,
            scope=self.scope,
            operator=Operator(W @ G @ W.conj().T),
            upper_bound=self.upper_bound,
            calibration_hash=self.calibration_hash,
            provenance=self.provenance,
            lower_bound=self.lower_bound,
        )


@dataclass(frozen=True)
class CanonicalConstraintSide:
    side_id: str
    constraint_id: str
    side: Literal["upper", "lower"]
    operator: Operator
    upper_bound: float


@dataclass(frozen=True)
class ConstraintRegistry:
    records: tuple[ConstraintInequality, ...]
    registry_version: str = "typed_linear_registry_v1"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "registry_version": self.registry_version,
            "records": [record.to_manifest() for record in self.records],
        }

    @property
    def registry_hash(self) -> str:
        return _stable_hash(self.to_manifest())

    def validate(self, dim: int) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("constraint registry dimension must be positive")
        if not str(self.registry_version).strip():
            raise ValueError("constraint registry version must be nonempty")
        seen: set[str] = set()
        for record in self.records:
            constraint_id = str(record.constraint_id).strip()
            if not constraint_id:
                raise ValueError("constraint_id must be nonempty")
            if constraint_id in seen:
                raise ValueError(f"duplicate constraint_id {constraint_id}")
            seen.add(constraint_id)
            if record.scope != "ambient_witness":
                raise ValueError(
                    f"constraint {constraint_id} has forbidden scope {record.scope}"
                )
            operator = np.asarray(record.operator.data, dtype=complex)
            if operator.shape != (dim, dim):
                raise ValueError(
                    f"constraint {constraint_id} has dimension {operator.shape}"
                )
            if not np.all(np.isfinite(operator)):
                raise ValueError(f"constraint {constraint_id} operator is nonfinite")
            if not np.allclose(operator, operator.conj().T, atol=1.0e-12, rtol=0.0):
                raise ValueError(f"constraint {constraint_id} operator is non-Hermitian")
            lower = record.lower_bound
            upper = record.upper_bound
            if lower is None and upper is None:
                raise ValueError(f"constraint {constraint_id} has no finite bound")
            if lower is not None and not np.isfinite(float(lower)):
                raise ValueError(f"constraint {constraint_id} lower bound is nonfinite")
            if upper is not None and not np.isfinite(float(upper)):
                raise ValueError(f"constraint {constraint_id} upper bound is nonfinite")
            if lower is not None and upper is not None and float(lower) > float(upper):
                raise ValueError(f"constraint {constraint_id} lower bound exceeds upper")
            if not str(record.calibration_hash).strip():
                raise ValueError(f"constraint {constraint_id} calibration hash is empty")
            if not str(record.provenance).strip():
                raise ValueError(f"constraint {constraint_id} provenance is empty")

    def canonical_sides(self) -> tuple[CanonicalConstraintSide, ...]:
        sides: list[CanonicalConstraintSide] = []
        for record in self.records:
            if record.upper_bound is not None:
                sides.append(
                    CanonicalConstraintSide(
                        side_id=f"{record.constraint_id}:upper",
                        constraint_id=record.constraint_id,
                        side="upper",
                        operator=record.operator,
                        upper_bound=float(record.upper_bound),
                    )
                )
            if record.lower_bound is not None:
                sides.append(
                    CanonicalConstraintSide(
                        side_id=f"{record.constraint_id}:lower",
                        constraint_id=record.constraint_id,
                        side="lower",
                        operator=Operator(-np.asarray(record.operator.data, dtype=complex)),
                        upper_bound=-float(record.lower_bound),
                    )
                )
        return tuple(sides)

    def conjugated(self, unitary: Operator) -> "ConstraintRegistry":
        return ConstraintRegistry(
            records=tuple(record.conjugated(unitary) for record in self.records),
            registry_version=self.registry_version,
        )


@dataclass(frozen=True)
class CalibrationBody:
    context_class: str
    radius: float
    whitening: np.ndarray
    calibration_hash: str
    stratum: str
    role: str = "calibrated_acceptance_body"

    def validate(self, size: int) -> None:
        matrix = np.asarray(self.whitening, dtype=float)
        if matrix.shape != (size, size):
            raise ValueError(
                f"{self.context_class} whitening shape {matrix.shape} does not match {size} entries"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("whitening matrix must be finite")
        if not np.isfinite(float(self.radius)) or float(self.radius) <= 0.0:
            raise ValueError("ILGC v1.1 operational bodies require a positive radius")
        if not str(self.calibration_hash).strip():
            raise ValueError("calibration body hash must be nonempty")
        if not str(self.stratum).strip():
            raise ValueError("calibration body stratum must be nonempty")
        if np.linalg.matrix_rank(matrix) != size:
            raise ValueError("ILGC v1.1 dual verification requires invertible whitening")

    def to_manifest(self) -> dict[str, Any]:
        matrix = np.ascontiguousarray(np.asarray(self.whitening, dtype=np.float64))
        return {
            "context_class": self.context_class,
            "radius": self.radius,
            "whitening_hash": hashlib.sha256(matrix.tobytes()).hexdigest()[:16],
            "shape": list(matrix.shape),
            "calibration_hash": self.calibration_hash,
            "stratum": self.stratum,
            "role": self.role,
        }


@dataclass(frozen=True)
class ExistencePrimalEvidence:
    density: DensityMatrix
    radial_upper_bound: float
    class_scores: Mapping[str, float]
    constraint_values: Mapping[str, float]
    trace_error: float
    min_eigenvalue: float
    repair_frobenius: float
    verifier_precision_dps: int
    proposal: Mapping[str, Any]

    @property
    def proof_matrix_hash(self) -> str:
        return complex_matrix_manifest(
            np.asarray(self.density.data, dtype=complex)
        )["sha256"]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "evidence_type": "ExistencePrimalEvidence",
            "proof_matrix": complex_matrix_manifest(
                np.asarray(self.density.data, dtype=complex)
            ),
            "proof_matrix_hash": self.proof_matrix_hash,
            "radial_upper_bound": self.radial_upper_bound,
            "class_scores": dict(self.class_scores),
            "constraint_values": dict(self.constraint_values),
            "trace_error": self.trace_error,
            "min_eigenvalue": self.min_eigenvalue,
            "repair_frobenius": self.repair_frobenius,
            "verifier_precision_dps": self.verifier_precision_dps,
            "proposal": dict(self.proposal),
        }


@dataclass(frozen=True)
class DualSeparationEvidence:
    radial_lower_bound: float
    separation_numerator: float
    calibration_support: float
    state_support_upper: float
    multiplier_repair: float
    separator_z: tuple[float, ...]
    constraint_multipliers: tuple[float, ...]
    constraint_multiplier_labels: tuple[str, ...]
    verifier_precision_dps: int
    proposal: Mapping[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "evidence_type": "DualSeparationEvidence",
            "radial_lower_bound": self.radial_lower_bound,
            "separation_numerator": self.separation_numerator,
            "calibration_support": self.calibration_support,
            "state_support_upper": self.state_support_upper,
            "multiplier_repair": self.multiplier_repair,
            "separator_z": list(self.separator_z),
            "constraint_multipliers": list(self.constraint_multipliers),
            "constraint_multiplier_labels": list(self.constraint_multiplier_labels),
            "verifier_precision_dps": self.verifier_precision_dps,
            "proposal": dict(self.proposal),
        }


@dataclass(frozen=True)
class ConstraintRegistrySeparationEvidence:
    separation_margin: float
    spectral_lower_bound: float
    registered_upper_bound: float
    multiplier_repair: float
    separator_z: tuple[float, ...]
    constraint_multipliers: tuple[float, ...]
    constraint_multiplier_labels: tuple[str, ...]
    verifier_precision_dps: int
    proposal: Mapping[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "evidence_type": "ConstraintRegistrySeparationEvidence",
            "separation_margin": self.separation_margin,
            "spectral_lower_bound": self.spectral_lower_bound,
            "registered_upper_bound": self.registered_upper_bound,
            "multiplier_repair": self.multiplier_repair,
            "separator_z": list(self.separator_z),
            "constraint_multipliers": list(self.constraint_multipliers),
            "constraint_multiplier_labels": list(self.constraint_multiplier_labels),
            "verifier_precision_dps": self.verifier_precision_dps,
            "proposal": dict(self.proposal),
        }


DualEvidence = DualSeparationEvidence | ConstraintRegistrySeparationEvidence


@dataclass(frozen=True)
class ExistenceClaim:
    semantic_status: SemanticStatus = "unresolved"
    evaluation_status: EvaluationStatus = "evaluated"
    verified_upper_bound: float | None = None
    verified_lower_bound: float | None = None
    primal_evidence: ExistencePrimalEvidence | None = None
    dual_evidence: DualEvidence | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    decision_reason: str | None = None

    @property
    def status(self) -> ExistenceStatus:
        if self.evaluation_status == "evaluated":
            return self.semantic_status
        return self.evaluation_status

    def to_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "semantic_status": self.semantic_status,
            "evaluation_status": self.evaluation_status,
            "verified_upper_bound": self.verified_upper_bound,
            "verified_lower_bound": self.verified_lower_bound,
            "evidence_type": (
                "ExistencePrimalEvidence"
                if self.primal_evidence is not None and self.dual_evidence is None
                else self.dual_evidence.to_manifest()["evidence_type"]
                if self.dual_evidence is not None and self.primal_evidence is None
                else None
            ),
            "evidence_types": [
                evidence_type
                for evidence_type, present in (
                    ("ExistencePrimalEvidence", self.primal_evidence is not None),
                    (
                        None
                        if self.dual_evidence is None
                        else self.dual_evidence.to_manifest()["evidence_type"],
                        self.dual_evidence is not None,
                    ),
                )
                if present and evidence_type is not None
            ],
            "decision_reason": self.decision_reason,
            "bound_interval": {
                "verified_lower_bound": self.verified_lower_bound,
                "verified_upper_bound": self.verified_upper_bound,
            },
            "primal_evidence": (
                None if self.primal_evidence is None else self.primal_evidence.to_manifest()
            ),
            "dual_evidence": (
                None if self.dual_evidence is None else self.dual_evidence.to_manifest()
            ),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class CanonicalWitness:
    density: DensityMatrix
    distance_upper_bound: float
    distance_lower_bound: float
    objective_gap: float
    representation_error_bound: float
    verifier_precision_dps: int
    certificate_instance_hash: str

    @property
    def rho(self) -> DensityMatrix:
        return self.density

    @property
    def downstream_scope(self) -> Literal["pipeline"]:
        return "pipeline"

    @property
    def witness_hash(self) -> str:
        return stable_hash(
            {
                "certificate_instance_hash": self.certificate_instance_hash,
                "density_sha256": complex_matrix_manifest(
                    np.asarray(self.density.data, dtype=complex)
                )["sha256"],
            }
        )

    def to_manifest(self) -> dict[str, Any]:
        density = np.asarray(self.density.data, dtype=complex)
        return {
            "object_type": "CanonicalWitness",
            "witness_hash": self.witness_hash,
            "certificate_instance_hash": self.certificate_instance_hash,
            "dim": int(density.shape[0]),
            "trace": float(np.real(np.trace(density))),
            "density_matrix": complex_matrix_manifest(density),
            "distance_upper_bound": self.distance_upper_bound,
            "distance_lower_bound": self.distance_lower_bound,
            "objective_gap": self.objective_gap,
            "representation_error_bound": self.representation_error_bound,
            "verifier_precision_dps": self.verifier_precision_dps,
        }


@dataclass(frozen=True)
class SelectorLowerBoundEvidence:
    distance_lower_bound: float
    beta: float
    center_value: float
    separator_norm: float
    auxiliary_floor: float
    separator_z: tuple[float, ...]
    constraint_multipliers: tuple[float, ...]
    constraint_multiplier_labels: tuple[str, ...]
    auxiliary_operator: np.ndarray
    verifier_precision_dps: int
    proposal: Mapping[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "evidence_type": "SelectorLowerBoundEvidence",
            "distance_lower_bound": self.distance_lower_bound,
            "beta": self.beta,
            "center_value": self.center_value,
            "separator_norm": self.separator_norm,
            "auxiliary_floor": self.auxiliary_floor,
            "separator_z": list(self.separator_z),
            "constraint_multipliers": list(self.constraint_multipliers),
            "constraint_multiplier_labels": list(self.constraint_multiplier_labels),
            "auxiliary_operator": complex_matrix_manifest(self.auxiliary_operator),
            "verifier_precision_dps": self.verifier_precision_dps,
            "proposal": dict(self.proposal),
        }


@dataclass(frozen=True)
class SelectorClaim:
    semantic_status: SelectorSemanticStatus = "unresolved"
    evaluation_status: SelectorEvaluationStatus = "evaluated"
    witness: CanonicalWitness | None = None
    feasible_evidence: ExistencePrimalEvidence | None = None
    lower_bound_evidence: SelectorLowerBoundEvidence | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> SelectorStatus:
        if self.evaluation_status == "evaluated":
            return self.semantic_status
        return self.evaluation_status

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "semantic_status": self.semantic_status,
            "evaluation_status": self.evaluation_status,
            "objective_gap": None,
            "representation_error_bound": None,
            "evidence_type": None,
            "feasible_evidence": (
                None
                if self.feasible_evidence is None
                else self.feasible_evidence.to_manifest()
            ),
            "lower_bound_evidence": (
                None
                if self.lower_bound_evidence is None
                else self.lower_bound_evidence.to_manifest()
            ),
            "diagnostics": dict(self.diagnostics),
        }
        if self.witness is not None:
            payload["evidence_type"] = "SelectorOptimalityEvidence"
            payload["objective_gap"] = self.witness.objective_gap
            payload["representation_error_bound"] = self.witness.representation_error_bound
            payload["canonical_witness"] = self.witness.to_manifest()
        return payload


@dataclass(frozen=True)
class ILGCCertificate:
    fiber: FiberSignature
    calibration_bodies: Mapping[str, CalibrationBody]
    constraint_registry: ConstraintRegistry
    precondition_status: PreconditionStatus
    observation_table_hash: str
    quantum_claim: ExistenceClaim
    bi_existence_claim: ExistenceClaim
    selector_claim: SelectorClaim
    emitted_objects: tuple[str, ...]
    blocked_objects: tuple[str, ...]
    solver_attempts: Mapping[str, Any]
    verifier_diagnostics: Mapping[str, Any]
    legacy_projection_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def certificate_instance_hash(self) -> str:
        return stable_hash(
            {
                "formulation_version": self.fiber.formulation_version,
                "fiber_hash": self.fiber.fiber_hash,
                "measurement_map_hash": self.fiber.measurement_map_hash,
                "observation_table_hash": self.observation_table_hash,
                "calibration_manifest_hash": calibration_bodies_hash(
                    self.calibration_bodies
                ),
                "constraint_registry_hash": self.constraint_registry.registry_hash,
            }
        )

    def to_artifact(self) -> dict[str, Any]:
        payload = {
            "formulation_version": self.fiber.formulation_version,
            "fiber_signature": self.fiber.to_manifest(),
            "fiber_hash": self.fiber.fiber_hash,
            "certificate_instance_hash": self.certificate_instance_hash,
            "calibration_manifest_hash": calibration_bodies_hash(
                self.calibration_bodies
            ),
            "effective_measurement_map_hash": self.fiber.measurement_map_hash,
            "observation_table_hash": self.observation_table_hash,
            "constraint_registry": self.constraint_registry.to_manifest(),
            "precondition_status": self.precondition_status,
            "quantum_claim": self.quantum_claim.to_manifest(),
            "bi_existence_claim": self.bi_existence_claim.to_manifest(),
            "selector_claim": self.selector_claim.to_manifest(),
            "emitted_objects": list(self.emitted_objects),
            "blocked_objects": list(self.blocked_objects),
            "solver_attempts": dict(self.solver_attempts),
            "verifier_diagnostics": dict(self.verifier_diagnostics),
            "legacy_projection_diagnostics": dict(self.legacy_projection_diagnostics),
        }
        if self.selector_claim.witness is not None:
            payload["canonical_witness"] = self.selector_claim.witness.to_manifest()
        return payload
