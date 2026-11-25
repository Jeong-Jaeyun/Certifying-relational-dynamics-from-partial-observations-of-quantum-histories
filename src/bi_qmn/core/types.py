from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

import numpy as np
from qiskit.quantum_info import DensityMatrix


Status = Literal["accept", "abstain", "blocked", "failure"]
OutcomeName = Literal[
    "ConstructionFailure",
    "ClockPreconditionFailure",
    "AdequacyFailure",
    "CoverageFailure",
    "CertificateAbstain",
    "NumericalAbstain",
    "CertifiedNoAtlas",
    "AtlasAccepted",
]


@dataclass(frozen=True)
class MeasurementContext:
    clock: Literal["D", "F"]
    system: Literal["X", "Y", "Z"]

    @property
    def context_id(self) -> str:
        return f"{self.clock},{self.system}"

    @property
    def context_class(self) -> Literal["diag", "coh"]:
        return "diag" if self.clock == "D" else "coh"


@dataclass
class MeasurementTable:
    probabilities: dict[str, dict[str, float]]
    contexts: list[MeasurementContext]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_density: DensityMatrix | None = None

    def copy_without_source(self) -> "MeasurementTable":
        return MeasurementTable(
            probabilities={k: dict(v) for k, v in self.probabilities.items()},
            contexts=list(self.contexts),
            metadata=dict(self.metadata),
            source_density=None,
        )


@dataclass(frozen=True)
class CertifiedWitness:
    rho: DensityMatrix
    R_Q: dict[str, float]
    R_BI: dict[str, float]
    Delta_con: dict[str, float]
    solver_diagnostics: dict[str, Any]
    witness_kind: Literal["canonical", "boundary_bootstrap"] = "canonical"
    downstream_scope: Literal["pipeline", "bootstrap_uncertainty"] = "pipeline"


@dataclass(frozen=True)
class CertifiedSlice:
    tau: int
    probability: float
    rho: DensityMatrix
    align: float
    witness_uncertainty: float = 0.0
    witness_hash: str | None = None
    certificate_instance_hash: str | None = None


@dataclass(frozen=True)
class VertexExclusion:
    tau: int
    reason: Literal[
        "SupportFail",
        "AlignFail",
        "WitnessUnstable",
        "SelectorEmissionUnqualified",
    ]


@dataclass(frozen=True)
class EdgeExclusion:
    edge: tuple[int, int]
    anchor: int
    reason: Literal["TransportFail", "DirectionalFail"]


@dataclass(frozen=True)
class AnchorExclusion:
    anchor: int
    reason: Literal["InsufficientSupport", "NoAdmissiblePath"]


@dataclass(frozen=True)
class CertifiedPath:
    anchor: int
    local_nodes: tuple[int, ...]
    physical_taus: tuple[int, ...]
    cost: float


@dataclass(frozen=True)
class AnchorAtlas:
    anchor: int
    local_vertices: tuple[int, ...]
    physical_taus: tuple[int, ...]
    admitted_edges: tuple[tuple[int, int], ...]
    chain: CertifiedPath


@dataclass(frozen=True)
class AnchorAggregate:
    median: float
    worst: float
    count: int


@dataclass(frozen=True)
class AnchorDetail:
    anchor: int
    cost: float
    path: tuple[int, ...]


@dataclass
class PipelineBase:
    status: Status
    earliest_cause: str | None
    emitted_objects: list[str] = field(default_factory=list)
    blocked_objects: list[str] = field(default_factory=list)

    @property
    def outcome_name(self) -> OutcomeName:
        return self.__class__.__name__  # type: ignore[return-value]


@dataclass
class ConstructionFailure(PipelineBase):
    reason: str = "construction_failure"


@dataclass
class ClockPreconditionFailure(PipelineBase):
    selected_clock: None = None
    stage_a_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdequacyFailure(PipelineBase):
    adequacy_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoverageFailure(PipelineBase):
    coverage_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class CertificateAbstain(PipelineBase):
    raw_qhat: MeasurementTable | None = None
    R_Q: dict[str, float] = field(default_factory=dict)
    Delta_con: dict[str, float] = field(default_factory=dict)
    cause: Literal["RQ_exceeded", "DeltaCon_exceeded"] = "RQ_exceeded"


@dataclass
class NumericalAbstain(PipelineBase):
    solver_diagnostics: dict[str, Any] = field(default_factory=dict)
    cause: Literal["numerical_abstention"] = "numerical_abstention"


@dataclass
class CertifiedNoAtlas(PipelineBase):
    witness: CertifiedWitness | None = None
    admitted_vertices: dict[int, CertifiedSlice] = field(default_factory=dict)
    excluded_vertices: list[VertexExclusion] = field(default_factory=list)
    admitted_edges_by_anchor: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    excluded_edges_by_anchor: dict[int, list[EdgeExclusion]] = field(default_factory=dict)
    reason: Literal["NoAnchorWithValidPath"] = "NoAnchorWithValidPath"


@dataclass
class AtlasAccepted(PipelineBase):
    witness: CertifiedWitness | None = None
    admitted_vertices: dict[int, CertifiedSlice] = field(default_factory=dict)
    excluded_vertices: list[VertexExclusion] = field(default_factory=list)
    admitted_edges_by_anchor: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    excluded_edges_by_anchor: dict[int, list[EdgeExclusion]] = field(default_factory=dict)
    anchor_atlases: dict[int, AnchorAtlas] = field(default_factory=dict)
    anchor_paths: dict[int, CertifiedPath] = field(default_factory=dict)
    excluded_anchors: list[AnchorExclusion] = field(default_factory=list)


PipelineOutcome = (
    ConstructionFailure
    | ClockPreconditionFailure
    | AdequacyFailure
    | CoverageFailure
    | CertificateAbstain
    | NumericalAbstain
    | CertifiedNoAtlas
    | AtlasAccepted
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, DensityMatrix):
        return {
            "dim": int(value.dim[0]) if isinstance(value.dim, tuple) else int(value.dim),
            "trace": float(np.real_if_close(np.trace(value.data))),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(list(value))
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    return value
