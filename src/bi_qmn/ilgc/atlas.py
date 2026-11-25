from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from bi_qmn.p0.atlas import AtlasComputation, AtlasFaults, build_atlas
from bi_qmn.core.types import MeasurementTable, to_jsonable
from .bootstrap import SelectorBootstrapStability
from .types import CanonicalWitness, ILGCCertificate


AtlasRouteStatus = Literal[
    "accepted",
    "atlas_rejected",
    "blocked_by_selector",
    "blocked_by_bootstrap",
]


@dataclass(frozen=True)
class ILGCAtlasRoute:
    status: AtlasRouteStatus
    atlas: AtlasComputation | None
    emitted_objects: tuple[str, ...]
    blocked_objects: tuple[str, ...]
    diagnostics: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "emitted_objects": list(self.emitted_objects),
            "blocked_objects": list(self.blocked_objects),
            "diagnostics": dict(self.diagnostics),
            "atlas": None if self.atlas is None else to_jsonable(self.atlas),
        }


def build_ilgc_atlas(
    certificate: ILGCCertificate,
    table: MeasurementTable,
    atlas_thresholds: dict,
    scale: dict,
    stability: SelectorBootstrapStability | None,
    *,
    faults: AtlasFaults | None = None,
) -> ILGCAtlasRoute:
    witness = certificate.selector_claim.witness
    if witness is None:
        return ILGCAtlasRoute(
            status="blocked_by_selector",
            atlas=None,
            emitted_objects=("certificate_statuses",),
            blocked_objects=(
                "canonical_witness",
                "conditional_slices",
                "vertices",
                "edges",
                "atlas",
                "paths",
            ),
            diagnostics={
                "selector_status": certificate.selector_claim.status,
                "reason": "canonical_witness_not_emitted",
            },
        )
    return build_ilgc_atlas_from_witness(
        witness,
        table,
        atlas_thresholds,
        scale,
        stability,
        faults=faults,
    )


def build_ilgc_atlas_from_witness(
    witness: CanonicalWitness,
    table: MeasurementTable,
    atlas_thresholds: dict,
    scale: dict,
    stability: SelectorBootstrapStability | None,
    *,
    faults: AtlasFaults | None = None,
) -> ILGCAtlasRoute:
    if not isinstance(witness, CanonicalWitness):
        raise TypeError("ILGC atlas requires a CanonicalWitness")
    if stability is None:
        return ILGCAtlasRoute(
            status="blocked_by_bootstrap",
            atlas=None,
            emitted_objects=("canonical_witness",),
            blocked_objects=("vertices", "edges", "atlas", "paths"),
            diagnostics={"reason": "selector_bootstrap_not_evaluated"},
        )

    uncertainties = (
        {
            tau: float(value)
            for tau, value in stability.distance_by_tau.items()
            if value is not None
        }
        if stability.qualified
        else None
    )
    atlas = build_atlas(
        witness,
        table,
        atlas_thresholds,
        scale,
        faults=faults,
        witness_uncertainties=uncertainties,
        selector_emission_lcb=(
            stability.selector_emission_lcb if stability.qualified else -1.0
        ),
        pi_sel_min=stability.pi_sel_min,
        selector_effective_count=stability.B_sel,
        selector_effective_min=stability.B_sel_min,
    )
    diagnostics = {
        "claim_scope": stability.claim_scope,
        "selector_emission_lcb": stability.selector_emission_lcb,
        "pi_sel_min": stability.pi_sel_min,
        "B_sel": stability.B_sel,
        "B_sel_min": stability.B_sel_min,
        "bootstrap_qualified": stability.qualified,
        "bootstrap_qualification_reasons": list(stability.qualification_reasons),
    }
    if not stability.qualified:
        return ILGCAtlasRoute(
            status="blocked_by_bootstrap",
            atlas=atlas,
            emitted_objects=(
                "canonical_witness",
                "selector_bootstrap_stability",
                "vertex_exclusions",
            ),
            blocked_objects=("admitted_vertices", "edges", "atlas", "paths"),
            diagnostics={**diagnostics, "reason": "selector_bootstrap_unqualified"},
        )
    if atlas.accepted:
        return ILGCAtlasRoute(
            status="accepted",
            atlas=atlas,
            emitted_objects=(
                "canonical_witness",
                "selector_bootstrap_stability",
                "admitted_vertices",
                "admitted_edges",
                "anchor_atlases",
                "anchor_paths",
            ),
            blocked_objects=(),
            diagnostics=diagnostics,
        )
    return ILGCAtlasRoute(
        status="atlas_rejected",
        atlas=atlas,
        emitted_objects=(
            "canonical_witness",
            "selector_bootstrap_stability",
            "vertex_exclusions",
            "edge_exclusions",
            "anchor_exclusions",
        ),
        blocked_objects=("atlas", "paths"),
        diagnostics=diagnostics,
    )
