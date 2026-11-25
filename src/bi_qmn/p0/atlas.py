from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from qiskit.quantum_info import DensityMatrix

from bi_qmn.core.linalg import bures_distance, fourier_matrix, kron, matrix_projector
from bi_qmn.core.noise import outcome_key
from bi_qmn.core.types import (
    AnchorAtlas,
    AnchorExclusion,
    CertifiedPath,
    CertifiedSlice,
    CertifiedWitness,
    EdgeExclusion,
    MeasurementTable,
    VertexExclusion,
)


PAULI = {
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


class AtlasWitness(Protocol):
    @property
    def rho(self) -> DensityMatrix: ...

    @property
    def downstream_scope(self) -> str: ...


@dataclass
class AtlasComputation:
    admitted_vertices: dict[int, CertifiedSlice]
    excluded_vertices: list[VertexExclusion]
    admitted_edges_by_anchor: dict[int, set[tuple[int, int]]]
    excluded_edges_by_anchor: dict[int, list[EdgeExclusion]]
    anchor_atlases: dict[int, AnchorAtlas]
    anchor_paths: dict[int, CertifiedPath]
    excluded_anchors: list[AnchorExclusion]

    @property
    def accepted(self) -> bool:
        return bool(self.anchor_atlases)


@dataclass(frozen=True)
class AtlasFaults:
    vertex_tau: int | None = None
    edge_anchor: int | None = None
    edge: tuple[int, int] | None = None


def _partial_trace_clock(block: np.ndarray, d_C: int = 4, d_S: int = 2) -> np.ndarray:
    reshaped = block.reshape(d_C, d_S, d_C, d_S)
    reduced = np.zeros((d_S, d_S), dtype=complex)
    for clock in range(d_C):
        reduced += reshaped[clock, :, clock, :]
    return reduced


def conditional_slices(
    witness: AtlasWitness,
    table: MeasurementTable,
    K: int = 4,
    witness_uncertainties: dict[int, float] | None = None,
) -> dict[int, CertifiedSlice]:
    F = fourier_matrix(K)
    rho = np.asarray(witness.rho.data, dtype=complex)
    slices: dict[int, CertifiedSlice] = {}
    for tau in range(K):
        P_C = matrix_projector(F[tau, :])
        P = kron(P_C, np.eye(2, dtype=complex))
        block = P @ rho @ P
        probability = float(np.real_if_close(np.trace(block), tol=1000))
        if probability <= 0.0:
            system_state = np.eye(2, dtype=complex) / 2.0
        else:
            system_state = _partial_trace_clock(block, d_C=K, d_S=2) / probability
        align = slice_alignment(system_state, tau, table)
        slices[tau] = CertifiedSlice(
            tau=tau,
            probability=probability,
            rho=DensityMatrix(system_state),
            align=align,
            witness_uncertainty=0.0 if witness_uncertainties is None else float(witness_uncertainties.get(tau, 0.0)),
            witness_hash=getattr(witness, "witness_hash", None),
            certificate_instance_hash=getattr(
                witness, "certificate_instance_hash", None
            ),
        )
    return slices


def _table_expectation(table: MeasurementTable, tau: int, observable: str) -> float | None:
    context_id = f"D,{observable}"
    if context_id not in table.probabilities:
        return None
    probs = table.probabilities[context_id]
    p0 = probs[outcome_key(tau, 0)]
    p1 = probs[outcome_key(tau, 1)]
    denom = p0 + p1
    if denom <= 0.0:
        return None
    return (p0 - p1) / denom


def slice_alignment(system_state: np.ndarray, tau: int, table: MeasurementTable) -> float:
    errors: list[float] = []
    for observable, pauli in PAULI.items():
        target = _table_expectation(table, tau, observable)
        if target is None:
            continue
        pred = float(np.real_if_close(np.trace(pauli @ system_state), tol=1000))
        errors.append((pred - target) ** 2)
    if not errors:
        return 0.0
    return float(np.sqrt(sum(errors) / len(errors)))


def admit_vertices(
    slices: dict[int, CertifiedSlice],
    p_min: float,
    theta_align: float,
    theta_wit: float,
    faults: AtlasFaults | None = None,
    selector_emission_lcb: float | None = None,
    pi_sel_min: float | None = None,
    selector_effective_count: int | None = None,
    selector_effective_min: int | None = None,
) -> tuple[dict[int, CertifiedSlice], list[VertexExclusion]]:
    admitted: dict[int, CertifiedSlice] = {}
    excluded: list[VertexExclusion] = []
    fault_tau = None if faults is None else faults.vertex_tau
    selector_gate_values = (
        selector_emission_lcb,
        pi_sel_min,
        selector_effective_count,
        selector_effective_min,
    )
    if any(value is not None for value in selector_gate_values) and any(
        value is None for value in selector_gate_values
    ):
        raise ValueError("selector emission gate requires all four values")
    selector_qualified = (
        True
        if selector_emission_lcb is None
        else float(selector_emission_lcb) >= float(pi_sel_min)
        and int(selector_effective_count) >= int(selector_effective_min)
    )
    for tau, certified_slice in slices.items():
        if not selector_qualified:
            excluded.append(
                VertexExclusion(tau=tau, reason="SelectorEmissionUnqualified")
            )
        elif tau == fault_tau:
            excluded.append(VertexExclusion(tau=tau, reason="AlignFail"))
        elif certified_slice.probability < p_min:
            excluded.append(VertexExclusion(tau=tau, reason="SupportFail"))
        elif certified_slice.align > theta_align:
            excluded.append(VertexExclusion(tau=tau, reason="AlignFail"))
        elif certified_slice.witness_uncertainty > theta_wit:
            excluded.append(VertexExclusion(tau=tau, reason="WitnessUnstable"))
        else:
            admitted[tau] = certified_slice
    return admitted, excluded


def _edge_cost(s0: CertifiedSlice, s1: CertifiedSlice) -> float:
    return bures_distance(np.asarray(s0.rho.data, dtype=complex), np.asarray(s1.rho.data, dtype=complex))


def build_anchor_atlas(
    admitted_vertices: dict[int, CertifiedSlice],
    K: int,
    anchor: int,
    local_window: list[int],
    n_loc: int,
    theta_tr: float,
    theta_dir: float,
    faults: AtlasFaults | None = None,
) -> tuple[AnchorAtlas | None, set[tuple[int, int]], list[EdgeExclusion], AnchorExclusion | None]:
    physical_taus = [(anchor + u) % K for u in local_window]
    local_vertices = [u for u, tau in zip(local_window, physical_taus, strict=True) if tau in admitted_vertices]
    if len(local_vertices) < n_loc:
        return None, set(), [], AnchorExclusion(anchor=anchor, reason="InsufficientSupport")

    chain_edges = list(zip(local_vertices[:-1], local_vertices[1:], strict=True))
    admitted_edges: set[tuple[int, int]] = set()
    excluded_edges: list[EdgeExclusion] = []
    for u, v in chain_edges:
        tau_u = (anchor + u) % K
        tau_v = (anchor + v) % K
        tr = _edge_cost(admitted_vertices[tau_u], admitted_vertices[tau_v])
        direction = abs(admitted_vertices[tau_u].probability - admitted_vertices[tau_v].probability)
        forced = (
            faults is not None
            and faults.edge_anchor == anchor
            and faults.edge == (u, v)
        )
        if forced or tr > theta_tr:
            excluded_edges.append(EdgeExclusion(edge=(u, v), anchor=anchor, reason="TransportFail"))
        elif direction > theta_dir:
            excluded_edges.append(EdgeExclusion(edge=(u, v), anchor=anchor, reason="DirectionalFail"))
        else:
            admitted_edges.add((u, v))

    if any(edge not in admitted_edges for edge in chain_edges):
        return None, admitted_edges, excluded_edges, AnchorExclusion(anchor=anchor, reason="NoAdmissiblePath")

    cost = sum(
        _edge_cost(
            admitted_vertices[(anchor + u) % K],
            admitted_vertices[(anchor + v) % K],
        )
        for u, v in chain_edges
    )
    path = CertifiedPath(
        anchor=anchor,
        local_nodes=tuple(local_vertices),
        physical_taus=tuple((anchor + u) % K for u in local_vertices),
        cost=cost,
    )
    atlas = AnchorAtlas(
        anchor=anchor,
        local_vertices=tuple(local_vertices),
        physical_taus=path.physical_taus,
        admitted_edges=tuple(sorted(admitted_edges)),
        chain=path,
    )
    return atlas, admitted_edges, excluded_edges, None


def build_atlas(
    witness: AtlasWitness,
    table: MeasurementTable,
    atlas_thresholds: dict,
    scale: dict,
    faults: AtlasFaults | None = None,
    witness_uncertainties: dict[int, float] | None = None,
    selector_emission_lcb: float | None = None,
    pi_sel_min: float | None = None,
    selector_effective_count: int | None = None,
    selector_effective_min: int | None = None,
) -> AtlasComputation:
    if witness.downstream_scope != "pipeline":
        raise ValueError("boundary bootstrap witness cannot be used for atlas emission")
    K = 4
    slices = conditional_slices(witness, table, K=K, witness_uncertainties=witness_uncertainties)
    admitted_vertices, excluded_vertices = admit_vertices(
        slices,
        p_min=float(atlas_thresholds["p_min"]),
        theta_align=float(atlas_thresholds["theta_align"]),
        theta_wit=float(atlas_thresholds["theta_wit"]),
        faults=faults,
        selector_emission_lcb=selector_emission_lcb,
        pi_sel_min=pi_sel_min,
        selector_effective_count=selector_effective_count,
        selector_effective_min=selector_effective_min,
    )
    admitted_edges_by_anchor: dict[int, set[tuple[int, int]]] = {}
    excluded_edges_by_anchor: dict[int, list[EdgeExclusion]] = {}
    anchor_atlases: dict[int, AnchorAtlas] = {}
    anchor_paths: dict[int, CertifiedPath] = {}
    excluded_anchors: list[AnchorExclusion] = []
    for anchor in range(K):
        atlas, admitted_edges, excluded_edges, anchor_exclusion = build_anchor_atlas(
            admitted_vertices=admitted_vertices,
            K=K,
            anchor=anchor,
            local_window=list(scale["local_window"]),
            n_loc=int(scale["n_loc"]),
            theta_tr=float(atlas_thresholds["theta_tr"]),
            theta_dir=float(atlas_thresholds["theta_dir"]),
            faults=faults,
        )
        admitted_edges_by_anchor[anchor] = admitted_edges
        excluded_edges_by_anchor[anchor] = excluded_edges
        if atlas is not None:
            anchor_atlases[anchor] = atlas
            anchor_paths[anchor] = atlas.chain
        if anchor_exclusion is not None:
            excluded_anchors.append(anchor_exclusion)
    return AtlasComputation(
        admitted_vertices=admitted_vertices,
        excluded_vertices=excluded_vertices,
        admitted_edges_by_anchor=admitted_edges_by_anchor,
        excluded_edges_by_anchor=excluded_edges_by_anchor,
        anchor_atlases=anchor_atlases,
        anchor_paths=anchor_paths,
        excluded_anchors=excluded_anchors,
    )
