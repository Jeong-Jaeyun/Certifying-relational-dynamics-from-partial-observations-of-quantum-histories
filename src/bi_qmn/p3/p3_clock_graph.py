"""Stage-A clock coordinate map and frozen relational graph for Model C v2 (development).

v1 bound the Stage-A selected-clock hash to the graph as provenance only; node coordinates were
not derived from that clock. v2 derives every node coordinate tau(a, u) = (a + u) mod K from the
selected clock register, exactly as the P1 atlas/path layer maps anchor-local offsets to
physical clock labels, and attaches relation types to the declared stride/direction.

Capacity-matched relation controls (same |V|, |E|, relation-type counts, parameter count and
feature dimension) are generated here:

* relation permutation  - relation labels shuffled across edges (R-GCN style edge-type shuffle)
* coordinate permutation - the node -> tau coordinate assignment is permuted
* degree-preserving rewire - Maslov & Sneppen (2002) double-edge swaps on the undirected skeleton

The graph is an estimator-input scaffold, not an atlas, and is never conditioned on a certificate,
selector, witness, path, query target, or test result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


class ClockFrameMismatch(ValueError):
    """The Stage-A selected clock differs from the register on which D/F contexts were measured."""


def _short_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class ClockCoordinateMap:
    stage_a_path: str
    stage_a_sha256: str
    selected_clock: str
    clock_qubits: tuple[int, ...]
    system_qubits: tuple[int, ...]
    K: int

    def tau(self, anchor: int, offset: int) -> int:
        return (int(anchor) + int(offset)) % self.K

    @property
    def map_hash(self) -> str:
        return _short_hash(
            {
                "stage_a_sha256": self.stage_a_sha256,
                "selected_clock": self.selected_clock,
                "clock_qubits": list(self.clock_qubits),
                "system_qubits": list(self.system_qubits),
                "K": self.K,
                "rule": "tau=(anchor+offset) mod K",
            }
        )


def _parse_selected_clock(selected: str) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    try:
        label, partition = selected.split(":", 1)
        clock_part, system_part = partition.split("|", 1)
        clock = tuple(int(value) for value in clock_part.split(",") if value != "")
        system = tuple(int(value) for value in system_part.split(",") if value != "")
    except ValueError as error:
        raise ValueError(f"unparseable Stage-A selected clock: {selected!r}") from error
    if not clock or not system or set(clock) & set(system):
        raise ValueError(f"invalid Stage-A clock/system partition: {selected!r}")
    return label, clock, system


def clock_coordinate_map_from_stage_a(
    stage_a: Mapping,
    *,
    stage_a_path: str,
    stage_a_sha256: str,
    generator_config: Mapping,
) -> ClockCoordinateMap:
    """Derive the clock coordinate map from an accepted Stage-A artifact.

    The map is accepted only if the selected register equals the register on which the frozen D/F
    measurement contexts are defined. A different selection would require new measurements, not
    an outcome relabelling, so it raises ``ClockFrameMismatch``.
    """
    if stage_a.get("status") != "accept" or not stage_a.get("selected_clock"):
        raise ValueError("Model C v2 requires an accepted Stage-A artifact with a selected clock")
    if stage_a.get("uses_estimator_performance") or stage_a.get("uses_path_quality"):
        raise ValueError("Stage-A selection must be estimator-free and path-free")
    selected = str(stage_a["selected_clock"])
    _, parsed_clock, parsed_system = _parse_selected_clock(selected)
    matching = [
        record
        for record in stage_a.get("records", [])
        if record.get("candidate_hash") == selected and record.get("admissible") is True
    ]
    if len(matching) != 1:
        raise ValueError("selected clock must correspond to exactly one admissible Stage-A record")
    record = matching[0]
    clock = tuple(int(value) for value in record["clock_qubits"])
    system = tuple(int(value) for value in record["system_qubits"])
    if clock != parsed_clock or system != parsed_system:
        raise ValueError("Stage-A selected-clock string and record partition disagree")
    K = 2 ** len(clock)
    dimensions = generator_config.get("dimensions", {})
    if int(dimensions.get("K", K)) != K:
        raise ValueError("selected clock dimension disagrees with the generator K")
    declared = generator_config["declared_clock"]
    if tuple(int(v) for v in declared["clock_qubits"]) != clock or tuple(int(v) for v in declared["system_qubits"]) != system:
        raise ClockFrameMismatch(
            "Stage-A selected a clock register different from the measured D/F frame; "
            "relabelling outcomes would not realise that clock"
        )
    return ClockCoordinateMap(
        stage_a_path=stage_a_path,
        stage_a_sha256=stage_a_sha256,
        selected_clock=selected,
        clock_qubits=clock,
        system_qubits=system,
        K=K,
    )


@dataclass(frozen=True)
class ClockEdge:
    source: int
    target: int
    relation: str
    stride: int
    sign: int
    scale: float

    @property
    def lag(self) -> int:
        """Signed clock lag carried by the relation type (forward +h, backward -h)."""
        return self.sign * self.stride


@dataclass(frozen=True)
class ClockCoordinateGraph:
    nodes: tuple[tuple[int, int], ...]
    node_tau: tuple[int, ...]
    edges: tuple[ClockEdge, ...]
    K: int
    coordinate_map_hash: str
    control: str = "declared"

    @property
    def relation_types(self) -> tuple[str, ...]:
        return tuple(sorted({edge.relation for edge in self.edges}))

    @property
    def graph_hash(self) -> str:
        return _short_hash(
            {
                "nodes": [list(node) for node in self.nodes],
                "node_tau": list(self.node_tau),
                "edges": [edge.__dict__ for edge in self.edges],
                "K": self.K,
                "coordinate_map_hash": self.coordinate_map_hash,
                "control": self.control,
            }
        )

    def capacity_signature(self) -> dict:
        counts: dict[str, int] = {}
        for edge in self.edges:
            counts[edge.relation] = counts.get(edge.relation, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), "relation_counts": dict(sorted(counts.items()))}


def relation_name(sign: int, stride: int) -> str:
    return f"{'forward' if sign > 0 else 'backward'}_h{int(stride)}"


def build_clock_coordinate_graph(
    coordinate_map: ClockCoordinateMap,
    *,
    anchors: Sequence[int],
    local_window: Sequence[int],
    h_max: int,
) -> ClockCoordinateGraph:
    """Nodes (a, u) with tau = (a + u) mod K; edges of stride 0 < v - u <= h_max (P1 scaffold)."""
    window = tuple(int(value) for value in local_window)
    anchor_list = tuple(int(value) for value in anchors)
    if len(window) < 2 or tuple(sorted(set(window))) != window:
        raise ValueError("local window must be strictly increasing with at least two offsets")
    if not anchor_list or len(set(anchor_list)) != len(anchor_list) or h_max < 1:
        raise ValueError("anchors must be distinct and h_max positive")
    nodes = tuple((anchor, offset) for anchor in anchor_list for offset in window)
    index = {node: position for position, node in enumerate(nodes)}
    node_tau = tuple(coordinate_map.tau(anchor, offset) for anchor, offset in nodes)
    edges: list[ClockEdge] = []
    for anchor in anchor_list:
        for left_position, left in enumerate(window):
            for right in window[left_position + 1 :]:
                stride = right - left
                if not 0 < stride <= h_max:
                    continue
                source, target = index[(anchor, left)], index[(anchor, right)]
                edges.append(ClockEdge(source, target, relation_name(+1, stride), stride, +1, 1.0 / stride))
                edges.append(ClockEdge(target, source, relation_name(-1, stride), stride, -1, 1.0 / stride))
    return ClockCoordinateGraph(
        nodes=nodes,
        node_tau=node_tau,
        edges=tuple(edges),
        K=coordinate_map.K,
        coordinate_map_hash=coordinate_map.map_hash,
    )


# ---------------------------------------------------------------------------
# Capacity-matched relation controls
# ---------------------------------------------------------------------------


def relation_permutation_control(graph: ClockCoordinateGraph, rng: np.random.Generator) -> ClockCoordinateGraph:
    labels = [(edge.relation, edge.stride, edge.sign, edge.scale) for edge in graph.edges]
    for _ in range(100):
        order = rng.permutation(len(labels))
        if np.any(order != np.arange(len(labels))) and any(labels[i][0] != labels[j][0] for j, i in enumerate(order)):
            break
    edges = tuple(
        ClockEdge(edge.source, edge.target, *labels[int(order[position])])
        for position, edge in enumerate(graph.edges)
    )
    return replace(graph, edges=edges, control="relation_permutation")


def coordinate_permutation_control(graph: ClockCoordinateGraph, rng: np.random.Generator) -> ClockCoordinateGraph:
    taus = np.asarray(graph.node_tau)
    for _ in range(100):
        permuted = taus[rng.permutation(taus.size)]
        if np.mean(permuted != taus) >= 0.5:
            break
    return replace(graph, node_tau=tuple(int(value) for value in permuted), control="coordinate_permutation")


def degree_preserving_rewire_control(
    graph: ClockCoordinateGraph,
    rng: np.random.Generator,
    *,
    swaps_per_edge: int = 10,
) -> ClockCoordinateGraph:
    """Maslov-Sneppen double-edge swaps on the undirected skeleton; stride types travel with pairs."""
    forward = [edge for edge in graph.edges if edge.sign > 0]
    pairs = [[edge.source, edge.target, edge.stride] for edge in forward]
    existing = {frozenset((u, v)) for u, v, _ in pairs}
    attempts = 0
    accepted = 0
    target_swaps = swaps_per_edge * len(pairs)
    while accepted < target_swaps and attempts < 50 * target_swaps:
        attempts += 1
        i, j = rng.choice(len(pairs), size=2, replace=False)
        u1, v1, h1 = pairs[i]
        u2, v2, h2 = pairs[j]
        if len({u1, v1, u2, v2}) < 4:
            continue
        new_a, new_b = frozenset((u1, v2)), frozenset((u2, v1))
        if new_a in existing or new_b in existing:
            continue
        existing.difference_update({frozenset((u1, v1)), frozenset((u2, v2))})
        existing.update({new_a, new_b})
        pairs[i] = [u1, v2, h1]
        pairs[j] = [u2, v1, h2]
        accepted += 1
    edges: list[ClockEdge] = []
    for u, v, stride in pairs:
        edges.append(ClockEdge(u, v, relation_name(+1, stride), stride, +1, 1.0 / stride))
        edges.append(ClockEdge(v, u, relation_name(-1, stride), stride, -1, 1.0 / stride))
    return replace(graph, edges=tuple(edges), control=f"degree_preserving_rewire:{accepted}")


def degree_sequence(graph: ClockCoordinateGraph) -> tuple[tuple[int, int], ...]:
    out_degree = [0] * len(graph.nodes)
    in_degree = [0] * len(graph.nodes)
    for edge in graph.edges:
        out_degree[edge.source] += 1
        in_degree[edge.target] += 1
    return tuple(zip(out_degree, in_degree))


CONTROL_BUILDERS = {
    "relation_permutation": relation_permutation_control,
    "coordinate_permutation": coordinate_permutation_control,
    "degree_preserving_rewire": degree_preserving_rewire_control,
}


# ---------------------------------------------------------------------------
# Finite relation generators v2: naive lattice (central-difference) Dirac form
# ---------------------------------------------------------------------------


def hermitian_unitary_gamma(latent_dim: int) -> np.ndarray:
    """A fixed Hermitian unitary (gamma^2 = I); sigma_x blocks plus a trailing 1 when r is odd."""
    if latent_dim < 2:
        raise ValueError("latent dimension must be at least two")
    gamma = np.zeros((latent_dim, latent_dim), dtype=complex)
    for block in range(latent_dim // 2):
        gamma[2 * block, 2 * block + 1] = 1.0
        gamma[2 * block + 1, 2 * block] = 1.0
    if latent_dim % 2:
        gamma[-1, -1] = 1.0
    return gamma


def relation_generator_v2(edge: ClockEdge, latent_dim: int) -> np.ndarray:
    """Forward hop +gamma, backward hop -gamma (naive lattice fermion; Wilson 1974, Kogut-Susskind 1975).

    Unlike the v1 cyclic-shift generators, forward and backward generators stay distinct at r = 2.
    """
    return float(edge.sign) * hermitian_unitary_gamma(latent_dim)
