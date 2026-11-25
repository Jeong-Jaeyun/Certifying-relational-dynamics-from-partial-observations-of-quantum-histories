from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from bi_qmn.core.generator import P0Instance
from bi_qmn.core.linalg import kron
from bi_qmn.core.measurement import effective_effect_map, measurement_table
from bi_qmn.core.types import MeasurementContext


PAULI = {
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _basis_projectors(clock_basis: str, K: int = 4) -> list[np.ndarray]:
    from bi_qmn.core.measurement import clock_projectors

    return clock_projectors(clock_basis, K)


def _signal_basis(kind: str, K: int = 4) -> list[np.ndarray]:
    if kind == "id":
        d_projectors = _basis_projectors("D", K)
        return [kron(P_C, PAULI[obs]) for P_C in d_projectors for obs in ("X", "Z")]
    if kind == "coh":
        f_projectors = _basis_projectors("F", K)
        return [kron(P_C, PAULI["Z"]) for P_C in f_projectors]
    raise ValueError(f"unknown signal basis kind {kind}")


def _allowed_context_ids(kind: str) -> set[str]:
    if kind == "id":
        return {"D,X", "D,Z"}
    if kind == "coh":
        return {"F,Z"}
    raise ValueError(f"unknown signal basis kind {kind}")


def _forward_matrix(
    contexts: Iterable[MeasurementContext],
    kind: str,
    eta: float = 0.0,
    K: int = 4,
) -> np.ndarray:
    context_list = [context for context in contexts if context.context_id in _allowed_context_ids(kind)]
    if not context_list:
        return np.zeros((0, len(_signal_basis(kind, K))), dtype=float)
    effects = effective_effect_map(context_list, K=K, eta=eta)
    basis = _signal_basis(kind, K)
    rows: list[list[float]] = []
    for context in context_list:
        for outcome in sorted(effects[context.context_id]):
            effect = np.asarray(effects[context.context_id][outcome].data, dtype=complex)
            rows.append([float(np.real_if_close(np.trace(effect @ B), tol=1000)) for B in basis])
    return np.asarray(rows, dtype=float)


def _rank_normalized_min_singular(matrix: np.ndarray, target_rank: int) -> float:
    if matrix.size == 0:
        return 0.0
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > 1.0e-10))
    if rank < target_rank:
        return 0.0
    nonzero = singular[:target_rank]
    denom = float(np.max(nonzero))
    if denom <= 0.0:
        return 0.0
    return float(np.min(nonzero) / denom)


def _separation_score(
    contexts: list[MeasurementContext],
    instances: Iterable[P0Instance] | None,
    kind: str,
    eta: float = 0.0,
) -> float:
    if instances is None:
        matrix = _forward_matrix(contexts, kind, eta=eta)
        return _rank_normalized_min_singular(matrix, len(_signal_basis(kind)))
    allowed = _allowed_context_ids(kind)
    selected = [context for context in contexts if context.context_id in allowed]
    vectors = []
    for instance in instances:
        table = measurement_table(instance.density, selected, K=instance.K, eta=eta, metadata={"eta": eta})
        vectors.append(
            np.asarray(
                [
                    table.probabilities[context.context_id][outcome]
                    for context in selected
                    for outcome in sorted(table.probabilities[context.context_id])
                ],
                dtype=float,
            )
        )
    distances = [
        float(np.linalg.norm(vectors[i] - vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    positive = [value for value in distances if value > 1.0e-10]
    if not positive:
        return 0.0
    reference = max(positive)
    return float(min(1.0, min(positive) / (0.05 * reference)))


def gamma_id_clean(
    contexts: Iterable[MeasurementContext],
    instances: Iterable[P0Instance] | None = None,
    eta: float = 0.0,
) -> float:
    matrix = _forward_matrix(contexts, "id", eta=eta)
    return _rank_normalized_min_singular(matrix, len(_signal_basis("id")))


def gamma_coh_clean(
    contexts: Iterable[MeasurementContext],
    instances: Iterable[P0Instance] | None = None,
    eta: float = 0.0,
) -> float:
    matrix = _forward_matrix(contexts, "coh", eta=eta)
    return _rank_normalized_min_singular(matrix, len(_signal_basis("coh")))


def zeta_id_tol(
    contexts: Iterable[MeasurementContext],
    instances: Iterable[P0Instance] | None = None,
    eta: float = 0.0,
) -> float:
    return _separation_score(list(contexts), instances, "id", eta=eta)


def zeta_coh_tol(
    contexts: Iterable[MeasurementContext],
    instances: Iterable[P0Instance] | None = None,
    eta: float = 0.0,
) -> float:
    return _separation_score(list(contexts), instances, "coh", eta=eta)


def adequacy_gates(
    contexts: Iterable[MeasurementContext],
    thresholds: dict,
    instances: Iterable[P0Instance] | None = None,
    eta: float = 0.0,
) -> dict[str, dict]:
    context_list = list(contexts)
    values = {
        "gamma_id_clean": (gamma_id_clean(context_list, instances=instances, eta=eta), thresholds["theta_id"]),
        "gamma_coh_clean": (gamma_coh_clean(context_list, instances=instances, eta=eta), thresholds["theta_coh"]),
        "zeta_id_tol": (zeta_id_tol(context_list, instances=instances, eta=eta), thresholds["theta_sep"]),
        "zeta_coh_tol": (zeta_coh_tol(context_list, instances=instances, eta=eta), thresholds["theta_coh_sep"]),
    }
    return {
        key: {
            "value": value,
            "threshold": threshold,
            "pass": value >= threshold,
            "method": (
                "operator_forward_map_singular_value"
                if instances is None
                else "construction_bank_separation"
            ),
        }
        for key, (value, threshold) in values.items()
    }


def adequacy_passes(
    contexts: Iterable[MeasurementContext],
    thresholds: dict,
    instances: Iterable[P0Instance] | None = None,
    eta: float = 0.0,
) -> bool:
    gates = adequacy_gates(contexts, thresholds, instances=instances, eta=eta)
    return all(bool(record["pass"]) for record in gates.values())
