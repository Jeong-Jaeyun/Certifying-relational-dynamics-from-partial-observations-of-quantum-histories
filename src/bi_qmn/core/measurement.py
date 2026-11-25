from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator

from bi_qmn.core.linalg import basis_vector, fourier_matrix, kron, matrix_projector, normalize_probabilities
from bi_qmn.core.noise import apply_readout_noise, outcome_key, parse_outcome_key
from bi_qmn.core.types import MeasurementContext, MeasurementTable


@dataclass(frozen=True)
class POVMEffect:
    context: MeasurementContext
    outcome: str
    operator: Operator


def parse_context(data: dict[str, str]) -> MeasurementContext:
    clock = data["clock"]
    system = data["system"]
    if clock not in {"D", "F"}:
        raise ValueError(f"unknown clock basis {clock}")
    if system not in {"X", "Y", "Z"}:
        raise ValueError(f"unknown system observable {system}")
    return MeasurementContext(clock=clock, system=system)  # type: ignore[arg-type]


def contexts_from_measurement_config(measurement_config: dict) -> dict[str, list[MeasurementContext]]:
    contexts = measurement_config["contexts"]
    return {
        "input": [parse_context(item) for item in contexts.get("input", [])],
        "compositional_heldout": [
            parse_context(item) for item in contexts.get("compositional_heldout", [])
        ],
        "operator_heldout": [parse_context(item) for item in contexts.get("operator_heldout", [])],
    }


def full_context_list(measurement_config: dict) -> list[MeasurementContext]:
    grouped = contexts_from_measurement_config(measurement_config)
    return grouped["input"] + grouped["compositional_heldout"] + grouped["operator_heldout"]


def clock_projectors(clock_basis: str, K: int) -> list[np.ndarray]:
    if clock_basis == "F":
        return [matrix_projector(basis_vector(K, r)) for r in range(K)]
    if clock_basis == "D":
        F = fourier_matrix(K)
        return [matrix_projector(F[tau, :]) for tau in range(K)]
    raise ValueError(f"unknown clock basis {clock_basis}")


def system_projectors(observable: str) -> list[np.ndarray]:
    if observable == "Z":
        plus = basis_vector(2, 0)
        minus = basis_vector(2, 1)
    elif observable == "X":
        plus = np.asarray([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
        minus = np.asarray([1.0, -1.0], dtype=complex) / np.sqrt(2.0)
    elif observable == "Y":
        plus = np.asarray([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
        minus = np.asarray([1.0, -1.0j], dtype=complex) / np.sqrt(2.0)
    else:
        raise ValueError(f"unknown system observable {observable}")
    return [matrix_projector(plus), matrix_projector(minus)]


def effects_for_context(context: MeasurementContext, K: int = 4) -> list[POVMEffect]:
    effects: list[POVMEffect] = []
    for clock_label, P_C in enumerate(clock_projectors(context.clock, K)):
        for system_bit, P_S in enumerate(system_projectors(context.system)):
            effects.append(
                POVMEffect(
                    context=context,
                    outcome=outcome_key(clock_label, system_bit),
                    operator=Operator(kron(P_C, P_S)),
                )
            )
    return effects


def effect_map(contexts: Iterable[MeasurementContext], K: int = 4) -> dict[str, dict[str, Operator]]:
    mapping: dict[str, dict[str, Operator]] = {}
    for context in contexts:
        mapping[context.context_id] = {
            effect.outcome: effect.operator for effect in effects_for_context(context, K)
        }
    return mapping


def readout_transition_probability(
    observed_outcome: str,
    true_outcome: str,
    K: int,
    eta: float,
) -> float:
    if eta <= 0.0:
        return 1.0 if observed_outcome == true_outcome else 0.0
    obs_clock, obs_system = parse_outcome_key(observed_outcome)
    true_clock, true_system = parse_outcome_key(true_outcome)
    if obs_clock == true_clock:
        p_clock = 1.0 - eta
    else:
        p_clock = eta / (K - 1)
    p_system = (1.0 - eta) if obs_system == true_system else eta
    return p_clock * p_system


def effective_effect_map(
    contexts: Iterable[MeasurementContext],
    K: int = 4,
    eta: float = 0.0,
) -> dict[str, dict[str, Operator]]:
    ideal = effect_map(contexts, K=K)
    if eta <= 0.0:
        return ideal
    effective: dict[str, dict[str, Operator]] = {}
    for context in contexts:
        context_id = context.context_id
        outcomes = list(ideal[context_id])
        effective[context_id] = {}
        for observed in outcomes:
            data = np.zeros_like(np.asarray(next(iter(ideal[context_id].values())).data, dtype=complex))
            for true in outcomes:
                data += readout_transition_probability(observed, true, K, eta) * np.asarray(
                    ideal[context_id][true].data,
                    dtype=complex,
                )
            effective[context_id][observed] = Operator(data)
    return effective


def probabilities_for_context(
    rho: DensityMatrix,
    context: MeasurementContext,
    K: int = 4,
    eta: float = 0.0,
    atol: float = 1.0e-12,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    rho_data = np.asarray(rho.data, dtype=complex)
    for effect in effects_for_context(context, K):
        value = np.trace(np.asarray(effect.operator.data, dtype=complex) @ rho_data)
        raw[effect.outcome] = float(np.real_if_close(value, tol=1000))
    ordered_keys = [outcome_key(c, s) for c in range(K) for s in range(2)]
    probs = normalize_probabilities([raw[key] for key in ordered_keys], atol=atol)
    distribution = dict(zip(ordered_keys, probs, strict=True))
    return apply_readout_noise(distribution, K=K, eta=eta)


def measurement_table(
    rho: DensityMatrix,
    contexts: Iterable[MeasurementContext],
    K: int = 4,
    eta: float = 0.0,
    metadata: dict | None = None,
) -> MeasurementTable:
    context_list = list(contexts)
    probabilities = {
        context.context_id: probabilities_for_context(rho, context, K=K, eta=eta)
        for context in context_list
    }
    return MeasurementTable(
        probabilities=probabilities,
        contexts=context_list,
        metadata=dict(metadata or {}),
        source_density=rho,
    )


def validate_measurement_table(
    table: MeasurementTable,
    eps_num: float = 1.0e-8,
) -> dict[str, object]:
    min_prob = 1.0
    max_norm_error = 0.0
    for context in table.contexts:
        probs = table.probabilities[context.context_id]
        min_prob = min(min_prob, min(probs.values()))
        max_norm_error = max(max_norm_error, abs(sum(probs.values()) - 1.0))
    return {
        "normalized": max_norm_error <= eps_num,
        "non_negative": min_prob >= -eps_num,
        "min_prob": min_prob,
        "max_norm_error": max_norm_error,
        "pass": max_norm_error <= eps_num and min_prob >= -eps_num,
    }
