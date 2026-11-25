from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator

from bi_qmn.core.linalg import kron, normalize_probabilities
from bi_qmn.core.measurement import clock_projectors, readout_transition_probability
from bi_qmn.core.noise import apply_readout_noise, outcome_key
from bi_qmn.core.types import MeasurementContext, MeasurementTable


PAULI_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


@dataclass(frozen=True)
class P1POVMEffect:
    context: MeasurementContext
    outcome: str
    operator: Operator


def parse_p1_context(data: dict[str, str], system_qubits: int = 2) -> MeasurementContext:
    clock = data["clock"]
    system = data["system"]
    if clock not in {"D", "F"}:
        raise ValueError(f"unknown P1 clock basis {clock}")
    if len(system) != system_qubits or any(char not in PAULI_1Q for char in system):
        raise ValueError(f"unknown P1 system Pauli string {system}")
    if set(system) == {"I"}:
        raise ValueError("all-identity P1 system observable is not a valid measurement context")
    return MeasurementContext(clock=clock, system=system)  # type: ignore[arg-type]


def contexts_from_p1_measurement_config(measurement_config: dict) -> dict[str, list[MeasurementContext]]:
    contexts = measurement_config["contexts"]
    system_qubits = int(measurement_config.get("system_qubits", 2))
    return {
        "input": [parse_p1_context(item, system_qubits=system_qubits) for item in contexts.get("input", [])],
        "compositional_heldout": [
            parse_p1_context(item, system_qubits=system_qubits)
            for item in contexts.get("compositional_heldout", [])
        ],
        "operator_heldout": [
            parse_p1_context(item, system_qubits=system_qubits) for item in contexts.get("operator_heldout", [])
        ],
    }


def full_p1_context_list(measurement_config: dict) -> list[MeasurementContext]:
    grouped = contexts_from_p1_measurement_config(measurement_config)
    return grouped["input"] + grouped["compositional_heldout"] + grouped["operator_heldout"]


def pauli_string_operator(label: str) -> np.ndarray:
    if not label:
        raise ValueError("Pauli string label cannot be empty")
    return kron(*(PAULI_1Q[char] for char in label))


def p1_system_projectors(pauli_label: str) -> list[np.ndarray]:
    observable = pauli_string_operator(pauli_label)
    dim = observable.shape[0]
    ident = np.eye(dim, dtype=complex)
    return [(ident + observable) / 2.0, (ident - observable) / 2.0]


def p1_effects_for_context(context: MeasurementContext, K: int = 8) -> list[P1POVMEffect]:
    effects: list[P1POVMEffect] = []
    for clock_label, P_C in enumerate(clock_projectors(context.clock, K)):
        for system_bit, P_S in enumerate(p1_system_projectors(context.system)):
            effects.append(
                P1POVMEffect(
                    context=context,
                    outcome=outcome_key(clock_label, system_bit),
                    operator=Operator(kron(P_C, P_S)),
                )
            )
    return effects


def p1_effect_map(contexts: Iterable[MeasurementContext], K: int = 8) -> dict[str, dict[str, Operator]]:
    mapping: dict[str, dict[str, Operator]] = {}
    for context in contexts:
        mapping[context.context_id] = {
            effect.outcome: effect.operator for effect in p1_effects_for_context(context, K)
        }
    return mapping


def p1_effective_effect_map(
    contexts: Iterable[MeasurementContext],
    K: int = 8,
    eta: float = 0.0,
) -> dict[str, dict[str, Operator]]:
    ideal = p1_effect_map(contexts, K=K)
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


def p1_probabilities_for_context(
    rho: DensityMatrix,
    context: MeasurementContext,
    K: int = 8,
    eta: float = 0.0,
    atol: float = 1.0e-12,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    rho_data = np.asarray(rho.data, dtype=complex)
    for effect in p1_effects_for_context(context, K):
        value = np.trace(np.asarray(effect.operator.data, dtype=complex) @ rho_data)
        raw[effect.outcome] = float(np.real_if_close(value, tol=1000))
    ordered_keys = [outcome_key(c, s) for c in range(K) for s in range(2)]
    probs = normalize_probabilities([raw[key] for key in ordered_keys], atol=atol)
    distribution = dict(zip(ordered_keys, probs, strict=True))
    return apply_readout_noise(distribution, K=K, eta=eta)


def p1_measurement_table(
    rho: DensityMatrix,
    contexts: Iterable[MeasurementContext],
    K: int = 8,
    eta: float = 0.0,
    metadata: dict | None = None,
) -> MeasurementTable:
    context_list = list(contexts)
    probabilities = {
        context.context_id: p1_probabilities_for_context(rho, context, K=K, eta=eta)
        for context in context_list
    }
    return MeasurementTable(
        probabilities=probabilities,
        contexts=context_list,
        metadata=dict(metadata or {}),
        source_density=rho,
    )
