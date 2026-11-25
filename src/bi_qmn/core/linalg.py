from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, Statevector


Array = np.ndarray


def qiskit_statevector(amplitudes: Array) -> Statevector:
    """Create a Qiskit Statevector through an initialized circuit."""
    vec = np.asarray(amplitudes, dtype=complex)
    norm = np.linalg.norm(vec)
    if not np.isclose(norm, 1.0):
        vec = vec / norm
    n_qubits = int(round(math.log2(vec.size)))
    circuit = QuantumCircuit(n_qubits)
    circuit.initialize(vec, list(range(n_qubits)))
    return Statevector.from_instruction(circuit)


def density_from_statevector(statevector: Statevector) -> DensityMatrix:
    return DensityMatrix(statevector)


def basis_vector(dim: int, index: int) -> Array:
    vec = np.zeros(dim, dtype=complex)
    vec[index] = 1.0
    return vec


def projector(vec: Array) -> Operator:
    data = np.outer(vec, np.conjugate(vec))
    return Operator(data)


def matrix_projector(vec: Array) -> Array:
    return np.outer(vec, np.conjugate(vec))


def kron(*ops: Array) -> Array:
    if not ops:
        raise ValueError("kron requires at least one operator")
    out = np.asarray(ops[0], dtype=complex)
    for op in ops[1:]:
        out = np.kron(out, np.asarray(op, dtype=complex))
    return out


def fourier_matrix(dim: int) -> Array:
    rows = []
    scale = 1.0 / math.sqrt(dim)
    for tau in range(dim):
        rows.append([scale * np.exp(-2j * np.pi * tau * r / dim) for r in range(dim)])
    return np.asarray(rows, dtype=complex)


def fourier_operator(dim: int) -> Operator:
    return Operator(fourier_matrix(dim))


def assert_unitary(operator: Operator, atol: float = 1.0e-10) -> None:
    data = np.asarray(operator.data, dtype=complex)
    ident = np.eye(data.shape[0], dtype=complex)
    if not np.allclose(data.conj().T @ data, ident, atol=atol):
        raise AssertionError("operator is not unitary")


def trace_real(op: Array, rho: DensityMatrix | Array) -> float:
    rho_data = rho.data if isinstance(rho, DensityMatrix) else rho
    value = np.trace(np.asarray(op, dtype=complex) @ np.asarray(rho_data, dtype=complex))
    return float(np.real_if_close(value, tol=1000))


def normalize_probabilities(values: Iterable[float], atol: float = 1.0e-12) -> list[float]:
    probs = [0.0 if abs(v) < atol else float(v) for v in values]
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("probability vector has non-positive mass")
    return [p / total for p in probs]


def matrix_sqrt_psd(rho: Array) -> Array:
    vals, vecs = np.linalg.eigh((rho + rho.conj().T) / 2.0)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def fidelity(rho: Array, sigma: Array) -> float:
    sr = matrix_sqrt_psd(rho)
    inner = sr @ sigma @ sr
    vals = np.linalg.eigvalsh((inner + inner.conj().T) / 2.0)
    return float(np.square(np.sum(np.sqrt(np.clip(vals, 0.0, None)))))


def bures_distance(rho: Array, sigma: Array) -> float:
    fid = min(1.0, max(0.0, fidelity(rho, sigma)))
    return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid))))


def as_density_matrix(data: Array) -> DensityMatrix:
    return DensityMatrix((data + data.conj().T) / 2.0)
