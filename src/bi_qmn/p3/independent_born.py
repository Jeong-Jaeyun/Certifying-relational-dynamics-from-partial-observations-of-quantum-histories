"""Independent Born-rule reference for the canonical finite Page--Wootters design (E6 cross-check).

Written from the definitions in documents/bi-qmn-ontological-contract_ver3.tex only
(eq. fourier-clock, eq. povm, eq. econ-instance). It imports numpy and nothing from the operational
measurement or estimator code, so agreement with that code is an independent check of conventions,
normalization and tolerances.

* Clock: ``|tau> = K^{-1/2} sum_r exp(-2 pi i r tau / K) |omega_r>`` (time basis, context ``D``) and
  ``|omega_r>`` (energy basis, context ``F``).
* System: two-qubit Pauli string ``P`` (first character acts on the first, more significant qubit);
  outcome bit ``b`` is the eigenvalue ``(-1)^b`` with projector ``(I + (-1)^b P) / 2``.
* Global effect ``|clock_l><clock_l| (x) (I + (-1)^b P)/2``, outcome key ``c{l}_s{b}``, clock factor first.
* History ``|Psi> = sum_j c_j |omega_{m_j}> (x) u_j``.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


PAULI_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli(label: str) -> np.ndarray:
    out = np.ones((1, 1), dtype=complex)
    for ch in label:
        out = np.kron(out, PAULI_1Q[ch])
    return out


def clock_vector(K: int, basis: str, index: int) -> np.ndarray:
    if basis == "F":
        v = np.zeros(K, dtype=complex)
        v[index] = 1.0
        return v
    if basis == "D":
        r = np.arange(K)
        return np.exp(-2j * math.pi * r * index / K) / math.sqrt(K)
    raise ValueError(f"unknown clock basis {basis}")


def history(modes: Sequence[int], coefficients: Sequence[complex], system_vectors: np.ndarray, K: int) -> np.ndarray:
    """``|Psi>`` as a ``(K, d_S)`` amplitude array: row ``k`` is the system vector attached to ``|omega_k>``."""
    u = np.asarray(system_vectors, dtype=complex)
    A = np.zeros((K, u.shape[0]), dtype=complex)
    for j, m in enumerate(modes):
        A[int(m)] += complex(coefficients[j]) * u[:, j]
    return A


def rotate_system(A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """``(I_C (x) U) |Psi>`` in the amplitude-array form."""
    return np.asarray(A, dtype=complex) @ np.asarray(U, dtype=complex).T


def context_table(A: np.ndarray, clock: str, system: str) -> dict[str, float]:
    K, d = A.shape
    P = pauli(system)
    proj = [(np.eye(d) + (-1) ** b * P) / 2.0 for b in (0, 1)]
    out = {}
    for l in range(K):
        w = clock_vector(K, clock, l).conj() @ A          # (<clock_l| (x) I) |Psi>
        for b in (0, 1):
            out[f"c{l}_s{b}"] = float(np.real(w.conj() @ proj[b] @ w))
    return out


def tables(A: np.ndarray, context_ids: Iterable[str]) -> dict[str, dict[str, float]]:
    """Tables for context ids ``"D,XI"``, ``"F,ZZ"``, ..."""
    out = {}
    for cid in context_ids:
        clock, system = cid.split(",")
        out[cid] = context_table(A, clock, system)
    return out


def conditional_states(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``p(tau)`` and ``rho_{S|tau}`` for the time basis."""
    K, d = A.shape
    p = np.zeros(K)
    rho = np.zeros((K, d, d), dtype=complex)
    for l in range(K):
        w = clock_vector(K, "D", l).conj() @ A
        p[l] = float(np.real(w.conj() @ w))
        rho[l] = np.outer(w, w.conj()) / p[l] if p[l] > 0 else 0.0
    return p, rho


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(np.asarray(rho) - np.asarray(sigma)))))


def pure_trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=complex).ravel(), np.asarray(b, dtype=complex).ravel()
    f = abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    return float(math.sqrt(max(0.0, 1.0 - f)))


def normalized_energy(A: np.ndarray, H_tot: np.ndarray) -> float:
    """``Tr[rho H_tot^2] / s_H^2`` with ``s_H^2 = Tr[H_tot^2] / d_CS`` (contract eq. econ-instance)."""
    psi = np.asarray(A, dtype=complex).reshape(-1)          # clock-major, matching H_C (x) I + I (x) H_S
    H = np.asarray(H_tot, dtype=complex)
    s2 = float(np.real(np.trace(H @ H))) / H.shape[0]
    Hpsi = H @ psi
    return float(np.real(np.vdot(Hpsi, Hpsi)) / (s2 * np.real(np.vdot(psi, psi))))


def max_table_difference(a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]], ids: Iterable[str] | None = None) -> float:
    ids = list(a) if ids is None else list(ids)
    return max((abs(float(a[c][o]) - float(b[c][o])) for c in ids for o in a[c]), default=0.0)
