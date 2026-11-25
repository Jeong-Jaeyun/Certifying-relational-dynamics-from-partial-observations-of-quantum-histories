"""Second design for paper 1: a larger clock, other active modes and other observed contexts.

The study asks where "dynamics visible, history not identified" arises outside the registered design, when one
added measurement resolves it, and how much of a finite-shot failure is an information limit. This module holds
the design and the deterministic tools; it never samples data.

Design (user specification, 2026-11-24):
* clock ``K = 16`` with energies ``omega_r = r``; time basis ``|tau> = K^-1/2 sum_r exp(-2 pi i r tau / K) |r>``;
* two-qubit system; active modes ``{0, 1, 4, 6}`` (``"golomb"``: all pair differences distinct) or ``{0, 1, 2, 3}``
  (``"collision"``: several branch pairs share a Fourier bin);
* observed ``D = {IX, XI, XX, YI, YX, ZI}``, ``F = {XZ, XY}``; their ``D`` Paulis share the commutant ``R = IX``;
* candidates for one added context: the unobserved ``D`` Paulis that anticommute with ``R``.

Every function is a deterministic function of its arguments.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

from bi_qmn.core.noise import outcome_key
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairEstimator, SharedBranchPairForwardModel

_P1 = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
       "Y": np.array([[0, -1j], [1j, 0]], dtype=complex), "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
ALL_PAULIS = ["".join(p) for p in itertools.product("IXYZ", repeat=2) if "".join(p) != "II"]
MODES = {"golomb": (0, 1, 4, 6), "collision": (0, 1, 2, 3)}
OBSERVED_D = ("IX", "XI", "XX", "YI", "YX", "ZI")
OBSERVED_F = ("XZ", "XY")
COMMUTANT = "IX"


def pauli(label: str) -> np.ndarray:
    return np.kron(_P1[label[0]], _P1[label[1]])


def anticommutes(a: str, b: str) -> bool:
    A, B = pauli(a), pauli(b)
    return bool(np.allclose(A @ B, -B @ A))


@dataclass(frozen=True)
class Design:
    name: str
    K: int = 16
    d_S: int = 4
    observed_D: tuple[str, ...] = OBSERVED_D
    observed_F: tuple[str, ...] = OBSERVED_F
    R: str = COMMUTANT
    modes: tuple[int, ...] = field(default=())

    @staticmethod
    def make(name: str) -> "Design":
        return Design(name=name, modes=MODES[name])

    @property
    def support(self) -> list[str]:
        return [f"D,{p}" for p in self.observed_D] + [f"F,{p}" for p in self.observed_F]

    @property
    def candidates(self) -> list[str]:
        """Unobserved ``D`` contexts whose Pauli anticommutes with ``R`` (the only ones that move along the circle)."""
        return [f"D,{p}" for p in ALL_PAULIS if p not in self.observed_D and anticommutes(p, self.R)]

    @property
    def unobserved_D(self) -> list[str]:
        return [f"D,{p}" for p in ALL_PAULIS if p not in self.observed_D]

    @property
    def all_contexts(self) -> list[str]:
        return [f"D,{p}" for p in ALL_PAULIS] + [f"F,{p}" for p in ALL_PAULIS]

    def pair_bins(self) -> dict[int, list[tuple[int, int]]]:
        out: dict[int, list[tuple[int, int]]] = {}
        for a, b in itertools.combinations(self.modes, 2):
            out.setdefault((b - a) % self.K, []).append((a, b))
        return out


def clock_time_vector(K: int, tau: int) -> np.ndarray:
    return np.exp(-2j * math.pi * np.arange(K) * tau / K) / math.sqrt(K)


def effects(design: Design, contexts: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    """Effect matrices on the joint space (clock factor first) for ``D,P`` and ``F,P`` contexts."""
    K, d = design.K, design.d_S
    out: dict[str, dict[str, np.ndarray]] = {}
    for cid in contexts:
        basis, P = cid.split(",")
        proj = {s: (np.eye(d) + (-1) ** s * pauli(P)) / 2 for s in (0, 1)}
        row = {}
        for l in range(K):
            if basis == "D":
                v = clock_time_vector(K, l)
            else:
                v = np.zeros(K, dtype=complex); v[l] = 1.0
            C = np.outer(v, v.conj())
            for s in (0, 1):
                row[outcome_key(l, s)] = np.kron(C, proj[s])
        out[cid] = row
    return out


_FW_CACHE: dict[str, SharedBranchPairForwardModel] = {}


def forward(design: Design) -> SharedBranchPairForwardModel:
    if design.name not in _FW_CACHE:
        _FW_CACHE[design.name] = SharedBranchPairForwardModel(design.modes, design.K, design.d_S, effects(design, design.all_contexts))
    return _FW_CACHE[design.name]


def estimator(design: Design, starts: int = 32, seed: int = 0) -> SharedBranchPairEstimator:
    return SharedBranchPairEstimator(design.modes, design.K, design.d_S, effects(design, design.all_contexts), starts=starts, seed=seed)


def branch_vectors(U: np.ndarray, probs: Sequence[float], phases: Sequence[float]) -> np.ndarray:
    """``V[:, j] = c_j u_j`` with ``c_j = sqrt(p_j) exp(i phi_j)`` and ``u_j`` the ``j``-th column of ``U``."""
    c = np.sqrt(np.asarray(probs, float)) * np.exp(1j * np.asarray(phases, float))
    n = len(c)
    return np.asarray(U, complex)[:, :n] * c[None, :]


def rotate(V: np.ndarray, R: str, t: float) -> np.ndarray:
    return expm(1j * t * pauli(R)) @ V


def tables(design: Design, V: np.ndarray, contexts: Sequence[str]) -> dict[str, np.ndarray]:
    return forward(design).tables(V, list(contexts))


def max_table_difference(design: Design, V1: np.ndarray, V2: np.ndarray, contexts: Sequence[str]) -> float:
    T1, T2 = tables(design, V1, contexts), tables(design, V2, contexts)
    return float(max(np.max(np.abs(T1[c] - T2[c])) for c in contexts))


def quadrature_matrix(design: Design, V: np.ndarray, Q: str) -> np.ndarray:
    """``M_Q = [a b]`` of ``q_Q(t) = qbar + a cos 2t + b sin 2t`` along ``exp(itR) V`` (``Q`` anticommuting with ``R``)."""
    cid = Q if "," in Q else f"D,{Q}"
    q0 = tables(design, V, [cid])[cid]
    q90 = tables(design, rotate(V, design.R, math.pi / 2), [cid])[cid]
    q45 = tables(design, rotate(V, design.R, math.pi / 4), [cid])[cid]
    a = (q0 - q90) / 2.0
    b = q45 - (q0 + q90) / 2.0
    return np.stack([a, b], axis=1)


def separation(design: Design, V: np.ndarray, Q: str) -> dict[str, float]:
    sv = np.linalg.svd(quadrature_matrix(design, V, Q), compute_uv=False)
    return {"sigma_max": float(sv[0]), "sigma_min": float(sv[1]), "rank": int(np.sum(sv > 1e-10 * max(1.0, sv[0])))}


def s_Q(design: Design, V: np.ndarray, Q: str) -> float:
    """Orbit-invariant sensitivity of the main text, Eq. (8), at branch vectors ``V``."""
    P = Q.split(",")[-1]
    X = V.conj().T @ pauli(P) @ V
    Y = V.conj().T @ (1j * pauli(P) @ pauli(design.R)) @ V
    n = V.shape[1]
    off = sum(abs(X[j, k]) ** 2 + abs(Y[j, k]) ** 2 for j in range(n) for k in range(j + 1, n))
    return float(math.sqrt(off + abs(np.trace(X)) ** 2 + abs(np.trace(Y)) ** 2))


def branch_components(U: np.ndarray, n: int, R: str, Fs: Sequence[str]) -> tuple[float, float]:
    """``max_j |<P>_j|`` and ``max_j |<iPR>_j|`` over observed ``F`` Paulis ``P`` that anticommute with ``R``."""
    Rm = pauli(R)
    a = b = 0.0
    for P in Fs:
        if not anticommutes(P, R):
            continue
        Pm = pauli(P)
        for j in range(n):
            u = U[:, j]
            a = max(a, abs(u.conj() @ Pm @ u)); b = max(b, abs(u.conj() @ (1j * Pm @ Rm) @ u))
    return float(a), float(b)


def haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    Zm = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2.0)
    Qm, Rm = np.linalg.qr(Zm)
    return Qm * (np.diag(Rm) / np.abs(np.diag(Rm)))[None, :]


def _anti_hermitian(theta: np.ndarray, d: int) -> np.ndarray:
    A = np.zeros((d, d), dtype=complex)
    iu = np.triu_indices(d, 1)
    k = len(iu[0])
    A[iu] = theta[:k] + 1j * theta[k:2 * k]
    A = A - A.conj().T
    A[np.diag_indices(d)] = 1j * theta[2 * k:]
    return A


def project_exact(U0: np.ndarray, design: Design, rng: np.random.Generator, tries: int = 8) -> tuple[np.ndarray, float]:
    """Least squares on ``U(d)``: ``<u_j|P|u_j> = <u_j|iPR|u_j> = 0`` for every observed anticommuting ``F`` Pauli."""
    d, n = U0.shape[0], len(design.modes)
    Rm = pauli(design.R)
    ops = []
    for P in design.observed_F:
        if anticommutes(P, design.R):
            ops += [pauli(P), 1j * pauli(P) @ Rm]

    def res(theta, Ub):
        U = expm(_anti_hermitian(theta, d)) @ Ub
        return np.array([np.real(U[:, j].conj() @ O @ U[:, j]) for O in ops for j in range(n)])

    best = (U0, math.inf)
    Ub = U0
    for _ in range(int(tries)):
        sol = least_squares(res, np.zeros(d * d), args=(Ub,), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        U = expm(_anti_hermitian(sol.x, d)) @ Ub
        r = float(np.max(np.abs(res(np.zeros(d * d), U))))
        if r < best[1]:
            best = (U, r)
        if r < 1e-13:
            break
        Ub = haar_unitary(d, rng) @ U0 if r > 1e-6 else U
    return best


def perturb(U: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    h = rng.normal(size=U.shape) + 1j * rng.normal(size=U.shape)
    h = (h + h.conj().T) / 2.0
    h = h / np.linalg.norm(h)
    return expm(1j * float(eps) * h) @ U


# ---------------------------------------------------------------------------------------------------------------
# Information: Bhattacharyya coefficients and the two-point (Le Cam) bound
# ---------------------------------------------------------------------------------------------------------------


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))))


def two_point_bound(design: Design, V0: np.ndarray, V1: np.ndarray, shots: Mapping[str, float], eval_contexts: Sequence[str]) -> dict[str, float]:
    """Lower bound on ``max_i E_i ||qhat - q_i||_2`` over the evaluation tables, for any estimator.

    ``v = TV(P_0^N, P_1^N) <= sqrt(1 - BC^2)`` with ``BC = prod_c BC_c^{N_c}``; the bound is
    ``(d/2) max(1 - sqrt(1 - BC^2), BC^2/2)`` with ``d`` the ``L_2`` distance of the evaluation tables.
    """
    ids = list(shots)
    T0, T1 = tables(design, V0, ids + list(eval_contexts)), tables(design, V1, ids + list(eval_contexts))
    log_bc = sum(float(shots[c]) * math.log(max(bhattacharyya(T0[c], T1[c]), 1e-300)) for c in ids)
    bc = math.exp(log_bc)
    v_upper = math.sqrt(max(0.0, 1.0 - bc * bc))
    d = float(math.sqrt(sum(float(np.sum((T0[c] - T1[c]) ** 2)) for c in eval_contexts)))
    return {"d": d, "bc": bc, "tv_upper": v_upper, "bound": 0.5 * d * max(1.0 - v_upper, 0.5 * bc * bc)}
