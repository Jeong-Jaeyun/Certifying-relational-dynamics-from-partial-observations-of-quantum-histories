"""Zero-energy history estimator: the SBP likelihood restricted to the kernel of a registered ``H_tot``.

M1 comparator for the information question raised after the Q1 verification run
(documents/M1-added-measurement-run.md, addendum). The shared branch-pair estimator
(:mod:`bi_qmn.p3.models.p3_shared_branch_pair`) is told the clock spectrum, ``d_S`` and the effects,
but **no system Hamiltonian**: its branch basis is free on the Stiefel manifold, so the D-commutant
orbit ``exp(itR)`` of type 3a is a support-equivalent family for it.

The BI certificate route, in contrast, registers the instance Hamiltonian
(``p1_constraint_registry``: ``<H_tot^2> / h_scale^2 <= theta_con``). This estimator is the matching
"Hamiltonian known" estimator. It fits the same pure history

    ``|Psi> = sum_j |m_j>_C (x) c_j u_j``

by maximum likelihood, but only inside the exact zero-energy subspace
``ker(H_tot B)``, where ``B`` embeds the active clock modes. The kernel is computed numerically from
the registered matrix; no structure of ``H_tot`` is assumed. For the P1 generator
(``H_tot = H_C (x) I + I (x) U diag(-omega_{m}) U^dag``) the kernel has one direction per active
mode, ``|m_j> (x) U[:, j]``, so only the branch weights and phases are fitted.

The hard kernel is the ``theta_con -> 0`` limit of the registered inequality. Along a 3a orbit the
registered energy is ``sin^2 t <R Psi|H_tot^2|R Psi> / h_scale^2``, so ``theta_con = 1e-6`` leaves an
arc of half-width about ``sqrt(theta_con / B)`` (about 3e-3 rad on the M1 bank). The hard version
ignores that arc.

The estimator is a function of the support table and the registered Hamiltonian only. Restart seeds
come from a hash of the support, so serial and parallel runs give identical tables.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from bi_qmn.core.types import MeasurementContext, MeasurementTable
from bi_qmn.p3.models.p3_shared_branch_pair import SharedBranchPairForwardModel


METHOD = "zero_energy_history_v1"


def mode_embedding(active_modes: Sequence[int], K: int, d_S: int) -> np.ndarray:
    """``B``: ``C^{n d_S} -> C^{K d_S}``, the same embedding as the SBP forward model (``x = V.T.reshape(-1)``)."""
    modes = [int(m) for m in active_modes]
    B = np.zeros((int(K) * int(d_S), len(modes) * int(d_S)), dtype=complex)
    for j, m in enumerate(modes):
        for s in range(int(d_S)):
            B[m * int(d_S) + s, j * int(d_S) + s] = 1.0
    return B


def zero_energy_kernel(H_tot: np.ndarray, h_scale: float, B: np.ndarray, relative_tolerance: float = 1.0e-9) -> np.ndarray:
    """Orthonormal basis (columns) of ``ker(H_tot B / h_scale)`` in the embedded coordinates."""
    M = np.asarray(H_tot, dtype=complex) @ B / float(h_scale)
    _, s, vh = np.linalg.svd(M)
    scale = max(1.0, float(s[0]) if s.size else 1.0)
    rank = int(np.sum(s > relative_tolerance * scale))
    return vh[rank:].conj().T


def registered_energy(x: np.ndarray, H_tot: np.ndarray, h_scale: float, B: np.ndarray) -> float:
    """``<Psi|H_tot^2|Psi> / (h_scale^2 <Psi|Psi>)`` for the embedded history ``x``."""
    psi = B @ np.asarray(x, dtype=complex)
    Hpsi = np.asarray(H_tot, dtype=complex) @ psi
    return float(np.vdot(Hpsi, Hpsi).real / (float(h_scale) ** 2 * np.vdot(psi, psi).real))


class ZeroEnergyHistoryEstimator:
    """Pure-history MLE inside the zero-energy subspace of a registered ``H_tot``."""

    method = METHOD

    def __init__(
        self,
        active_modes: Sequence[int],
        K: int,
        d_S: int,
        effects: Mapping[str, Mapping[str, np.ndarray]],
        H_tot: np.ndarray,
        h_scale: float,
        *,
        starts: int = 32,
        seed: int = 0,
        probability_floor: float = 1.0e-6,
        likelihood_floor: float = 1.0e-12,
        near_optimal_nats: float = 1.0,
        max_iterations: int = 4000,
        kernel_relative_tolerance: float = 1.0e-9,
    ):
        self.forward = SharedBranchPairForwardModel(active_modes, K, d_S, effects)
        H = np.asarray(H_tot, dtype=complex)
        if H.shape != (self.forward.K * self.forward.d_S,) * 2:
            raise ValueError("H_tot must act on the K * d_S clock-system space of the forward model")
        self.H_tot = H
        self.h_scale = float(h_scale)
        self.B = mode_embedding(self.forward.modes, self.forward.K, self.forward.d_S)
        self.N = zero_energy_kernel(H, self.h_scale, self.B, kernel_relative_tolerance)
        if self.N.shape[1] == 0:
            raise ValueError("the registered Hamiltonian has no zero-energy history on the active clock modes")
        self.AN = {cid: np.einsum("ia,oij,jb->oab", self.N.conj(), A, self.N) for cid, A in self.forward.A.items()}
        self.starts = int(starts)
        self.seed = int(seed)
        self.probability_floor = float(probability_floor)
        self.likelihood_floor = float(likelihood_floor)
        self.near_optimal_nats = float(near_optimal_nats)
        self.max_iterations = int(max_iterations)
        if self.starts < 1:
            raise ValueError("at least one start is required")
        digest = hashlib.sha256(f"{METHOD}:{self.starts}:{self.seed}:{self.probability_floor}:{self.likelihood_floor}:{self.near_optimal_nats}:{self.max_iterations}:{self.forward.effects_hash}".encode("ascii"))
        digest.update(np.ascontiguousarray(np.round(H / self.h_scale, 12)).view(np.float64).tobytes())
        self.checkpoint_hash = f"{METHOD}:{digest.hexdigest()[:16]}"
        self.last_fit: dict[str, Any] | None = None

    @property
    def kernel_dimension(self) -> int:
        return int(self.N.shape[1])

    def fit_records(self, records: Any) -> "ZeroEnergyHistoryEstimator":  # noqa: ARG002 - nothing is learned across records
        return self

    def branch_vectors(self, y: np.ndarray) -> np.ndarray:
        """``V[:, j] = c_j u_j`` of the normalized kernel coordinates ``y``."""
        x = self.N @ (np.asarray(y, dtype=complex) / np.linalg.norm(y))
        return x.reshape(self.forward.n, self.forward.d_S).T

    def _objective(self, w: np.ndarray, A: np.ndarray, counts: np.ndarray) -> tuple[float, np.ndarray]:
        k = self.kernel_dimension
        v = w[:k] + 1j * w[k:]
        s = float(np.vdot(v, v).real)
        Av = A @ v
        q = np.einsum("a,oa->o", v.conj(), Av).real / s
        qf = q + self.likelihood_floor
        value = -float(counts @ np.log(qf))
        weights = -counts / qf
        G = (2.0 / s) * (weights[:, None] * (Av - q[:, None] * v[None, :])).sum(axis=0)
        return value, np.concatenate([G.real, G.imag])

    def _support_arrays(self, support: Mapping[str, Mapping[str, float]], n_shots: float) -> tuple[list[str], np.ndarray, np.ndarray]:
        ids = sorted(support)
        A = np.concatenate([self.AN[cid] for cid in ids])
        counts = np.concatenate([[float(support[cid].get(o, 0.0)) * n_shots for o in self.forward.outcomes] for cid in ids])
        return ids, A, counts

    def _seed_for(self, ids: list[str], counts: np.ndarray) -> int:
        digest = hashlib.sha256(f"{METHOD}:{self.seed}:{','.join(ids)}".encode("utf-8"))
        digest.update(np.ascontiguousarray(np.round(counts, 9)).tobytes())
        return int(digest.hexdigest()[:16], 16)

    def fit_support(self, support: Mapping[str, Mapping[str, float]], n_shots: float = 1.0) -> dict[str, Any]:
        ids, A, counts = self._support_arrays(support, float(n_shots))
        rng = np.random.default_rng(self._seed_for(ids, counts))
        k = self.kernel_dimension
        runs = []
        for _ in range(self.starts):
            w0 = rng.normal(0.0, 1.0, 2 * k)
            res = minimize(self._objective, w0, args=(A, counts), jac=True, method="L-BFGS-B",
                           options={"maxiter": self.max_iterations, "ftol": 1.0e-13, "gtol": 1.0e-9})
            runs.append((float(res.fun), res.x.copy(), bool(res.success)))
        runs.sort(key=lambda item: item[0])
        best_value, best_w = runs[0][0], runs[0][1]
        y = best_w[:k] + 1j * best_w[k:]
        V = self.branch_vectors(y)
        near = [r for r in runs if r[0] <= best_value + self.near_optimal_nats]
        x = V.T.reshape(-1)
        fit = {
            "method": METHOD,
            "objective_nats": best_value,
            "starts": self.starts,
            "starts_converged": int(sum(r[2] for r in runs)),
            "starts_within_near_optimal": len(near),
            "kernel_dimension": k,
            "registered_energy": registered_energy(x, self.H_tot, self.h_scale, self.B),
            "populations": (np.linalg.norm(V, axis=0) ** 2).tolist(),
            "branch_vectors_real_imag": [V.real.tolist(), V.imag.tolist()],
            "n_shots": float(n_shots),
            "support_context_ids": ids,
        }
        self.last_fit = fit
        return fit

    def predict_table(self, support_table: MeasurementTable, contexts: list[MeasurementContext]) -> MeasurementTable:
        """Same output convention as the SBP estimator (clip, floor, renormalize per context)."""
        fit = self.fit_support(support_table.probabilities, float(support_table.metadata.get("n_shots", 1.0)))
        V = np.asarray(fit["branch_vectors_real_imag"][0]) + 1j * np.asarray(fit["branch_vectors_real_imag"][1])
        tables = self.forward.tables(V, [c.context_id for c in contexts])
        probabilities = {}
        for context in contexts:
            values = np.clip(tables[context.context_id], 0.0, None) + self.probability_floor
            values = values / values.sum()
            probabilities[context.context_id] = dict(zip(self.forward.outcomes, values.tolist(), strict=True))
        return MeasurementTable(probabilities=probabilities, contexts=list(contexts),
                                metadata={**dict(support_table.metadata), "zero_energy_history_fit": fit}, source_density=None)


def orbit_energy_coefficient(V: np.ndarray, R: np.ndarray, H_tot: np.ndarray, h_scale: float, B: np.ndarray) -> float:
    """``B`` in ``<H_tot^2>(t) / h_scale^2 = B sin^2 t`` along ``exp(itR) V`` for a zero-energy ``V`` (``R^2 = I``)."""
    x = np.asarray(R @ np.asarray(V, dtype=complex), dtype=complex).T.reshape(-1)
    return registered_energy(x, H_tot, h_scale, B)


def feasible_half_width(B_coefficient: float, theta_con: float) -> float:
    """Half-width in ``t`` of the orbit arc with ``B sin^2 t <= theta_con`` (``pi/2`` when the whole orbit is feasible)."""
    if B_coefficient <= theta_con:
        return math.pi / 2.0
    return float(math.asin(math.sqrt(theta_con / B_coefficient)))
