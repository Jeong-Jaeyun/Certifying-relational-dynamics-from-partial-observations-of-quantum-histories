"""Shared branch-pair estimator (P3 D1-v6): fixed clock frequencies, coefficients shared across contexts.

Major revision 2026-11-20 (formulation patch v3). D1-RDN showed that Model A recovers only about
half of the ``tau``-modulation of unobserved D contexts. The declared-family MLE recovers all of it,
but it knows the registered unitaries. This estimator sits between them. It is told only

* the clock dimension ``K`` and the active clock modes ``m_j`` (the declared clock spectrum),
* the system dimension ``d_S``, and
* the measurement effects of every context (the declared measurement map),

and fits a pure zero-energy history

    ``|Psi> = sum_j |m_j>_C (x) c_j u_j``,  with ``u_j`` orthonormal,

to the support counts by maximum likelihood. The branch basis ``u_j`` is free on the Stiefel
manifold: no unitary list, no system Hamiltonian, no real-ness. Orthonormality is the
Page--Wootters constraint with non-degenerate branch eigenvectors.

Why it transfers to unobserved contexts. For a D (time-basis) context with Pauli ``P`` write
``g_P(tau) = K [q_P(tau, 0) - q_P(tau, 1)]``. Then

    ``(F_tau g_P)_{m_k - m_j} = conj(c_j) c_k <u_j|P|u_k>``,

so each branch pair ``(j, k)`` sits in one fixed DFT bin, and its coefficient is shared by every
Pauli through ``<u_j|P|u_k>``. The fitted ``(u, c)`` from observed contexts therefore predicts the
modulation of unobserved ones. Purity also links the pairs: ``conj(c_1) c_3 = conj(c_1) c_0 *
conj(c_0) c_3 / |c_0|^2``, so a pair that no observed Pauli touches can still be inferred when both
of its branches connect to a third one.

What it cannot do. If the observed Paulis leave a branch disconnected from the others, its phase is
not identified by the support (formulation patch v3, D1-v6 note). The likelihood is then flat in
that phase up to sampling noise, and the fit returns a noise-determined phase. Restarts do not
reveal this: they all converge to the same noise-pinned optimum. ``pair_observability`` reports it
from the support alone instead. For each branch pair it gives
``s_jk = sqrt(sum_{P observed} |<u_j|P|u_k>|^2)`` in the *fitted* basis; a pair below the threshold
is hidden, and a branch whose two pairs are both hidden is disconnected.

The estimator is a function of the support table only. Restart seeds come from a hash of the
support, so serial and parallel runs give identical tables.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from bi_qmn.core.noise import outcome_key
from bi_qmn.core.types import MeasurementContext, MeasurementTable


METHOD = "shared_branch_pair_v1"


def _anti_hermitian_basis(n: int) -> np.ndarray:
    """Real basis of the ``n x n`` anti-Hermitian matrices, shape ``(n*n, n, n)``."""
    basis = []
    for k in range(n):
        for l in range(k + 1, n):
            B = np.zeros((n, n), dtype=complex)
            B[k, l], B[l, k] = 1.0, -1.0
            basis.append(B)
            B = np.zeros((n, n), dtype=complex)
            B[k, l], B[l, k] = 1.0j, 1.0j
            basis.append(B)
    for k in range(n):
        B = np.zeros((n, n), dtype=complex)
        B[k, k] = 1.0j
        basis.append(B)
    return np.stack(basis)


class SharedBranchPairForwardModel:
    """Tables of ``sum_j |m_j> (x) v_j`` through the declared effects, ``q_o = x^H A_o x``."""

    def __init__(self, active_modes: Sequence[int], K: int, d_S: int, effects: Mapping[str, Mapping[str, np.ndarray]]):
        self.modes = tuple(int(m) for m in active_modes)
        self.K = int(K)
        self.d_S = int(d_S)
        self.n = len(self.modes)
        self.outcomes = [outcome_key(l, b) for l in range(self.K) for b in (0, 1)]
        dim = self.K * self.d_S
        B = np.zeros((dim, self.n * self.d_S), dtype=complex)
        for j, m in enumerate(self.modes):
            for s in range(self.d_S):
                B[m * self.d_S + s, j * self.d_S + s] = 1.0
        self.A: dict[str, np.ndarray] = {}
        for cid, row in effects.items():
            mats = [np.asarray(row[o], dtype=complex) for o in self.outcomes]
            if any(M.shape != (dim, dim) for M in mats):
                raise ValueError(f"effect dimension mismatch for {cid}")
            self.A[cid] = np.stack([B.conj().T @ M @ B for M in mats])
        digest = hashlib.sha256(f"{self.modes}:{self.K}:{self.d_S}".encode("ascii"))
        for cid in sorted(self.A):
            digest.update(cid.encode("utf-8"))
            digest.update(np.ascontiguousarray(self.A[cid]).view(np.float64).tobytes())
        self.effects_hash = digest.hexdigest()[:16]
        # System observable of a time-basis context: sum_tau (E_{tau,0} - E_{tau,1}) = I_C (x) P, read off block (0, 0).
        self.system_observable: dict[str, np.ndarray] = {}
        for cid, A in self.A.items():
            if cid.startswith("D,"):
                total = sum(A[2 * l] - A[2 * l + 1] for l in range(self.K))
                self.system_observable[cid] = total[: self.d_S, : self.d_S]
        self.pairs = [(j, k) for j in range(self.n) for k in range(j + 1, self.n)]

    def vector(self, V: np.ndarray) -> np.ndarray:
        return np.asarray(V, dtype=complex).T.reshape(-1)

    def tables(self, V: np.ndarray, context_ids: Iterable[str]) -> dict[str, np.ndarray]:
        x = self.vector(V)
        return {cid: np.einsum("i,oij,j->o", x.conj(), self.A[cid], x).real for cid in context_ids}

    def pair_bin(self, pair: tuple[int, int]) -> int:
        j, k = pair
        return (self.modes[k] - self.modes[j]) % self.K

    def pair_observability(self, V: np.ndarray, observed_D: Iterable[str], threshold: float) -> dict[str, Any]:
        """Support coverage of each branch pair in the basis of ``V`` (fitted or true)."""
        V = np.asarray(V, dtype=complex)
        norms = np.linalg.norm(V, axis=0)
        u = V / np.where(norms > 0, norms, 1.0)[None, :]
        ids = [cid for cid in observed_D if cid in self.system_observable]
        strength = {}
        for j, k in self.pairs:
            strength[f"{self.modes[j]}-{self.modes[k]}"] = float(math.sqrt(sum(abs(u[:, j].conj() @ self.system_observable[cid] @ u[:, k]) ** 2 for cid in ids)))
        hidden = [label for label, s in strength.items() if s < threshold]
        disconnected = []
        for b, m in enumerate(self.modes):
            touching = [f"{self.modes[j]}-{self.modes[k]}" for j, k in self.pairs if b in (j, k)]
            if all(label in hidden for label in touching):
                disconnected.append(int(m))
        mask_class = "branch_disconnected" if disconnected else "pair_hidden_connected" if hidden else "all_pairs_observed"
        return {"strength": strength, "hidden_pairs": hidden, "disconnected_modes": disconnected, "mask_class": mask_class, "threshold": float(threshold)}


class SharedBranchPairEstimator:
    """Pure-history MLE with a free orthonormal branch basis; see the module docstring."""

    method = METHOD

    def __init__(
        self,
        active_modes: Sequence[int],
        K: int,
        d_S: int,
        effects: Mapping[str, Mapping[str, np.ndarray]],
        *,
        starts: int = 32,
        seed: int = 0,
        probability_floor: float = 1.0e-6,
        likelihood_floor: float = 1.0e-12,
        near_optimal_nats: float = 1.0,
        max_iterations: int = 4000,
        pair_observability_threshold: float = 0.05,
    ):
        self.forward = SharedBranchPairForwardModel(active_modes, K, d_S, effects)
        self.starts = int(starts)
        self.seed = int(seed)
        self.probability_floor = float(probability_floor)
        self.likelihood_floor = float(likelihood_floor)
        self.near_optimal_nats = float(near_optimal_nats)
        self.max_iterations = int(max_iterations)
        self.pair_observability_threshold = float(pair_observability_threshold)
        if self.starts < 1:
            raise ValueError("at least one start is required")
        self.basis = _anti_hermitian_basis(self.forward.d_S)
        settings = f"{METHOD}:{self.starts}:{self.seed}:{self.probability_floor}:{self.likelihood_floor}:{self.near_optimal_nats}:{self.max_iterations}:{self.pair_observability_threshold}:{self.forward.effects_hash}"
        self.checkpoint_hash = f"{METHOD}:{hashlib.sha256(settings.encode('ascii')).hexdigest()[:16]}"
        self.last_fit: dict[str, Any] | None = None

    # -- interface parity with the P1 kernel estimators -----------------------------------------
    def fit_records(self, records: Any) -> "SharedBranchPairEstimator":  # noqa: ARG002 - nothing is learned across records
        return self

    # -- parametrization ------------------------------------------------------------------------
    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        d, n = self.forward.d_S, self.forward.n
        nb = d * d
        X = np.tensordot(theta[:nb], self.basis, axes=1)
        z = np.concatenate([[0.0], theta[nb : nb + n - 1]])
        phases = np.concatenate([[0.0], theta[nb + n - 1 : nb + 2 * (n - 1)]])
        p = np.exp(z - z.max())
        p = p / p.sum()
        c = np.sqrt(p) * np.exp(1j * phases)
        return X, p, phases, c, expm(X)

    def branch_vectors(self, theta: np.ndarray) -> np.ndarray:
        """``V[:, j] = c_j u_j``."""
        _, _, _, c, W = self._unpack(theta)
        return W[:, : self.forward.n] * c[None, :]

    def _objective(self, theta: np.ndarray, A: np.ndarray, counts: np.ndarray) -> tuple[float, np.ndarray]:
        d, n = self.forward.d_S, self.forward.n
        nb = d * d
        X, p, _, c, W = self._unpack(theta)
        Q = W[:, :n]
        V = Q * c[None, :]
        x = V.T.reshape(-1)
        Ax = A @ x
        q = np.einsum("i,oi->o", x.conj(), Ax).real
        qf = q + self.likelihood_floor
        value = -float(counts @ np.log(qf))
        Gx = -2.0 * ((counts / qf)[:, None] * Ax).sum(axis=0)     # dL = Re(Gx^H dx)
        GV = Gx.reshape(n, d).T
        GQ = GV * c.conj()[None, :]
        GW = np.zeros((d, d), dtype=complex)
        GW[:, :n] = GQ
        GX = expm_frechet(X.conj().T, GW, compute_expm=False)
        grad = np.empty_like(theta)
        grad[:nb] = np.real(np.einsum("ij,kij->k", GX.conj(), self.basis))
        Gc = np.einsum("aj,aj->j", Q.conj(), GV)
        dphi = np.real(Gc.conj() * 1j * c)
        dp = np.real(Gc.conj() * c) / (2.0 * p)
        dz = p * (dp - float(dp @ p))
        grad[nb : nb + n - 1] = dz[1:]
        grad[nb + n - 1 :] = dphi[1:]
        return value, grad

    def _support_arrays(self, support: Mapping[str, Mapping[str, float]], n_shots: float) -> tuple[list[str], np.ndarray, np.ndarray]:
        ids = sorted(support)
        A = np.concatenate([self.forward.A[cid] for cid in ids])
        counts = np.concatenate([[float(support[cid].get(o, 0.0)) * n_shots for o in self.forward.outcomes] for cid in ids])
        return ids, A, counts

    def _seed_for(self, ids: list[str], counts: np.ndarray) -> int:
        digest = hashlib.sha256(f"{self.seed}:{','.join(ids)}".encode("utf-8"))
        digest.update(np.ascontiguousarray(np.round(counts, 9)).tobytes())
        return int(digest.hexdigest()[:16], 16)

    def fit_support(self, support: Mapping[str, Mapping[str, float]], n_shots: float = 1.0) -> dict[str, Any]:
        ids, A, counts = self._support_arrays(support, float(n_shots))
        rng = np.random.default_rng(self._seed_for(ids, counts))
        d, n = self.forward.d_S, self.forward.n
        size = d * d + 2 * (n - 1)
        runs = []
        for _ in range(self.starts):
            theta0 = np.concatenate([rng.normal(0.0, 1.0, d * d), np.zeros(n - 1), rng.uniform(0.0, 2.0 * math.pi, n - 1)])
            res = minimize(self._objective, theta0, args=(A, counts), jac=True, method="L-BFGS-B",
                           options={"maxiter": self.max_iterations, "ftol": 1.0e-13, "gtol": 1.0e-9})
            runs.append((float(res.fun), res.x.copy(), bool(res.success), int(res.nit)))
        runs.sort(key=lambda item: item[0])
        best_value, best_theta = runs[0][0], runs[0][1]
        assert best_theta.shape == (size,)
        V = self.branch_vectors(best_theta)
        near = [r for r in runs if r[0] <= best_value + self.near_optimal_nats]
        _, p, phases, _, _ = self._unpack(best_theta)
        fit = {
            "method": METHOD,
            "objective_nats": best_value,
            "starts": self.starts,
            "starts_converged": int(sum(r[2] for r in runs)),
            "starts_within_near_optimal": len(near),
            "pair_observability": self.forward.pair_observability(V, ids, self.pair_observability_threshold),
            "populations": p.tolist(),
            "relative_phases": np.mod(phases, 2.0 * math.pi).tolist(),
            "branch_vectors_real_imag": [V.real.tolist(), V.imag.tolist()],
            "n_shots": float(n_shots),
            "support_context_ids": ids,
        }
        self.last_fit = fit
        return fit

    def predict_table(self, support_table: MeasurementTable, contexts: list[MeasurementContext]) -> MeasurementTable:
        support = support_table.probabilities
        n_shots = float(support_table.metadata.get("n_shots", 1.0))
        fit = self.fit_support(support, n_shots)
        V = np.asarray(fit["branch_vectors_real_imag"][0]) + 1j * np.asarray(fit["branch_vectors_real_imag"][1])
        tables = self.forward.tables(V, [c.context_id for c in contexts])
        probabilities = {}
        for context in contexts:
            values = np.clip(tables[context.context_id], 0.0, None) + self.probability_floor
            values = values / values.sum()
            probabilities[context.context_id] = dict(zip(self.forward.outcomes, values.tolist(), strict=True))
        return MeasurementTable(
            probabilities=probabilities,
            contexts=list(contexts),
            metadata={**dict(support_table.metadata), "shared_branch_pair_fit": fit},
            source_density=None,
        )


# ---------------------------------------------------------------------------
# Comparator: within-context projection onto the declared clock spectrum
# ---------------------------------------------------------------------------


def declared_spectrum_bins(active_modes: Sequence[int], K: int) -> list[int]:
    """DFT bins that a zero-energy history can occupy: ``{(m_k - m_j) mod K}`` (bin 0 included)."""
    return sorted({(int(b) - int(a)) % int(K) for a in active_modes for b in active_modes})


class DeclaredSpectrumProjection:
    """Wrap a table estimator and project its output onto the declared clock spectrum, context by context.

    D (time-basis) contexts: the clock marginal is set flat (``1/K``), and
    ``g_P(tau) = K [q(tau, 0) - q(tau, 1)]`` keeps only the DFT bins in ``declared_spectrum_bins``.
    F (energy-basis) contexts: mass on inactive clock modes is removed. The projection uses no
    other context, so it isolates the effect of the frequency constraint from any transfer between
    contexts. Entries are then clipped at ``probability_floor`` and renormalized; the number of
    clipped entries is reported.
    """

    def __init__(self, base: Any, active_modes: Sequence[int], K: int, *, probability_floor: float = 1.0e-6):
        self.base = base
        self.K = int(K)
        self.modes = tuple(int(m) for m in active_modes)
        self.bins = declared_spectrum_bins(self.modes, self.K)
        self.probability_floor = float(probability_floor)
        self.outcomes = [outcome_key(l, b) for l in range(self.K) for b in (0, 1)]
        base_hash = getattr(base, "checkpoint_hash", type(base).__name__)
        self.checkpoint_hash = f"declared_spectrum_projection_v1:{base_hash}:{self.bins}"

    def fit_records(self, records: Any) -> "DeclaredSpectrumProjection":
        self.base.fit_records(records)
        return self

    def project(self, table: MeasurementTable) -> tuple[MeasurementTable, int]:
        mask = np.zeros(self.K, dtype=bool)
        mask[self.bins] = True
        inactive = np.ones(self.K, dtype=bool)
        inactive[list(self.modes)] = False
        clipped = 0
        probabilities = {}
        for context in table.contexts:
            row = table.probabilities[context.context_id]
            q = np.asarray([[float(row[outcome_key(l, b)]) for b in (0, 1)] for l in range(self.K)])
            if context.clock == "D":
                g = self.K * (q[:, 0] - q[:, 1])
                F = np.fft.fft(g)
                F[~mask] = 0.0
                g = np.fft.ifft(F).real
                q = np.stack([(1.0 + g) / (2.0 * self.K), (1.0 - g) / (2.0 * self.K)], axis=1)
                clipped += int(np.sum(q < self.probability_floor))
            else:
                q[inactive, :] = 0.0
                clipped += int(np.sum(q[~inactive, :] < self.probability_floor))
            values = q.reshape(-1)
            values = np.clip(values, self.probability_floor, None)
            values = values / values.sum()
            probabilities[context.context_id] = dict(zip(self.outcomes, values.tolist(), strict=True))
        projected = MeasurementTable(probabilities=probabilities, contexts=list(table.contexts), metadata={**dict(table.metadata), "declared_spectrum_projection": {"bins": self.bins, "clipped_entries": clipped}}, source_density=None)
        return projected, clipped

    def predict_table(self, support_table: MeasurementTable, contexts: list[MeasurementContext]) -> MeasurementTable:
        return self.project(self.base.predict_table(support_table, list(contexts)))[0]
