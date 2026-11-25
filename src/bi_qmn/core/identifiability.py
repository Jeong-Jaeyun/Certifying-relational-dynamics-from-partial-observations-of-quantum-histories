"""Native identifiability of the declared Page--Wootters construction (estimator-free).

Added in the 2026-11-20 major revision, which followed the PW-QDELS-P2 identifiability study
(experiments E1--E3: reflection symmetry, shared latent phase, and the BI-QMN support bridge).
Everything here acts on exact states or exact tables of the declared construction. Nothing is
learned and nothing is calibrated.

Facts implemented here and checked in ``tests/core/test_identifiability.py``:

1. **Energy pinching.** ``energy_dephase`` (pinching in the clock energy basis, ``E``) leaves the
   constraint residual ``eps_con`` unchanged, because ``H_tot^2`` is block diagonal in that basis.
   It leaves every F (energy-basis) context unchanged. It replaces every D (time-basis) context
   by its average over the clock label ``tau``.
2. **Where coherence lives.** On stationary (zero-energy) states, F contexts carry only branch
   populations and branch system states. All inter-branch coherence, and therefore every
   ``tau``-dependence of the conditional system states, is the ``tau``-modulation of the D
   contexts.
3. **Separability.** ``E(rho)`` is classical-quantum in the clock energy basis, hence separable.
   On the exact zero-energy space of the P1 construction, states have maximally correlated form.
   For those states: separable == PPT == ``E``-invariant == every D context is ``tau``-flat.
   The last equivalence needs the measured set to detect every populated mode pair.
4. **Dihedral symmetry.** Complex conjugation in the product computational basis is time
   reversal. It maps a table by ``tau -> -tau`` on D contexts and flips the outcome bit for
   system words with an odd number of ``Y``. Shifting relative phases by the clock spacing
   translates D contexts cyclically. The seeded P1 family (real system unitaries, uniform relative
   phases) has the same law under both maps, so orientation and time origin are frame
   conventions.
5. **Modulation bound.** For states whose clock marginal is energy diagonal (all zero-energy
   states of the construction),
   ``||(I - T) q_diag||_2^2 = sum_{P in X_D} Var_tau <P>_tau / (2K)
   <= d_S (1 - Tr rho_S^2) / (2K)``.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from bi_qmn.core.linalg import fourier_matrix, kron
from bi_qmn.core.noise import outcome_key, parse_outcome_key
from bi_qmn.core.types import MeasurementContext


# ---------------------------------------------------------------------------
# State maps
# ---------------------------------------------------------------------------


def _array(rho) -> np.ndarray:
    return np.asarray(getattr(rho, "data", rho), dtype=complex)


def energy_dephase(rho, d_C: int, d_S: int) -> np.ndarray:
    """Pinching in the clock energy basis: ``sum_r (|r><r| x I) rho (|r><r| x I)``."""
    data = _array(rho)
    if data.shape != (d_C * d_S, d_C * d_S):
        raise ValueError("state dimension does not match d_C * d_S")
    out = np.zeros_like(data)
    for r in range(d_C):
        block = slice(r * d_S, (r + 1) * d_S)
        out[block, block] = data[block, block]
    return out


def tau_dephase(rho, d_C: int, d_S: int) -> np.ndarray:
    """Pinching in the clock-label (``tau``) basis; the P1 ``gamma=0`` negative control."""
    data = _array(rho)
    F = fourier_matrix(d_C)
    out = np.zeros_like(data)
    for tau in range(d_C):
        row = F[tau, :].reshape(-1, 1)
        P = kron(row @ row.conj().T, np.eye(d_S, dtype=complex))
        out += P @ data @ P
    return (out + out.conj().T) / 2.0


def is_energy_block_diagonal(rho, d_C: int, d_S: int, atol: float = 1.0e-12) -> bool:
    data = _array(rho)
    return bool(np.max(np.abs(data - energy_dephase(data, d_C, d_S))) <= atol)


def min_partial_transpose_eigenvalue(rho, d_C: int, d_S: int) -> float:
    """Smallest eigenvalue of the partial transpose on the system factor (PPT test)."""
    data = _array(rho).reshape(d_C, d_S, d_C, d_S)
    pt = data.transpose(0, 3, 2, 1).reshape(d_C * d_S, d_C * d_S)
    return float(np.linalg.eigvalsh((pt + pt.conj().T) / 2.0).min())


def system_marginal(rho, d_C: int, d_S: int) -> np.ndarray:
    return np.einsum("rarb->ab", _array(rho).reshape(d_C, d_S, d_C, d_S))


# ---------------------------------------------------------------------------
# Table maps (tables are {context_id: {outcome_key: probability}})
# ---------------------------------------------------------------------------


Table = Mapping[str, Mapping[str, float]]


def _y_parity(system_word: str) -> int:
    return system_word.count("Y") % 2


def tau_mean_table(table: Table, contexts: Sequence[MeasurementContext], K: int) -> dict[str, dict[str, float]]:
    """``T``: each D context replaced by its mean over ``tau`` (per system bit); F unchanged."""
    out: dict[str, dict[str, float]] = {}
    for context in contexts:
        row = table[context.context_id]
        if context.clock == "D":
            means = [float(np.mean([row[outcome_key(l, b)] for l in range(K)])) for b in (0, 1)]
            out[context.context_id] = {outcome_key(l, b): means[b] for l in range(K) for b in (0, 1)}
        else:
            out[context.context_id] = {key: float(value) for key, value in row.items()}
    return out


def time_reverse_table(table: Table, contexts: Sequence[MeasurementContext], K: int) -> dict[str, dict[str, float]]:
    """Table of the complex-conjugated state.

    D contexts map ``tau -> -tau mod K``. Both bases flip the system bit when the word has an odd
    number of ``Y``.
    """
    out: dict[str, dict[str, float]] = {}
    for context in contexts:
        flip = _y_parity(context.system)
        row = table[context.context_id]
        new: dict[str, float] = {}
        for key, value in row.items():
            label, bit = parse_outcome_key(key)
            target_label = (-label) % K if context.clock == "D" else label
            new[outcome_key(target_label, bit ^ flip)] = float(value)
        out[context.context_id] = new
    return out


def translate_table(table: Table, contexts: Sequence[MeasurementContext], K: int, shift: int) -> dict[str, dict[str, float]]:
    """Cyclic clock-label translation on D contexts: ``p'(tau) = p(tau - shift)``; F unchanged."""
    out: dict[str, dict[str, float]] = {}
    for context in contexts:
        row = table[context.context_id]
        if context.clock == "D":
            out[context.context_id] = {
                outcome_key((parse_outcome_key(key)[0] + shift) % K, parse_outcome_key(key)[1]): float(value)
                for key, value in row.items()
            }
        else:
            out[context.context_id] = {key: float(value) for key, value in row.items()}
    return out


def translated_phases(phases: Sequence[float], active_modes: Sequence[int], K: int, shift: int) -> np.ndarray:
    """Relative phases whose D table is ``translate_table(..., shift)`` of the original."""
    modes = np.asarray(active_modes, dtype=float)
    return np.asarray(phases, dtype=float) - 2.0 * math.pi * modes * shift / K


def class_vector(table: Table, contexts: Sequence[MeasurementContext], context_class: str, K: int) -> np.ndarray:
    keys = [outcome_key(l, b) for l in range(K) for b in (0, 1)]
    return np.asarray(
        [float(table[c.context_id][key]) for c in contexts if c.context_class == context_class for key in keys],
        dtype=float,
    )


def tau_modulation(table: Table, contexts: Sequence[MeasurementContext], K: int) -> dict[str, object]:
    """Per-context and diag-class ``||(I - T) q||_2``. F contexts are zero by definition of ``T``."""
    mean = tau_mean_table(table, contexts, K)
    keys = [outcome_key(l, b) for l in range(K) for b in (0, 1)]
    per_context = {}
    for context in contexts:
        if context.clock != "D":
            continue
        diff = np.asarray([table[context.context_id][k] - mean[context.context_id][k] for k in keys], dtype=float)
        per_context[context.context_id] = float(np.linalg.norm(diff))
    total = float(math.sqrt(sum(v * v for v in per_context.values())))
    return {"per_context": per_context, "diag_class": total}


def tau_modulation_upper_bound(d_S: int, K: int, rho_S_purity: float) -> float:
    """Bound 5 of the module docstring, given ``Tr rho_S^2``."""
    return float(math.sqrt(max(0.0, d_S * (1.0 - rho_S_purity)) / (2.0 * K)))


def universal_tau_modulation_bound(d_S: int, K: int, populated_levels: int) -> float:
    """Largest possible diag-class modulation for any state supported on ``populated_levels`` branches."""
    return tau_modulation_upper_bound(d_S, K, 1.0 / populated_levels)


# ---------------------------------------------------------------------------
# Which D contexts can see which branch coherence
# ---------------------------------------------------------------------------


def pauli_word(label: str) -> np.ndarray:
    pauli = {
        "I": np.eye(2, dtype=complex),
        "X": np.asarray([[0, 1], [1, 0]], dtype=complex),
        "Y": np.asarray([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.asarray([[1, 0], [0, -1]], dtype=complex),
    }
    return kron(*(pauli[ch] for ch in label))


def pair_detection_matrix(system_unitary: np.ndarray, n_active: int, system_words: Iterable[str]) -> dict[str, dict[str, float]]:
    """``|<u_j|P|u_k>|`` for active branch pairs ``j<k``.

    The D context with word ``P`` detects coherence between branches ``j,k`` iff this is nonzero
    (active-mode differences are distinct mod K for the declared construction).
    """
    U = np.asarray(system_unitary, dtype=complex)
    out: dict[str, dict[str, float]] = {}
    for word in system_words:
        P = pauli_word(word)
        out[word] = {f"{j}{k}": float(abs(U[:, j].conj() @ P @ U[:, k])) for j, k in combinations(range(n_active), 2)}
    return out


def tau_flat_words(system_unitary: np.ndarray, n_active: int, system_words: Iterable[str], atol: float = 1.0e-12) -> list[str]:
    """D-context system words that are exactly ``tau``-flat for every state of the construction."""
    matrix = pair_detection_matrix(system_unitary, n_active, system_words)
    return sorted(word for word, pairs in matrix.items() if max(pairs.values()) <= atol)


def undetected_pairs(system_unitary: np.ndarray, n_active: int, system_words: Iterable[str], atol: float = 1.0e-12) -> list[str]:
    """Active branch pairs whose coherence no measured D context can see."""
    matrix = pair_detection_matrix(system_unitary, n_active, system_words)
    pairs = [f"{j}{k}" for j, k in combinations(range(n_active), 2)]
    return [pair for pair in pairs if all(matrix[word][pair] <= atol for word in matrix)]


def difference_harmonics_distinct(active_modes: Sequence[int], K: int) -> bool:
    """Active-mode differences are pairwise distinct up to sign mod K (no harmonic aliasing)."""
    diffs = []
    for a, b in combinations(active_modes, 2):
        d = (b - a) % K
        diffs.append(min(d, K - d))
    return len(set(diffs)) == len(diffs) and all(d != 0 for d in diffs)


# ---------------------------------------------------------------------------
# Contract adequacy precheck: as written (v2) and as revised (v3)
# ---------------------------------------------------------------------------


def _herm_vec(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X.real.ravel(), X.imag.ravel()])


def _orthonormal_columns(columns: list[np.ndarray], rtol: float = 1.0e-10) -> np.ndarray:
    if not columns:
        return np.zeros((0, 0))
    M = np.stack(columns, axis=1)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    keep = s > rtol * max(1.0, float(s[0]))
    return U[:, keep]


def _forward_rows(effects: Sequence[np.ndarray]) -> np.ndarray:
    # Tr[E X] = <E^*, X> elementwise; for Hermitian E this is Re(E)·Re(X) + Im(E)·Im(X).
    return np.stack([_herm_vec(np.asarray(E, dtype=complex)) for E in effects], axis=0)


def adequacy_precheck(
    densities: Sequence[np.ndarray],
    d_effects: Sequence[np.ndarray],
    f_effects: Sequence[np.ndarray],
    d_C: int,
    d_S: int,
) -> dict[str, object]:
    """Contract precheck values on a construction bank.

    ``contract_v2``: ``inf ||A_F(Y)||`` over unit ``Y`` in ``(I - D_tau) V_sig``, as written in
    contract ver2, Eq. (gamma-coh-clean).
    ``revised_v3``: ``inf ||A_D(Y)||`` over unit ``Y`` in ``(I - E) V_sig``. This is the
    energy-coherent part of the signal space, seen by the time-basis contexts.
    Also returned: ``||A_F||`` restricted to ``(I - E) V_sig``. Fact 1 makes it exactly zero.
    """
    rhos = [np.asarray(_array(r)) for r in densities]
    diffs = [r - rhos[0] for r in rhos[1:]]
    v_sig = _orthonormal_columns([_herm_vec(X) for X in diffs])
    dim = d_C * d_S

    def unvec(v: np.ndarray) -> np.ndarray:
        return v[: dim * dim].reshape(dim, dim) + 1j * v[dim * dim :].reshape(dim, dim)

    basis = [unvec(v_sig[:, i]) for i in range(v_sig.shape[1])]
    v_coh_tau = _orthonormal_columns([_herm_vec(X - tau_dephase(X, d_C, d_S)) for X in basis])
    v_coh_E = _orthonormal_columns([_herm_vec(X - energy_dephase(X, d_C, d_S)) for X in basis])
    A_F = _forward_rows(f_effects)
    A_D = _forward_rows(d_effects)

    def inf_gain(A: np.ndarray, V: np.ndarray) -> float:
        if V.size == 0:
            return float("nan")
        return float(np.linalg.svd(A @ V, compute_uv=False).min()) if V.shape[1] <= A.shape[0] else 0.0

    def sup_gain(A: np.ndarray, V: np.ndarray) -> float:
        if V.size == 0:
            return float("nan")
        return float(np.linalg.svd(A @ V, compute_uv=False).max())

    return {
        "signal_dimension": int(v_sig.shape[1]),
        "tau_coherent_dimension": int(v_coh_tau.shape[1]),
        "energy_coherent_dimension": int(v_coh_E.shape[1]),
        "contract_v2_gamma_coh": inf_gain(A_F, v_coh_tau),
        "revised_v3_gamma_coh": inf_gain(A_D, v_coh_E),
        "f_contexts_on_energy_coherent_subspace_max_gain": sup_gain(A_F, v_coh_E),
    }
