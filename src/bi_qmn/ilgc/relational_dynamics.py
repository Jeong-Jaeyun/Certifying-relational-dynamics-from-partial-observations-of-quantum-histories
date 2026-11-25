"""Relational-dynamics necessity (RDN): does a certified body force ``tau``-dependence?

ILGC certifies **existence**: some quantum state (Q), or some constraint-compatible state (BI),
reproduces every calibrated class body. Existence is not necessity. Let ``E`` be pinching in the
clock energy basis. For every state ``rho``:

* ``eps_con(E rho) = eps_con(rho)``, because ``H_tot^2`` is block diagonal in the clock energy basis;
* the coh (energy-basis F) class is unchanged;
* every diag (time-basis D) context is replaced by its mean over ``tau``;
* ``E rho = sum_r |r><r| x sigma_r`` is classical-quantum, hence separable. Its conditional system
  state ``rho_{S|tau}`` does not depend on ``tau``: a *static* relational history.

So if any BI-feasible state exists, the question "do the certified tables require relational
dynamics?" reduces to whether an ``E``-invariant state also fits the bodies. RDN is adjudicated
with the same three-valued, proof-carrying discipline as ILGC existence:

``certified``
    Every ``E``-invariant table is ``tau``-flat on the diag class, and ``tau``-flat vectors form a
    linear subspace. If the whitened distance from ``q_hat_diag`` to that subspace exceeds
    ``r_diag``, no ``E``-invariant state lies in the diag body. With identity whitening the
    distance is ``||(I - T) q_hat_diag||_2`` and the comparison is verified at ``precision_dps``.
``refuted``
    An explicit ``E``-invariant state (for example ``E`` of the verified BI primal state) is PSD,
    has unit trace, satisfies every registered constraint, and lies inside every class body. All
    of this is verified at ``precision_dps``.
``unresolved``
    Neither holds.

This is a report channel added by formulation patch v3. It never changes Q/BI/selector/atlas
statuses. It decides whether a certificate may be cited as evidence of relational dynamics,
clock-system coherence, or entanglement.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import mpmath as mp
import numpy as np

from bi_qmn.core.noise import parse_outcome_key


RDN_VERSION = "rdn_v1"
Key = tuple[str, str]  # (context_id, outcome)


def _groups(keys: Sequence[Key]) -> list[list[int]]:
    """Index groups spanning the ``tau``-flat subspace. One group per (D context, system bit).

    Keys of non-D contexts each form their own group: they carry no ``tau`` structure.
    """
    grouped: "OrderedDict[tuple, list[int]]" = OrderedDict()
    for index, (context_id, outcome) in enumerate(keys):
        if context_id.startswith("D,"):
            _, bit = parse_outcome_key(outcome)
            grouped.setdefault((context_id, bit), []).append(index)
        else:
            grouped[(context_id, outcome)] = [index]
    return list(grouped.values())


def tau_flat_basis(keys: Sequence[Key]) -> np.ndarray:
    groups = _groups(keys)
    B = np.zeros((len(keys), len(groups)))
    for column, members in enumerate(groups):
        B[members, column] = 1.0
    return B


def tau_flat_distance(target: np.ndarray, keys: Sequence[Key], whitening: np.ndarray | None = None) -> float:
    """``min_c ||L (B c - target)||_2`` (float64). Identity ``L`` gives ``||(I - T) target||``."""
    target = np.asarray(target, dtype=float)
    B = tau_flat_basis(keys)
    L = np.eye(len(keys)) if whitening is None else np.asarray(whitening, dtype=float)
    LB = L @ B
    coeff, *_ = np.linalg.lstsq(LB, L @ target, rcond=None)
    return float(np.linalg.norm(LB @ coeff - L @ target))


def _tau_flat_distance_sq_mp(target: np.ndarray, keys: Sequence[Key]) -> mp.mpf:
    """Exact-arithmetic (``mp.dps``) squared distance for identity whitening."""
    values = [mp.mpf(float(v)) for v in np.asarray(target, dtype=float)]
    total = mp.mpf(0)
    for members in _groups(keys):
        mean = mp.fsum(values[i] for i in members) / len(members)
        total += mp.fsum((values[i] - mean) ** 2 for i in members)
    return total


def _is_identity(matrix: np.ndarray | None, size: int) -> bool:
    return matrix is None or (np.asarray(matrix).shape == (size, size) and np.array_equal(np.asarray(matrix, dtype=float), np.eye(size)))


def energy_block_candidate(rho: np.ndarray, d_C: int, d_S: int) -> list[np.ndarray]:
    """Diagonal clock-energy blocks of ``E(rho)`` (the classical-quantum candidate)."""
    data = np.asarray(rho, dtype=complex)
    blocks = []
    for r in range(d_C):
        s = slice(r * d_S, (r + 1) * d_S)
        block = data[s, s]
        blocks.append((block + block.conj().T) / 2.0)
    return blocks


def _mp_block(block: np.ndarray) -> mp.matrix:
    return mp.matrix([[mp.mpc(complex(block[i, j]).real, complex(block[i, j]).imag) for j in range(block.shape[1])] for i in range(block.shape[0])])


def _mp_trace_product_blocks(effect: np.ndarray, blocks_mp: list[mp.matrix], d_S: int) -> mp.mpf:
    total = mp.mpc(0)
    for r, block in enumerate(blocks_mp):
        s = slice(r * d_S, (r + 1) * d_S)
        e = np.asarray(effect, dtype=complex)[s, s]
        for i in range(d_S):
            for j in range(d_S):
                if e[i, j] != 0:
                    total += mp.mpc(e[i, j].real, e[i, j].imag) * block[j, i]
    return mp.re(total)


def verify_classical_candidate(
    blocks: Sequence[np.ndarray],
    *,
    class_keys: Mapping[str, Sequence[Key]],
    targets: Mapping[str, np.ndarray],
    radii: Mapping[str, float],
    effects: Mapping[Key, np.ndarray],
    constraints: Sequence[tuple[np.ndarray, float | None, float | None]] = (),
    whitening: Mapping[str, np.ndarray] | None = None,
    d_S: int,
    precision_dps: int = 80,
    interior_floor: float = 1.0e-12,
) -> dict[str, Any]:
    """Verify an ``E``-invariant candidate ``sum_r |r><r| x blocks[r]`` against every obligation.

    In high precision the candidate is renormalized to unit trace and then mixed with the maximally
    mixed state, ``(1 - floor) sigma + floor I / d``. The mixing works like ILGC's PSD interior
    repair: float64 rounding of a rank-deficient state cannot then fake a negative eigenvalue. Both
    operations keep the state ``E``-invariant and separable, and the verified (repaired) state is
    the evidence.
    """
    with mp.workdps(precision_dps):
        mp_blocks = [_mp_block(np.asarray(b, dtype=complex)) for b in blocks]
        trace = mp.fsum(mp.re(b[i, i]) for b in mp_blocks for i in range(b.rows))
        if trace <= 0:
            return {"valid": False, "reason": "nonpositive_trace"}
        floor = mp.mpf(interior_floor)
        dim = len(mp_blocks) * d_S
        mp_blocks = [(1 - floor) * (b / trace) + (floor / dim) * mp.eye(d_S) for b in mp_blocks]
        min_eig = min(min(mp.re(v) for v in mp.eighe(b, eigvals_only=True)) for b in mp_blocks)
        psd = bool(min_eig >= 0)
        class_report = {}
        inside = psd
        for cls, keys in class_keys.items():
            pred = [_mp_trace_product_blocks(effects[key], mp_blocks, d_S) for key in keys]
            residual = [pred[i] - mp.mpf(float(targets[cls][i])) for i in range(len(keys))]
            L = None if whitening is None else whitening.get(cls)
            if _is_identity(L, len(keys)):
                dist_sq = mp.fsum(x * x for x in residual)
            else:
                Lm = np.asarray(L, dtype=float)
                dist_sq = mp.fsum(mp.fsum(mp.mpf(float(Lm[a, b])) * residual[b] for b in range(len(keys))) ** 2 for a in range(len(keys)))
            radius = mp.mpf(float(radii[cls]))
            ok = bool(dist_sq <= radius * radius)
            inside = inside and ok
            class_report[cls] = {"distance": float(mp.sqrt(dist_sq)), "radius": float(radius), "normalized": float(mp.sqrt(dist_sq) / radius) if radius > 0 else None, "inside": ok}
        constraint_report = []
        for G, lower, upper in constraints:
            value = _mp_trace_product_blocks(G, mp_blocks, d_S)
            ok = (lower is None or value >= mp.mpf(float(lower))) and (upper is None or value <= mp.mpf(float(upper)))
            inside = inside and bool(ok)
            constraint_report.append({"value": float(value), "lower": lower, "upper": upper, "satisfied": bool(ok)})
        return {
            "valid": bool(inside),
            "psd": psd,
            "min_block_eigenvalue": float(min_eig),
            "renormalized_from_trace": float(trace),
            "interior_floor": float(interior_floor),
            "classes": class_report,
            "constraints": constraint_report,
            "precision_dps": precision_dps,
            "energy_block_diagonal_by_construction": True,
        }


def optimal_classical_candidate(
    *,
    class_keys: Mapping[str, Sequence[Key]],
    targets: Mapping[str, np.ndarray],
    radii: Mapping[str, float],
    effects: Mapping[Key, np.ndarray],
    constraints: Sequence[tuple[np.ndarray, float | None, float | None]] = (),
    d_C: int,
    d_S: int,
    constraint_safety: float = 0.5,
    eigenvalue_floor: float = 1.0e-12,
) -> dict[str, Any]:
    """Proposal only: the ``E``-invariant state minimizing the largest normalized class distance.

    The proposal is solved with an interior margin (registered upper bounds scaled by
    ``constraint_safety``, block eigenvalues at least ``eigenvalue_floor``), so rounding does not
    push it across a boundary. Its output must pass ``verify_classical_candidate`` before it counts
    as evidence. The verification uses the unscaled bounds.
    """
    import cvxpy as cp

    blocks = [cp.Variable((d_S, d_S), hermitian=True) for _ in range(d_C)]
    t = cp.Variable(nonneg=True)
    cons = [b >> eigenvalue_floor * np.eye(d_S) for b in blocks]
    cons.append(sum(cp.real(cp.trace(b)) for b in blocks) == 1)

    vecs = [cp.vec(b, order="F") for b in blocks]

    def forward(operators: Sequence[np.ndarray]):
        # Tr[E sigma_r] = vec_F(E_rr^T) . vec_F(sigma_r), summed over clock energy blocks r.
        rows = []
        for r in range(d_C):
            s = slice(r * d_S, (r + 1) * d_S)
            rows.append(np.stack([np.asarray(E, dtype=complex)[s, s].T.flatten(order="F") for E in operators]))
        return cp.real(sum(rows[r] @ vecs[r] for r in range(d_C)))

    for cls, keys in class_keys.items():
        pred = forward([effects[key] for key in keys])
        cons.append(cp.norm(pred - np.asarray(targets[cls], dtype=float), 2) <= t * float(radii[cls]))
    for G, lower, upper in constraints:
        value = forward([G])[0]
        if upper is not None:
            cons.append(value <= constraint_safety * float(upper))
        if lower is not None:
            cons.append(value >= float(lower))
    problem = cp.Problem(cp.Minimize(t), cons)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return {"status": problem.status, "blocks": None, "objective": None}
    out = []
    for b in blocks:
        value = np.asarray(b.value, dtype=complex)
        value = (value + value.conj().T) / 2.0
        w, v = np.linalg.eigh(value)
        out.append((v * np.clip(w, eigenvalue_floor, None)) @ v.conj().T)
    return {"status": problem.status, "blocks": out, "objective": float(t.value), "constraint_safety": constraint_safety}


def adjudicate_relational_dynamics(
    *,
    class_keys: Mapping[str, Sequence[Key]],
    targets: Mapping[str, np.ndarray],
    radii: Mapping[str, float],
    effects: Mapping[Key, np.ndarray] | None = None,
    candidates: Sequence[tuple[str, Sequence[np.ndarray]]] = (),
    constraints: Sequence[tuple[np.ndarray, float | None, float | None]] = (),
    whitening: Mapping[str, np.ndarray] | None = None,
    d_S: int = 4,
    precision_dps: int = 80,
    diag_class: str = "diag",
) -> dict[str, Any]:
    """Three-valued RDN decision for one record (see module docstring)."""
    keys = list(class_keys[diag_class])
    target = np.asarray(targets[diag_class], dtype=float)
    radius = float(radii[diag_class])
    L = None if whitening is None else whitening.get(diag_class)
    if _is_identity(L, len(keys)):
        with mp.workdps(precision_dps):
            dist_sq = _tau_flat_distance_sq_mp(target, keys)
            lower_ok = bool(dist_sq > mp.mpf(radius) ** 2)
            distance = float(mp.sqrt(dist_sq))
        method = "identity_whitening_exact_group_means"
    else:
        distance = tau_flat_distance(target, keys, L)
        lower_ok = bool(distance > radius * (1.0 + 1.0e-9))
        method = "general_whitening_float64_with_relative_margin_1e-9"
    report: dict[str, Any] = {
        "rdn_version": RDN_VERSION,
        "tau_flat_distance": distance,
        "radius_diag": radius,
        "kappa": distance / radius if radius > 0 else float("inf"),
        "lower_bound_method": method,
        "candidates": [],
    }
    if lower_ok:
        report["status"] = "certified"
        return report
    if effects is not None:
        for name, blocks in candidates:
            check = verify_classical_candidate(
                blocks, class_keys=class_keys, targets=targets, radii=radii, effects=effects,
                constraints=constraints, whitening=whitening, d_S=d_S, precision_dps=precision_dps,
            )
            report["candidates"].append({"name": name, **check})
            if check["valid"]:
                report["status"] = "refuted"
                report["refuting_candidate"] = name
                return report
    report["status"] = "unresolved"
    return report
