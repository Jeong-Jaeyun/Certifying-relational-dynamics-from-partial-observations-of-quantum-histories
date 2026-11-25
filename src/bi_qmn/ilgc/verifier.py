from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mpmath as mp
import numpy as np
from qiskit.quantum_info import DensityMatrix

from .solver import DensityProposal, SeparatorProposal
from .types import (
    CalibrationBody,
    ConstraintRegistrySeparationEvidence,
    ConstraintRegistry,
    DualEvidence,
    DualSeparationEvidence,
    ExistencePrimalEvidence,
    MeasurementModel,
    SelectorLowerBoundEvidence,
)


DEFAULT_PRECISION_DPS = 80
TRACE_REPRESENTATION_TOLERANCE = 1.0e-12
PSD_INTERIOR_FLOOR = 1.0e-11


@dataclass(frozen=True)
class PrimalVerification:
    evidence: ExistencePrimalEvidence | None
    profile_valid: bool
    accepted: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DualVerification:
    evidence: DualEvidence | None
    refuted: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SelectorCandidateVerification:
    density: DensityMatrix | None
    distance_upper_bound: float | None
    radial_evidence: ExistencePrimalEvidence | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SelectorLowerVerification:
    distance_lower_bound: float | None
    evidence: SelectorLowerBoundEvidence | None
    diagnostics: dict[str, Any]


def _mp_scalar(value: float | complex) -> mp.mpf | mp.mpc:
    number = complex(value)
    if abs(number.imag) == 0.0:
        return mp.mpf(number.real)
    return mp.mpc(number.real, number.imag)


def _mp_matrix(value: np.ndarray) -> mp.matrix:
    array = np.asarray(value, dtype=complex)
    return mp.matrix(
        [
            [_mp_scalar(array[row, column]) for column in range(array.shape[1])]
            for row in range(array.shape[0])
        ]
    )


def _mp_vector(value: np.ndarray) -> mp.matrix:
    array = np.asarray(value).reshape(-1)
    return mp.matrix([_mp_scalar(item) for item in array])


def _trace_product(left: mp.matrix, right: mp.matrix) -> mp.mpc:
    dim = left.rows
    return mp.fsum(
        left[row, column] * right[column, row]
        for row in range(dim)
        for column in range(dim)
    )


def _hermitian_eigenvalues(matrix: mp.matrix) -> list[mp.mpf]:
    values = mp.eighe(matrix, eigvals_only=True)
    return [mp.re(values[index]) for index in range(values.rows)]


def _frobenius_norm(matrix: mp.matrix) -> mp.mpf:
    return mp.sqrt(
        mp.fsum(
            abs(matrix[row, column]) ** 2
            for row in range(matrix.rows)
            for column in range(matrix.cols)
        )
    )


def _validate_precision(precision_dps: int) -> None:
    if not isinstance(precision_dps, int) or precision_dps <= 0:
        raise ValueError("precision_dps must be a positive integer")


def _validate_common_inputs(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
) -> None:
    registry.validate(model.hilbert_dim)
    if set(bodies) != set(model.context_classes):
        raise ValueError("calibration body classes do not match measurement model")
    for context_class in model.context_classes:
        bodies[context_class].validate(len(model.indices_for_class(context_class)))


def _validate_density_array(candidate: np.ndarray, expected_dim: int) -> np.ndarray:
    raw = np.asarray(candidate, dtype=complex)
    if raw.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"density matrix shape {raw.shape} does not match {(expected_dim, expected_dim)}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("density matrix must be finite")
    return raw


def sanitize_density_proposal(
    candidate: np.ndarray,
    *,
    expected_dim: int,
    psd_floor: float = PSD_INTERIOR_FLOOR,
) -> tuple[DensityMatrix, float]:
    raw = _validate_density_array(candidate, expected_dim)
    if not np.isfinite(float(psd_floor)) or psd_floor < 0.0:
        raise ValueError("psd_floor must be finite and nonnegative")
    hermitian = (raw + raw.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    clipped = np.maximum(eigenvalues, 0.0)
    if float(np.sum(clipped)) <= 0.0:
        clipped = np.ones_like(clipped)
    projected = (eigenvectors * clipped) @ eigenvectors.conj().T
    projected /= float(np.real(np.trace(projected)))
    dim = projected.shape[0]
    floor = min(float(psd_floor), 0.25 / dim)
    repaired = (1.0 - dim * floor) * projected + floor * np.eye(dim, dtype=complex)
    repaired = (repaired + repaired.conj().T) / 2.0
    repaired /= float(np.real(np.trace(repaired)))
    repair_norm = float(np.linalg.norm(repaired - raw, ord="fro"))
    return DensityMatrix(repaired), repair_norm


def repair_density_matrix(
    candidate: np.ndarray,
    *,
    psd_floor: float = PSD_INTERIOR_FLOOR,
) -> tuple[DensityMatrix, float]:
    raw = np.asarray(candidate)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("density proposal must be a square matrix")
    return sanitize_density_proposal(
        raw,
        expected_dim=int(raw.shape[0]),
        psd_floor=psd_floor,
    )


def _predictions_mp(model: MeasurementModel, rho: mp.matrix) -> list[mp.mpf]:
    predictions: list[mp.mpf] = []
    for entry in model.entries:
        effect = _mp_matrix(np.asarray(entry.effect.data, dtype=complex))
        predictions.append(mp.re(_trace_product(effect, rho)))
    return predictions


def _class_scores_mp(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    predictions: list[mp.mpf],
) -> dict[str, mp.mpf]:
    scores: dict[str, mp.mpf] = {}
    for context_class in model.context_classes:
        indices = model.indices_for_class(context_class)
        body = bodies[context_class]
        residual = mp.matrix(
            [
                predictions[index] - mp.mpf(model.entries[index].target)
                for index in indices
            ]
        )
        whitened = _mp_matrix(np.asarray(body.whitening, dtype=float)) * residual
        norm = mp.sqrt(mp.fsum(abs(whitened[index]) ** 2 for index in range(whitened.rows)))
        scores[context_class] = norm / mp.mpf(body.radius)
    return scores


def verify_primal_evidence(
    proof_matrix: DensityMatrix | np.ndarray,
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    precision_dps: int = DEFAULT_PRECISION_DPS,
    repair_frobenius: float = 0.0,
    proposal_metadata: dict[str, Any] | None = None,
) -> PrimalVerification:
    try:
        _validate_precision(precision_dps)
        _validate_common_inputs(model, bodies, registry)
        raw = _validate_density_array(
            (
                np.asarray(proof_matrix.data, dtype=complex)
                if isinstance(proof_matrix, DensityMatrix)
                else np.asarray(proof_matrix, dtype=complex)
            ),
            model.hilbert_dim,
        )
        if not np.allclose(raw, raw.conj().T, atol=1.0e-12, rtol=0.0):
            raise ValueError("proof matrix must be Hermitian")
        if not np.isfinite(float(repair_frobenius)) or repair_frobenius < 0.0:
            raise ValueError("repair_frobenius must be finite and nonnegative")
        density = DensityMatrix(raw.copy())
    except Exception as exc:
        return PrimalVerification(
            evidence=None,
            profile_valid=False,
            accepted=False,
            diagnostics={"reason": "primal_evidence_malformed", "error": str(exc)},
        )

    try:
        with mp.workdps(precision_dps):
            rho = _mp_matrix(raw)
            trace_value = mp.re(mp.fsum(rho[index, index] for index in range(rho.rows)))
            trace_error = abs(trace_value - 1)
            eigenvalues = _hermitian_eigenvalues(rho)
            min_eigenvalue = min(eigenvalues)
            predictions = _predictions_mp(model, rho)
            scores_mp = _class_scores_mp(model, bodies, predictions)
            constraint_values_mp: dict[str, mp.mpf] = {}
            constraints_ok = True
            for record in registry.records:
                operator = _mp_matrix(np.asarray(record.operator.data, dtype=complex))
                value = mp.re(_trace_product(operator, rho))
                constraint_values_mp[record.constraint_id] = value
                if record.lower_bound is not None and value < mp.mpf(record.lower_bound):
                    constraints_ok = False
                if record.upper_bound is not None and value > mp.mpf(record.upper_bound):
                    constraints_ok = False
            density_ok = (
                trace_error <= mp.mpf(TRACE_REPRESENTATION_TOLERANCE)
                and min_eigenvalue >= 0
            )
            radial_upper = max(scores_mp.values(), default=mp.mpf("0"))
            profile_valid = bool(density_ok and constraints_ok)
            accepted = bool(profile_valid and radial_upper <= 1)
            evidence = ExistencePrimalEvidence(
                density=density,
                radial_upper_bound=float(radial_upper),
                class_scores={key: float(value) for key, value in scores_mp.items()},
                constraint_values={
                    key: float(value) for key, value in constraint_values_mp.items()
                },
                trace_error=float(trace_error),
                min_eigenvalue=float(min_eigenvalue),
                repair_frobenius=float(repair_frobenius),
                verifier_precision_dps=precision_dps,
                proposal=dict(proposal_metadata or {"source": "proof_replay"}),
            )
            return PrimalVerification(
                evidence=evidence,
                profile_valid=profile_valid,
                accepted=accepted,
                diagnostics={
                    "density_ok": density_ok,
                    "constraints_ok": constraints_ok,
                    "trace_error": float(trace_error),
                    "min_eigenvalue": float(min_eigenvalue),
                    "radial_upper_bound": float(radial_upper),
                    "proof_matrix_hash": evidence.proof_matrix_hash,
                    "proof_matrix_modified": False,
                },
            )
    except Exception as exc:
        return PrimalVerification(
            evidence=None,
            profile_valid=False,
            accepted=False,
            diagnostics={"reason": "primal_replay_failed", "error": str(exc)},
        )


def verify_primal_proposal(
    proposal: DensityProposal,
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    precision_dps: int = DEFAULT_PRECISION_DPS,
) -> PrimalVerification:
    try:
        _validate_precision(precision_dps)
        if not np.isfinite(float(proposal.raw_objective)):
            raise ValueError("density proposal raw objective must be finite")
        density, repair_norm = sanitize_density_proposal(
            proposal.density,
            expected_dim=model.hilbert_dim,
        )
    except Exception as exc:
        return PrimalVerification(
            evidence=None,
            profile_valid=False,
            accepted=False,
            diagnostics={"reason": "density_sanitization_failed", "error": str(exc)},
        )
    return verify_primal_evidence(
        density,
        model,
        bodies,
        registry,
        precision_dps=precision_dps,
        repair_frobenius=repair_norm,
        proposal_metadata={
            "solver": proposal.solver,
            "solver_status": proposal.status,
            "attempt_index": proposal.attempt_index,
            "raw_objective": proposal.raw_objective,
            "sanitizer": "density_projection_with_declared_psd_floor_v1",
        },
    )


def _separator_components_mp(
    proposal: SeparatorProposal,
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
) -> tuple[mp.matrix, mp.mpf, mp.mpf, list[mp.mpf], float]:
    z = [mp.mpf(value) for value in np.asarray(proposal.z, dtype=float)]
    raw_mu = np.asarray(proposal.multipliers, dtype=float)
    repaired_mu = np.maximum(raw_mu, 0.0)
    multiplier_repair = float(np.linalg.norm(repaired_mu - raw_mu))
    mu = [mp.mpf(value) for value in repaired_mu]

    dim = model.hilbert_dim
    B = mp.matrix(dim, dim)
    for index, entry in enumerate(model.entries):
        B += z[index] * _mp_matrix(np.asarray(entry.effect.data, dtype=complex))
    for index, side in enumerate(registry.canonical_sides()):
        B -= mu[index] * _mp_matrix(np.asarray(side.operator.data, dtype=complex))

    calibration_support = mp.mpf("0")
    for context_class in model.context_classes:
        indices = model.indices_for_class(context_class)
        body = bodies[context_class]
        L_transpose = _mp_matrix(np.asarray(body.whitening, dtype=float).T)
        z_class = mp.matrix([z[index] for index in indices])
        dual_vector = mp.lu_solve(L_transpose, z_class)
        norm = mp.sqrt(
            mp.fsum(abs(dual_vector[index]) ** 2 for index in range(dual_vector.rows))
        )
        calibration_support += mp.mpf(body.radius) * norm

    q_dot_z = mp.fsum(
        mp.mpf(entry.target) * z[index] for index, entry in enumerate(model.entries)
    )
    return B, calibration_support, q_dot_z, mu, multiplier_repair


def _validate_separator_proposal(
    proposal: SeparatorProposal,
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    precision_dps: int,
    require_auxiliary: bool,
) -> None:
    _validate_precision(precision_dps)
    _validate_common_inputs(model, bodies, registry)
    z = np.asarray(proposal.z, dtype=float)
    if z.shape != (len(model.entries),) or not np.all(np.isfinite(z)):
        raise ValueError("separator z must be a finite measurement-length vector")
    multipliers = np.asarray(proposal.multipliers, dtype=float)
    if multipliers.shape != (len(registry.canonical_sides()),) or not np.all(
        np.isfinite(multipliers)
    ):
        raise ValueError("separator multipliers must match the canonical constraint sides")
    if not np.isfinite(float(proposal.raw_objective)):
        raise ValueError("separator raw objective must be finite")
    auxiliary = proposal.auxiliary_operator
    if require_auxiliary and auxiliary is None:
        raise ValueError("selector separator requires an auxiliary operator")
    if auxiliary is not None:
        array = np.asarray(auxiliary, dtype=complex)
        expected = (model.hilbert_dim, model.hilbert_dim)
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise ValueError("auxiliary operator must be a finite ambient square matrix")
        if not np.allclose(array, array.conj().T, atol=1.0e-12, rtol=0.0):
            raise ValueError("auxiliary operator must be Hermitian")


def verify_radial_separator(
    proposal: SeparatorProposal,
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    precision_dps: int = DEFAULT_PRECISION_DPS,
) -> DualVerification:
    try:
        _validate_separator_proposal(
            proposal,
            model,
            bodies,
            registry,
            precision_dps=precision_dps,
            require_auxiliary=False,
        )
        sides = registry.canonical_sides()
        with mp.workdps(precision_dps):
            B, support, q_dot_z, mu, multiplier_repair = _separator_components_mp(
                proposal, model, bodies, registry
            )
            eigenvalues = _hermitian_eigenvalues(B)
            state_support = max(eigenvalues)
            state_support += mp.fsum(
                mu[index] * mp.mpf(side.upper_bound)
                for index, side in enumerate(sides)
            )
            numerator = q_dot_z - state_support
            if support <= mp.mpf("1e-50"):
                if sides and numerator > 0:
                    G_sum = mp.matrix(model.hilbert_dim, model.hilbert_dim)
                    for index, side in enumerate(sides):
                        G_sum += mu[index] * _mp_matrix(
                            np.asarray(side.operator.data, dtype=complex)
                        )
                    spectral_lower = min(_hermitian_eigenvalues(G_sum))
                    registered_upper = mp.fsum(
                        mu[index] * mp.mpf(side.upper_bound)
                        for index, side in enumerate(sides)
                    )
                    margin = spectral_lower - registered_upper
                    evidence = ConstraintRegistrySeparationEvidence(
                        separation_margin=float(margin),
                        spectral_lower_bound=float(spectral_lower),
                        registered_upper_bound=float(registered_upper),
                        multiplier_repair=multiplier_repair,
                        separator_z=tuple(
                            float(value) for value in np.asarray(proposal.z, dtype=float)
                        ),
                        constraint_multipliers=tuple(float(value) for value in mu),
                        constraint_multiplier_labels=tuple(
                            side.side_id for side in sides
                        ),
                        verifier_precision_dps=precision_dps,
                        proposal={
                            "solver": proposal.solver,
                            "solver_status": proposal.status,
                            "attempt_index": proposal.attempt_index,
                            "raw_objective": proposal.raw_objective,
                        },
                    )
                    return DualVerification(
                        evidence=evidence,
                        refuted=bool(margin > 0),
                        diagnostics={
                            "constraint_registry_separation_margin": float(margin),
                            "spectral_lower_bound": float(spectral_lower),
                            "registered_upper_bound": float(registered_upper),
                        },
                    )
                return DualVerification(
                    evidence=None,
                    refuted=False,
                    diagnostics={"reason": "zero_calibration_support"},
                )
            lower_bound = numerator / support
            evidence = DualSeparationEvidence(
                radial_lower_bound=float(lower_bound),
                separation_numerator=float(numerator),
                calibration_support=float(support),
                state_support_upper=float(state_support),
                multiplier_repair=multiplier_repair,
                separator_z=tuple(
                    float(value) for value in np.asarray(proposal.z, dtype=float)
                ),
                constraint_multipliers=tuple(float(value) for value in mu),
                constraint_multiplier_labels=tuple(side.side_id for side in sides),
                verifier_precision_dps=precision_dps,
                proposal={
                    "solver": proposal.solver,
                    "solver_status": proposal.status,
                    "attempt_index": proposal.attempt_index,
                    "raw_objective": proposal.raw_objective,
                },
            )
            return DualVerification(
                evidence=evidence,
                refuted=bool(lower_bound > 1),
                diagnostics={
                    "radial_lower_bound": float(lower_bound),
                    "calibration_support": float(support),
                    "separation_numerator": float(numerator),
                },
            )
    except Exception as exc:
        return DualVerification(
            evidence=None,
            refuted=False,
            diagnostics={"reason": "dual_verification_failed", "error": str(exc)},
        )


def verify_selector_candidate(
    proposal: DensityProposal,
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    precision_dps: int = DEFAULT_PRECISION_DPS,
) -> SelectorCandidateVerification:
    primal = verify_primal_proposal(
        proposal,
        model,
        bodies,
        registry,
        precision_dps=precision_dps,
    )
    if not primal.accepted or primal.evidence is None:
        return SelectorCandidateVerification(
            density=None,
            distance_upper_bound=None,
            radial_evidence=primal.evidence,
            diagnostics={
                "reason": "selector_candidate_not_in_accepted_set",
                **primal.diagnostics,
                "constraint_values": (
                    {}
                    if primal.evidence is None
                    else dict(primal.evidence.constraint_values)
                ),
                "constraint_bounds": {
                    record.constraint_id: {
                        "lower": record.lower_bound,
                        "upper": record.upper_bound,
                    }
                    for record in registry.records
                },
            },
        )
    with mp.workdps(precision_dps):
        rho = _mp_matrix(np.asarray(primal.evidence.density.data, dtype=complex))
        rho0 = mp.eye(model.hilbert_dim) / model.hilbert_dim
        distance = _frobenius_norm(rho - rho0)
    return SelectorCandidateVerification(
        density=primal.evidence.density,
        distance_upper_bound=float(distance),
        radial_evidence=primal.evidence,
        diagnostics={
            "distance_upper_bound": float(distance),
            "proposal": dict(primal.evidence.proposal),
        },
    )


def verify_selector_lower_bound(
    proposal: SeparatorProposal,
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    precision_dps: int = DEFAULT_PRECISION_DPS,
) -> SelectorLowerVerification:
    try:
        _validate_separator_proposal(
            proposal,
            model,
            bodies,
            registry,
            precision_dps=precision_dps,
            require_auxiliary=True,
        )
        sides = registry.canonical_sides()
        with mp.workdps(precision_dps):
            B_measure, support, q_dot_z, mu, _ = _separator_components_mp(
                proposal, model, bodies, registry
            )
            auxiliary = (
                mp.matrix(model.hilbert_dim, model.hilbert_dim)
                if proposal.auxiliary_operator is None
                else _mp_matrix(np.asarray(proposal.auxiliary_operator, dtype=complex))
            )
            auxiliary_floor = min(_hermitian_eigenvalues(auxiliary))
            B = B_measure + auxiliary
            beta = q_dot_z - support - mp.fsum(
                mu[index] * mp.mpf(side.upper_bound)
                for index, side in enumerate(sides)
            ) + auxiliary_floor
            rho0 = mp.eye(model.hilbert_dim) / model.hilbert_dim
            center_value = mp.re(_trace_product(B, rho0))
            norm = _frobenius_norm(B)
            if norm <= mp.mpf("1e-50"):
                return SelectorLowerVerification(
                    distance_lower_bound=None,
                    evidence=None,
                    diagnostics={"reason": "zero_selector_separator_norm"},
                )
            lower = max(mp.mpf("0"), (beta - center_value) / norm)
            repaired_multipliers = np.maximum(
                np.asarray(proposal.multipliers, dtype=float),
                0.0,
            )
            auxiliary_array = (
                np.zeros((model.hilbert_dim, model.hilbert_dim), dtype=complex)
                if proposal.auxiliary_operator is None
                else np.asarray(proposal.auxiliary_operator, dtype=complex)
            )
            evidence = SelectorLowerBoundEvidence(
                distance_lower_bound=float(lower),
                beta=float(beta),
                center_value=float(center_value),
                separator_norm=float(norm),
                auxiliary_floor=float(auxiliary_floor),
                separator_z=tuple(
                    float(value) for value in np.asarray(proposal.z, dtype=float)
                ),
                constraint_multipliers=tuple(
                    float(value) for value in repaired_multipliers
                ),
                constraint_multiplier_labels=tuple(side.side_id for side in sides),
                auxiliary_operator=auxiliary_array,
                verifier_precision_dps=precision_dps,
                proposal={
                    "solver": proposal.solver,
                    "solver_status": proposal.status,
                    "attempt_index": proposal.attempt_index,
                    "raw_objective": proposal.raw_objective,
                },
            )
            return SelectorLowerVerification(
                distance_lower_bound=float(lower),
                evidence=evidence,
                diagnostics={
                    "distance_lower_bound": float(lower),
                    "beta": float(beta),
                    "center_value": float(center_value),
                    "separator_norm": float(norm),
                    "auxiliary_floor": float(auxiliary_floor),
                    "solver": proposal.solver,
                    "solver_status": proposal.status,
                    "attempt_index": proposal.attempt_index,
                    "raw_objective": proposal.raw_objective,
                },
            )
    except Exception as exc:
        return SelectorLowerVerification(
            distance_lower_bound=None,
            evidence=None,
            diagnostics={"reason": "selector_lower_verification_failed", "error": str(exc)},
        )


def selector_error_bound(distance_upper: float, distance_lower: float) -> tuple[float, float]:
    if (
        not np.isfinite(float(distance_upper))
        or not np.isfinite(float(distance_lower))
        or float(distance_upper) < 0.0
        or float(distance_lower) < 0.0
    ):
        raise ValueError("selector distance bounds must be finite and nonnegative")
    upper_squared = float(distance_upper) ** 2
    lower_squared = float(distance_lower) ** 2
    gap = upper_squared - lower_squared
    if gap < -1.0e-10:
        raise ValueError("verified selector lower bound exceeds primal upper bound")
    gap = max(0.0, gap)
    return gap, math.sqrt(gap)
