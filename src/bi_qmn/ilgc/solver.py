from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
import warnings

import cvxpy as cp
import numpy as np

from .types import CalibrationBody, ConstraintRegistry, MeasurementModel


@dataclass(frozen=True)
class DensityProposal:
    density: np.ndarray
    raw_objective: float
    solver: str
    status: str
    attempt_index: int


@dataclass(frozen=True)
class SeparatorProposal:
    z: np.ndarray
    multipliers: np.ndarray
    raw_objective: float
    solver: str
    status: str
    attempt_index: int
    auxiliary_operator: np.ndarray | None = None


@dataclass(frozen=True)
class ProposalBatch:
    proposals: tuple[DensityProposal | SeparatorProposal, ...]
    attempts: tuple[dict[str, Any], ...]


def _attempt_specs(
    backend_labels: tuple[str, ...] | None = None,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    specs = (
        ("CLARABEL", cp.CLARABEL, {}),
        (
            "CLARABEL_refine",
            cp.CLARABEL,
            {
                "max_iter": 1000,
                "tol_gap_abs": 1.0e-9,
                "tol_gap_rel": 1.0e-9,
                "tol_feas": 1.0e-9,
                "tol_infeas_abs": 1.0e-9,
                "tol_infeas_rel": 1.0e-9,
            },
        ),
        ("SCS", cp.SCS, {"eps": 1.0e-7, "max_iters": 30000}),
    )
    if backend_labels is None:
        return specs
    known = {label for label, _, _ in specs}
    unknown = set(backend_labels) - known
    if unknown:
        raise ValueError(f"unknown ILGC proposal backends {sorted(unknown)}")
    return tuple(spec for spec in specs if spec[0] in backend_labels)


def _solve(
    problem: cp.Problem,
    snapshot,
    *,
    backend_labels: tuple[str, ...] | None = None,
) -> ProposalBatch:
    proposals: list[DensityProposal | SeparatorProposal] = []
    attempts: list[dict[str, Any]] = []
    for attempt_index, (label, solver, options) in enumerate(_attempt_specs(backend_labels)):
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = problem.solve(
                    solver=solver,
                    verbose=False,
                    warm_start=True,
                    **options,
                )
            finite_value = (
                None
                if value is None or not np.isfinite(float(value))
                else float(value)
            )
            attempt: dict[str, Any] = {
                "solver": label,
                "status": str(problem.status),
                "raw_objective": finite_value,
                "warning_count": len(caught),
                "wall_clock_seconds": time.perf_counter() - started,
            }
            if caught:
                attempt["warnings"] = [
                    {
                        "category": warning.category.__name__,
                        "message": str(warning.message),
                    }
                    for warning in caught
                ]
            proposal = snapshot(label, str(problem.status), attempt_index, finite_value)
            if proposal is not None:
                proposals.append(proposal)
            attempts.append(attempt)
        except Exception as exc:  # pragma: no cover - backend availability is environment-specific
            attempts.append(
                {
                    "solver": label,
                    "status": "exception",
                    "error": str(exc),
                    "warning_count": 0,
                    "wall_clock_seconds": time.perf_counter() - started,
                }
            )
    return ProposalBatch(proposals=tuple(proposals), attempts=tuple(attempts))


def _predictions(rho: cp.Variable, model: MeasurementModel) -> cp.Expression:
    values = [
        cp.real(cp.trace(np.asarray(entry.effect.data, dtype=complex) @ rho))
        for entry in model.entries
    ]
    return cp.hstack(values)


def _calibration_constraints(
    predicted: cp.Expression,
    radial_scale: cp.Expression,
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    *,
    relative_guard: float = 1.0e-7,
) -> list[cp.Constraint]:
    targets = model.targets()
    constraints: list[cp.Constraint] = []
    for context_class in model.context_classes:
        indices = list(model.indices_for_class(context_class))
        body = bodies[context_class]
        residual = predicted[indices] - targets[indices]
        proposal_radius = float(body.radius) * (1.0 - float(relative_guard))
        constraints.append(
            cp.norm(np.asarray(body.whitening, dtype=float) @ residual, 2)
            <= radial_scale * proposal_radius
        )
    return constraints


def _registry_constraints(
    rho: cp.Variable,
    registry: ConstraintRegistry,
    *,
    skip_constraint_ids: frozenset[str] = frozenset(),
) -> list[cp.Constraint]:
    constraints: list[cp.Constraint] = []
    for side in registry.canonical_sides():
        if side.constraint_id in skip_constraint_ids:
            continue
        bound = float(side.upper_bound)
        guard = min(1.0e-8, 0.1 * abs(bound)) if bound != 0.0 else 0.0
        # Tight physical constraints otherwise differ from the radial objective by
        # six or more orders of magnitude and are poorly resolved by conic backends.
        constraint_scale = max(abs(bound), 1.0e-8)
        constraints.append(
            cp.real(
                cp.trace(
                    np.asarray(side.operator.data, dtype=complex) @ rho
                )
            )
            / constraint_scale
            <= (bound - guard) / constraint_scale
        )
    return constraints


def _p0_kernel_reduction(
    registry: ConstraintRegistry,
    dim: int,
) -> tuple[np.ndarray | None, frozenset[str]]:
    """Use the known P0 stationarity kernel for numerically tight H_tot^2 bounds."""
    for record in registry.records:
        if (
            record.provenance
            not in {
                "P0Instance.H_tot_squared_over_h_scale_squared",
                "P1Instance.H_tot_squared_over_h_scale_squared",
            }
            or record.lower_bound is not None
            or record.upper_bound is None
            or float(record.upper_bound) > 1.0e-5
        ):
            continue
        operator = np.asarray(record.operator.data, dtype=complex)
        eigenvalues, eigenvectors = np.linalg.eigh(operator)
        if np.min(eigenvalues) < -1.0e-10:
            continue
        kernel = eigenvectors[:, eigenvalues <= 1.0e-12]
        if 0 < kernel.shape[1] < dim:
            return kernel, frozenset({record.constraint_id})
    return None, frozenset()


def solve_radial_proposals(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    backend_labels: tuple[str, ...] | None = None,
) -> ProposalBatch:
    dim = model.hilbert_dim
    kernel, skipped_constraints = _p0_kernel_reduction(registry, dim)
    rho_variable = cp.Variable(
        (dim, dim) if kernel is None else (kernel.shape[1], kernel.shape[1]),
        hermitian=True,
    )
    rho = rho_variable if kernel is None else kernel @ rho_variable @ kernel.conj().T
    radial_scale = cp.Variable(nonneg=True)
    predicted = _predictions(rho, model)
    constraints: list[cp.Constraint] = [
        rho_variable >> 0,
        cp.real(cp.trace(rho)) == 1.0,
        *_calibration_constraints(predicted, radial_scale, model, bodies),
        *_registry_constraints(
            rho,
            registry,
            skip_constraint_ids=skipped_constraints,
        ),
    ]
    problem = cp.Problem(cp.Minimize(radial_scale), constraints)

    def snapshot(label: str, status: str, attempt_index: int, value: Any):
        if rho_variable.value is None or value is None:
            return None
        return DensityProposal(
            density=np.asarray(rho.value, dtype=complex).copy(),
            raw_objective=float(value),
            solver=label,
            status=status,
            attempt_index=attempt_index,
        )

    return _solve(problem, snapshot, backend_labels=backend_labels)


def _dual_support_expression(
    z: cp.Variable,
    model: MeasurementModel,
    registry: ConstraintRegistry,
    multipliers: cp.Variable | None,
) -> cp.Expression:
    dim = model.hilbert_dim
    B = cp.Constant(np.zeros((dim, dim), dtype=complex))
    for index, entry in enumerate(model.entries):
        B = B + z[index] * np.asarray(entry.effect.data, dtype=complex)
    if multipliers is not None:
        for index, side in enumerate(registry.canonical_sides()):
            B = B - multipliers[index] * np.asarray(side.operator.data, dtype=complex)
    return cp.hermitian_wrap(B)


def _calibration_dual_support(
    z: cp.Variable,
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
) -> cp.Expression:
    terms: list[cp.Expression] = []
    for context_class in model.context_classes:
        indices = list(model.indices_for_class(context_class))
        body = bodies[context_class]
        dual_map = np.linalg.inv(np.asarray(body.whitening, dtype=float).T)
        terms.append(float(body.radius) * cp.norm(dual_map @ z[indices], 2))
    return cp.sum(terms)


def solve_radial_separator_proposals(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    backend_labels: tuple[str, ...] | None = None,
) -> ProposalBatch:
    n_entries = len(model.entries)
    z = cp.Variable(n_entries)
    sides = registry.canonical_sides()
    multipliers = cp.Variable(len(sides), nonneg=True) if sides else None
    t = cp.Variable()
    B = _dual_support_expression(z, model, registry, multipliers)
    calibration_support = _calibration_dual_support(z, model, bodies)
    multiplier_bound = (
        0.0
        if multipliers is None
        else cp.sum(
            cp.multiply(
                np.asarray([side.upper_bound for side in sides], dtype=float),
                multipliers,
            )
        )
    )
    targets = model.targets()
    constraints: list[cp.Constraint] = [
        calibration_support <= 1.0,
        cp.hermitian_wrap(t * np.eye(model.hilbert_dim) - B) >> 0,
    ]
    if multipliers is not None:
        constraints.append(cp.sum(multipliers) <= 1.0)
    objective = cp.Maximize(targets @ z - t - multiplier_bound)
    problem = cp.Problem(objective, constraints)

    def snapshot(label: str, status: str, attempt_index: int, value: Any):
        if z.value is None or value is None:
            return None
        mu = (
            np.zeros(0, dtype=float)
            if multipliers is None or multipliers.value is None
            else np.asarray(multipliers.value, dtype=float).copy()
        )
        return SeparatorProposal(
            z=np.asarray(z.value, dtype=float).copy(),
            multipliers=mu,
            raw_objective=float(value),
            solver=label,
            status=status,
            attempt_index=attempt_index,
            auxiliary_operator=None,
        )

    return _solve(problem, snapshot, backend_labels=backend_labels)


def solve_selector_proposals(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    backend_labels: tuple[str, ...] | None = None,
    body_guard: float = 1.0e-7,
    use_kernel_reduction: bool = True,
) -> ProposalBatch:
    dim = model.hilbert_dim
    kernel, skipped_constraints = (
        _p0_kernel_reduction(registry, dim)
        if use_kernel_reduction
        and any(record.provenance.startswith("P1Instance.") for record in registry.records)
        else (None, frozenset())
    )
    rho_variable = cp.Variable(
        (dim, dim) if kernel is None else (kernel.shape[1], kernel.shape[1]),
        hermitian=True,
    )
    rho = rho_variable if kernel is None else kernel @ rho_variable @ kernel.conj().T
    predicted = _predictions(rho, model)
    constraints: list[cp.Constraint] = [
        rho_variable >> 0,
        cp.real(cp.trace(rho)) == 1.0,
        *_calibration_constraints(
            predicted,
            cp.Constant(1.0),
            model,
            bodies,
            relative_guard=body_guard,
        ),
        *_registry_constraints(
            rho,
            registry,
            skip_constraint_ids=skipped_constraints,
        ),
    ]
    rho0 = np.eye(dim, dtype=complex) / dim
    delta = rho - rho0
    objective = cp.Minimize(
        cp.sum_squares(cp.real(delta)) + cp.sum_squares(cp.imag(delta))
    )
    problem = cp.Problem(objective, constraints)

    def snapshot(label: str, status: str, attempt_index: int, value: Any):
        if rho_variable.value is None or value is None:
            return None
        return DensityProposal(
            density=np.asarray(rho.value, dtype=complex).copy(),
            raw_objective=float(value),
            solver=label,
            status=status,
            attempt_index=attempt_index,
        )

    return _solve(problem, snapshot, backend_labels=backend_labels)


def solve_selector_lower_bound_proposals(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    backend_labels: tuple[str, ...] | None = None,
) -> ProposalBatch:
    n_entries = len(model.entries)
    z = cp.Variable(n_entries)
    sides = registry.canonical_sides()
    multipliers = cp.Variable(len(sides), nonneg=True) if sides else None
    B_measure = _dual_support_expression(z, model, registry, multipliers)
    auxiliary = cp.Variable(
        (model.hilbert_dim, model.hilbert_dim),
        hermitian=True,
    )
    auxiliary_floor = cp.Variable()
    B = cp.hermitian_wrap(B_measure + auxiliary)
    calibration_support = _calibration_dual_support(z, model, bodies)
    multiplier_bound = (
        0.0
        if multipliers is None
        else cp.sum(
            cp.multiply(
                np.asarray([side.upper_bound for side in sides], dtype=float),
                multipliers,
            )
        )
    )
    rho0 = np.eye(model.hilbert_dim, dtype=complex) / model.hilbert_dim
    center_value = cp.real(cp.trace(B @ rho0))
    constraints = [
        cp.norm(B, "fro") <= 1.0,
        cp.hermitian_wrap(auxiliary - auxiliary_floor * np.eye(model.hilbert_dim)) >> 0,
    ]
    objective = cp.Maximize(
        model.targets() @ z
        - calibration_support
        - multiplier_bound
        + auxiliary_floor
        - center_value
    )
    problem = cp.Problem(objective, constraints)

    def snapshot(label: str, status: str, attempt_index: int, value: Any):
        if z.value is None or auxiliary.value is None or value is None:
            return None
        mu = (
            np.zeros(0, dtype=float)
            if multipliers is None or multipliers.value is None
            else np.asarray(multipliers.value, dtype=float).copy()
        )
        return SeparatorProposal(
            z=np.asarray(z.value, dtype=float).copy(),
            multipliers=mu,
            raw_objective=float(value),
            solver=label,
            status=status,
            attempt_index=attempt_index,
            auxiliary_operator=np.asarray(auxiliary.value, dtype=complex).copy(),
        )

    return _solve(problem, snapshot, backend_labels=backend_labels)
