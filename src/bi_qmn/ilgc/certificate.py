from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from .solver import (
    DensityProposal,
    ProposalBatch,
    SeparatorProposal,
    solve_radial_proposals,
    solve_radial_separator_proposals,
    solve_selector_lower_bound_proposals,
    solve_selector_proposals,
)
from .types import (
    CalibrationBody,
    CanonicalWitness,
    ConstraintRegistry,
    ExistenceClaim,
    ExistenceStatus,
    FiberSignature,
    ILGCCertificate,
    MeasurementModel,
    SelectorClaim,
    calibration_bodies_hash,
    stable_hash,
)
from .verifier import (
    DEFAULT_PRECISION_DPS,
    selector_error_bound,
    verify_primal_proposal,
    verify_radial_separator,
    verify_selector_candidate,
    verify_selector_lower_bound,
)


DOWNSTREAM_WITNESS_OBJECTS = (
    "canonical_witness",
    "conditional_slices",
    "vertices",
    "edges",
    "atlas",
    "paths",
)


def adjudicate_verified_bounds(
    verified_upper_bound: float | None,
    verified_lower_bound: float | None,
    *,
    nonradial_dual_refuted: bool = False,
) -> tuple[ExistenceStatus, bool]:
    upper = (
        None
        if verified_upper_bound is None or not math.isfinite(verified_upper_bound)
        else float(verified_upper_bound)
    )
    lower = (
        None
        if verified_lower_bound is None or not math.isfinite(verified_lower_bound)
        else float(verified_lower_bound)
    )
    primal_accept = upper is not None and upper <= 1.0
    dual_refute = bool(nonradial_dual_refuted or (lower is not None and lower > 1.0))
    contradiction = bool(primal_accept and dual_refute)
    if contradiction:
        return "unresolved", True
    if primal_accept:
        return "primal_verified", False
    if dual_refute:
        return "dual_refuted", False
    return "unresolved", False


def _attempt_manifest(batch: ProposalBatch) -> list[dict[str, Any]]:
    return [dict(attempt) for attempt in batch.attempts]


def _select_best_dual(verifications):
    radial = [
        item
        for item in verifications
        if item.evidence is not None
        and hasattr(item.evidence, "radial_lower_bound")
    ]
    registry = [
        item
        for item in verifications
        if item.evidence is not None
        and hasattr(item.evidence, "separation_margin")
    ]
    groups = (
        ([item for item in radial if item.refuted], "radial_lower_bound"),
        ([item for item in registry if item.refuted], "separation_margin"),
        (radial, "radial_lower_bound"),
        (registry, "separation_margin"),
    )
    for candidates, attribute in groups:
        if candidates:
            return max(candidates, key=lambda item: getattr(item.evidence, attribute))
    return None


def _compute_existence_claim(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    precision_dps: int,
    proposal_backends: tuple[str, ...] | None,
) -> tuple[ExistenceClaim, dict[str, Any]]:
    primal_batch = solve_radial_proposals(
        model,
        bodies,
        registry,
        backend_labels=proposal_backends,
    )
    primal_verifications = [
        verify_primal_proposal(
            proposal,
            model,
            bodies,
            registry,
            precision_dps=precision_dps,
        )
        for proposal in primal_batch.proposals
        if isinstance(proposal, DensityProposal)
    ]
    valid_primal = [
        verification
        for verification in primal_verifications
        if verification.profile_valid and verification.evidence is not None
    ]
    best_primal = (
        min(valid_primal, key=lambda item: item.evidence.radial_upper_bound)
        if valid_primal
        else None
    )

    dual_batch = ProposalBatch(proposals=(), attempts=())
    dual_verifications = []
    if best_primal is None or not best_primal.accepted:
        dual_batch = solve_radial_separator_proposals(
            model,
            bodies,
            registry,
            backend_labels=proposal_backends,
        )
        dual_verifications = [
            verify_radial_separator(
                proposal,
                model,
                bodies,
                registry,
                precision_dps=precision_dps,
            )
            for proposal in dual_batch.proposals
            if isinstance(proposal, SeparatorProposal)
        ]
    valid_dual = [
        verification for verification in dual_verifications if verification.evidence is not None
    ]
    best_dual = _select_best_dual(valid_dual)

    verified_upper_bound = (
        None if best_primal is None else best_primal.evidence.radial_upper_bound
    )
    verified_lower_bound = (
        None
        if best_dual is None
        else getattr(best_dual.evidence, "radial_lower_bound", None)
    )
    decision_status, contradiction = adjudicate_verified_bounds(
        verified_upper_bound,
        verified_lower_bound,
        nonradial_dual_refuted=bool(
            best_dual is not None
            and best_dual.refuted
            and verified_lower_bound is None
        ),
    )
    diagnostics = {
        "primal_proposals": len(primal_batch.proposals),
        "primal_verified_candidates": len(valid_primal),
        "dual_proposals": len(dual_batch.proposals),
        "dual_verified_candidates": len(valid_dual),
        "verifier_contradiction": contradiction,
        "primal_verifications": [
            verification.diagnostics for verification in primal_verifications
        ],
        "dual_verifications": [
            verification.diagnostics for verification in dual_verifications
        ],
    }

    reason = (
        "primal_dual_verifier_contradiction"
        if contradiction
        else "verified_primal_membership"
        if decision_status == "primal_verified"
        else "verified_dual_separation"
        if decision_status == "dual_refuted"
        else "no_verified_decision_bound"
    )
    claim = ExistenceClaim(
        semantic_status=decision_status,
        evaluation_status="evaluated",
        verified_upper_bound=verified_upper_bound,
        verified_lower_bound=verified_lower_bound,
        primal_evidence=(None if best_primal is None else best_primal.evidence),
        dual_evidence=(None if best_dual is None else best_dual.evidence),
        diagnostics={**diagnostics, "reason": reason},
        decision_reason=reason,
    )

    attempts = {
        "radial_primal": _attempt_manifest(primal_batch),
        "radial_dual": _attempt_manifest(dual_batch),
    }
    return claim, attempts


def _compute_selector_claim(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    *,
    selector_error_budget: float,
    precision_dps: int,
    certificate_instance_hash: str,
    proposal_backends: tuple[str, ...] | None,
    selector_candidate_body_guard: float,
    selector_kernel_reduction: bool,
) -> tuple[SelectorClaim, dict[str, Any]]:
    candidate_batch = solve_selector_proposals(
        model,
        bodies,
        registry,
        backend_labels=proposal_backends,
        body_guard=selector_candidate_body_guard,
        use_kernel_reduction=selector_kernel_reduction,
    )
    candidate_verifications = [
        verify_selector_candidate(
            proposal,
            model,
            bodies,
            registry,
            precision_dps=precision_dps,
        )
        for proposal in candidate_batch.proposals
        if isinstance(proposal, DensityProposal)
    ]
    valid_candidates = [
        verification
        for verification in candidate_verifications
        if verification.density is not None
        and verification.distance_upper_bound is not None
    ]
    best_candidate = (
        min(valid_candidates, key=lambda item: item.distance_upper_bound)
        if valid_candidates
        else None
    )

    lower_batch = solve_selector_lower_bound_proposals(
        model,
        bodies,
        registry,
        backend_labels=proposal_backends,
    )
    lower_verifications = [
        verify_selector_lower_bound(
            proposal,
            model,
            bodies,
            registry,
            precision_dps=precision_dps,
        )
        for proposal in lower_batch.proposals
        if isinstance(proposal, SeparatorProposal)
    ]
    valid_lowers = [
        verification
        for verification in lower_verifications
        if verification.distance_lower_bound is not None
        and verification.evidence is not None
    ]
    best_lower = (
        max(valid_lowers, key=lambda item: item.distance_lower_bound)
        if valid_lowers
        else None
    )

    diagnostics: dict[str, Any] = {
        "selector_error_budget": float(selector_error_budget),
        "selector_candidate_body_guard": float(selector_candidate_body_guard),
        "candidate_proposals": len(candidate_batch.proposals),
        "candidate_verified": len(valid_candidates),
        "lower_bound_proposals": len(lower_batch.proposals),
        "lower_bounds_verified": len(valid_lowers),
        "candidate_verifications": [
            verification.diagnostics for verification in candidate_verifications
        ],
        "lower_bound_verifications": [
            verification.diagnostics for verification in lower_verifications
        ],
    }
    if best_candidate is None or best_lower is None:
        claim = SelectorClaim(
            semantic_status="unresolved",
            feasible_evidence=(
                None if best_candidate is None else best_candidate.radial_evidence
            ),
            lower_bound_evidence=(None if best_lower is None else best_lower.evidence),
            diagnostics={**diagnostics, "reason": "selector_bound_missing"},
        )
    else:
        try:
            gap, error_bound = selector_error_bound(
                best_candidate.distance_upper_bound,
                best_lower.distance_lower_bound,
            )
        except ValueError as exc:
            claim = SelectorClaim(
                semantic_status="unresolved",
                feasible_evidence=best_candidate.radial_evidence,
                lower_bound_evidence=best_lower.evidence,
                diagnostics={
                    **diagnostics,
                    "reason": "selector_bound_contradiction",
                    "error": str(exc),
                },
            )
        else:
            diagnostics.update(
                {
                    "distance_upper_bound": best_candidate.distance_upper_bound,
                    "distance_lower_bound": best_lower.distance_lower_bound,
                    "objective_gap": gap,
                    "representation_error_bound": error_bound,
                }
            )
            if error_bound <= float(selector_error_budget):
                witness = CanonicalWitness(
                    density=best_candidate.density,
                    distance_upper_bound=best_candidate.distance_upper_bound,
                    distance_lower_bound=best_lower.distance_lower_bound,
                    objective_gap=gap,
                    representation_error_bound=error_bound,
                    verifier_precision_dps=precision_dps,
                    certificate_instance_hash=certificate_instance_hash,
                )
                claim = SelectorClaim(
                    semantic_status="verified",
                    witness=witness,
                    feasible_evidence=best_candidate.radial_evidence,
                    lower_bound_evidence=best_lower.evidence,
                    diagnostics=diagnostics,
                )
            else:
                claim = SelectorClaim(
                    semantic_status="unresolved",
                    feasible_evidence=best_candidate.radial_evidence,
                    lower_bound_evidence=best_lower.evidence,
                    diagnostics={**diagnostics, "reason": "selector_error_budget_exceeded"},
                )

    attempts = {
        "selector_primal": _attempt_manifest(candidate_batch),
        "selector_lower_bound": _attempt_manifest(lower_batch),
    }
    return claim, attempts


def _validate_inputs(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    fiber: FiberSignature,
) -> None:
    if fiber.hilbert_dim != model.hilbert_dim:
        raise ValueError("fiber/model Hilbert dimensions differ")
    if fiber.measurement_map_hash != model.effective_map_hash:
        raise ValueError("fiber measurement-map hash does not match model")
    missing = set(model.context_classes) - set(bodies)
    extra = set(bodies) - set(model.context_classes)
    if missing or extra:
        raise ValueError(
            f"calibration body classes mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for context_class in model.context_classes:
        body = bodies[context_class]
        if body.context_class != context_class:
            raise ValueError(
                f"calibration body key {context_class} contains class {body.context_class}"
            )
        if body.stratum != fiber.stratum:
            raise ValueError(
                f"calibration body stratum {body.stratum} does not match fiber {fiber.stratum}"
            )
        body.validate(len(model.indices_for_class(context_class)))
    registry.validate(model.hilbert_dim)


def _certificate_instance_hash(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    fiber: FiberSignature,
) -> str:
    return stable_hash(
        {
            "formulation_version": fiber.formulation_version,
            "fiber_hash": fiber.fiber_hash,
            "measurement_map_hash": fiber.measurement_map_hash,
            "observation_table_hash": model.observation_table_hash,
            "calibration_manifest_hash": calibration_bodies_hash(bodies),
            "constraint_registry_hash": registry.registry_hash,
        }
    )


def compute_certificate(
    model: MeasurementModel,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    fiber: FiberSignature,
    *,
    selector_error_budget: float = 1.0e-4,
    precision_dps: int = DEFAULT_PRECISION_DPS,
    precondition_status: str = "mechanical_input_verified",
    proposal_backends: tuple[str, ...] | None = None,
    selector_candidate_body_guard: float = 1.0e-7,
    selector_precision_dps: int | None = None,
    selector_proposal_backends: tuple[str, ...] | None = None,
    selector_kernel_reduction: bool = True,
) -> ILGCCertificate:
    try:
        _validate_inputs(model, bodies, registry, fiber)
        if (
            not math.isfinite(float(selector_error_budget))
            or float(selector_error_budget) < 0.0
        ):
            raise ValueError("selector_error_budget must be finite and nonnegative")
        if not isinstance(precision_dps, int) or precision_dps <= 0:
            raise ValueError("precision_dps must be a positive integer")
        if precondition_status not in {
            "mechanical_input_verified",
            "scientific_calibration_verified",
        }:
            raise ValueError("unknown precondition status")
        if proposal_backends is not None and not proposal_backends:
            raise ValueError("proposal_backends cannot be empty")
        if selector_proposal_backends is not None and not selector_proposal_backends:
            raise ValueError("selector_proposal_backends cannot be empty")
        if selector_precision_dps is not None and (
            not isinstance(selector_precision_dps, int) or selector_precision_dps <= 0
        ):
            raise ValueError("selector_precision_dps must be a positive integer when provided")
        if not isinstance(selector_kernel_reduction, bool):
            raise ValueError("selector_kernel_reduction must be boolean")
        if not 0.0 <= float(selector_candidate_body_guard) < 1.0:
            raise ValueError("selector_candidate_body_guard must be in [0, 1)")
    except (TypeError, ValueError) as exc:
        unsupported = ExistenceClaim(
            semantic_status="unresolved",
            evaluation_status="unsupported",
            diagnostics={"reason": "input_contract_failed", "error": str(exc)},
            decision_reason="input_contract_failed",
        )
        return ILGCCertificate(
            fiber=fiber,
            calibration_bodies=bodies,
            constraint_registry=registry,
            precondition_status="unsupported",
            observation_table_hash=model.observation_table_hash,
            quantum_claim=unsupported,
            bi_existence_claim=replace(unsupported),
            selector_claim=SelectorClaim(
                semantic_status="unresolved",
                evaluation_status="not_evaluated",
                diagnostics={"reason": "blocked_by_input_contract"},
            ),
            emitted_objects=("fiber_manifest", "input_contract_diagnostic"),
            blocked_objects=DOWNSTREAM_WITNESS_OBJECTS,
            solver_attempts={},
            verifier_diagnostics={"input_contract_error": str(exc)},
        )

    selector_precision = (
        precision_dps if selector_precision_dps is None else selector_precision_dps
    )
    selector_backends = (
        proposal_backends
        if selector_proposal_backends is None
        else selector_proposal_backends
    )
    quantum_registry = ConstraintRegistry(records=())
    instance_hash = _certificate_instance_hash(model, bodies, registry, fiber)
    quantum_claim, quantum_attempts = _compute_existence_claim(
        model,
        bodies,
        quantum_registry,
        precision_dps=precision_dps,
        proposal_backends=proposal_backends,
    )

    if quantum_claim.status != "primal_verified":
        upstream_reason = (
            "quantum_gluing_refuted"
            if quantum_claim.status == "dual_refuted"
            else f"quantum_gluing_{quantum_claim.status}"
        )
        bi_claim = ExistenceClaim(
            semantic_status="unresolved",
            evaluation_status="blocked_upstream",
            diagnostics={"reason": upstream_reason},
            decision_reason=upstream_reason,
        )
        selector_claim = SelectorClaim(
            semantic_status="unresolved",
            evaluation_status="blocked_upstream",
            diagnostics={"reason": upstream_reason},
        )
        bi_attempts: dict[str, Any] = {}
        selector_attempts: dict[str, Any] = {}
    else:
        bi_claim, bi_attempts = _compute_existence_claim(
            model,
            bodies,
            registry,
            precision_dps=precision_dps,
            proposal_backends=proposal_backends,
        )
        if bi_claim.status == "primal_verified":
            selector_claim, selector_attempts = _compute_selector_claim(
                model,
                bodies,
                registry,
                selector_error_budget=selector_error_budget,
                precision_dps=selector_precision,
                certificate_instance_hash=instance_hash,
                proposal_backends=selector_backends,
                selector_candidate_body_guard=selector_candidate_body_guard,
                selector_kernel_reduction=selector_kernel_reduction,
            )
        else:
            selector_claim = SelectorClaim(
                semantic_status="unresolved",
                evaluation_status="blocked_upstream",
                diagnostics={"reason": f"bi_existence_{bi_claim.status}"},
            )
            selector_attempts = {}

    emitted = ["fiber_manifest", "calibration_bodies"]
    if quantum_claim.status == "primal_verified":
        emitted.append("quantum_primal_evidence")
    elif quantum_claim.status == "dual_refuted":
        emitted.append("quantum_dual_evidence")
    if bi_claim.status == "primal_verified":
        emitted.append("bi_existence_primal_evidence")
    elif bi_claim.status == "dual_refuted":
        emitted.append("bi_existence_dual_evidence")
    if selector_claim.status == "verified":
        emitted.extend(("selector_optimality_evidence", "canonical_witness"))
        blocked: tuple[str, ...] = ()
    else:
        blocked = DOWNSTREAM_WITNESS_OBJECTS

    return ILGCCertificate(
        fiber=fiber,
        calibration_bodies=bodies,
        constraint_registry=registry,
        precondition_status=precondition_status,
        observation_table_hash=model.observation_table_hash,
        quantum_claim=quantum_claim,
        bi_existence_claim=bi_claim,
        selector_claim=selector_claim,
        emitted_objects=tuple(emitted),
        blocked_objects=blocked,
        solver_attempts={
            "quantum": quantum_attempts,
            "bi_existence": bi_attempts,
            "selector": selector_attempts,
        },
        verifier_diagnostics={
            "precision_dps": precision_dps,
            "selector_precision_dps": selector_precision,
            "selector_error_budget": selector_error_budget,
            "semantic_rule": "solver_labels_are_diagnostic_only",
            "proposal_backends": (
                None if proposal_backends is None else list(proposal_backends)
            ),
            "selector_proposal_backends": (
                None if selector_backends is None else list(selector_backends)
            ),
            "selector_kernel_reduction": selector_kernel_reduction,
            "selector_candidate_body_guard": selector_candidate_body_guard,
        },
    )
