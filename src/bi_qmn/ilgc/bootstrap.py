from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time
from typing import Any, Literal, Mapping

import numpy as np
from scipy.stats import norm

from bi_qmn.p0.atlas import conditional_slices
from bi_qmn.p0.estimator import FreeMeasurementTableEstimator
from bi_qmn.core.generator import P0Instance
from bi_qmn.core.linalg import fourier_matrix, kron, matrix_projector
from bi_qmn.core.observations import ObservationRecord, sample_shot_batch
from bi_qmn.core.types import MeasurementContext
from .certificate import compute_certificate
from .model import (
    bind_measurement_observation,
    effective_measurement_map_from_p0,
    observation_table_from_p0,
    p0_constraint_registry,
    p0_fiber_signature,
)
from .types import CalibrationBody, ConstraintRegistry, ILGCCertificate, stable_hash


@dataclass(frozen=True)
class EmbeddingMetric:
    covariance: np.ndarray
    inverse_covariance: np.ndarray
    shrinkage: float
    eigenvalue_floor: float
    sample_count: int
    metric_hash: str

    def distance(self, left: np.ndarray, right: np.ndarray) -> float:
        delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        squared = float(delta @ self.inverse_covariance @ delta)
        return math.sqrt(max(0.0, squared))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "method": "clean_construction_covariance_v1",
            "metric_hash": self.metric_hash,
            "sample_count": self.sample_count,
            "shrinkage": self.shrinkage,
            "eigenvalue_floor": self.eigenvalue_floor,
            "covariance": np.asarray(self.covariance, dtype=float).tolist(),
            "inverse_covariance": np.asarray(
                self.inverse_covariance,
                dtype=float,
            ).tolist(),
        }


@dataclass(frozen=True)
class SelectorBootstrapStability:
    claim_scope: Literal["scientific", "mechanical_conformance"]
    B_boot: int
    B_sel: int
    B_bi: int
    bi_emission_rate: float
    bi_emission_lcb: float
    selector_given_bi_rate: float
    selector_given_bi_lcb: float
    selector_emission_rate: float
    selector_emission_lcb: float
    pi_sel_min: float
    B_sel_min: int
    unresolved_rate: float
    max_unresolved_rate: float
    distance_quantile: float
    distance_by_tau: Mapping[int, float | None]
    canonical_density_frobenius_quantile: float | None
    qualified: bool
    qualification_reasons: tuple[str, ...]
    status_counts: Mapping[str, int]
    replicate_records: tuple[Mapping[str, Any], ...]
    embedding_metric: EmbeddingMetric
    base_fiber_hash: str
    wall_clock_seconds: float

    def to_manifest(self) -> dict[str, Any]:
        return {
            "object_type": "SelectorBootstrapStability",
            "claim_scope": self.claim_scope,
            "B_boot": self.B_boot,
            "B_sel": self.B_sel,
            "B_bi": self.B_bi,
            "B_eff": self.B_sel,
            "selector_emission_rate": self.selector_emission_rate,
            "selector_emission_lcb": self.selector_emission_lcb,
            "bi_emission_rate": self.bi_emission_rate,
            "bi_emission_lcb": self.bi_emission_lcb,
            "selector_given_bi_rate": self.selector_given_bi_rate,
            "selector_given_bi_lcb": self.selector_given_bi_lcb,
            "pi_sel_min": self.pi_sel_min,
            "B_sel_min": self.B_sel_min,
            "unresolved_rate": self.unresolved_rate,
            "max_unresolved_rate": self.max_unresolved_rate,
            "distance_quantile": self.distance_quantile,
            "distance_by_tau": {
                str(tau): value for tau, value in sorted(self.distance_by_tau.items())
            },
            "canonical_density_frobenius_quantile": (
                self.canonical_density_frobenius_quantile
            ),
            "qualified": self.qualified,
            "qualification_reasons": list(self.qualification_reasons),
            "status_counts": dict(self.status_counts),
            "replicates": [dict(record) for record in self.replicate_records],
            "embedding_metric": self.embedding_metric.to_manifest(),
            "base_fiber_hash": self.base_fiber_hash,
            "wall_clock_seconds": self.wall_clock_seconds,
            "semantic_fallback": False,
        }


def _conditional_states(density: np.ndarray, K: int) -> list[np.ndarray]:
    F = fourier_matrix(K)
    rho = np.asarray(density, dtype=complex)
    states: list[np.ndarray] = []
    for tau in range(K):
        projector = kron(matrix_projector(F[tau, :]), np.eye(2, dtype=complex))
        block = projector @ rho @ projector
        probability = float(np.real(np.trace(block)))
        reduced = block.reshape(K, 2, K, 2).trace(axis1=0, axis2=2)
        if probability <= 0.0:
            states.append(np.eye(2, dtype=complex) / 2.0)
        else:
            states.append(reduced / probability)
    return states


def chi0_qubit(density: np.ndarray) -> np.ndarray:
    rho = np.asarray(density, dtype=complex)
    paulis = (
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    )
    return np.asarray(
        [float(np.real(np.trace(pauli @ rho))) / math.sqrt(2.0) for pauli in paulis],
        dtype=float,
    )


def build_clean_embedding_metric(
    instances: list[P0Instance],
    *,
    shrinkage: float,
    eigenvalue_floor: float,
) -> EmbeddingMetric:
    if not instances:
        raise ValueError("embedding metric requires construction instances")
    if not 0.0 <= shrinkage <= 1.0 or eigenvalue_floor <= 0.0:
        raise ValueError("invalid embedding metric regularization")
    embeddings = np.asarray(
        [
            chi0_qubit(state)
            for instance in instances
            for state in _conditional_states(
                np.asarray(instance.density.data, dtype=complex),
                instance.K,
            )
        ],
        dtype=float,
    )
    empirical = np.cov(embeddings, rowvar=False, ddof=1)
    isotropic_scale = float(np.trace(empirical) / empirical.shape[0])
    shrunk = (1.0 - shrinkage) * empirical + shrinkage * isotropic_scale * np.eye(
        empirical.shape[0],
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    covariance = (eigenvectors * clipped) @ eigenvectors.T
    inverse = (eigenvectors * (1.0 / clipped)) @ eigenvectors.T
    metric_hash = stable_hash(
        {
            "method": "clean_construction_covariance_v1",
            "shrinkage": shrinkage,
            "eigenvalue_floor": eigenvalue_floor,
            "sample_count": int(embeddings.shape[0]),
            "covariance": covariance.tolist(),
        }
    )
    return EmbeddingMetric(
        covariance=covariance,
        inverse_covariance=inverse,
        shrinkage=float(shrinkage),
        eigenvalue_floor=float(eigenvalue_floor),
        sample_count=int(embeddings.shape[0]),
        metric_hash=metric_hash,
    )


def wilson_lcb(successes: int, total: int, alpha: float) -> float:
    if total <= 0:
        return 0.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("Wilson alpha must be in (0, 1)")
    z = float(norm.ppf(1.0 - alpha))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, float(center - margin))


def validate_selector_bootstrap_scope(
    bodies: Mapping[str, CalibrationBody],
    claim_scope: Literal["scientific", "mechanical_conformance"],
) -> None:
    if claim_scope not in {"scientific", "mechanical_conformance"}:
        raise ValueError("invalid selector bootstrap claim scope")
    if claim_scope == "scientific" and any(
        body.role != "split_conformal_calibrated_body" for body in bodies.values()
    ):
        raise ValueError(
            "scientific selector bootstrap requires qualified split-conformal bodies"
        )


def summarize_selector_bootstrap(
    replicate_records: list[Mapping[str, Any]],
    distance_samples: Mapping[int, list[float]],
    *,
    alpha_wit: float,
    emission_lcb_alpha: float,
    pi_sel_min: float,
    B_sel_min: int,
    max_unresolved_rate: float,
    embedding_metric: EmbeddingMetric,
    base_fiber_hash: str,
    wall_clock_seconds: float,
    claim_scope: Literal["scientific", "mechanical_conformance"],
) -> SelectorBootstrapStability:
    if claim_scope not in {"scientific", "mechanical_conformance"}:
        raise ValueError("invalid selector bootstrap claim scope")
    if not 0.0 < alpha_wit < 1.0:
        raise ValueError("bootstrap alpha_wit must be in (0, 1)")
    if not 0.0 <= pi_sel_min <= 1.0 or B_sel_min < 0:
        raise ValueError("invalid selector-emission qualification gate")
    if not 0.0 <= max_unresolved_rate <= 1.0:
        raise ValueError("max_unresolved_rate must be in [0, 1]")
    B_boot = len(replicate_records)
    B_sel = sum(
        1 for record in replicate_records if record["selector_status"] == "verified"
    )
    B_bi = sum(
        1 for record in replicate_records if record["bi_status"] == "primal_verified"
    )
    emission_rate = B_sel / max(1, B_boot)
    emission_lcb = wilson_lcb(B_sel, B_boot, emission_lcb_alpha)
    bi_emission_rate = B_bi / max(1, B_boot)
    bi_emission_lcb = wilson_lcb(B_bi, B_boot, emission_lcb_alpha)
    selector_given_bi_rate = B_sel / max(1, B_bi)
    selector_given_bi_lcb = wilson_lcb(B_sel, B_bi, emission_lcb_alpha)
    unresolved = sum(
        1
        for record in replicate_records
        if "unresolved"
        in {
            str(record["quantum_status"]),
            str(record["bi_status"]),
            str(record["selector_status"]),
        }
        or "unsupported"
        in {
            str(record["quantum_status"]),
            str(record["bi_status"]),
        }
    )
    unresolved_rate = unresolved / max(1, B_boot)
    quantile = 1.0 - alpha_wit
    distances: dict[int, float | None] = {}
    reasons: list[str] = []
    for tau, values in sorted(distance_samples.items()):
        if len(values) != B_sel:
            reasons.append(f"distance_count_mismatch_tau_{tau}")
            distances[tau] = None
        elif not values:
            reasons.append(f"distance_unavailable_tau_{tau}")
            distances[tau] = None
        else:
            distances[tau] = float(
                np.quantile(np.asarray(values, dtype=float), quantile, method="higher")
            )
    density_distances = [
        float(record["canonical_density_frobenius_distance"])
        for record in replicate_records
        if record.get("canonical_density_frobenius_distance") is not None
    ]
    density_quantile = (
        None
        if not density_distances
        else float(
            np.quantile(
                np.asarray(density_distances, dtype=float),
                quantile,
                method="higher",
            )
        )
    )
    if len(density_distances) != B_sel:
        reasons.append("canonical_density_distance_count_mismatch")
    if B_sel < B_sel_min:
        reasons.append("selector_effective_count_below_minimum")
    if emission_lcb < pi_sel_min:
        reasons.append("selector_emission_lcb_below_minimum")
    if unresolved_rate > max_unresolved_rate:
        reasons.append("unresolved_rate_above_maximum")
    status_counts: Counter[str] = Counter()
    for record in replicate_records:
        status_counts[f"Q:{record['quantum_status']}"] += 1
        status_counts[f"BI:{record['bi_status']}"] += 1
        status_counts[f"SEL:{record['selector_status']}"] += 1
    return SelectorBootstrapStability(
        claim_scope=claim_scope,
        B_boot=B_boot,
        B_sel=B_sel,
        B_bi=B_bi,
        bi_emission_rate=bi_emission_rate,
        bi_emission_lcb=bi_emission_lcb,
        selector_given_bi_rate=selector_given_bi_rate,
        selector_given_bi_lcb=selector_given_bi_lcb,
        selector_emission_rate=emission_rate,
        selector_emission_lcb=emission_lcb,
        pi_sel_min=float(pi_sel_min),
        B_sel_min=int(B_sel_min),
        unresolved_rate=unresolved_rate,
        max_unresolved_rate=float(max_unresolved_rate),
        distance_quantile=quantile,
        distance_by_tau=distances,
        canonical_density_frobenius_quantile=density_quantile,
        qualified=not reasons,
        qualification_reasons=tuple(reasons),
        status_counts=dict(status_counts),
        replicate_records=tuple(dict(record) for record in replicate_records),
        embedding_metric=embedding_metric,
        base_fiber_hash=base_fiber_hash,
        wall_clock_seconds=float(wall_clock_seconds),
    )


def _qhat_for_record(
    record: ObservationRecord,
    estimator: FreeMeasurementTableEstimator,
    contexts: list[MeasurementContext],
    *,
    bootstrap_replicate: int | None = None,
):
    metadata = {
        "eta": record.eta,
        "regime": record.regime,
        "instance_index": record.instance_index,
        "replicate": record.replicate,
        "record_index": record.metadata["record_index"],
        "estimated": True,
    }
    if bootstrap_replicate is not None:
        metadata["bootstrap_replicate"] = bootstrap_replicate
    return estimator.predict_table(record.support.table, contexts, metadata=metadata)


def _replicate_manifest(
    index: int,
    certificate: ILGCCertificate,
    distances: Mapping[int, float] | None,
    canonical_density_frobenius_distance: float | None = None,
) -> dict[str, Any]:
    return {
        "replicate": index,
        "fiber_hash": certificate.fiber.fiber_hash,
        "observation_table_hash": certificate.observation_table_hash,
        "quantum_status": certificate.quantum_claim.status,
        "bi_status": certificate.bi_existence_claim.status,
        "selector_status": certificate.selector_claim.status,
        "quantum_verified_upper": certificate.quantum_claim.verified_upper_bound,
        "quantum_verified_lower": certificate.quantum_claim.verified_lower_bound,
        "bi_verified_upper": certificate.bi_existence_claim.verified_upper_bound,
        "bi_verified_lower": certificate.bi_existence_claim.verified_lower_bound,
        "selector_error_bound": (
            None
            if certificate.selector_claim.witness is None
            else certificate.selector_claim.witness.representation_error_bound
        ),
        "distance_by_tau": (
            None
            if distances is None
            else {str(tau): value for tau, value in sorted(distances.items())}
        ),
        "canonical_density_frobenius_distance": (
            canonical_density_frobenius_distance
        ),
        "semantic_fallback": False,
    }


def estimate_selector_bootstrap(
    record: ObservationRecord,
    estimator: FreeMeasurementTableEstimator,
    contexts: list[MeasurementContext],
    instance: P0Instance,
    bodies: dict[str, CalibrationBody],
    registry: ConstraintRegistry,
    embedding_metric: EmbeddingMetric,
    *,
    selector_error_budget: float,
    precision_dps: int,
    B_boot: int,
    seed: int,
    alpha_wit: float,
    emission_lcb_alpha: float,
    pi_sel_min: float,
    B_sel_min: int,
    max_unresolved_rate: float,
    claim_scope: Literal["scientific", "mechanical_conformance"] = (
        "mechanical_conformance"
    ),
) -> tuple[ILGCCertificate, SelectorBootstrapStability]:
    if B_boot <= 0:
        raise ValueError("selector bootstrap requires B_boot > 0")
    validate_selector_bootstrap_scope(bodies, claim_scope)
    start = time.perf_counter()
    base_table = _qhat_for_record(record, estimator, contexts)
    frozen_map = effective_measurement_map_from_p0(
        contexts,
        K=instance.K,
        eta=float(record.eta),
    )
    base_observation = observation_table_from_p0(
        base_table,
        frozen_map,
        stratum=record.regime,
        estimator_checkpoint_hash=estimator.checkpoint_hash,
        support_query_hash=stable_hash(
            {"record_index": record.metadata["record_index"], "role": "base"}
        ),
    )
    base_model = bind_measurement_observation(
        frozen_map,
        base_observation,
        binding_mode="frozen_map_estimator_observation",
    )
    base_fiber = p0_fiber_signature(
        instance,
        base_model,
        checkpoint_hash=estimator.checkpoint_hash,
        stratum=record.regime,
        observation_id="p0_estimator_support_query_v1",
    )
    base_certificate = compute_certificate(
        base_model,
        bodies,
        registry,
        base_fiber,
        selector_error_budget=selector_error_budget,
        precision_dps=precision_dps,
        precondition_status=(
            "scientific_calibration_verified"
            if claim_scope == "scientific"
            else "mechanical_input_verified"
        ),
    )
    base_witness = base_certificate.selector_claim.witness
    if base_witness is None:
        raise ValueError("selector bootstrap requires a verified canonical base witness")
    base_slices = conditional_slices(base_witness, base_table, K=instance.K)
    base_embeddings = {
        tau: chi0_qubit(np.asarray(certified_slice.rho.data, dtype=complex))
        for tau, certified_slice in base_slices.items()
    }
    base_density = np.asarray(base_witness.density.data, dtype=complex)

    rng = np.random.default_rng(seed)
    distance_samples: dict[int, list[float]] = {
        tau: [] for tau in sorted(base_embeddings)
    }
    replicate_records: list[dict[str, Any]] = []
    for bootstrap_index in range(B_boot):
        support = sample_shot_batch(
            record.support.table,
            list(record.support.table.contexts),
            int(record.support.table.metadata["n_shots"]),
            rng,
            shot_start=(
                int(record.metadata["record_index"]) * 50_000_000
                + bootstrap_index * 100_000
            ),
            role="ilgc_bootstrap_support",
        )
        bootstrap_record = ObservationRecord(
            split=record.split,
            regime=record.regime,
            instance_index=record.instance_index,
            replicate=record.replicate,
            eta=record.eta,
            exact_table=record.exact_table,
            support=support,
            query=record.query,
            metadata=dict(record.metadata),
        )
        table = _qhat_for_record(
            bootstrap_record,
            estimator,
            contexts,
            bootstrap_replicate=bootstrap_index,
        )
        observation = observation_table_from_p0(
            table,
            frozen_map,
            stratum=record.regime,
            estimator_checkpoint_hash=estimator.checkpoint_hash,
            support_query_hash=stable_hash(
                {
                    "record_index": record.metadata["record_index"],
                    "bootstrap_replicate": bootstrap_index,
                }
            ),
        )
        model = bind_measurement_observation(
            frozen_map,
            observation,
            binding_mode="frozen_map_estimator_observation",
        )
        fiber = p0_fiber_signature(
            instance,
            model,
            checkpoint_hash=estimator.checkpoint_hash,
            stratum=record.regime,
            observation_id="p0_estimator_bootstrap_support_v1",
        )
        certificate = compute_certificate(
            model,
            bodies,
            registry,
            fiber,
            selector_error_budget=selector_error_budget,
            precision_dps=precision_dps,
            precondition_status=(
                "scientific_calibration_verified"
                if claim_scope == "scientific"
                else "mechanical_input_verified"
            ),
        )
        witness = certificate.selector_claim.witness
        replicate_distances: dict[int, float] | None = None
        density_distance: float | None = None
        if witness is not None:
            density_distance = float(
                np.linalg.norm(
                    np.asarray(witness.density.data, dtype=complex) - base_density,
                    ord="fro",
                )
            )
            slices = conditional_slices(witness, table, K=instance.K)
            replicate_distances = {}
            for tau, certified_slice in slices.items():
                embedding = chi0_qubit(
                    np.asarray(certified_slice.rho.data, dtype=complex)
                )
                distance = embedding_metric.distance(embedding, base_embeddings[tau])
                distance_samples[tau].append(distance)
                replicate_distances[tau] = distance
        replicate_records.append(
            _replicate_manifest(
                bootstrap_index,
                certificate,
                replicate_distances,
                density_distance,
            )
        )

    stability = summarize_selector_bootstrap(
        replicate_records,
        distance_samples,
        alpha_wit=alpha_wit,
        emission_lcb_alpha=emission_lcb_alpha,
        pi_sel_min=pi_sel_min,
        B_sel_min=B_sel_min,
        max_unresolved_rate=max_unresolved_rate,
        embedding_metric=embedding_metric,
        base_fiber_hash=base_fiber.fiber_hash,
        wall_clock_seconds=time.perf_counter() - start,
        claim_scope=claim_scope,
    )
    return base_certificate, stability


def p0_registry_for_bootstrap(
    instance: P0Instance,
    generator_thresholds: Mapping[str, Any],
    *,
    calibration_hash: str,
) -> ConstraintRegistry:
    return p0_constraint_registry(
        instance,
        float(generator_thresholds["theta_con"]),
        calibration_hash=calibration_hash,
    )
