"""Indexed local-to-global certificate formulation for BI-QMN."""

from .atlas import ILGCAtlasRoute, build_ilgc_atlas, build_ilgc_atlas_from_witness
from .atlas_calibration import (
    AtlasCalibrationSample,
    AtlasThresholdBank,
    AtlasThresholdCell,
    fit_atlas_threshold_bank,
)
from .atlas_calibration_runner import run_p0_ilgc_atlas_calibration
from .bootstrap import (
    EmbeddingMetric,
    SelectorBootstrapStability,
    build_clean_embedding_metric,
    estimate_selector_bootstrap,
    summarize_selector_bootstrap,
    validate_selector_bootstrap_scope,
    wilson_lcb,
)
from .calibration import (
    CalibrationCell,
    ILGCCalibrationBank,
    fit_calibration_bank,
    prepare_p0_ilgc_calibration,
    run_p0_ilgc_calibration,
)
from .certificate import adjudicate_verified_bounds, compute_certificate
from .scientific import run_p0_ilgc_scientific_evaluation
from .model import (
    bind_measurement_observation,
    calibration_bodies_from_radii,
    effective_measurement_map_from_p0,
    measurement_model_from_p0,
    observation_table_from_p0,
    p0_constraint_registry,
    p0_fiber_signature,
)
from .types import (
    CalibrationBody,
    CanonicalWitness,
    ConstraintInequality,
    ConstraintRegistrySeparationEvidence,
    ConstraintRegistry,
    DualSeparationEvidence,
    ExistenceClaim,
    ExistencePrimalEvidence,
    FiberSignature,
    ILGCCertificate,
    MeasurementModel,
    SelectorClaim,
    SelectorLowerBoundEvidence,
)
from .verifier import sanitize_density_proposal, verify_primal_evidence

__all__ = [
    "CalibrationBody",
    "CalibrationCell",
    "CanonicalWitness",
    "ConstraintInequality",
    "ConstraintRegistrySeparationEvidence",
    "ConstraintRegistry",
    "DualSeparationEvidence",
    "ExistenceClaim",
    "ExistencePrimalEvidence",
    "FiberSignature",
    "EmbeddingMetric",
    "ILGCAtlasRoute",
    "AtlasCalibrationSample",
    "AtlasThresholdBank",
    "AtlasThresholdCell",
    "ILGCCalibrationBank",
    "ILGCCertificate",
    "MeasurementModel",
    "SelectorClaim",
    "SelectorBootstrapStability",
    "SelectorLowerBoundEvidence",
    "adjudicate_verified_bounds",
    "build_clean_embedding_metric",
    "build_ilgc_atlas",
    "build_ilgc_atlas_from_witness",
    "fit_atlas_threshold_bank",
    "run_p0_ilgc_atlas_calibration",
    "bind_measurement_observation",
    "calibration_bodies_from_radii",
    "compute_certificate",
    "effective_measurement_map_from_p0",
    "estimate_selector_bootstrap",
    "fit_calibration_bank",
    "measurement_model_from_p0",
    "observation_table_from_p0",
    "p0_constraint_registry",
    "p0_fiber_signature",
    "prepare_p0_ilgc_calibration",
    "run_p0_ilgc_calibration",
    "run_p0_ilgc_scientific_evaluation",
    "sanitize_density_proposal",
    "summarize_selector_bootstrap",
    "validate_selector_bootstrap_scope",
    "wilson_lcb",
    "verify_primal_evidence",
]
