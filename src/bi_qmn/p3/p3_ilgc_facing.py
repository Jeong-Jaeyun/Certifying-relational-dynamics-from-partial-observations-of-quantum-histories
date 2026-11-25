"""ILGC-facing development metrics computed without any SDP.

ILGC certifies the full 20-context table (input, compositional, operator) split into the
``diag`` (time-basis D) and ``coh`` (energy-basis F) context classes. Each class body is a ball
around the estimator table whose radius is the split-conformal order statistic of the per-block
maximum ``||q_exact - q_hat||_2`` (identity whitening), and a body is scientifically qualified only
if ``radius / sqrt(dim) <= ceiling`` (``p1_ilgc_calibration._fit_calibration_bank`` and
``_anti_vacuity_audit``). These functions compute exactly those quantities on development banks,
where exact tables are available, so model development can target what certification will see.

``SupportPassthroughPredictor`` reproduces the P1 kernel estimators' rule for observed contexts
(``p1_estimator._smoothed(support)``) and uses the wrapped model only for unobserved contexts.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
import yaml

from bi_qmn.stats import p3_statistics as engine
from bi_qmn.core.observations import ShotBatch
from bi_qmn.p1.model.p1_banks import P1ObservationRecord
from bi_qmn.p1.p1_estimator import _smoothed
from bi_qmn.p3.p3_records import P3ObservationRecord
from bi_qmn.p3.models.p3_relational_dirac import P3_OUTCOMES
from bi_qmn.core.types import MeasurementContext, MeasurementTable


CONTEXT_CLASSES = ("diag", "coh")


# ---------------------------------------------------------------------------
# Targets frozen in the P1 ILGC operational configuration
# ---------------------------------------------------------------------------


def load_ilgc_targets(config_dir: str | Path = "config", filename: str = "p1_ilgc_operational_r4.yaml") -> dict[str, Any]:
    values = yaml.safe_load((Path(config_dir) / filename).read_text(encoding="utf-8"))
    body = values["calibration_body"]
    if body["whitening"]["method"] != "identity_weighting_v1":
        raise ValueError("ILGC-facing metrics assume identity whitening")
    return {
        "source": filename,
        "strata": dict(values["strata"]),
        "alpha": {key: float(value) for key, value in body["alpha"].items()},
        "minimum_construction_blocks": int(body["minimum_construction_blocks"]),
        "normalized_radius_ceiling": {
            stratum: {cls: float(value) for cls, value in by_class.items()}
            for stratum, by_class in values["anti_vacuity"]["normalized_radius_ceiling"].items()
        },
    }


# ---------------------------------------------------------------------------
# Prediction interface
# ---------------------------------------------------------------------------


class _VectorModel(Protocol):
    def predict_vectors(self, table: MeasurementTable, contexts: list[MeasurementContext]) -> np.ndarray: ...


def predict_full_table(model: Any, support_table: MeasurementTable, contexts: list[MeasurementContext]) -> MeasurementTable:
    """Uniform table prediction for P1 kernel estimators (predict_table) and Model C (predict_vectors)."""
    if hasattr(model, "predict_table"):
        return model.predict_table(support_table, list(contexts))
    vectors = model.predict_vectors(support_table, list(contexts))
    return MeasurementTable(
        probabilities={
            context.context_id: dict(zip(P3_OUTCOMES, row.tolist(), strict=True))
            for context, row in zip(contexts, vectors, strict=True)
        },
        contexts=list(contexts),
        metadata=dict(support_table.metadata),
        source_density=None,
    )


class SupportPassthroughPredictor:
    """Observed contexts -> smoothed support frequencies; unobserved contexts -> wrapped model."""

    def __init__(self, base: Any):
        self.base = base

    @property
    def checkpoint_hash(self) -> str:
        return f"support_passthrough:{getattr(self.base, 'checkpoint_hash', 'unknown')}"

    def predict_table(self, support_table: MeasurementTable, contexts: list[MeasurementContext]) -> MeasurementTable:
        unobserved = [c for c in contexts if c.context_id not in support_table.probabilities]
        predicted = predict_full_table(self.base, support_table, unobserved) if unobserved else None
        probabilities: dict[str, dict[str, float]] = {}
        for context in contexts:
            if context.context_id in support_table.probabilities:
                probabilities[context.context_id] = _smoothed(dict(support_table.probabilities[context.context_id]))
            else:
                probabilities[context.context_id] = dict(predicted.probabilities[context.context_id])
        return MeasurementTable(probabilities=probabilities, contexts=list(contexts), metadata=dict(support_table.metadata), source_density=None)


# ---------------------------------------------------------------------------
# Adapter so the P1 kernel estimators (Model A / Model B) can be fitted on P3 development records
# ---------------------------------------------------------------------------


def _merge_batches(batches: Iterable[ShotBatch], role: str) -> ShotBatch:
    probabilities: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    shot_ids: dict[str, Any] = {}
    contexts: list[MeasurementContext] = []
    metadata: dict[str, Any] = {}
    for batch in batches:
        for context in batch.table.contexts:
            if context.context_id in probabilities:
                raise ValueError(f"context {context.context_id} appears in more than one query batch")
            contexts.append(context)
            probabilities[context.context_id] = dict(batch.table.probabilities[context.context_id])
            counts[context.context_id] = dict(batch.counts[context.context_id])
            shot_ids[context.context_id] = batch.shot_ids[context.context_id]  # keep compact ranges compact
        metadata = {**batch.table.metadata, **metadata}
    table = MeasurementTable(probabilities=probabilities, contexts=contexts, metadata={**metadata, "role": role}, source_density=None)
    return ShotBatch(table=table, shot_ids=shot_ids, counts=counts)


def p1_record_from_p3(record: P3ObservationRecord) -> P1ObservationRecord:
    """Support unchanged; query = disjoint union of the shot, compositional and operator target batches."""
    bank_id = str(record.metadata.get("bank_id", record.split))
    return P1ObservationRecord(
        split=record.split,
        deficit=record.deficit,
        gamma=record.gamma,
        instance_index=record.instance_index,
        replicate=record.replicate,
        exact_table=record.exact_table,
        support=record.support,
        query=_merge_batches((record.query_shot, record.query_comp, record.query_op), "p3_merged_query"),
        metadata={
            **record.metadata,
            "construction_bank_id": bank_id,
            "construction_fingerprint": f"{bank_id}:{record.instance_index}",
        },
    )


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def class_dimensions(contexts: list[MeasurementContext]) -> dict[str, int]:
    dims: dict[str, int] = defaultdict(int)
    for context in contexts:
        dims[context.context_class] += len(P3_OUTCOMES)
    return dict(dims)


def class_residuals(
    prediction: MeasurementTable,
    exact: MeasurementTable,
    contexts: list[MeasurementContext],
    *,
    observed_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, dict[str, np.ndarray]]:
    """Residuals ``exact - prediction`` per context class, for all contexts and for unobserved ones."""
    all_values: dict[str, list[float]] = defaultdict(list)
    unobserved: dict[str, list[float]] = defaultdict(list)
    for context in contexts:
        for outcome in P3_OUTCOMES:
            value = float(exact.probabilities[context.context_id][outcome]) - float(prediction.probabilities[context.context_id][outcome])
            all_values[context.context_class].append(value)
            if context.context_id not in observed_ids:
                unobserved[context.context_class].append(value)
    return {
        cls: {"all": np.asarray(all_values[cls], dtype=float), "unobs": np.asarray(unobserved.get(cls, []), dtype=float)}
        for cls in all_values
    }


def record_scores(model: Any, record: P3ObservationRecord, contexts: list[MeasurementContext], *, support_table: MeasurementTable | None = None) -> dict[str, dict[str, float]]:
    table = record.support.table if support_table is None else support_table
    prediction = predict_full_table(model, table, contexts)
    residuals = class_residuals(prediction, record.exact_table, contexts, observed_ids=set(table.probabilities))
    return {cls: {kind: float(np.linalg.norm(values)) for kind, values in parts.items()} for cls, parts in residuals.items()}


def conformal_radius(block_values: Iterable[float], alpha: float) -> dict[str, Any]:
    values = sorted(float(v) for v in block_values)
    n = len(values)
    k = engine.split_conformal_order_index(n, alpha) if n else None
    return {"n_blocks": n, "order_index": k, "radius": None if k is None else values[k - 1]}


def sharpness(radius: float | None, dim: int, ceiling: float) -> dict[str, Any]:
    if radius is None:
        return {"normalized_radius": None, "ceiling": ceiling, "qualified": False, "reason": "insufficient_blocks"}
    normalized = radius / math.sqrt(dim)
    return {"normalized_radius": normalized, "ceiling": ceiling, "qualified": bool(normalized <= ceiling)}


def body_report(
    block_scores: Mapping[str, Mapping[str, Mapping[int, float]]],
    dims: Mapping[str, int],
    targets: Mapping[str, Any],
    *,
    split_seed: int,
) -> dict[str, Any]:
    """Conformal radius, sharpness gate and split-half coverage audit per (stratum, class).

    ``block_scores[stratum][class][block] -> score`` (already the per-block maximum).
    """
    report: dict[str, Any] = {}
    for stratum, by_class in block_scores.items():
        ceilings = targets["normalized_radius_ceiling"][stratum]
        report[stratum] = {}
        for cls, scores in by_class.items():
            alpha = float(targets["alpha"][cls])
            blocks = sorted(scores)
            full = conformal_radius([scores[b] for b in blocks], alpha)
            rng = np.random.default_rng(split_seed)
            order = rng.permutation(len(blocks))
            half = len(blocks) // 2
            fit_blocks = [blocks[i] for i in order[:half]]
            audit_blocks = [blocks[i] for i in order[half:]]
            fit = conformal_radius([scores[b] for b in fit_blocks], alpha)
            audit = None
            if fit["radius"] is not None and audit_blocks:
                exceed = int(sum(scores[b] > fit["radius"] for b in audit_blocks))
                audit = {
                    "fit_blocks": len(fit_blocks),
                    "audit_blocks": len(audit_blocks),
                    "radius": fit["radius"],
                    "exceedances": exceed,
                    "empirical_coverage": 1.0 - exceed / len(audit_blocks),
                    "exchangeability_tail_probability": engine.exceedance_tail_probability(len(fit_blocks), int(fit["order_index"]), len(audit_blocks), exceed) if exceed else 1.0,
                }
            report[stratum][cls] = {
                "dimension": int(dims[cls]),
                "alpha": alpha,
                "conformal": full,
                "sharpness": sharpness(full["radius"], int(dims[cls]), float(ceilings[cls])),
                "mean_block_score": float(np.mean([scores[b] for b in blocks])) if blocks else None,
                "split_half_coverage_audit": audit,
            }
    return report


def non_inferiority_margin(targets: Mapping[str, Any], dims: Mapping[str, int], *, fraction: float, reference_stratum: str) -> dict[str, float]:
    ceilings = targets["normalized_radius_ceiling"][reference_stratum]
    return {cls: float(fraction * ceilings[cls] * math.sqrt(dims[cls])) for cls in CONTEXT_CLASSES}
