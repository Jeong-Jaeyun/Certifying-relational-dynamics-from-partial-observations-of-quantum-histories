from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bi_qmn.core.config import load_p0_config, output_path
from bi_qmn.core.observations import ObservationRecord, build_observation_records, records_by_split
from bi_qmn.core.measurement import effects_for_context, full_context_list, measurement_table
from bi_qmn.core.types import MeasurementContext, MeasurementTable, to_jsonable


PAULI_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _pauli_basis_3q() -> list[np.ndarray]:
    basis = []
    labels = ["I", "X", "Y", "Z"]
    for a in labels:
        for b in labels:
            for c in labels:
                basis.append(np.kron(np.kron(PAULI_1Q[a], PAULI_1Q[b]), PAULI_1Q[c]))
    return basis


PAULI_BASIS_3Q = _pauli_basis_3q()


def context_descriptor(context: MeasurementContext) -> np.ndarray:
    """Fixed Pauli-coefficient descriptor of the POVM context, not a learned name embedding."""
    coeffs: list[float] = []
    for effect in effects_for_context(context, K=4):
        data = np.asarray(effect.operator.data, dtype=complex)
        for pauli in PAULI_BASIS_3Q:
            coeff = np.trace(pauli.conj().T @ data) / 8.0
            coeffs.append(float(np.real_if_close(coeff, tol=1000)))
    return np.asarray(coeffs, dtype=float)


def _descriptor_key(context: MeasurementContext) -> tuple[float, ...]:
    return tuple(np.round(context_descriptor(context), decimals=10).tolist())


def support_vector(table: MeasurementTable, input_contexts: list[MeasurementContext]) -> np.ndarray:
    values: list[float] = []
    for context in input_contexts:
        probs = table.probabilities[context.context_id]
        for outcome in sorted(probs):
            values.append(float(probs[outcome]))
    return np.asarray(values, dtype=float)


@dataclass
class _MemoryRecord:
    support: np.ndarray
    predictions: dict[tuple[float, ...], dict[str, float]]


class FreeMeasurementTableEstimator:
    def __init__(self, input_contexts: list[MeasurementContext]):
        self.input_contexts = input_contexts
        self.records: list[_MemoryRecord] = []
        self._checkpoint_hash: str | None = None

    @property
    def checkpoint_hash(self) -> str:
        if self._checkpoint_hash is None:
            raise RuntimeError("estimator has not been fitted")
        return self._checkpoint_hash

    def _freeze_checkpoint(self, provenance: list[dict]) -> None:
        digest = hashlib.sha256()
        digest.update(b"FreeMeasurementTableEstimator_v1")
        for context in self.input_contexts:
            digest.update(context.context_id.encode("utf-8"))
        for item in provenance:
            digest.update(
                json.dumps(
                    to_jsonable(item),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        for record in self.records:
            for value in record.support:
                digest.update(float(value).hex().encode("ascii"))
            for descriptor, probabilities in sorted(record.predictions.items()):
                for value in descriptor:
                    digest.update(float(value).hex().encode("ascii"))
                for outcome, probability in sorted(probabilities.items()):
                    digest.update(outcome.encode("utf-8"))
                    digest.update(float(probability).hex().encode("ascii"))
        self._checkpoint_hash = digest.hexdigest()[:16]

    def fit(self, tables: list[MeasurementTable]) -> "FreeMeasurementTableEstimator":
        self.records = []
        provenance: list[dict] = []
        for table in tables:
            predictions = {
                _descriptor_key(context): dict(table.probabilities[context.context_id])
                for context in table.contexts
            }
            self.records.append(_MemoryRecord(support=support_vector(table, self.input_contexts), predictions=predictions))
            provenance.append(dict(table.metadata))
        if not self.records:
            raise ValueError("estimator requires at least one training table")
        self._freeze_checkpoint(provenance)
        return self

    def fit_records(self, records: list[ObservationRecord]) -> "FreeMeasurementTableEstimator":
        self.records = []
        provenance: list[dict] = []
        for record in records:
            predictions = {
                _descriptor_key(context): dict(record.query.table.probabilities[context.context_id])
                for context in record.query.table.contexts
            }
            self.records.append(
                _MemoryRecord(
                    support=support_vector(record.support.table, self.input_contexts),
                    predictions=predictions,
                )
            )
            provenance.append(
                {
                    "split": record.split,
                    "regime": record.regime,
                    "instance_index": record.instance_index,
                    "replicate": record.replicate,
                    "seed": record.metadata.get("seed"),
                    "record_index": record.metadata.get("record_index"),
                }
            )
        if not self.records:
            raise ValueError("estimator requires at least one training record")
        self._freeze_checkpoint(provenance)
        return self

    def predict(self, support_table: MeasurementTable, query_context: MeasurementContext) -> dict[str, float]:
        if query_context.context_id in support_table.probabilities:
            return _smoothed(dict(support_table.probabilities[query_context.context_id]))
        query = support_vector(support_table, self.input_contexts)
        distances = [float(np.linalg.norm(record.support - query)) for record in self.records]
        record = self.records[int(np.argmin(distances))]
        key = _descriptor_key(query_context)
        if key not in record.predictions:
            raise KeyError("query context descriptor not present in fitted table memory")
        return dict(record.predictions[key])

    def predict_table(
        self,
        support_table: MeasurementTable,
        contexts: list[MeasurementContext],
        metadata: dict | None = None,
    ) -> MeasurementTable:
        return MeasurementTable(
            probabilities={context.context_id: self.predict(support_table, context) for context in contexts},
            contexts=list(contexts),
            metadata=dict(metadata or support_table.metadata),
            source_density=None,
        )


def _smoothed(distribution: dict[str, float], eps: float = 1.0e-9) -> dict[str, float]:
    total = sum(distribution.values()) + eps * len(distribution)
    return {key: (float(value) + eps) / total for key, value in distribution.items()}


def nll_distribution(true_probs: dict[str, float], pred_probs: dict[str, float], eps: float = 1.0e-12) -> float:
    return -sum(float(p) * math.log(max(eps, float(pred_probs[k]))) for k, p in true_probs.items())


def average_distribution(tables: list[MeasurementTable], context: MeasurementContext) -> dict[str, float]:
    keys = sorted(tables[0].probabilities[context.context_id])
    avg = {key: 0.0 for key in keys}
    for table in tables:
        for key in keys:
            avg[key] += table.probabilities[context.context_id][key] / len(tables)
    return avg


def lcb_normal(values: list[float], alpha: float = 0.05) -> float:
    if not values:
        return float("-inf")
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return float(arr[0])
    z = 1.6448536269514722 if alpha == 0.05 else 1.6448536269514722
    return float(np.mean(arr) - z * np.std(arr, ddof=1) / np.sqrt(arr.size))


def evaluate_records_noncollapse(
    estimator: FreeMeasurementTableEstimator,
    train_records: list[ObservationRecord],
    test_records: list[ObservationRecord],
    input_contexts: list[MeasurementContext],
    comp_context: MeasurementContext,
    op_context: MeasurementContext,
    alpha: float,
) -> dict:
    train_query_tables = [record.query.table for record in train_records]
    const_by_context = {
        context.context_id: average_distribution(train_query_tables, context)
        for context in input_contexts + [comp_context, op_context]
    }
    delta_shot_records: list[float] = []
    delta_comp_records: list[float] = []
    delta_op_records: list[float] = []
    for record in test_records:
        shot_deltas = []
        for context in input_contexts:
            pred = estimator.predict(record.support.table, context)
            true = record.query.table.probabilities[context.context_id]
            shot_deltas.append(
                nll_distribution(true, const_by_context[context.context_id])
                - nll_distribution(true, pred)
            )
        pred_comp = estimator.predict(record.support.table, comp_context)
        pred_op = estimator.predict(record.support.table, op_context)
        delta_shot_records.append(float(np.mean(shot_deltas)))
        delta_comp_records.append(
            nll_distribution(record.query.table.probabilities[comp_context.context_id], const_by_context[comp_context.context_id])
            - nll_distribution(record.query.table.probabilities[comp_context.context_id], pred_comp)
        )
        delta_op_records.append(
            nll_distribution(record.query.table.probabilities[op_context.context_id], const_by_context[op_context.context_id])
            - nll_distribution(record.query.table.probabilities[op_context.context_id], pred_op)
        )
    return {
        "Delta_shot": float(np.mean(delta_shot_records)),
        "Delta_comp": float(np.mean(delta_comp_records)),
        "Delta_op": float(np.mean(delta_op_records)),
        "LCB_Delta_shot": lcb_normal(delta_shot_records, alpha=alpha),
        "LCB_Delta_comp": lcb_normal(delta_comp_records, alpha=alpha),
        "LCB_Delta_op": lcb_normal(delta_op_records, alpha=alpha),
        "per_record": {
            "Delta_shot": delta_shot_records,
            "Delta_comp": delta_comp_records,
            "Delta_op": delta_op_records,
        },
    }


def evaluate_free_table_estimator(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    config = load_p0_config(config_dir)
    from bi_qmn.core.measurement import contexts_from_measurement_config

    parsed = contexts_from_measurement_config(config.measurement)
    input_contexts = parsed["input"]
    comp_context = parsed["compositional_heldout"][0]
    op_context = parsed["operator_heldout"][0]
    records = build_observation_records(
        config.generator,
        config.measurement,
        seed=int(config.seeds.get("shot_seed", 0)),
    )
    train_records = records_by_split(records, "train")
    test_records = records_by_split(records, "test")
    estimator = FreeMeasurementTableEstimator(input_contexts).fit_records(train_records)
    alpha = float(config.measurement["observation"]["lcb_alpha"])
    noncollapse = evaluate_records_noncollapse(
        estimator,
        train_records,
        test_records,
        input_contexts,
        comp_context,
        op_context,
        alpha=alpha,
    )
    split = {
        "train": sorted({record.instance_index for record in train_records}),
        "calibration": sorted({record.instance_index for record in records_by_split(records, "calibration")}),
        "test": sorted({record.instance_index for record in test_records}),
    }
    artifact = {
        "stage": "estimator",
        "config_hash": config.config_hash,
        "status": "accept"
        if noncollapse["LCB_Delta_shot"] > 0.0 and noncollapse["LCB_Delta_comp"] > 0.0
        else "failure",
        "estimator": "FreeMeasurementTableEstimator",
        "descriptor": "fixed Pauli coefficients of Qiskit POVM effects",
        "split": split,
        "heldout": {
            **noncollapse,
        },
        "notes": {
            "output_head": "free measurement table",
            "direct_state_head": False,
            "context_name_embedding": False,
            "shot_level_status": "same-context disjoint support/query shot batches",
            "test_table_seen_during_fit": False,
        },
    }
    with output_path(output_dir, "p0_estimator.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(artifact), fh, indent=2, sort_keys=True)
    return artifact
