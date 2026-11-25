from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from bi_qmn.p1.model.p1_banks import P1ObservationRecord
from bi_qmn.core.types import MeasurementContext, MeasurementTable


def p1_support_vector(table: MeasurementTable, input_contexts: list[MeasurementContext]) -> np.ndarray:
    values: list[float] = []
    for context in input_contexts:
        if context.context_id not in table.probabilities:
            values.extend([0.0] * 16)
            values.append(0.0)
            continue
        probs = table.probabilities[context.context_id]
        for outcome in sorted(probs):
            values.append(float(probs[outcome]))
        values.append(1.0)
    return np.asarray(values, dtype=float)


@dataclass
class _P1MemoryRecord:
    support: np.ndarray
    predictions: dict[str, dict[str, float]]


@dataclass
class _P1KernelRecord:
    construction_key: str
    support_probabilities: dict[str, dict[str, float]]
    predictions: dict[str, dict[str, float]]


class P1FreeTableEstimator:
    def __init__(self, input_contexts: list[MeasurementContext]):
        self.input_contexts = input_contexts
        self.records: list[_P1MemoryRecord] = []
        self.checkpoint_hash = "unfitted"

    def fit_records(self, records: list[P1ObservationRecord]) -> "P1FreeTableEstimator":
        self.records = []
        for record in records:
            predictions = {
                context.context_id: dict(record.query.table.probabilities[context.context_id])
                for context in record.query.table.contexts
            }
            self.records.append(
                _P1MemoryRecord(
                    support=p1_support_vector(record.support.table, self.input_contexts),
                    predictions=predictions,
                )
            )
        if not self.records:
            raise ValueError("P1 estimator requires at least one training record")
        digest = hashlib.sha256()
        for memory in self.records:
            digest.update(np.ascontiguousarray(memory.support, dtype=np.float64).tobytes())
            for descriptor, probabilities in sorted(memory.predictions.items()):
                digest.update(repr(descriptor).encode("ascii"))
                for outcome, value in sorted(probabilities.items()):
                    digest.update(outcome.encode("ascii"))
                    digest.update(float(value).hex().encode("ascii"))
        self.checkpoint_hash = f"p1_memory_v1:{digest.hexdigest()[:16]}"
        return self

    def predict(self, support_table: MeasurementTable, query_context: MeasurementContext) -> dict[str, float]:
        if query_context.context_id in support_table.probabilities:
            return _smoothed(dict(support_table.probabilities[query_context.context_id]))
        query = p1_support_vector(support_table, self.input_contexts)
        distances = [float(np.linalg.norm(record.support - query)) for record in self.records]
        record = self.records[int(np.argmin(distances))]
        return dict(record.predictions[query_context.context_id])

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


class P1KernelFreeTableEstimator:
    """Adaptive kernel estimator whose outputs remain unconstrained free tables."""

    def __init__(
        self,
        input_contexts: list[MeasurementContext],
        *,
        neighbor_count: int = 16,
        distance_power: float = 4.0,
        probability_floor: float = 1.0e-6,
        distance_mode: str = "observed_overlap",
    ):
        if neighbor_count <= 0 or distance_power <= 0.0 or probability_floor <= 0.0:
            raise ValueError("invalid P1 kernel-estimator configuration")
        self.input_contexts = list(input_contexts)
        self.neighbor_count = int(neighbor_count)
        self.distance_power = float(distance_power)
        self.probability_floor = float(probability_floor)
        if distance_mode not in {"observed_overlap", "flat_masked"}:
            raise ValueError("unknown P1 kernel distance mode")
        self.distance_mode = distance_mode
        self.records: list[_P1KernelRecord] = []
        self._support_matrix = np.empty((0, len(self.input_contexts) * 17), dtype=float)
        self.checkpoint_hash = "unfitted"

    def fit_records(self, records: list[P1ObservationRecord]) -> "P1KernelFreeTableEstimator":
        if not records:
            raise ValueError("P1 kernel estimator requires at least one training record")

        # Keep the highest-information shot batch for each construction. This makes every
        # training donor comparable on whatever subset is observed at inference time.
        selected: dict[str, P1ObservationRecord] = {}
        for record in records:
            key = str(
                record.metadata.get("construction_fingerprint")
                or record.exact_table.metadata.get("construction_fingerprint")
                or f"legacy-index:{record.instance_index}"
            )
            incumbent = selected.get(key)
            score = (len(record.support.table.contexts), int(record.metadata.get("n_shot", 0)))
            incumbent_score = (
                (-1, -1)
                if incumbent is None
                else (
                    len(incumbent.support.table.contexts),
                    int(incumbent.metadata.get("n_shot", 0)),
                )
            )
            if incumbent is None or score > incumbent_score:
                selected[key] = record

        self.records = []
        for key, record in sorted(selected.items()):
            self.records.append(
                _P1KernelRecord(
                    construction_key=key,
                    support_probabilities={
                        context_id: dict(probabilities)
                        for context_id, probabilities in record.support.table.probabilities.items()
                    },
                    predictions={
                        context.context_id: dict(
                            record.query.table.probabilities[context.context_id]
                        )
                        for context in record.query.table.contexts
                    },
                )
            )

        self._support_matrix = np.vstack(
            [
                p1_support_vector(
                    MeasurementTable(
                        probabilities=memory.support_probabilities,
                        contexts=[
                            context
                            for context in self.input_contexts
                            if context.context_id in memory.support_probabilities
                        ],
                    ),
                    self.input_contexts,
                )
                for memory in self.records
            ]
        )

        digest = hashlib.sha256()
        digest.update(b"p1_adaptive_kernel_free_table_v2")
        digest.update(str(self.neighbor_count).encode("ascii"))
        digest.update(float(self.distance_power).hex().encode("ascii"))
        digest.update(float(self.probability_floor).hex().encode("ascii"))
        digest.update(self.distance_mode.encode("ascii"))
        for memory in self.records:
            digest.update(memory.construction_key.encode("utf-8"))
            for context_id, probabilities in sorted(memory.support_probabilities.items()):
                digest.update(context_id.encode("ascii"))
                for outcome, value in sorted(probabilities.items()):
                    digest.update(outcome.encode("ascii"))
                    digest.update(float(value).hex().encode("ascii"))
            for context_id, probabilities in sorted(memory.predictions.items()):
                digest.update(context_id.encode("ascii"))
                for outcome, value in sorted(probabilities.items()):
                    digest.update(outcome.encode("ascii"))
                    digest.update(float(value).hex().encode("ascii"))
        self.checkpoint_hash = f"p1_kernel_free_v2:{digest.hexdigest()[:16]}"
        return self

    def _weights(self, support_table: MeasurementTable) -> list[tuple[_P1KernelRecord, float]]:
        query = p1_support_vector(support_table, self.input_contexts)
        columns: list[int] = []
        if self.distance_mode == "flat_masked":
            columns.extend(range(len(self.input_contexts) * 17))
        else:
            for index, context in enumerate(self.input_contexts):
                if context.context_id in support_table.probabilities:
                    columns.extend(range(index * 17, index * 17 + 16))
        if not columns:
            raise ValueError("P1 kernel estimator requires at least one observed context")
        column_array = np.asarray(columns, dtype=int)
        difference = self._support_matrix[:, column_array] - query[column_array]
        distances = np.sqrt(np.mean(difference * difference, axis=1))
        order = np.argsort(distances, kind="stable")[: self.neighbor_count]
        selected = [(float(distances[index]), self.records[int(index)]) for index in order]
        if not selected:
            raise ValueError("P1 kernel estimator found no comparable training support")
        exact = [memory for distance, memory in selected if distance <= 1.0e-14]
        if exact:
            weight = 1.0 / len(exact)
            return [(memory, weight) for memory in exact]
        raw = np.asarray(
            [1.0 / max(distance, 1.0e-12) ** self.distance_power for distance, _ in selected],
            dtype=float,
        )
        raw /= float(np.sum(raw))
        return [(memory, float(weight)) for (_, memory), weight in zip(selected, raw, strict=True)]

    def predict(self, support_table: MeasurementTable, query_context: MeasurementContext) -> dict[str, float]:
        if query_context.context_id in support_table.probabilities:
            return _smoothed(dict(support_table.probabilities[query_context.context_id]))
        return self._predict_with_weights(query_context.context_id, self._weights(support_table))

    def _predict_with_weights(
        self,
        context_id: str,
        weighted: list[tuple[_P1KernelRecord, float]],
    ) -> dict[str, float]:
        outcomes = sorted(weighted[0][0].predictions[context_id])
        prediction = {
            outcome: sum(weight * float(memory.predictions[context_id][outcome]) for memory, weight in weighted)
            for outcome in outcomes
        }
        values = np.asarray([max(self.probability_floor, prediction[outcome]) for outcome in outcomes], dtype=float)
        values /= float(np.sum(values))
        return dict(zip(outcomes, values.tolist(), strict=True))

    def predict_table(
        self,
        support_table: MeasurementTable,
        contexts: list[MeasurementContext],
        metadata: dict | None = None,
    ) -> MeasurementTable:
        weighted = self._weights(support_table)
        probabilities: dict[str, dict[str, float]] = {}
        for context in contexts:
            if context.context_id in support_table.probabilities:
                probabilities[context.context_id] = _smoothed(
                    dict(support_table.probabilities[context.context_id])
                )
            else:
                probabilities[context.context_id] = self._predict_with_weights(
                    context.context_id,
                    weighted,
                )
        return MeasurementTable(
            probabilities=probabilities,
            contexts=list(contexts),
            metadata=dict(metadata or support_table.metadata),
            source_density=None,
        )


class P1MatchedGenericComparator(P1KernelFreeTableEstimator):
    """Capacity-matched kernel control without observed-overlap geometry."""

    def __init__(
        self,
        input_contexts: list[MeasurementContext],
        *,
        neighbor_count: int = 8,
        distance_power: float = 2.0,
        probability_floor: float = 1.0e-6,
    ):
        super().__init__(
            input_contexts,
            neighbor_count=neighbor_count,
            distance_power=distance_power,
            probability_floor=probability_floor,
            distance_mode="flat_masked",
        )


def nll_distribution(true_probs: dict[str, float], pred_probs: dict[str, float], eps: float = 1.0e-12) -> float:
    return -sum(float(p) * math.log(max(eps, float(pred_probs[k]))) for k, p in true_probs.items())


def average_distribution(records: list[P1ObservationRecord], context: MeasurementContext) -> dict[str, float]:
    keys = sorted(records[0].query.table.probabilities[context.context_id])
    avg = {key: 0.0 for key in keys}
    for record in records:
        probs = record.query.table.probabilities[context.context_id]
        for key in keys:
            avg[key] += float(probs[key]) / len(records)
    return avg


def lcb_normal(values: list[float], alpha: float = 0.05) -> float:
    if not values:
        return float("-inf")
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return float(arr[0])
    z = 1.6448536269514722
    return float(np.mean(arr) - z * np.std(arr, ddof=1) / np.sqrt(arr.size))


def evaluate_p1_noncollapse(
    estimator: P1FreeTableEstimator,
    train_records: list[P1ObservationRecord],
    test_records: list[P1ObservationRecord],
    input_contexts: list[MeasurementContext],
    comp_contexts: list[MeasurementContext],
    op_contexts: list[MeasurementContext],
    alpha: float,
) -> dict:
    contexts = input_contexts + comp_contexts + op_contexts
    const = {context.context_id: average_distribution(train_records, context) for context in contexts}
    per_record = {"Delta_shot": [], "Delta_comp": [], "Delta_op": []}
    for record in test_records:
        prediction = estimator.predict_table(record.support.table, contexts)
        shot = []
        for context in input_contexts:
            if context.context_id not in record.support.table.probabilities:
                continue
            true = record.query.table.probabilities[context.context_id]
            pred = prediction.probabilities[context.context_id]
            shot.append(nll_distribution(true, const[context.context_id]) - nll_distribution(true, pred))
        comp = []
        for context in comp_contexts:
            true = record.query.table.probabilities[context.context_id]
            pred = prediction.probabilities[context.context_id]
            comp.append(nll_distribution(true, const[context.context_id]) - nll_distribution(true, pred))
        op = []
        for context in op_contexts:
            true = record.query.table.probabilities[context.context_id]
            pred = prediction.probabilities[context.context_id]
            op.append(nll_distribution(true, const[context.context_id]) - nll_distribution(true, pred))
        per_record["Delta_shot"].append(float(np.mean(shot)))
        per_record["Delta_comp"].append(float(np.mean(comp)))
        per_record["Delta_op"].append(float(np.mean(op)))
    return {
        "Delta_shot": float(np.mean(per_record["Delta_shot"])),
        "Delta_comp": float(np.mean(per_record["Delta_comp"])),
        "Delta_op": float(np.mean(per_record["Delta_op"])),
        "LCB_Delta_shot": lcb_normal(per_record["Delta_shot"], alpha),
        "LCB_Delta_comp": lcb_normal(per_record["Delta_comp"], alpha),
        "LCB_Delta_op": lcb_normal(per_record["Delta_op"], alpha),
        "per_record": per_record,
    }


def evaluate_p1_matched_advantage(
    estimator: P1KernelFreeTableEstimator,
    comparator: P1MatchedGenericComparator,
    test_records: list[P1ObservationRecord],
    input_contexts: list[MeasurementContext],
    comp_contexts: list[MeasurementContext],
    op_contexts: list[MeasurementContext],
    alpha: float,
) -> dict:
    contexts = input_contexts + comp_contexts + op_contexts
    per_record = {"Adv_shot": [], "Adv_comp": [], "Adv_op": []}
    groups = {
        "Adv_shot": input_contexts,
        "Adv_comp": comp_contexts,
        "Adv_op": op_contexts,
    }
    for record in test_records:
        primary = estimator.predict_table(record.support.table, contexts)
        generic = comparator.predict_table(record.support.table, contexts)
        for key, selected_contexts in groups.items():
            values = []
            for context in selected_contexts:
                if key == "Adv_shot" and context.context_id not in record.support.table.probabilities:
                    continue
                truth = record.query.table.probabilities[context.context_id]
                values.append(
                    nll_distribution(truth, generic.probabilities[context.context_id])
                    - nll_distribution(truth, primary.probabilities[context.context_id])
                )
            per_record[key].append(float(np.mean(values)))
    result = {key: float(np.mean(values)) for key, values in per_record.items()}
    result.update({f"LCB_{key}": lcb_normal(values, alpha) for key, values in per_record.items()})
    result["per_record"] = per_record
    return result


def _smoothed(distribution: dict[str, float], eps: float = 1.0e-9) -> dict[str, float]:
    total = sum(distribution.values()) + eps * len(distribution)
    return {key: (float(value) + eps) / total for key, value in distribution.items()}
