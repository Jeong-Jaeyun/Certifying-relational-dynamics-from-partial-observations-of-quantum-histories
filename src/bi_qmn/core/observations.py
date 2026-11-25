from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bi_qmn.core.generator import P0Instance, construction_bank
from bi_qmn.core.measurement import full_context_list, measurement_table
from bi_qmn.core.types import MeasurementContext
from bi_qmn.core.types import MeasurementTable


def support_query_ids(
    input_contexts: list[MeasurementContext],
    query_contexts: list[MeasurementContext],
) -> tuple[set[str], set[str]]:
    support = {context.context_id for context in input_contexts}
    query = {context.context_id for context in query_contexts}
    return support, query


def support_query_disjoint(
    input_contexts: list[MeasurementContext],
    query_contexts: list[MeasurementContext],
) -> bool:
    support, query = support_query_ids(input_contexts, query_contexts)
    return support.isdisjoint(query)


@dataclass(frozen=True)
class ShotBatch:
    table: MeasurementTable
    shot_ids: dict[str, tuple[int, ...]]
    counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class ObservationRecord:
    split: str
    regime: str
    instance_index: int
    replicate: int
    eta: float
    exact_table: MeasurementTable
    support: ShotBatch
    query: ShotBatch
    metadata: dict


def sample_shot_batch(
    exact_table: MeasurementTable,
    contexts: list[MeasurementContext],
    n_shots: int,
    rng: np.random.Generator,
    shot_start: int,
    role: str,
) -> ShotBatch:
    probabilities: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    shot_ids: dict[str, tuple[int, ...]] = {}
    next_shot = shot_start
    for context in contexts:
        context_id = context.context_id
        outcomes = sorted(exact_table.probabilities[context_id])
        probs = np.asarray([exact_table.probabilities[context_id][outcome] for outcome in outcomes], dtype=float)
        probs = probs / np.sum(probs)
        draw = rng.multinomial(n_shots, probs)
        counts[context_id] = {outcome: int(count) for outcome, count in zip(outcomes, draw, strict=True)}
        probabilities[context_id] = {
            outcome: int(count) / float(n_shots) for outcome, count in zip(outcomes, draw, strict=True)
        }
        shot_ids[context_id] = tuple(range(next_shot, next_shot + n_shots))
        next_shot += n_shots
    table = MeasurementTable(
        probabilities=probabilities,
        contexts=list(contexts),
        metadata={
            **exact_table.metadata,
            "sampled": True,
            "role": role,
            "n_shots": n_shots,
        },
        source_density=None,
    )
    return ShotBatch(table=table, shot_ids=shot_ids, counts=counts)


def sample_nested_shot_batches(
    exact_table: MeasurementTable,
    contexts: list[MeasurementContext],
    shot_counts: list[int] | tuple[int, ...],
    rng: np.random.Generator,
    shot_start: int,
    role: str,
) -> dict[int, ShotBatch]:
    """Sample one maximum-shot stream and expose deterministic prefix batches."""
    levels = tuple(sorted({int(value) for value in shot_counts}))
    if not levels or levels[0] <= 0:
        raise ValueError("nested shot levels must be positive")
    maximum = levels[-1]
    probabilities = {level: {} for level in levels}
    counts = {level: {} for level in levels}
    shot_ids = {level: {} for level in levels}
    next_shot = int(shot_start)
    for context in contexts:
        context_id = context.context_id
        outcomes = sorted(exact_table.probabilities[context_id])
        probs = np.asarray(
            [exact_table.probabilities[context_id][outcome] for outcome in outcomes],
            dtype=float,
        )
        probs /= float(np.sum(probs))
        draws = rng.choice(len(outcomes), size=maximum, replace=True, p=probs)
        for level in levels:
            prefix = draws[:level]
            values = np.bincount(prefix, minlength=len(outcomes))
            counts[level][context_id] = {
                outcome: int(value)
                for outcome, value in zip(outcomes, values, strict=True)
            }
            probabilities[level][context_id] = {
                outcome: int(value) / float(level)
                for outcome, value in zip(outcomes, values, strict=True)
            }
            shot_ids[level][context_id] = tuple(range(next_shot, next_shot + level))
        next_shot += maximum
    return {
        level: ShotBatch(
            table=MeasurementTable(
                probabilities=probabilities[level],
                contexts=list(contexts),
                metadata={
                    **exact_table.metadata,
                    "sampled": True,
                    "role": role,
                    "n_shots": level,
                    "nested_shot_parent_size": maximum,
                },
                source_density=None,
            ),
            shot_ids=shot_ids[level],
            counts=counts[level],
        )
        for level in levels
    }


def shot_batches_disjoint(record: ObservationRecord) -> bool:
    support_ids = {shot_id for ids in record.support.shot_ids.values() for shot_id in ids}
    query_ids = {shot_id for ids in record.query.shot_ids.values() for shot_id in ids}
    return support_ids.isdisjoint(query_ids)


def build_observation_records(
    generator_config: dict,
    measurement_config: dict,
    seed: int,
) -> list[ObservationRecord]:
    observation = measurement_config["observation"]
    contexts = full_context_list(measurement_config)
    from bi_qmn.core.measurement import contexts_from_measurement_config

    grouped = contexts_from_measurement_config(measurement_config)
    support_contexts = grouped["input"]
    query_contexts = contexts
    split_indices = {
        "train": list(observation["train_indices"]),
        "calibration": list(observation["calibration_indices"]),
        "test": list(observation["test_indices"]),
    }
    split_replicates = {
        "train": int(observation["train_replicates"]),
        "calibration": int(observation["calibration_replicates"]),
        "test": int(observation["test_replicates"]),
    }
    bank = construction_bank(generator_config)
    records: list[ObservationRecord] = []
    global_record = 0
    for regime, regime_config in observation["regimes"].items():
        eta = float(regime_config["eta"])
        for split, indices in split_indices.items():
            for instance_index in indices:
                instance = bank[int(instance_index)]
                exact = measurement_table(
                    instance.density,
                    contexts,
                    K=instance.K,
                    eta=eta,
                    metadata={
                        "instance": instance.label,
                        "instance_index": int(instance_index),
                        "eta": eta,
                        "regime": regime,
                    },
                )
                for replicate in range(split_replicates[split]):
                    record_seed = seed + 100_000 * global_record + 997 * int(instance_index) + replicate
                    rng = np.random.default_rng(record_seed)
                    support = sample_shot_batch(
                        exact,
                        support_contexts,
                        int(observation["n_support"]),
                        rng,
                        shot_start=global_record * 10_000_000,
                        role="support",
                    )
                    query = sample_shot_batch(
                        exact,
                        query_contexts,
                        int(observation["n_query"]),
                        rng,
                        shot_start=global_record * 10_000_000 + 5_000_000,
                        role="query",
                    )
                    record = ObservationRecord(
                        split=split,
                        regime=regime,
                        instance_index=int(instance_index),
                        replicate=replicate,
                        eta=eta,
                        exact_table=exact,
                        support=support,
                        query=query,
                        metadata={"seed": record_seed, "record_index": global_record},
                    )
                    if not shot_batches_disjoint(record):
                        raise AssertionError("support/query shot ids overlap")
                    records.append(record)
                    global_record += 1
    return records


def records_by_split(records: list[ObservationRecord], split: str) -> list[ObservationRecord]:
    return [record for record in records if record.split == split]


def records_by_regime(records: list[ObservationRecord], regime: str) -> list[ObservationRecord]:
    return [record for record in records if record.regime == regime]
