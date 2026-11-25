from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from bi_qmn.core.observations import ShotBatch, sample_shot_batch
from bi_qmn.p1.model.p1_banks import support_contexts_for_deficit
from bi_qmn.p1.model.p1_generator import clock_label_dephasing_control, construction_bank_p1
from bi_qmn.p1.model.p1_measurement import contexts_from_p1_measurement_config, full_p1_context_list, p1_measurement_table
from bi_qmn.core.types import MeasurementTable


@dataclass(frozen=True)
class P3ObservationRecord:
    """Model C record with disjoint support and explicitly typed target batches."""

    split: str
    deficit: str
    gamma: float
    instance_index: int
    replicate: int
    exact_table: MeasurementTable
    support: ShotBatch
    query_shot: ShotBatch
    query_comp: ShotBatch
    query_op: ShotBatch
    metadata: dict


def shot_index_set(batch: ShotBatch) -> set[tuple[str, int]]:
    return {
        (context_id, shot_id)
        for context_id, shot_ids in batch.shot_ids.items()
        for shot_id in shot_ids
    }


def p3_target_split_valid(record: P3ObservationRecord) -> bool:
    support = shot_index_set(record.support)
    target_batches = (record.query_shot, record.query_comp, record.query_op)
    targets = [shot_index_set(batch) for batch in target_batches]
    if any(support.intersection(target) for target in targets):
        return False
    if any(targets[left].intersection(targets[right]) for left in range(3) for right in range(left + 1, 3)):
        return False

    support_context_ids = set(record.support.table.probabilities)
    if set(record.query_shot.table.probabilities) != support_context_ids:
        return False
    if support_context_ids.intersection(record.query_comp.table.probabilities):
        return False
    if support_context_ids.intersection(record.query_op.table.probabilities):
        return False
    return not set(record.query_comp.table.probabilities).intersection(record.query_op.table.probabilities)


def p3_target_index_manifest(records: list[P3ObservationRecord]) -> dict:
    digest = hashlib.sha256()
    channels = {"support": 0, "shot": 0, "comp": 0, "op": 0}
    for record in records:
        digest.update(
            f"{record.split}:{record.deficit}:{record.gamma}:{record.instance_index}:{record.replicate}".encode("ascii")
        )
        for name, batch in (
            ("support", record.support),
            ("shot", record.query_shot),
            ("comp", record.query_comp),
            ("op", record.query_op),
        ):
            indices = sorted(shot_index_set(batch))
            channels[name] += len(indices)
            digest.update(name.encode("ascii"))
            for context_id, shot_id in indices:
                digest.update(f"{context_id}:{shot_id}".encode("ascii"))
    return {
        "record_count": len(records),
        "shot_index_counts": channels,
        "target_index_hash": digest.hexdigest()[:16],
        "target_channels": ["shot", "comp", "op"],
        "all_records_disjoint": all(p3_target_split_valid(record) for record in records),
    }


def build_p3_observation_records(
    generator_config: dict,
    measurement_config: dict,
    seeds: dict,
    *,
    include_postconstruction_controls: bool = False,
) -> list[P3ObservationRecord]:
    """Build Model C records without changing the historical P1 record definition."""
    bank = construction_bank_p1(generator_config)
    grouped = contexts_from_p1_measurement_config(measurement_config)
    contexts = full_p1_context_list(measurement_config)
    splits = measurement_config["p1_b"]["splits"]
    split_indices = {
        "train": [int(value) for value in splits["train_indices"]],
        "calibration": [int(value) for value in splits["calibration_indices"]],
        "test": [int(value) for value in splits["test_indices"]],
    }
    split_replicates = {
        "train": int(splits["train_replicates"]),
        "calibration": int(splits["calibration_replicates"]),
        "test": int(splits["test_replicates"]),
    }
    gammas = [float(measurement_config["noise"]["calibration_gamma"])]
    if include_postconstruction_controls:
        gammas.extend(float(value) for value in measurement_config["noise"]["negative_control_gamma"])

    records: list[P3ObservationRecord] = []
    record_index = 0
    for gamma in gammas:
        for deficit, deficit_config in measurement_config["observation_deficit"].items():
            n_shot = int(deficit_config["n_shot"])
            for split, indices in split_indices.items():
                for instance_index in indices:
                    instance = bank[instance_index]
                    rho = instance.density if gamma == 1.0 else clock_label_dephasing_control(
                        instance.density,
                        gamma=gamma,
                        K=instance.K,
                        d_S=instance.d_S,
                    )
                    exact = p1_measurement_table(
                        rho,
                        contexts,
                        K=instance.K,
                        eta=float(measurement_config["noise"]["readout_eta"]),
                        metadata={
                            "instance": instance.label,
                            "instance_index": instance_index,
                            "gamma": gamma,
                            "deficit": deficit,
                            "eta": float(measurement_config["noise"]["readout_eta"]),
                        },
                    )
                    support_contexts = support_contexts_for_deficit(
                        measurement_config,
                        deficit,
                        seed_key=f"p3:{split}:{deficit}:{gamma}:{instance_index}",
                    )
                    for replicate in range(split_replicates[split]):
                        rng = np.random.default_rng(
                            int(seeds.get("shot_seed", 0))
                            + 200_000 * record_index
                            + 1_009 * replicate
                            + 97 * instance_index
                        )
                        shot_start = record_index * 100_000_000
                        record = P3ObservationRecord(
                            split=split,
                            deficit=deficit,
                            gamma=gamma,
                            instance_index=instance_index,
                            replicate=replicate,
                            exact_table=exact,
                            support=sample_shot_batch(
                                exact,
                                support_contexts,
                                n_shots=n_shot,
                                rng=rng,
                                shot_start=shot_start,
                                role="p3_support",
                            ),
                            query_shot=sample_shot_batch(
                                exact,
                                support_contexts,
                                n_shots=n_shot,
                                rng=rng,
                                shot_start=shot_start + 20_000_000,
                                role="p3_query_shot",
                            ),
                            query_comp=sample_shot_batch(
                                exact,
                                grouped["compositional_heldout"],
                                n_shots=n_shot,
                                rng=rng,
                                shot_start=shot_start + 40_000_000,
                                role="p3_query_comp",
                            ),
                            query_op=sample_shot_batch(
                                exact,
                                grouped["operator_heldout"],
                                n_shots=n_shot,
                                rng=rng,
                                shot_start=shot_start + 60_000_000,
                                role="p3_query_op",
                            ),
                            metadata={
                                "record_index": record_index,
                                "n_shot": n_shot,
                                "model_c_coherent_core": gamma == 1.0,
                            },
                        )
                        if not p3_target_split_valid(record):
                            raise AssertionError("P3 support and target indices overlap")
                        records.append(record)
                        record_index += 1
    return records


def filter_p3_records(
    records: list[P3ObservationRecord],
    *,
    split: str | None = None,
    gamma: float | None = None,
    deficit: str | None = None,
) -> list[P3ObservationRecord]:
    return [
        record
        for record in records
        if (split is None or record.split == split)
        and (gamma is None or record.gamma == gamma)
        and (deficit is None or record.deficit == deficit)
    ]
