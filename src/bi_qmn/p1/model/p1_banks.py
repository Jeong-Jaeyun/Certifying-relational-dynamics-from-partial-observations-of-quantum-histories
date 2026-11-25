from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from bi_qmn.core.observations import ShotBatch, sample_shot_batch
from bi_qmn.p1.model.p1_generator import (
    P1Instance,
    clock_label_dephasing_control,
    construction_bank_p1,
    p1_construction_fingerprint,
)
from bi_qmn.p1.model.p1_measurement import contexts_from_p1_measurement_config, full_p1_context_list, p1_measurement_table
from bi_qmn.core.types import MeasurementContext, MeasurementTable


@dataclass(frozen=True)
class P1ObservationRecord:
    split: str
    deficit: str
    gamma: float
    instance_index: int
    replicate: int
    exact_table: MeasurementTable
    support: ShotBatch
    query: ShotBatch
    metadata: dict


def support_contexts_for_deficit(
    measurement_config: dict,
    deficit: str,
    seed_key: str,
) -> list[MeasurementContext]:
    grouped = contexts_from_p1_measurement_config(measurement_config)
    deficit_config = measurement_config["observation_deficit"][deficit]
    diag_contexts = [context for context in grouped["input"] if context.clock == "D"]
    coh_contexts = [context for context in grouped["input"] if context.clock == "F"]
    return _select_contexts(diag_contexts, float(deficit_config["cov_diag"]), seed_key + ":diag") + _select_contexts(
        coh_contexts,
        float(deficit_config["cov_coh"]),
        seed_key + ":coh",
    )


def build_p1_observation_records(
    generator_config: dict,
    measurement_config: dict,
    seeds: dict,
    include_negative_controls: bool = True,
) -> list[P1ObservationRecord]:
    bank = construction_bank_p1(generator_config)
    contexts = full_p1_context_list(measurement_config)
    splits = measurement_config["p1_b"]["splits"]
    split_indices = {
        "train": [int(v) for v in splits["train_indices"]],
        "calibration": [int(v) for v in splits["calibration_indices"]],
        "test": [int(v) for v in splits["test_indices"]],
    }
    split_replicates = {
        "train": int(splits["train_replicates"]),
        "calibration": int(splits["calibration_replicates"]),
        "test": int(splits["test_replicates"]),
    }
    gammas = [float(measurement_config["noise"]["calibration_gamma"])]
    if include_negative_controls:
        gammas.extend(float(v) for v in measurement_config["noise"]["negative_control_gamma"])

    records: list[P1ObservationRecord] = []
    record_index = 0
    for gamma in gammas:
        for deficit in measurement_config["observation_deficit"]:
            n_shot = int(measurement_config["observation_deficit"][deficit]["n_shot"])
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
                        seed_key=f"{split}:{deficit}:{gamma}:{instance_index}",
                    )
                    for replicate in range(split_replicates[split]):
                        rng = np.random.default_rng(
                            int(seeds.get("shot_seed", 0))
                            + 100_000 * record_index
                            + 1_009 * replicate
                            + 97 * instance_index
                        )
                        shot_start = record_index * 100_000_000
                        support = sample_shot_batch(
                            exact,
                            support_contexts,
                            n_shots=n_shot,
                            rng=rng,
                            shot_start=shot_start,
                            role="support",
                        )
                        query = sample_shot_batch(
                            exact,
                            contexts,
                            n_shots=n_shot,
                            rng=rng,
                            shot_start=shot_start + 50_000_000,
                            role="query",
                        )
                        records.append(
                            P1ObservationRecord(
                                split=split,
                                deficit=deficit,
                                gamma=gamma,
                                instance_index=instance_index,
                                replicate=replicate,
                                exact_table=exact,
                                support=support,
                                query=query,
                                metadata={"record_index": record_index, "n_shot": n_shot},
                            )
                        )
                        record_index += 1
    return records


def build_p1_records_for_construction_bank(
    instances: list[P1Instance],
    measurement_config: dict,
    seeds: dict,
    *,
    split: str,
    bank_id: str,
    replicates: int = 1,
    gamma: float = 1.0,
    record_offset: int = 0,
) -> list[P1ObservationRecord]:
    """Build one disjoint observation split for an explicit construction bank."""
    if not instances or replicates <= 0:
        raise ValueError("explicit P1 observation bank requires instances and replicates")
    contexts = full_p1_context_list(measurement_config)
    records: list[P1ObservationRecord] = []
    record_index = int(record_offset)
    for deficit in measurement_config["observation_deficit"]:
        n_shot = int(measurement_config["observation_deficit"][deficit]["n_shot"])
        for instance_index, instance in enumerate(instances):
            rho = instance.density if gamma == 1.0 else clock_label_dephasing_control(
                instance.density,
                gamma=gamma,
                K=instance.K,
                d_S=instance.d_S,
            )
            fingerprint = p1_construction_fingerprint(instance)
            exact = p1_measurement_table(
                rho,
                contexts,
                K=instance.K,
                eta=float(measurement_config["noise"]["readout_eta"]),
                metadata={
                    "instance": instance.label,
                    "instance_index": instance_index,
                    "construction_fingerprint": fingerprint,
                    "construction_bank_id": bank_id,
                    "gamma": gamma,
                    "deficit": deficit,
                    "eta": float(measurement_config["noise"]["readout_eta"]),
                },
            )
            support_contexts = support_contexts_for_deficit(
                measurement_config,
                deficit,
                seed_key=f"{bank_id}:{split}:{deficit}:{gamma}:{fingerprint}",
            )
            for replicate in range(replicates):
                rng = np.random.default_rng(
                    int(seeds.get("shot_seed", 0))
                    + 100_000 * record_index
                    + 1_009 * replicate
                    + 97 * instance_index
                )
                shot_start = record_index * 100_000_000
                support = sample_shot_batch(
                    exact,
                    support_contexts,
                    n_shots=n_shot,
                    rng=rng,
                    shot_start=shot_start,
                    role="support",
                )
                query = sample_shot_batch(
                    exact,
                    contexts,
                    n_shots=n_shot,
                    rng=rng,
                    shot_start=shot_start + 50_000_000,
                    role="query",
                )
                records.append(
                    P1ObservationRecord(
                        split=split,
                        deficit=deficit,
                        gamma=gamma,
                        instance_index=instance_index,
                        replicate=replicate,
                        exact_table=exact,
                        support=support,
                        query=query,
                        metadata={
                            "record_index": record_index,
                            "n_shot": n_shot,
                            "construction_bank_id": bank_id,
                            "construction_fingerprint": fingerprint,
                        },
                    )
                )
                record_index += 1
    return records


def filter_records(
    records: list[P1ObservationRecord],
    *,
    split: str | None = None,
    gamma: float | None = None,
    deficit: str | None = None,
) -> list[P1ObservationRecord]:
    out = records
    if split is not None:
        out = [record for record in out if record.split == split]
    if gamma is not None:
        out = [record for record in out if record.gamma == gamma]
    if deficit is not None:
        out = [record for record in out if record.deficit == deficit]
    return out


def p1_banks_hash(records: list[P1ObservationRecord]) -> str:
    payload = "|".join(
        f"{r.split}:{r.deficit}:{r.gamma}:{r.instance_index}:{r.replicate}:{len(r.support.table.contexts)}"
        for r in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _select_contexts(contexts: list[MeasurementContext], coverage: float, seed_key: str) -> list[MeasurementContext]:
    if coverage >= 1.0:
        return list(contexts)
    if coverage <= 0.0:
        return []
    n_keep = max(1, int(round(len(contexts) * coverage)))
    rng = np.random.default_rng(int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16))
    indices = sorted(rng.choice(len(contexts), size=n_keep, replace=False).tolist())
    return [contexts[index] for index in indices]
