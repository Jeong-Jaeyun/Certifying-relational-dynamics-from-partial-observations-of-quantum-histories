from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .types import stable_hash


ATLAS_METRICS = (
    "p_min",
    "theta_align",
    "theta_wit",
    "theta_tr",
    "theta_dir",
)
LOWER_METRICS = {"p_min"}


@dataclass(frozen=True)
class AtlasCalibrationSample:
    stratum: str
    construction_block: int
    values: Mapping[str, float]

    def validate(self) -> None:
        if not self.stratum:
            raise ValueError("atlas calibration sample requires a stratum")
        if self.construction_block < 0:
            raise ValueError("atlas calibration construction block must be nonnegative")
        if set(self.values) != set(ATLAS_METRICS):
            raise ValueError("atlas calibration sample has an unexpected metric set")
        for name, value in self.values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"atlas calibration metric {name} must be finite")
        if not 0.0 <= float(self.values["p_min"]) <= 1.0:
            raise ValueError("atlas calibration support metric must be in [0, 1]")
        if any(float(self.values[name]) < 0.0 for name in ATLAS_METRICS[1:]):
            raise ValueError("atlas calibration upper metrics must be nonnegative")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "construction_block": self.construction_block,
            "values": {name: float(self.values[name]) for name in ATLAS_METRICS},
        }


@dataclass(frozen=True)
class AtlasThresholdCell:
    stratum: str
    thresholds: Mapping[str, float]
    means: Mapping[str, float]
    standard_deviations: Mapping[str, float]
    block_count: int
    qualified: bool
    qualification_reasons: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "thresholds": {name: float(self.thresholds[name]) for name in ATLAS_METRICS},
            "means": {name: float(self.means[name]) for name in ATLAS_METRICS},
            "standard_deviations": {
                name: float(self.standard_deviations[name]) for name in ATLAS_METRICS
            },
            "construction_block_count": self.block_count,
            "qualified": self.qualified,
            "qualification_reasons": list(self.qualification_reasons),
        }


@dataclass(frozen=True)
class AtlasThresholdBank:
    method: str
    minimum_construction_blocks: int
    kappa: Mapping[str, float]
    cells: Mapping[str, AtlasThresholdCell]
    samples: tuple[AtlasCalibrationSample, ...]
    calibration_manifest_hash: str

    @property
    def qualified(self) -> bool:
        return bool(self.cells) and all(cell.qualified for cell in self.cells.values())

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.to_manifest())

    def thresholds_for_stratum(self, stratum: str) -> dict[str, float]:
        cell = self.cells.get(stratum)
        if cell is None:
            raise KeyError(f"missing atlas threshold cell for {stratum}")
        if not cell.qualified:
            raise ValueError(f"atlas threshold cell for {stratum} is unqualified")
        return {name: float(cell.thresholds[name]) for name in ATLAS_METRICS}

    def to_manifest(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "minimum_construction_blocks": self.minimum_construction_blocks,
            "kappa": {name: float(self.kappa[name]) for name in ATLAS_METRICS},
            "calibration_manifest_hash": self.calibration_manifest_hash,
            "qualified": self.qualified,
            "sample_hash": stable_hash([sample.to_manifest() for sample in self.samples]),
            "cells": {
                stratum: self.cells[stratum].to_manifest()
                for stratum in sorted(self.cells)
            },
            "samples": [sample.to_manifest() for sample in self.samples],
        }


def fit_atlas_threshold_bank(
    samples: Iterable[AtlasCalibrationSample],
    config: Mapping[str, Any],
    *,
    calibration_manifest_hash: str,
) -> AtlasThresholdBank:
    """Freeze per-stratum atlas gates from construction-block calibration samples.

    The lower support gate uses ``mean - kappa * sample_std``; all failure metrics
    use ``mean + kappa * sample_std``.  A construction block contributes exactly one
    vector, so multiple records cannot silently inflate the apparent calibration size.
    """
    method = str(config.get("method", ""))
    if method != "construction_seed_blocked_mean_std_v1":
        raise ValueError("unsupported atlas calibration method")
    minimum_blocks = int(config.get("minimum_construction_blocks", 0))
    raw_kappa = config.get("kappa")
    if minimum_blocks <= 1 or not isinstance(raw_kappa, Mapping):
        raise ValueError("invalid atlas calibration configuration")
    kappa = {name: float(raw_kappa[name]) for name in ATLAS_METRICS}
    if any(not math.isfinite(value) or value < 0.0 for value in kappa.values()):
        raise ValueError("atlas calibration kappas must be finite and nonnegative")
    if not calibration_manifest_hash:
        raise ValueError("atlas calibration requires a body calibration manifest hash")

    ordered = tuple(
        sorted(samples, key=lambda sample: (sample.stratum, sample.construction_block))
    )
    grouped: dict[str, dict[int, AtlasCalibrationSample]] = defaultdict(dict)
    for sample in ordered:
        sample.validate()
        if sample.construction_block in grouped[sample.stratum]:
            raise ValueError("atlas calibration received duplicate construction blocks")
        grouped[sample.stratum][sample.construction_block] = sample

    cells: dict[str, AtlasThresholdCell] = {}
    for stratum, by_block in sorted(grouped.items()):
        block_samples = list(by_block.values())
        reasons: list[str] = []
        if len(block_samples) < minimum_blocks:
            reasons.append("insufficient_construction_blocks")
        means: dict[str, float] = {}
        deviations: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        for name in ATLAS_METRICS:
            values = np.asarray([sample.values[name] for sample in block_samples], dtype=float)
            mean = float(np.mean(values))
            deviation = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
            if not math.isfinite(mean) or not math.isfinite(deviation):
                reasons.append(f"nonfinite_{name}_summary")
                mean = 0.0
                deviation = 0.0
            means[name] = mean
            deviations[name] = deviation
            if name in LOWER_METRICS:
                thresholds[name] = float(np.clip(mean - kappa[name] * deviation, 0.0, 1.0))
            else:
                thresholds[name] = max(0.0, mean + kappa[name] * deviation)
        cells[stratum] = AtlasThresholdCell(
            stratum=stratum,
            thresholds=thresholds,
            means=means,
            standard_deviations=deviations,
            block_count=len(block_samples),
            qualified=not reasons,
            qualification_reasons=tuple(sorted(set(reasons))),
        )
    return AtlasThresholdBank(
        method=method,
        minimum_construction_blocks=minimum_blocks,
        kappa=kappa,
        cells=cells,
        samples=ordered,
        calibration_manifest_hash=calibration_manifest_hash,
    )
