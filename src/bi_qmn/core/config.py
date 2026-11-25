from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class P0Config:
    generator: dict[str, Any]
    measurement: dict[str, Any]
    seeds: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class P1Config:
    generator: dict[str, Any]
    measurement: dict[str, Any]
    seeds: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class P2Config:
    generator: dict[str, Any]
    measurement: dict[str, Any]
    seeds: dict[str, Any]
    p2: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class P0ILGCConfig:
    values: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class P0ILGCOperationalConfig:
    values: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class P1ILGCOperationalConfig:
    values: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class P3ModelCConfig:
    values: dict[str, Any]
    config_hash: str


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return data


def load_p0_config(config_dir: str | Path = "config") -> P0Config:
    root = Path(config_dir)
    generator = _read_yaml(root / "p0_generator.yaml")
    measurement = _read_yaml(root / "p0_measurement.yaml")
    seeds = _read_yaml(root / "p0_seeds.yaml")
    payload = {"generator": generator, "measurement": measurement, "seeds": seeds}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return P0Config(
        generator=generator,
        measurement=measurement,
        seeds=seeds,
        config_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )


def load_p1_config(config_dir: str | Path = "config") -> P1Config:
    root = Path(config_dir)
    generator = _read_yaml(root / "p1_generator.yaml")
    measurement = _read_yaml(root / "p1_measurement.yaml")
    seeds = _read_yaml(root / "p1_seeds.yaml")
    payload = {"generator": generator, "measurement": measurement, "seeds": seeds}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return P1Config(
        generator=generator,
        measurement=measurement,
        seeds=seeds,
        config_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )


def load_p2_config(config_dir: str | Path = "config") -> P2Config:
    root = Path(config_dir)
    generator = _read_yaml(root / "p1_generator.yaml")
    measurement = _read_yaml(root / "p1_measurement.yaml")
    seeds = _read_yaml(root / "p1_seeds.yaml")
    p2 = _read_yaml(root / "p2.yaml")
    payload = {"generator": generator, "measurement": measurement, "seeds": seeds, "p2": p2}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return P2Config(
        generator=generator,
        measurement=measurement,
        seeds=seeds,
        p2=p2,
        config_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )


def load_p0_ilgc_config(config_dir: str | Path = "config") -> P0ILGCConfig:
    root = Path(config_dir)
    values = _read_yaml(root / "p0_ilgc.yaml")
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return P0ILGCConfig(
        values=values,
        config_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )


def load_p0_ilgc_operational_config(
    config_dir: str | Path = "config",
) -> P0ILGCOperationalConfig:
    root = Path(config_dir)
    values = _read_yaml(root / "p0_ilgc_operational.yaml")
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return P0ILGCOperationalConfig(
        values=values,
        config_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )


def load_p1_ilgc_operational_config(
    config_dir: str | Path = "config",
    filename: str = "p1_ilgc_operational.yaml",
) -> P1ILGCOperationalConfig:
    root = Path(config_dir)
    values = _read_yaml(root / filename)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return P1ILGCOperationalConfig(
        values=values,
        config_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )


def load_p3_model_c_config(config_dir: str | Path = "config") -> P3ModelCConfig:
    root = Path(config_dir)
    values = _read_yaml(root / "p3_model_c.yaml")
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return P3ModelCConfig(
        values=values,
        config_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )


def output_path(output_dir: str | Path, name: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / name
