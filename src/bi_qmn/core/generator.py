from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import json
from typing import Any, Iterable, Mapping

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, Statevector

from bi_qmn.core.linalg import density_from_statevector, qiskit_statevector


@dataclass(frozen=True)
class P0Instance:
    alpha: float
    phi: float
    active_modes: tuple[int, int]
    K: int
    d_C: int
    d_S: int
    omega: tuple[float, ...]
    statevector: Statevector
    density: DensityMatrix
    H_C: Operator
    H_S: Operator
    H_tot: Operator
    h_scale: float

    @property
    def label(self) -> str:
        return f"alpha={self.alpha:.12g};phi={self.phi:.12g}"


def _cs_index(clock_index: int, system_index: int, d_S: int) -> int:
    return clock_index * d_S + system_index


def build_p0_instance(
    alpha: float,
    phi: float,
    active_modes: Iterable[int],
    omega: Iterable[float],
    d_C: int = 4,
    d_S: int = 2,
    K: int = 4,
) -> P0Instance:
    modes = tuple(int(v) for v in active_modes)
    if len(modes) != 2:
        raise ValueError("P0 v0 requires exactly two active modes")
    omega_tuple = tuple(float(v) for v in omega)
    if len(omega_tuple) != d_C:
        raise ValueError("omega must have d_C entries")

    amplitudes = np.zeros(d_C * d_S, dtype=complex)
    amplitudes[_cs_index(modes[0], 0, d_S)] = math.cos(alpha)
    amplitudes[_cs_index(modes[1], 1, d_S)] = np.exp(1j * phi) * math.sin(alpha)
    statevector = qiskit_statevector(amplitudes)
    density = density_from_statevector(statevector)

    H_C_data = np.diag(omega_tuple).astype(complex)
    H_S_data = np.diag([-omega_tuple[modes[0]], -omega_tuple[modes[1]]]).astype(complex)
    H_tot_data = np.kron(H_C_data, np.eye(d_S)) + np.kron(np.eye(d_C), H_S_data)
    nonzero_scale = max(1.0, float(np.max(np.abs(np.linalg.eigvalsh(H_tot_data)))))

    return P0Instance(
        alpha=float(alpha),
        phi=float(phi),
        active_modes=(modes[0], modes[1]),
        K=int(K),
        d_C=int(d_C),
        d_S=int(d_S),
        omega=omega_tuple,
        statevector=statevector,
        density=density,
        H_C=Operator(H_C_data),
        H_S=Operator(H_S_data),
        H_tot=Operator(H_tot_data),
        h_scale=nonzero_scale,
    )


def construction_bank(generator_config: dict) -> list[P0Instance]:
    seed_spec = generator_config.get("seeded_construction_bank")
    if seed_spec is not None:
        return seeded_construction_bank(generator_config, seed_spec)
    dims = generator_config["dimensions"]
    grid = generator_config["construction_grid"]
    instances: list[P0Instance] = []
    for alpha in grid["alpha"]:
        for phi in grid["phi"]:
            instances.append(
                build_p0_instance(
                    alpha=float(alpha),
                    phi=float(phi),
                    active_modes=generator_config["active_modes"],
                    omega=generator_config["omega"],
                    d_C=int(dims["d_C"]),
                    d_S=int(dims["d_S"]),
                    K=int(dims["K"]),
                )
            )
    return instances


def seeded_construction_bank(
    generator_config: Mapping[str, Any],
    seed_spec: Mapping[str, Any],
) -> list[P0Instance]:
    """Build a reproducible, seed-indexed bank without changing the declared grid."""
    if seed_spec.get("method") != "seeded_uniform_alpha_phi_v1":
        raise ValueError("unsupported seeded construction-bank method")
    count = int(seed_spec["block_count"])
    seed = int(seed_spec["seed"])
    alpha_range = tuple(float(value) for value in seed_spec["alpha_range"])
    phi_range = tuple(float(value) for value in seed_spec["phi_range"])
    if count <= 0 or seed < 0:
        raise ValueError("seeded construction bank requires a positive count and seed")
    if len(alpha_range) != 2 or len(phi_range) != 2:
        raise ValueError("seeded construction bank ranges must have two endpoints")
    if not 0.0 < alpha_range[0] < alpha_range[1] < math.pi / 2.0:
        raise ValueError("seeded construction alpha range must lie in (0, pi/2)")
    if not 0.0 <= phi_range[0] < phi_range[1] <= 2.0 * math.pi:
        raise ValueError("seeded construction phi range must lie in [0, 2pi]")

    dims = generator_config["dimensions"]
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(alpha_range[0], alpha_range[1], size=count)
    phis = rng.uniform(phi_range[0], phi_range[1], size=count)
    return [
        build_p0_instance(
            alpha=float(alpha),
            phi=float(phi),
            active_modes=generator_config["active_modes"],
            omega=generator_config["omega"],
            d_C=int(dims["d_C"]),
            d_S=int(dims["d_S"]),
            K=int(dims["K"]),
        )
        for alpha, phi in zip(alphas, phis, strict=True)
    ]


def construction_fingerprint(instance: P0Instance) -> str:
    payload = {
        "alpha": instance.alpha,
        "phi": instance.phi,
        "active_modes": instance.active_modes,
        "omega": instance.omega,
        "d_C": instance.d_C,
        "d_S": instance.d_S,
        "K": instance.K,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def constraint_residual(rho: DensityMatrix, H_tot: Operator, h_scale: float) -> float:
    H = np.asarray(H_tot.data, dtype=complex)
    H2 = H @ H
    value = np.trace(np.asarray(rho.data, dtype=complex) @ H2) / (h_scale * h_scale)
    return float(np.real_if_close(value, tol=1000))


def anti_aliasing_margin(instance: P0Instance, c_amp: float) -> dict[str, float | bool]:
    r0, r1 = instance.active_modes
    gcd_ok = math.gcd(abs(r1 - r0), instance.K) == 1
    amp = abs(math.cos(instance.alpha) * math.sin(instance.alpha))
    return {
        "gcd_ok": gcd_ok,
        "amplitude_ok": amp >= c_amp,
        "amplitude": amp,
        "margin": amp - c_amp,
    }


def validate_static_anti_aliasing(instances: Iterable[P0Instance], c_amp: float) -> dict[str, object]:
    records = [anti_aliasing_margin(instance, c_amp) for instance in instances]
    return {
        "gcd_ok": all(bool(r["gcd_ok"]) for r in records),
        "amplitude_ok": all(bool(r["amplitude_ok"]) for r in records),
        "min_margin": min(float(r["margin"]) for r in records),
        "records": records,
    }
