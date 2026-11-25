from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, Statevector

from bi_qmn.core.generator import constraint_residual
from bi_qmn.core.linalg import density_from_statevector, fourier_matrix, kron, matrix_projector, qiskit_statevector


@dataclass(frozen=True)
class P1Instance:
    coefficient_label: str
    phase_label: str
    system_unitary_label: str
    active_modes: tuple[int, ...]
    coefficients: tuple[complex, ...]
    K: int
    d_C: int
    d_S: int
    omega: tuple[float, ...]
    system_unitary: np.ndarray
    statevector: Statevector
    density: DensityMatrix
    H_C: Operator
    H_S: Operator
    H_tot: Operator
    h_scale: float

    @property
    def label(self) -> str:
        return (
            f"coeff={self.coefficient_label};phase={self.phase_label};"
            f"U={self.system_unitary_label};R={','.join(str(r) for r in self.active_modes)}"
        )


def _cs_index(clock_index: int, system_index: int, d_S: int) -> int:
    return clock_index * d_S + system_index


def _single_qubit_h() -> np.ndarray:
    return np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)


def _single_qubit_ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.asarray([[c, -s], [s, c]], dtype=complex)


def _cnot_01() -> np.ndarray:
    data = np.eye(4, dtype=complex)
    data[2, 2] = 0.0
    data[3, 3] = 0.0
    data[2, 3] = 1.0
    data[3, 2] = 1.0
    return data


def system_unitary_from_config(item: dict) -> tuple[str, np.ndarray]:
    label = str(item["label"])
    kind = str(item["kind"])
    I = np.eye(2, dtype=complex)
    if kind == "identity":
        data = np.eye(4, dtype=complex)
    elif kind == "h0":
        data = kron(_single_qubit_h(), I)
    elif kind == "cnot_h0":
        data = _cnot_01() @ kron(_single_qubit_h(), I)
    elif kind == "cnot_ry":
        alpha = float(item["alpha"])
        beta = float(item["beta"])
        data = _cnot_01() @ kron(_single_qubit_ry(alpha), _single_qubit_ry(beta))
    else:
        raise ValueError(f"unknown P1 system unitary kind {kind}")
    if not np.allclose(data.conj().T @ data, np.eye(4, dtype=complex), atol=1.0e-10):
        raise ValueError(f"P1 system unitary {label} is not unitary")
    return label, data


def _coefficients(probabilities: Iterable[float], phases: Iterable[float]) -> tuple[complex, ...]:
    probs = tuple(float(v) for v in probabilities)
    phase_values = tuple(float(v) for v in phases)
    if len(probs) != len(phase_values):
        raise ValueError("coefficient probabilities and phases must have equal length")
    if any(v <= 0.0 for v in probs):
        raise ValueError("active-mode probabilities must be positive")
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("coefficient probabilities must have positive mass")
    return tuple(math.sqrt(p / total) * np.exp(1j * phi) for p, phi in zip(probs, phase_values, strict=True))


def build_p1_instance(
    coefficient_label: str,
    probabilities: Iterable[float],
    phase_label: str,
    phases: Iterable[float],
    system_unitary_label: str,
    system_unitary: np.ndarray,
    active_modes: Iterable[int],
    omega: Iterable[float],
    d_C: int = 8,
    d_S: int = 4,
    K: int = 8,
) -> P1Instance:
    modes = tuple(int(v) for v in active_modes)
    if len(modes) < 3:
        raise ValueError("P1 requires at least three active modes")
    if len(modes) > d_S:
        raise ValueError("P1 active mode count cannot exceed d_S in the v1 construction")
    if len(set(modes)) != len(modes):
        raise ValueError("active modes must be distinct")
    omega_tuple = tuple(float(v) for v in omega)
    if len(omega_tuple) != d_C:
        raise ValueError("omega must have d_C entries")

    coeffs = _coefficients(probabilities, phases)
    if len(coeffs) != len(modes):
        raise ValueError("coefficient profile length must match active mode count")

    U = np.asarray(system_unitary, dtype=complex)
    if U.shape != (d_S, d_S):
        raise ValueError("P1 system unitary shape must be d_S x d_S")

    amplitudes = np.zeros(d_C * d_S, dtype=complex)
    for j, (mode, coeff) in enumerate(zip(modes, coeffs, strict=True)):
        amplitudes[mode * d_S : (mode + 1) * d_S] += coeff * U[:, j]
    statevector = qiskit_statevector(amplitudes)
    density = density_from_statevector(statevector)

    H_C_data = np.diag(omega_tuple).astype(complex)
    system_eigs = np.zeros(d_S, dtype=float)
    for j, mode in enumerate(modes):
        system_eigs[j] = -omega_tuple[mode]
    unused = [r for r in range(d_C) if r not in modes]
    for j in range(len(modes), d_S):
        system_eigs[j] = -omega_tuple[unused[j - len(modes)]]
    H_S_data = U @ np.diag(system_eigs).astype(complex) @ U.conj().T
    H_tot_data = kron(H_C_data, np.eye(d_S, dtype=complex)) + kron(np.eye(d_C, dtype=complex), H_S_data)
    h_scale = float(math.sqrt(max(1.0e-30, np.trace(H_tot_data @ H_tot_data).real / (d_C * d_S))))

    return P1Instance(
        coefficient_label=str(coefficient_label),
        phase_label=str(phase_label),
        system_unitary_label=str(system_unitary_label),
        active_modes=modes,
        coefficients=coeffs,
        K=int(K),
        d_C=int(d_C),
        d_S=int(d_S),
        omega=omega_tuple,
        system_unitary=U,
        statevector=statevector,
        density=density,
        H_C=Operator(H_C_data),
        H_S=Operator(H_S_data),
        H_tot=Operator(H_tot_data),
        h_scale=h_scale,
    )


def construction_bank_p1(generator_config: dict) -> list[P1Instance]:
    dims = generator_config["dimensions"]
    grid = generator_config["construction_grid"]
    unitaries = [system_unitary_from_config(item) for item in grid["system_unitaries"]]
    instances: list[P1Instance] = []
    for coeff_profile in grid["coefficient_profiles"]:
        for phase_profile in grid["phase_profiles"]:
            for unitary_label, unitary in unitaries:
                instances.append(
                    build_p1_instance(
                        coefficient_label=str(coeff_profile["label"]),
                        probabilities=coeff_profile["probabilities"],
                        phase_label=str(phase_profile["label"]),
                        phases=phase_profile["phases"],
                        system_unitary_label=unitary_label,
                        system_unitary=unitary,
                        active_modes=generator_config["active_modes"],
                        omega=generator_config["omega"],
                        d_C=int(dims["d_C"]),
                        d_S=int(dims["d_S"]),
                        K=int(dims["K"]),
                    )
                )
    return instances


def p1_construction_fingerprint(instance: P1Instance) -> str:
    digest = hashlib.sha256()
    digest.update(instance.label.encode("utf-8"))
    for value in (
        np.asarray(instance.coefficients, dtype=np.complex128),
        np.asarray(instance.system_unitary, dtype=np.complex128),
        np.asarray(instance.H_tot.data, dtype=np.complex128),
    ):
        digest.update(np.ascontiguousarray(value).view(np.float64).tobytes())
    return digest.hexdigest()[:16]


def seeded_construction_bank_p1(
    generator_config: dict,
    bank_spec: Mapping[str, object],
) -> list[P1Instance]:
    """Draw a reproducible P1 bank from the registered coefficient/phase family."""
    required = {"bank_id", "seed", "block_count", "dirichlet_concentration"}
    missing = required - set(bank_spec)
    if missing:
        raise ValueError(f"P1 seeded bank specification is missing {sorted(missing)}")
    block_count = int(bank_spec["block_count"])
    concentration = float(bank_spec["dirichlet_concentration"])
    if block_count <= 0 or concentration <= 0.0:
        raise ValueError("P1 seeded bank requires positive block count and concentration")

    rng = np.random.default_rng(int(bank_spec["seed"]))
    grid = generator_config["construction_grid"]
    unitaries = [system_unitary_from_config(item) for item in grid["system_unitaries"]]
    modes = tuple(int(value) for value in generator_config["active_modes"])
    min_probability = float(
        bank_spec.get(
            "minimum_mode_probability",
            generator_config["thresholds"]["c_mode"],
        )
    )
    phase_range = tuple(float(value) for value in bank_spec.get("phase_range", [0.0, 2.0 * math.pi]))
    if len(phase_range) != 2 or phase_range[0] >= phase_range[1]:
        raise ValueError("P1 seeded bank phase_range must be increasing")
    if min_probability * len(modes) >= 1.0:
        raise ValueError("P1 seeded bank minimum mode probability is infeasible")

    instances: list[P1Instance] = []
    for index in range(block_count):
        probabilities = None
        for _ in range(10000):
            proposal = rng.dirichlet(np.full(len(modes), concentration, dtype=float))
            if float(np.min(proposal)) >= min_probability:
                probabilities = proposal
                break
        if probabilities is None:
            raise RuntimeError("failed to sample a P1 coefficient profile above c_mode")
        phases = np.zeros(len(modes), dtype=float)
        phases[1:] = rng.uniform(phase_range[0], phase_range[1], size=len(modes) - 1)
        unitary_label, unitary = unitaries[int(rng.integers(0, len(unitaries)))]
        instances.append(
            build_p1_instance(
                coefficient_label=f"{bank_spec['bank_id']}:coeff:{index}",
                probabilities=probabilities,
                phase_label=f"{bank_spec['bank_id']}:phase:{index}",
                phases=phases,
                system_unitary_label=unitary_label,
                system_unitary=unitary,
                active_modes=modes,
                omega=generator_config["omega"],
                d_C=int(generator_config["dimensions"]["d_C"]),
                d_S=int(generator_config["dimensions"]["d_S"]),
                K=int(generator_config["dimensions"]["K"]),
            )
        )
    fingerprints = [p1_construction_fingerprint(instance) for instance in instances]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("P1 seeded construction bank contains duplicate fingerprints")
    return instances


def anti_aliasing_margin_p1(instance: P1Instance, c_mode: float) -> dict[str, float | bool]:
    differences = [
        abs(r1 - r0)
        for index, r0 in enumerate(instance.active_modes)
        for r1 in instance.active_modes[index + 1 :]
    ]
    gcd_value = instance.K
    for diff in differences:
        gcd_value = math.gcd(gcd_value, diff)
    mode_probs = [abs(c) ** 2 for c in instance.coefficients]
    min_mode_probability = float(min(mode_probs))
    return {
        "gcd_ok": gcd_value == 1,
        "gcd_value": gcd_value,
        "mode_probability_ok": min_mode_probability >= c_mode,
        "min_mode_probability": min_mode_probability,
        "margin": min_mode_probability - c_mode,
    }


def validate_p1_construction_bank(instances: Iterable[P1Instance], eps_num: float, c_mode: float) -> dict[str, object]:
    records = []
    for instance in instances:
        residual = constraint_residual(instance.density, instance.H_tot, instance.h_scale)
        alias = anti_aliasing_margin_p1(instance, c_mode)
        records.append(
            {
                "label": instance.label,
                "constraint_residual": residual,
                **alias,
                "pass": residual <= eps_num and bool(alias["gcd_ok"]) and bool(alias["mode_probability_ok"]),
            }
        )
    return {
        "count": len(records),
        "max_constraint_residual": max((float(r["constraint_residual"]) for r in records), default=float("inf")),
        "min_mode_margin": min((float(r["margin"]) for r in records), default=float("-inf")),
        "pass": all(bool(r["pass"]) for r in records),
        "records": records,
    }


def clock_label_dephase(rho: DensityMatrix, K: int = 8, d_S: int = 4) -> DensityMatrix:
    """Apply full dephasing in the relational clock-label basis."""
    F = fourier_matrix(K)
    rho_data = np.asarray(rho.data, dtype=complex)
    out = np.zeros_like(rho_data)
    for tau in range(K):
        P_C = matrix_projector(F[tau, :])
        P = kron(P_C, np.eye(d_S, dtype=complex))
        out += P @ rho_data @ P
    return DensityMatrix((out + out.conj().T) / 2.0)


def clock_label_dephasing_control(rho: DensityMatrix, gamma: float, K: int = 8, d_S: int = 4) -> DensityMatrix:
    """Post-construction negative control, not a coherent-core generator."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    dephased = clock_label_dephase(rho, K=K, d_S=d_S)
    data = gamma * np.asarray(rho.data, dtype=complex) + (1.0 - gamma) * np.asarray(dephased.data, dtype=complex)
    return DensityMatrix((data + data.conj().T) / 2.0)


def dephasing_residual_audit(instances: Iterable[P1Instance], gammas: Iterable[float]) -> dict[str, object]:
    records = []
    by_gamma: dict[str, list[float]] = {str(float(gamma)): [] for gamma in gammas}
    for instance in instances:
        clean = constraint_residual(instance.density, instance.H_tot, instance.h_scale)
        for gamma in gammas:
            controlled = clock_label_dephasing_control(instance.density, float(gamma), K=instance.K, d_S=instance.d_S)
            residual = constraint_residual(controlled, instance.H_tot, instance.h_scale)
            by_gamma[str(float(gamma))].append(residual)
            records.append(
                {
                    "label": instance.label,
                    "gamma": float(gamma),
                    "epsilon_con": residual,
                    "clean_epsilon_con": clean,
                    "delta_epsilon_con": residual - clean,
                }
            )
    return {
        "role": "post_construction_negative_control_audit",
        "threshold_source": "excluded_from_p1_calibration",
        "by_gamma": {
            gamma: {
                "count": len(values),
                "min": float(min(values)) if values else None,
                "median": float(np.median(np.asarray(values, dtype=float))) if values else None,
                "max": float(max(values)) if values else None,
            }
            for gamma, values in by_gamma.items()
        },
        "records": records,
    }
