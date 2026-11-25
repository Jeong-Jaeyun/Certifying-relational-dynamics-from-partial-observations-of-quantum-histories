from __future__ import annotations


def apply_readout_noise(
    distribution: dict[str, float],
    K: int,
    eta: float,
) -> dict[str, float]:
    """Apply independent symmetric clock-label and system-bit readout noise."""
    if eta <= 0.0:
        return {k: float(v) for k, v in distribution.items()}
    if not 0.0 <= eta < 1.0:
        raise ValueError("readout eta must be in [0, 1)")

    noisy = {key: 0.0 for key in distribution}
    for key, prob in distribution.items():
        clock_label, system_bit = parse_outcome_key(key)
        for out_clock in range(K):
            if out_clock == clock_label:
                p_clock = 1.0 - eta
            else:
                p_clock = eta / (K - 1)
            for out_system in (0, 1):
                p_system = (1.0 - eta) if out_system == system_bit else eta
                noisy[outcome_key(out_clock, out_system)] += prob * p_clock * p_system
    return noisy


def outcome_key(clock_label: int, system_bit: int) -> str:
    return f"c{int(clock_label)}_s{int(system_bit)}"


def parse_outcome_key(key: str) -> tuple[int, int]:
    left, right = key.split("_")
    return int(left[1:]), int(right[1:])
