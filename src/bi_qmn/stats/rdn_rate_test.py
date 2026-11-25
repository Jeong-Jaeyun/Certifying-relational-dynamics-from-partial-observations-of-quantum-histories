"""Exact one-sample binomial decision and power for the P3 RDN certification-rate claim.

Decided by the user on 2026-11-20 (formulation patch v3 §5, statistical protocol v2):

* A block succeeds iff ILGC BI existence is ``primal_verified`` **and** RDN is ``certified``.
  The denominator is every evaluation block of the stratum. Refuted, unresolved and BI-unverified
  blocks count as failures.
* The primary claim is: under the declared generative distribution, the success rate exceeds
  ``p0 = 0.80`` in **both** the medium and the low stratum. Each stratum uses a one-sided exact
  test at ``alpha = 0.05``, equivalently a one-sided 95% Clopper--Pearson lower bound ``> p0``.
  Requiring both (an intersection--union test) keeps the claim's level at ``alpha``. Declaring a
  single passing stratum as its own confirmation would need a separate multiplicity rule.
* Planning power is computed exactly. The joint power of the two strata is bounded below by
  ``1 - sum(1 - power_s)`` (Bonferroni/Frechet), which needs no independence assumption.

This module is separate from ``p3_statistics`` so the engine-v2 conformance artifact, which hashes
that source file, stays valid.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from scipy.stats import beta, binom


def critical_count(n: int, p0: float, alpha: float) -> int | None:
    """Smallest ``k`` with ``P(X >= k | n, p0) <= alpha``. Reject ``H0: p <= p0`` iff ``X >= k``."""
    n = int(n)
    if binom.sf(n - 1, n, p0) > alpha:
        return None
    k = int(binom.isf(alpha, n, p0)) + 1  # isf start point, then exact correction on the lattice
    k = min(max(k, 0), n)
    while k > 0 and binom.sf(k - 2, n, p0) <= alpha:
        k -= 1
    while binom.sf(k - 1, n, p0) > alpha:
        k += 1
    return k


def clopper_pearson_lower(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact (Clopper--Pearson) lower confidence bound for a binomial rate."""
    if k <= 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, k, n - k + 1))


def decide_rate(successes: int, n: int, *, p0: float = 0.80, alpha: float = 0.05) -> dict:
    """Exact one-sided decision. ``passes`` iff the lower bound exceeds ``p0``.

    For the one-sided Clopper--Pearson bound this is the same as ``successes >= critical_count``.
    """
    if not 0 <= successes <= n or n <= 0:
        raise ValueError("invalid binomial counts")
    lower = clopper_pearson_lower(successes, n, 1.0 - alpha)
    k = critical_count(n, p0, alpha)
    passes = bool(lower > p0)
    if k is not None and passes != (successes >= k):
        raise AssertionError("Clopper-Pearson bound and exact critical count disagree")
    return {
        "successes": int(successes),
        "n": int(n),
        "observed_rate": successes / n,
        "one_sided_lower_bound": lower,
        "p0": float(p0),
        "alpha": float(alpha),
        "critical_count": k,
        "p_value": float(binom.sf(successes - 1, n, p0)),
        "passes": passes,
    }


def intersection_union(decisions: Mapping[str, Mapping[str, object]]) -> dict:
    """The primary claim holds iff every stratum's test passes. Level = per-stratum alpha.

    The claim's p-value for the graphical family procedure is the maximum per-stratum p-value.
    """
    return {
        "strata": dict(decisions),
        "claim_passes": all(bool(d["passes"]) for d in decisions.values()),
        "p_value": max(float(d["p_value"]) for d in decisions.values()),
    }


def exact_power(n: int, p1: float, *, p0: float = 0.80, alpha: float = 0.05) -> float:
    k = critical_count(n, p0, alpha)
    return 0.0 if k is None else float(binom.sf(k - 1, n, p1))


def joint_power_lower_bound(powers: Iterable[float]) -> float:
    """Bonferroni/Frechet lower bound on P(all strata pass); valid under any dependence."""
    powers = list(powers)
    return max(0.0, 1.0 - sum(1.0 - p for p in powers))


def plan(n_per_stratum: int, planning_rates: Mapping[str, float], *, p0: float = 0.80, alpha: float = 0.05) -> dict:
    powers = {s: exact_power(n_per_stratum, p, p0=p0, alpha=alpha) for s, p in planning_rates.items()}
    product = 1.0
    for value in powers.values():
        product *= value
    return {
        "n_per_stratum": int(n_per_stratum),
        "critical_count": critical_count(n_per_stratum, p0, alpha),
        "exact_size": float(binom.sf(critical_count(n_per_stratum, p0, alpha) - 1, n_per_stratum, p0)),
        "planning_rates": dict(planning_rates),
        "per_stratum_power": powers,
        "joint_power_lower_bound": joint_power_lower_bound(powers.values()),
        "joint_power_if_independent": product,
    }


def minimal_sample_size(
    planning_rates: Mapping[str, float],
    *,
    joint_target: float,
    p0: float = 0.80,
    alpha: float = 0.05,
    n_max: int = 2000,
    sawtooth_window: int = 50,
) -> dict:
    """Smallest n whose joint lower bound reaches the target. Also returns the sawtooth-safe n.

    The sawtooth-safe n is the smallest n from which every n' in ``[n, n + window]`` also reaches the
    target. Exact binomial power is not monotone in n.
    """
    ok = [joint_power_lower_bound(exact_power(n, p, p0=p0, alpha=alpha) for p in planning_rates.values()) >= joint_target for n in range(1, n_max + sawtooth_window + 1)]
    first = next((i + 1 for i, v in enumerate(ok[:n_max]) if v), None)
    safe = next((i + 1 for i in range(n_max) if all(ok[i : i + sawtooth_window + 1])), None)
    return {"first_n": first, "sawtooth_safe_n": safe, "window": sawtooth_window, "joint_target": joint_target}


def block_successes(bi_status: Sequence[str], rdn_status: Sequence[str]) -> int:
    """Success = BI ``primal_verified`` and RDN ``certified``. Every block stays in the denominator."""
    if len(bi_status) != len(rdn_status):
        raise ValueError("BI and RDN status vectors must cover the same blocks")
    return sum(1 for b, r in zip(bi_status, rdn_status) if b == "primal_verified" and r == "certified")
