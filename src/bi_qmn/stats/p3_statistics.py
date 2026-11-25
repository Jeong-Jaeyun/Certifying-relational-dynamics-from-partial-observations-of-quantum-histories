"""P3-0 versioned paired-bound statistical engine (literature-adopted, v2).

This module implements the statistical obligations listed in
``documents/P2-P3-branchpoint.md`` section 11 (P3-0) using established methods instead of
new ad hoc rules. Every public function is deterministic given its inputs. The literature basis
and the conformance evidence are recorded in ``documents/P3-0-statistical-engine-v2.md``.

Scope boundary: nothing in this module may be used to officially re-adjudicate a frozen P1/P2
artifact. Historical artifacts can only be passed through ``historical_diagnostic`` which always
returns ``official_status_changed = False``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln


ENGINE_VERSION = "p3_paired_bound_engine_v2"
_TOL = 1.0e-12


class NonReclassificationError(RuntimeError):
    """Raised when the v2 engine is asked to issue an official verdict outside its protocol."""


# ---------------------------------------------------------------------------
# 1. Exact nonparametric bootstrap for lattice-valued paired endpoints
#    (Efron & Tibshirani 1993; Davison & Hinkley 1997, sec. 2.5 / 5.2.2)
# ---------------------------------------------------------------------------


def _lattice_counts(values: Sequence[float], unit: float) -> tuple[int, np.ndarray]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("exact bootstrap requires a non-empty one-dimensional vector")
    if unit <= 0.0:
        raise ValueError("lattice unit must be positive")
    scaled = data / unit
    integers = np.rint(scaled)
    if np.max(np.abs(scaled - integers)) > 1.0e-9:
        raise ValueError("values are not on the declared lattice; exact enumeration is undefined")
    integers = integers.astype(np.int64)
    low = int(np.min(integers))
    counts = np.bincount(integers - low).astype(float)
    return low, counts


def _poly_power(base: np.ndarray, exponent: int) -> np.ndarray:
    result = np.ones(1, dtype=float)
    power = base.copy()
    remaining = int(exponent)
    while remaining > 0:
        if remaining & 1:
            result = np.convolve(result, power)
        remaining >>= 1
        if remaining:
            power = np.convolve(power, power)
    return result


def exact_bootstrap_mean_distribution(values: Sequence[float], *, unit: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact distribution of the bootstrap mean (n i.i.d. draws from the empirical law).

    The resampled sum of n draws from a lattice-valued empirical distribution is the n-fold
    convolution of that distribution. This replaces Monte Carlo resampling by deterministic
    enumeration, so the bound no longer depends on a replicate count or a seed.
    """
    low, counts = _lattice_counts(values, unit)
    n = int(np.sum(counts))
    pmf = _poly_power(counts / float(n), n)
    pmf = np.clip(pmf, 0.0, None)
    pmf = pmf / float(np.sum(pmf))
    sums = np.arange(pmf.size, dtype=float) + n * low
    return sums * unit / n, pmf


def _inverse_cdf(atoms: np.ndarray, pmf: np.ndarray, level: float) -> float:
    cdf = np.cumsum(pmf)
    index = int(np.searchsorted(cdf, level - _TOL, side="left"))
    return float(atoms[min(index, atoms.size - 1)])


def exact_bootstrap_bound(
    values: Sequence[float],
    *,
    alpha: float,
    direction: str,
    unit: float = 1.0,
) -> float:
    """Exact percentile-bootstrap one-sided bound (inverse-CDF quantile of the exact law)."""
    if direction not in {"lower", "upper"}:
        raise ValueError("direction must be 'lower' or 'upper'")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    atoms, pmf = exact_bootstrap_mean_distribution(values, unit=unit)
    level = alpha if direction == "lower" else 1.0 - alpha
    return _inverse_cdf(atoms, pmf, level)


def monte_carlo_bound_distribution(
    values: Sequence[float],
    *,
    alpha: float,
    direction: str,
    replicates: int,
    unit: float = 1.0,
    mass_floor: float = 1.0e-12,
) -> dict:
    """Exact law of the Monte Carlo ``np.quantile(..., method='higher')`` bootstrap bound.

    With B replicates the reported bound is the order statistic X_(j), j = ceil(q (B-1)) (0-based).
    P(X_(j) <= x) = P(Binomial(B, F(x)) >= j + 1), where F is the exact bootstrap CDF. This turns
    the "same paired vector, different seed, different verdict" phenomenon into a computable
    Monte Carlo error manifest (Davison & Hinkley 1997, sec. 2.5.2).
    """
    if replicates <= 1:
        raise ValueError("Monte Carlo bootstrap requires at least two replicates")
    atoms, pmf = exact_bootstrap_mean_distribution(values, unit=unit)
    q = alpha if direction == "lower" else 1.0 - alpha
    order = int(math.ceil(q * (replicates - 1) - 1.0e-12))
    cdf = np.clip(np.cumsum(pmf), 0.0, 1.0)
    below = stats.binom.sf(order, replicates, cdf)  # P(#{X <= x} >= order + 1)
    mass = np.diff(np.concatenate([[0.0], below]))
    exact = exact_bootstrap_bound(values, alpha=alpha, direction=direction, unit=unit)
    keep = mass > mass_floor
    distribution = {float(a): float(m) for a, m in zip(atoms[keep], mass[keep], strict=True)}
    disagreement = float(sum(m for a, m in distribution.items() if abs(a - exact) > 1.0e-12))
    return {
        "exact_bound": exact,
        "replicates": int(replicates),
        "order_statistic_index_zero_based": order,
        "monte_carlo_bound_distribution": distribution,
        "probability_monte_carlo_differs_from_exact": disagreement,
    }


# ---------------------------------------------------------------------------
# 2. Common-random-number Monte Carlo bootstrap (shared resampling indices)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommonResampleBootstrap:
    replicates: int
    seed: int
    n: int
    index_hash: str
    indices: np.ndarray

    @classmethod
    def draw(cls, n: int, *, replicates: int, seed: int) -> "CommonResampleBootstrap":
        if n <= 0 or replicates <= 1:
            raise ValueError("common bootstrap requires n > 0 and at least two replicates")
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, n, size=(int(replicates), int(n)))
        digest = hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()[:16]
        return cls(replicates=int(replicates), seed=int(seed), n=int(n), index_hash=digest, indices=indices)

    def bound(self, values: Sequence[float], *, alpha: float, direction: str) -> float:
        data = np.asarray(values, dtype=float)
        if data.shape != (self.n,):
            raise ValueError("every paired endpoint must be indexed by the same n blocks")
        means = np.mean(data[self.indices], axis=1)
        quantile = alpha if direction == "lower" else 1.0 - alpha
        return float(np.quantile(means, quantile, method="higher"))

    def bounds(self, endpoints: Mapping[str, Sequence[float]], *, alpha: float, directions: Mapping[str, str]) -> dict:
        return {
            "engine_version": ENGINE_VERSION,
            "replicates": self.replicates,
            "seed": self.seed,
            "shared_index_hash": self.index_hash,
            "bounds": {
                name: self.bound(values, alpha=alpha, direction=directions[name])
                for name, values in endpoints.items()
            },
        }


# ---------------------------------------------------------------------------
# 3. Paired binary endpoints: exact conditional McNemar and Tango score bound
#    (McNemar 1947; Tango 1998; Fagerland, Lydersen & Laake 2014)
# ---------------------------------------------------------------------------


def ternary_counts(values: Sequence[float]) -> tuple[int, int, int]:
    data = np.asarray(values, dtype=float)
    if not np.all(np.isin(data, (-1.0, 0.0, 1.0))):
        raise ValueError("paired binary differences must lie in {-1, 0, +1}")
    return int(np.sum(data == 1.0)), int(np.sum(data == -1.0)), int(data.size)


def mcnemar_exact_one_sided(n_plus: int, n_minus: int, *, alternative: str, mid_p: bool = False) -> float:
    """Conditional exact McNemar test of p(+1) = p(-1) on the discordant pairs."""
    discordant = int(n_plus) + int(n_minus)
    if discordant == 0:
        return 1.0
    if alternative == "greater":
        tail, point = stats.binom.sf(n_plus, discordant, 0.5), stats.binom.pmf(n_plus, discordant, 0.5)
    elif alternative == "less":
        tail, point = stats.binom.cdf(n_plus - 1, discordant, 0.5), stats.binom.pmf(n_plus, discordant, 0.5)
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    return float(tail + (0.5 if mid_p else 1.0) * point)


def tango_score_statistic(n_plus: int, n_minus: int, n: int, delta0: float) -> float:
    """Tango (1998) score statistic for H0: p(+1) - p(-1) = delta0 (paired design)."""
    b, c, total = float(n_plus), float(n_minus), float(n)
    a_coef = 2.0 * total
    b_coef = -b - c + (2.0 * total - b + c) * delta0
    c_coef = -c * delta0 * (1.0 - delta0)
    disc = max(b_coef * b_coef - 4.0 * a_coef * c_coef, 0.0)
    p21 = (math.sqrt(disc) - b_coef) / (2.0 * a_coef)
    variance = total * (2.0 * p21 + delta0 * (1.0 - delta0))
    numerator = b - c - total * delta0
    if variance <= 0.0:
        if abs(numerator) < 1.0e-12:
            return 0.0
        return math.copysign(math.inf, numerator)
    return numerator / math.sqrt(variance)


def tango_score_bound(n_plus: int, n_minus: int, n: int, *, alpha: float, direction: str) -> float:
    """One-sided Tango score confidence bound for the paired difference p(+1) - p(-1)."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = float(stats.norm.ppf(1.0 - alpha))
    target = z if direction == "lower" else -z
    edge = 1.0 - 1.0e-12

    def f(delta: float) -> float:
        return tango_score_statistic(n_plus, n_minus, n, delta) - target

    left, right = f(-edge), f(edge)
    if direction == "lower":
        if left <= 0.0:
            return -1.0
        if right >= 0.0:
            return 1.0
    else:
        if right >= 0.0:
            return 1.0
        if left <= 0.0:
            return -1.0
    return float(optimize.brentq(f, -edge, edge, xtol=1.0e-12, rtol=1.0e-12, maxiter=500))


def tango_rejects_at_margin(n_plus: np.ndarray, n_minus: np.ndarray, n: int, *, alpha: float, margin: float, direction: str) -> np.ndarray:
    """Vectorised decision ``bound beyond margin`` without root finding (Z is monotone in delta)."""
    b = np.asarray(n_plus, dtype=float)
    c = np.asarray(n_minus, dtype=float)
    total = float(n)
    a_coef = 2.0 * total
    b_coef = -b - c + (2.0 * total - b + c) * margin
    c_coef = -c * margin * (1.0 - margin)
    p21 = (np.sqrt(np.maximum(b_coef * b_coef - 4.0 * a_coef * c_coef, 0.0)) - b_coef) / (2.0 * a_coef)
    variance = total * (2.0 * p21 + margin * (1.0 - margin))
    numerator = b - c - total * margin
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stat = np.where(variance > 0.0, numerator / np.sqrt(np.maximum(variance, 1.0e-300)), np.sign(numerator) * np.inf)
    z_stat = np.where((variance <= 0.0) & (numerator == 0.0), 0.0, z_stat)
    critical = float(stats.norm.ppf(1.0 - alpha))
    # lower bound > margin  <=>  Z(margin) > z ; upper bound < margin <=> Z(margin) < -z
    return z_stat > critical if direction == "lower" else z_stat < -critical


# ---------------------------------------------------------------------------
# 4. Exact power and sample size for paired ternary endpoints
#    (Connor 1987; Chernick & Liu 2002 saw-tooth power)
# ---------------------------------------------------------------------------


def connor_sample_size(p_plus: float, p_minus: float, *, alpha: float, power: float) -> float:
    """Connor (1987) asymptotic sample size for a one-sided paired-proportion test at margin 0."""
    psi = p_plus + p_minus
    delta = p_plus - p_minus
    if delta == 0.0 or psi <= 0.0:
        return math.inf
    z_a = float(stats.norm.ppf(1.0 - alpha))
    z_b = float(stats.norm.ppf(power))
    return (z_a * math.sqrt(psi) + z_b * math.sqrt(max(psi - delta * delta, 0.0))) ** 2 / (delta * delta)


def _trinomial_grid(n: int, p_plus: float, p_minus: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, c = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    valid = b + c <= n
    b, c = b[valid], c[valid]
    p_zero = max(0.0, 1.0 - p_plus - p_minus)
    log_pmf = gammaln(n + 1.0) - gammaln(b + 1.0) - gammaln(c + 1.0) - gammaln(n - b - c + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pmf = (
            log_pmf
            + np.where(b > 0, b * np.log(max(p_plus, 1.0e-300)), 0.0)
            + np.where(c > 0, c * np.log(max(p_minus, 1.0e-300)), 0.0)
            + np.where(n - b - c > 0, (n - b - c) * np.log(max(p_zero, 1.0e-300)), 0.0)
        )
    return b, c, np.exp(log_pmf)


def paired_ternary_power(
    n: int,
    p_plus: float,
    p_minus: float,
    *,
    alpha: float,
    margin: float,
    direction: str,
    method: str = "tango",
) -> float:
    """Exact (enumerated) probability that the one-sided bound clears ``margin``."""
    b, c, pmf = _trinomial_grid(int(n), float(p_plus), float(p_minus))
    if method == "tango":
        decision = tango_rejects_at_margin(b, c, n, alpha=alpha, margin=margin, direction=direction)
    elif method == "mcnemar_exact":
        if margin != 0.0:
            raise ValueError("the conditional exact McNemar test is defined at margin 0 only")
        discordant = b + c
        if direction == "lower":
            pvalues = stats.binom.sf(b - 1, discordant, 0.5)
        else:
            pvalues = stats.binom.cdf(b, discordant, 0.5)
        decision = (discordant > 0) & (pvalues <= alpha)
    elif method == "exact_bootstrap":
        decision = np.zeros(b.size, dtype=bool)
        for index, (x, y) in enumerate(zip(b, c)):
            vector = [1.0] * int(x) + [-1.0] * int(y) + [0.0] * int(n - x - y)
            bound = exact_bootstrap_bound(vector, alpha=alpha, direction=direction)
            decision[index] = bound > margin if direction == "lower" else bound < margin
    else:
        raise ValueError(f"unknown power method: {method}")
    return float(np.sum(pmf[decision]))


def minimal_sample_size(
    p_plus: float,
    p_minus: float,
    *,
    alpha: float,
    power: float,
    margin: float,
    direction: str,
    method: str = "tango",
    n_min: int = 8,
    n_max: int = 1024,
    step: int = 1,
) -> dict:
    """First n reaching ``power`` and the saw-tooth-safe n after which power never drops below it."""
    curve: dict[int, float] = {}
    first = None
    for n in range(int(n_min), int(n_max) + 1, int(step)):
        value = paired_ternary_power(n, p_plus, p_minus, alpha=alpha, margin=margin, direction=direction, method=method)
        curve[n] = value
        if first is None and value >= power:
            first = n
    safe = None
    for n in sorted(curve, reverse=True):
        if curve[n] < power:
            break
        safe = n
    return {
        "method": method,
        "first_n_reaching_power": first,
        "sawtooth_safe_n": safe,
        "n_max_searched": int(n_max),
        "power_at_safe_n": None if safe is None else curve[safe],
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# 5. Multiplicity: Holm (1979) and the graphical procedure (Bretz et al. 2009)
# ---------------------------------------------------------------------------


def holm(pvalues: Mapping[str, float], *, alpha: float) -> dict[str, bool]:
    ordered = sorted(pvalues, key=lambda name: (pvalues[name], name))
    rejected = {name: False for name in pvalues}
    m = len(ordered)
    for rank, name in enumerate(ordered):
        if pvalues[name] <= alpha / (m - rank):
            rejected[name] = True
        else:
            break
    return rejected


def graphical_procedure(
    pvalues: Mapping[str, float],
    weights: Mapping[str, float],
    transitions: Mapping[str, Mapping[str, float]],
    *,
    alpha: float,
) -> dict[str, bool]:
    """Sequentially rejective graphical multiple test (Bretz, Maurer, Brannath & Posch 2009)."""
    active = list(pvalues)
    w = {name: float(weights.get(name, 0.0)) for name in active}
    g = {j: {k: float(transitions.get(j, {}).get(k, 0.0)) for k in active if k != j} for j in active}
    if sum(w.values()) > 1.0 + 1.0e-12:
        raise ValueError("graphical weights must sum to at most one")
    rejected = {name: False for name in active}
    while True:
        candidates = [j for j in active if pvalues[j] <= w[j] * alpha + 1.0e-15]
        if not candidates:
            return rejected
        j = sorted(candidates, key=lambda name: (pvalues[name], name))[0]
        rejected[j] = True
        remaining = [k for k in active if k != j]
        new_w = {l: w[l] + w[j] * g[j].get(l, 0.0) for l in remaining}
        new_g: dict[str, dict[str, float]] = {}
        for l in remaining:
            new_g[l] = {}
            for k in remaining:
                if k == l:
                    continue
                denominator = 1.0 - g[l].get(j, 0.0) * g[j].get(l, 0.0)
                new_g[l][k] = 0.0 if denominator <= 0.0 else (g[l].get(k, 0.0) + g[l].get(j, 0.0) * g[j].get(k, 0.0)) / denominator
        active, w, g = remaining, new_w, new_g


# ---------------------------------------------------------------------------
# 6. Group-sequential accumulation: Lan & DeMets (1983) alpha spending
# ---------------------------------------------------------------------------


def obrien_fleming_spending(t: float, alpha: float) -> float:
    if t <= 0.0:
        return 0.0
    return float(2.0 - 2.0 * stats.norm.cdf(stats.norm.ppf(1.0 - alpha / 2.0) / math.sqrt(min(t, 1.0))))


def pocock_spending(t: float, alpha: float) -> float:
    if t <= 0.0:
        return 0.0
    return float(alpha * math.log(1.0 + (math.e - 1.0) * min(t, 1.0)))


def lan_demets_boundaries(
    information_fractions: Sequence[float],
    *,
    alpha: float,
    spending: Callable[[float, float], float] = obrien_fleming_spending,
) -> list[float]:
    """One-sided efficacy z-boundaries from an alpha-spending function (independent increments)."""
    fractions = [float(value) for value in information_fractions]
    if any(b <= a for a, b in zip(fractions, fractions[1:])) or fractions[0] <= 0.0 or fractions[-1] > 1.0 + 1.0e-12:
        raise ValueError("information fractions must be strictly increasing in (0, 1]")
    boundaries: list[float] = []
    spent_previous = 0.0
    for k, t_k in enumerate(fractions):
        increment = spending(t_k, alpha) - spent_previous
        if k == 0:
            boundaries.append(float(stats.norm.isf(increment)))
        else:
            ts = fractions[: k + 1]
            cov = np.array([[math.sqrt(min(a, b) / max(a, b)) for b in ts] for a in ts])
            previous = np.asarray(boundaries, dtype=float)
            no_cross_before = stats.multivariate_normal(mean=np.zeros(k), cov=cov[:k, :k]).cdf(previous)
            joint = stats.multivariate_normal(mean=np.zeros(k + 1), cov=cov)

            def excess(c: float) -> float:
                return (no_cross_before - float(joint.cdf(np.append(previous, c)))) - increment

            boundaries.append(float(optimize.brentq(excess, -2.0, 10.0, xtol=1.0e-7)))
        spent_previous = spending(t_k, alpha)
    return boundaries


# ---------------------------------------------------------------------------
# 7. Paired sign-flip randomization test (Fisher 1935; Phipson & Smyth 2010)
# ---------------------------------------------------------------------------


def sign_flip_test(
    deltas: Sequence[float],
    *,
    alternative: str = "greater",
    exact_max_n: int = 16,
    mc_draws: int = 20000,
    seed: int = 0,
) -> dict:
    data = np.asarray(deltas, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("sign-flip test requires a non-empty vector")
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'")
    observed = float(np.mean(data))
    orient = 1.0 if alternative == "greater" else -1.0
    if data.size <= exact_max_n:
        signs = np.asarray(list(itertools.product((1.0, -1.0), repeat=data.size)))
        null = signs @ data / data.size
        p_value = float(np.mean(orient * null >= orient * observed - 1.0e-12))
        method = "exact_enumeration"
    else:
        rng = np.random.default_rng(int(seed))
        signs = rng.choice((1.0, -1.0), size=(int(mc_draws), data.size))
        null = signs @ data / data.size
        exceed = int(np.sum(orient * null >= orient * observed - 1.0e-12))
        p_value = float((exceed + 1) / (mc_draws + 1))
        method = "monte_carlo_plus_one"
    return {"mean": observed, "p_value": p_value, "alternative": alternative, "method": method, "n": int(data.size)}


# ---------------------------------------------------------------------------
# 8. Order-statistic calibration: split conformal and training-conditional (PAC) validity
#    (Wilks 1941; Vovk, Gammerman & Shafer 2005; Vovk 2012; Lei et al. 2018)
# ---------------------------------------------------------------------------


def split_conformal_order_index(n: int, alpha: float) -> int | None:
    k = int(math.ceil((n + 1) * (1.0 - alpha) - 1.0e-12))
    return k if k <= n else None


def training_conditional_failure_probability(n: int, k: int, epsilon: float) -> float:
    """delta = P(coverage of the k-th order statistic < 1 - epsilon); coverage ~ Beta(k, n + 1 - k)."""
    if not 1 <= k <= n:
        raise ValueError("order index must lie in 1..n")
    return float(stats.beta.cdf(1.0 - epsilon, k, n + 1 - k))


def pac_order_index(n: int, epsilon: float, delta: float) -> int | None:
    for k in range(1, n + 1):
        if training_conditional_failure_probability(n, k, epsilon) <= delta:
            return k
    return None


def minimum_calibration_size_for_maximum(epsilon: float, delta: float) -> int:
    """Wilks (1941): the sample maximum covers 1 - epsilon with probability >= 1 - delta."""
    return int(math.ceil(math.log(delta) / math.log(1.0 - epsilon)))


def exceedance_tail_probability(n: int, k: int, m: int, observed: int) -> float:
    """P(at least ``observed`` of m exchangeable test values exceed the k-th of n calibration values).

    The exceedance count is Beta-binomial(m, n + 1 - k, k) under exchangeability.
    """
    return float(stats.betabinom.sf(int(observed) - 1, int(m), n + 1 - k, k))


# ---------------------------------------------------------------------------
# 9. Non-reclassification guard
# ---------------------------------------------------------------------------


def official_verdict_allowed(protocol: Mapping[str, object]) -> None:
    """Raise unless a protocol froze this engine version before any confirmatory data were opened."""
    if protocol.get("statistical_engine_version") != ENGINE_VERSION:
        raise NonReclassificationError("protocol did not freeze the v2 statistical engine")
    if not protocol.get("frozen_before_confirmatory_data", False):
        raise NonReclassificationError("v2 verdicts require a protocol frozen before confirmatory data")
    if protocol.get("historical_artifact", False):
        raise NonReclassificationError("historical P1/P2 artifacts may not be re-adjudicated by the v2 engine")


def historical_diagnostic(name: str, payload: Mapping[str, object]) -> dict:
    return {
        "name": name,
        "engine_version": ENGINE_VERSION,
        "role": "report_only_historical_diagnostic",
        "official_status_changed": False,
        **dict(payload),
    }
