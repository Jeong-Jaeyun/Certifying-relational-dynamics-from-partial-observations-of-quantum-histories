"""Support-only identifiability screen for the shared branch-pair estimator (P3 D1-v7).

D1-v6 found two ways a low-stratum support can fail to identify the state (run report §4,
``bi_qmn.p3.runs.p3_d1_v6_tail_diagnostics``). Each failure put one calibration record in the score
tail and, under a block-maximum conformal radius, removed the whole stratum.

* **Pauli-commutant degeneracy.** Every observed context measures a two-qubit Pauli ``P`` on the
  system. If every observed ``P`` commutes with one non-identity Pauli ``R``, then
  ``u_j -> exp(i theta R) u_j`` leaves every support table unchanged, for every state. Unobserved
  contexts are not protected. The commutant of a set of Paulis is spanned by the Paulis that commute
  with all of them, so checking single Paulis is complete for system-unitary symmetries. The check
  depends only on the context list.
* **Disconnected branch.** For a D (time-basis) context with Pauli ``P``, write
  ``F_P[b] = sum_tau (q_P(tau, 0) - q_P(tau, 1)) exp(-2 pi i b tau / K)``. Branch pair ``(j, k)`` sits
  alone in bin ``b_jk = (m_k - m_j) mod K``, where ``F_P[b_jk] = conj(c_j) c_k <u_j|P|u_k>``. The
  observed power of the pair is ``S_jk = sum_{P observed} |F_P[b_jk]|^2``. If both pairs that touch
  a branch carry no power, the support does not identify that branch's phase.

The power test uses a null distribution that comes from shot noise alone. Each shot adds one
unit-modulus term ``+-exp(-2 pi i b tau / K)`` to ``F_P[b]``. When the true coefficient is zero and
the clock marginal is flat (zero-energy histories), ``N |F_P[b]|^2`` is approximately ``Exp(1)``.
The sum over ``n_D`` observed D contexts is then approximately ``Gamma(n_D, 1)``. A pair is
*hidden* when ``S_jk`` does not exceed the ``1 - epsilon`` quantile of ``Gamma(n_D, 1) / N``. A
branch whose two pairs are both hidden is *disconnected*. Given a truly disconnected branch, the
test misses it with probability at most ``2 epsilon``.

A record **passes** when neither failure is detected. The screen is a function of the support
alone: its context list, its D tables, and its shot count. Selecting calibration and evaluation
blocks with it therefore keeps them exchangeable within the screened-in population.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import gamma

from bi_qmn.core.noise import outcome_key


SCREEN_VERSION = "support_identifiability_screen_v1"
TWO_QUBIT_PAULIS = tuple(a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II")


def paulis_commute(a: str, b: str) -> bool:
    """Two Pauli strings commute iff they anticommute on an even number of tensor factors."""
    return sum(1 for x, y in zip(a, b, strict=True) if x != "I" and y != "I" and x != y) % 2 == 0


def pauli_commutant(paulis: Iterable[str]) -> list[str]:
    """Non-identity two-qubit Paulis that commute with every Pauli in ``paulis``."""
    paulis = list(paulis)
    return [r for r in TWO_QUBIT_PAULIS if all(paulis_commute(r, p) for p in paulis)]


def support_paulis(context_ids: Iterable[str]) -> list[str]:
    """System Pauli labels of ``"D,XY"``-style context ids (D and F contexts alike)."""
    return sorted({cid.split(",")[1] for cid in context_ids})


def null_power_threshold(n_d: int, n_shots: float, epsilon: float) -> float:
    """``1 - epsilon`` quantile of ``Gamma(n_d, 1) / n_shots``; infinite when no D context is observed."""
    if n_d <= 0:
        return math.inf
    if not 0.0 < epsilon < 1.0 or n_shots <= 0:
        raise ValueError("epsilon must lie in (0, 1) and n_shots must be positive")
    return float(gamma.ppf(1.0 - epsilon, a=int(n_d)) / float(n_shots))


class SupportIdentifiabilityScreen:
    """Abstain when the support cannot identify the shared branch-pair state (see module docstring)."""

    version = SCREEN_VERSION

    def __init__(self, active_modes: Sequence[int], K: int, *, epsilon: float = 1.0e-6):
        self.modes = tuple(int(m) for m in active_modes)
        self.K = int(K)
        self.epsilon = float(epsilon)
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie in (0, 1)")
        self.pairs = [(j, k) for j in range(len(self.modes)) for k in range(j + 1, len(self.modes))]
        self.bins = {pair: (self.modes[pair[1]] - self.modes[pair[0]]) % self.K for pair in self.pairs}
        signed = [b for b in self.bins.values()] + [(-b) % self.K for b in self.bins.values()]
        if 0 in signed or len(set(signed)) != len(signed):
            raise ValueError("branch pairs must occupy distinct non-zero DFT bins (and distinct from their conjugates)")
        self.outcomes = [(outcome_key(l, 0), outcome_key(l, 1)) for l in range(self.K)]

    def label(self, pair: tuple[int, int]) -> str:
        return f"{self.modes[pair[0]]}-{self.modes[pair[1]]}"

    def pair_power(self, support: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        """Observed power ``S_jk`` of every branch pair, summed over the observed D contexts."""
        power = {self.label(p): 0.0 for p in self.pairs}
        phase = np.exp(-2j * math.pi * np.arange(self.K) / self.K)
        for cid in sorted(support):
            if not cid.startswith("D,"):
                continue
            row = support[cid]
            g = np.asarray([float(row.get(a, 0.0)) - float(row.get(b, 0.0)) for a, b in self.outcomes])
            for pair, b in self.bins.items():
                power[self.label(pair)] += float(abs(np.sum(g * phase ** b)) ** 2)
        return power

    def evaluate(self, support: Mapping[str, Mapping[str, float]], n_shots: float) -> dict[str, Any]:
        ids = sorted(support)
        n_d = sum(1 for cid in ids if cid.startswith("D,"))
        commutant = pauli_commutant(support_paulis(ids))
        power = self.pair_power(support)
        threshold = null_power_threshold(n_d, float(n_shots), self.epsilon)
        hidden = sorted(label for label, s in power.items() if s <= threshold)
        disconnected = []
        for b, m in enumerate(self.modes):
            touching = [self.label(p) for p in self.pairs if b in p]
            if all(label in hidden for label in touching):
                disconnected.append(int(m))
        reasons = (["pauli_commutant"] if commutant else []) + (["branch_disconnected"] if disconnected else [])
        return {
            "version": SCREEN_VERSION,
            "passes": not reasons,
            "reasons": reasons,
            "pauli_commutant": commutant,
            "pair_power": power,
            "power_threshold": threshold,
            "hidden_pairs": hidden,
            "disconnected_modes": disconnected,
            "n_D": n_d,
            "n_shots": float(n_shots),
            "epsilon": self.epsilon,
        }


def first_screened_in(passes: Sequence[bool], size: int) -> dict[str, Any]:
    """The first ``size`` passing positions in the declared order, or ``available = False`` if too few pass."""
    chosen = [i for i, ok in enumerate(passes) if ok][: int(size)]
    available = len(chosen) == int(size)
    return {"positions": chosen, "available": available, "scanned_through": (chosen[-1] + 1) if available else len(passes), "screened_out_before": (chosen[-1] + 1 - len(chosen)) if available else sum(not ok for ok in passes)}
