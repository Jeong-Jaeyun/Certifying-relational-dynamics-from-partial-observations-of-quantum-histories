"""E1 power simulation from development data only (plan §4 E1, §6).

Primary contrast proposed for the sealed E1: ``Delta = CVaR_0.9(C) - CVaR_0.9(A_s)`` of the diag error on
the common held-out set ``E``, screened-in 3a, decided by the lower end of a paired bootstrap interval
against a minimum important effect ``Delta_min``.

The development records of the declared M1 run (``m1_added_measurement_v1``) stand in for the population:
a simulated confirmatory bank draws ``n`` records with replacement, and the decision is the declared
paired bootstrap on that bank. The development effect is very large, so the scenarios shrink it:

* ``shrink`` lambda: every record's A_s error is moved toward its C error, ``A_s + lambda (C - A_s)``;
* ``failure`` f: a random fraction f of records lose the added-context gain entirely (A_s error := C error),
  which hits the tail directly;
* ``natural``: a flag-conditioned, screened-in mixture with the development 3a share (3a and 3b records),
  the operational population when the 3a label is not known.

Nothing here opens a bank; it reads the declared development artifact only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from bi_qmn.core.provenance import file_sha256


METHOD = "m1_e1_power_v1"
SOURCE = Path("outputs") / "m1_added_measurement" / "m1_added_measurement.json"
OUTPUT = Path("outputs") / "m1_e1_power" / "m1_e1_power.json"
LEVEL = 0.9
SIZES = (32, 48, 64, 96, 128, 192, 256)
DELTA_MIN = (0.0, 0.0403, 0.1, 0.2)        # 0, one frozen low diag radius, and two larger margins
SHRINK = (0.0, 0.5, 0.8, 0.9, 0.95)
FAILURE = (0.1, 0.2, 0.3, 0.5)


def tail_count(n: int, level: float = LEVEL) -> int:
    return max(1, math.ceil(round((1.0 - level) * n, 9)))


def cvar_rows(x: np.ndarray, level: float = LEVEL) -> np.ndarray:
    """CVaR of every row of ``x`` (mean of the ``ceil((1-level) n)`` largest entries)."""
    m = tail_count(x.shape[-1], level)
    return np.mean(np.partition(x, x.shape[-1] - m, axis=-1)[..., -m:], axis=-1)


def lower_bounds(c: np.ndarray, s: np.ndarray, rng: np.random.Generator, resamples: int, alpha: float) -> np.ndarray:
    """Paired bootstrap lower bound of ``CVaR(c) - CVaR(s)`` for each simulated bank (rows)."""
    S, n = c.shape
    out = np.empty(S)
    for k in range(S):
        idx = rng.integers(0, n, size=(resamples, n))
        d = cvar_rows(c[k][idx]) - cvar_rows(s[k][idx])
        out[k] = np.quantile(d, alpha)
    return out


def simulate(c_pop: np.ndarray, s_pop: np.ndarray, n: int, *, sims: int, resamples: int, alpha: float, rng: np.random.Generator,
             strata: np.ndarray | None = None, share: float | None = None) -> dict[str, Any]:
    if strata is None:
        pick = rng.integers(0, c_pop.size, size=(sims, n))
    else:
        a, b = np.flatnonzero(strata), np.flatnonzero(~strata)
        is_a = rng.random((sims, n)) < float(share)
        pick = np.where(is_a, a[rng.integers(0, a.size, size=(sims, n))], b[rng.integers(0, b.size, size=(sims, n))])
    c, s = c_pop[pick], s_pop[pick]
    lb = lower_bounds(c, s, rng, resamples, alpha)
    point = cvar_rows(c) - cvar_rows(s)
    out = {"n": int(n), "tail_count": tail_count(n), "point_median": float(np.median(point)), "lower_bound_quantiles": {q: float(np.quantile(lb, q)) for q in (0.05, 0.5)},
           "power": {f"{dm:g}": float(np.mean(lb > dm)) for dm in DELTA_MIN}}
    if strata is not None:
        out["expected_3a_in_bank"] = float(share) * n
    return out


def run_m1_e1_power(source: str | Path = SOURCE, output: str | Path = OUTPUT, *, sims: int = 1000, resamples: int = 1000, alpha: float = 0.025, seed: int = 61001,
                    sizes: Sequence[int] = SIZES, write: bool = True, progress=print) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output)
    if write and output.exists():
        raise FileExistsError(f"power artifact already exists: {output}")
    src = json.loads(Path(source).read_text(encoding="utf-8"))
    rec = src["records"]
    get = lambda rows, arm: np.asarray([r["errors"][arm]["E"]["diag"] for r in rows])   # noqa: E731
    r3a = [r for r in rec if r["class"] == "3a"]
    C, S, A0, F = get(r3a, "C"), get(r3a, "A_s"), get(r3a, "A0"), get(r3a, "A_f")
    rng = np.random.default_rng(seed)
    dev = {"n": len(r3a), "delta": float(cvar_rows(C) - cvar_rows(S)), "C_cvar": float(cvar_rows(C)), "A_s_cvar": float(cvar_rows(S))}
    # leave-one-record-out influence on the development contrast
    loo = [float(cvar_rows(np.delete(C, k)) - cvar_rows(np.delete(S, k))) for k in range(C.size)]
    dev["leave_one_out"] = {"min": min(loo), "max": max(loo)}
    scenarios: dict[str, Any] = {}
    for lam in SHRINK:
        s_pop = S + lam * (C - S)
        scenarios[f"shrink_{lam:g}"] = {"population_delta": float(cvar_rows(C) - cvar_rows(s_pop)),
                                        "by_n": [simulate(C, s_pop, n, sims=sims, resamples=resamples, alpha=alpha, rng=rng) for n in sizes]}
        if progress:
            progress(f"shrink {lam}: done")
    for f in FAILURE:
        # failure is drawn per simulated record: duplicate the population with a failure flag
        fail = rng.random(C.size * 20) < f
        base = np.tile(np.arange(C.size), 20)
        c_pop, s_pop = C[base], np.where(fail, C[base], S[base])
        scenarios[f"failure_{f:g}"] = {"population_delta": float(cvar_rows(c_pop) - cvar_rows(s_pop)),
                                       "by_n": [simulate(c_pop, s_pop, n, sims=sims, resamples=resamples, alpha=alpha, rng=rng) for n in sizes]}
        if progress:
            progress(f"failure {f}: done")
    # natural flag-conditioned mixture (development screened-in share of 3a)
    led = src["bank_ledger"]
    share = led["screened_in"]["3a"] / (led["screened_in"]["3a"] + led["screened_in"]["3b"])
    r3b = [r for r in rec if r["class"] == "3b"]
    c_mix = np.concatenate([C, get(r3b, "C")])
    s_mix = np.concatenate([S, get(r3b, "A_s")])
    strata = np.concatenate([np.ones(len(r3a), bool), np.zeros(len(r3b), bool)])
    scenarios["natural_mixture"] = {"share_3a": share, "by_n": [simulate(c_mix, s_mix, n, sims=sims, resamples=resamples, alpha=alpha, rng=rng, strata=strata, share=share)
                                                                 for n in (128, 192, 256, 384, 512)]}
    # secondary contrasts at the development effect (no shrinkage)
    secondary = {"A0_minus_A_s": [simulate(A0, S, n, sims=sims, resamples=resamples, alpha=alpha, rng=rng) for n in (64, 128)],
                 "A_f_minus_A_s": [simulate(F, S, n, sims=sims, resamples=resamples, alpha=alpha, rng=rng) for n in (64, 128, 256)]}
    artifact = {
        "artifact_type": "m1_e1_power", "method": METHOD, "status": "development_only",
        "source_artifact": Path(source).as_posix(), "source_artifact_sha256": file_sha256(Path(source)),
        "settings": {"level": LEVEL, "alpha_lower": alpha, "sims": sims, "resamples": resamples, "seed": seed, "delta_min": list(DELTA_MIN), "sizes": list(sizes)},
        "development": dev, "scenarios": scenarios, "secondary": secondary,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "wall_seconds": time.perf_counter() - started,
    }
    if write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
