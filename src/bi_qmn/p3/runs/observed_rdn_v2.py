"""Observed RDN v2: the both-margin conditional test of relational dynamics.

Protocols: documents/observed-RDN-v2-protocol.md (development) and documents/observed-RDN-v2-sealed-preregistration.md
(sealed sensitivity and the retrospective S3-support analysis). Each is frozen by its own declaration.

**The test.** For every observed time-basis (D) context the counts ``N(tau, s)`` form a K x 2 table. Under trusted
product measurements, a conditional system state that does not depend on the clock reading makes the table a
product ``p(tau) p(s)`` (conditional independence of clock and system outcome). Conditioning on both margins of
every table removes every nuisance parameter, *including the clock marginal*: for binary ``s`` the counts
``N(., 1)`` are multivariate hypergeometric with category sizes ``r_tau`` and draw size ``c_1``. The likelihood-ratio
statistic ``G = 2 sum N log(N n / (r_tau c_s))`` summed over the observed D contexts is compared with ``B`` draws
from that law; the rank p-value ``(1 + #{G_b >= G_obs}) / (B + 1)`` has false-certification probability at most
``alpha`` for every state with a clock-independent conditional state, any clock marginal and any product readout
channel. The v1 test (uniform clock marginal, column totals only) is computed on the same counts for comparison.

**Stages.**
* ``dev`` (config ``observed_rdn_v2.yaml``): a fresh development bank of controlled truths, including static
  histories seen with a non-uniform clock marginal or an asymmetric clock readout (nulls for v2, not for v1) and
  a joint clock-system readout error (outside the product-readout assumption), plus the regenerated E5 and
  G-guard supports joined with the model-based route.
* ``sealed`` (config ``observed_rdn_v2_sealed.yaml``): a sealed bank with preregistered per-stratum endpoints
  (exact one-sided binomial tests, intersection-union), and, as a separately labelled post hoc analysis, the
  fixed rule applied to the supports of the S3 sealed evaluation bank, with an orbit analysis of S3 failures.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from qiskit.quantum_info import DensityMatrix
from scipy.stats import binom, chi2
import yaml

from bi_qmn.core import identifiability as ident
from bi_qmn.core.config import load_p1_config
from bi_qmn.core.observations import sample_shot_batch
from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.core.types import MeasurementTable
from bi_qmn.ilgc.relational_dynamics import tau_flat_distance
from bi_qmn.p1.model.p1_banks import support_contexts_for_deficit
from bi_qmn.p1.model.p1_generator import construction_bank_p1, seeded_construction_bank_p1
from bi_qmn.p1.model.p1_measurement import full_p1_context_list, p1_measurement_table
from bi_qmn.p3 import p3_dev_harness as harness
from bi_qmn.p3.p3_support_screen import SupportIdentifiabilityScreen
from bi_qmn.p3.runs import e5_rdn_specificity as e5
from bi_qmn.p3.runs import m1_added_measurement as am
from bi_qmn.p3.runs import m1_added_measurement_posthoc as ph
from bi_qmn.p3.runs import m1_d_commutant_check as dc
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs import observed_rdn as orn
from bi_qmn.p3.runs import p3_s3_confirmatory as s3
from bi_qmn.p3.runs.p3_d1_v6_development import _execute, _keys, _table_dict, _true_branch_vectors
from bi_qmn.p3.runs.p3_model_c_development_v2 import physical_fingerprint


CONFIGS = {"dev": "observed_rdn_v2.yaml", "sealed": "observed_rdn_v2_sealed.yaml"}
DECLARATIONS = {"dev": "observed_rdn_v2_declaration.json", "sealed": "observed_rdn_v2_sealed_declaration.json"}
ARTIFACTS = {"dev": "observed_rdn_v2.json", "sealed": "observed_rdn_v2_sealed.json"}
STATUS = {"dev": "development_only", "sealed": "sealed_confirmatory"}
RUNNER = "runners/p3/run_observed_rdn_v2.py"
DEFICITS = ("medium", "low")
RESERVED = (53500, 53999)
GENERATIVE_KEYS = ("dirichlet_concentration", "minimum_mode_probability", "phase_range")
_W: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# The test (pure)
# ---------------------------------------------------------------------------


def g_independence(N: np.ndarray) -> float:
    """``2 sum N log(N n / (r c))`` over the positive cells of one table (0 for an empty table)."""
    N = np.asarray(N, dtype=float)
    n = float(N.sum())
    if n <= 0.0:
        return 0.0
    E = np.outer(N.sum(axis=1), N.sum(axis=0)) / n
    nz = N > 0
    return float(2.0 * np.sum(N[nz] * np.log(N[nz] / E[nz])))


def g_both(counts: Mapping[str, np.ndarray]) -> float:
    return float(sum(g_independence(counts[cid]) for cid in sorted(counts)))


def _degenerate(N: np.ndarray) -> bool:
    """A table whose conditional law is a point mass (one nonzero row or one nonzero column)."""
    r, c = N.sum(axis=1), N.sum(axis=0)
    return int(np.count_nonzero(r)) < 2 or int(np.count_nonzero(c)) < 2


def null_draws_both(counts: Mapping[str, np.ndarray], B: int, rng: np.random.Generator) -> np.ndarray:
    """``G`` under the conditional law given both margins of every table (binary system outcome)."""
    G = np.zeros(int(B))
    for cid in sorted(counts):
        N = np.asarray(counts[cid], dtype=np.int64)
        if N.shape[1] != 2:
            raise ValueError("the both-margin sampler is written for binary system outcomes")
        if _degenerate(N):
            continue
        r = N.sum(axis=1)
        n = int(r.sum())
        c1 = int(N[:, 1].sum())
        c0 = n - c1
        x = rng.multivariate_hypergeometric(r, c1, size=int(B)).astype(float)
        y = r[None, :].astype(float) - x
        rr = np.where(r > 0, r, 1).astype(float)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(x > 0, x * np.log(x * n / (rr * c1)), 0.0)
            ty = np.where(y > 0, y * np.log(y * n / (rr * c0)), 0.0)
        G += 2.0 * (tx.sum(axis=1) + ty.sum(axis=1))
    return G


def both_margin_test(counts: Mapping[str, np.ndarray], B: int, rng: np.random.Generator) -> dict[str, Any]:
    G_obs = g_both(counts)
    G_null = null_draws_both(counts, B, rng)
    tol = 1.0e-9 * max(1.0, abs(G_obs))
    df = sum((int(np.count_nonzero(N.sum(axis=1))) - 1) * (int(np.count_nonzero(N.sum(axis=0))) - 1) for N in counts.values())
    return {"G": G_obs, "p_monte_carlo": float((1 + np.sum(G_null >= G_obs - tol)) / (int(B) + 1)), "p_chi2": float(chi2.sf(G_obs, df)) if df > 0 else 1.0,
            "df": int(df), "null_mean": float(np.mean(G_null)), "contexts": len(counts),
            "degenerate_contexts": int(sum(_degenerate(np.asarray(N)) for N in counts.values()))}


def run_tests(counts: Mapping[str, np.ndarray], K: int, B: int, seed_text: str) -> dict[str, Any]:
    """The v2 test (primary) and the v1 uniform-clock test on the same counts, with independent Monte Carlo streams."""
    v2 = both_margin_test(counts, B, np.random.default_rng(_seed(seed_text + ":v2")))
    v1 = orn.observed_rdn_test(counts, K, B, np.random.default_rng(_seed(seed_text + ":v1")))
    return {"v2": v2, "v1": v1}


def binomial_endpoint(k: int, n: int, p0: float, direction: str, alpha: float) -> dict[str, Any]:
    """Exact one-sided binomial test: ``greater`` tests p > p0, ``less`` tests p < p0."""
    if n <= 0:
        return {"k": int(k), "n": int(n), "p_value": None, "passes": False, "ci95": None}
    if direction == "greater":
        p = float(binom.sf(int(k) - 1, int(n), float(p0)))
    elif direction == "less":
        p = float(binom.cdf(int(k), int(n), float(p0)))
    else:
        raise ValueError(direction)
    return {"k": int(k), "n": int(n), "rate": k / n, "p0": float(p0), "direction": direction, "alpha": float(alpha),
            "p_value": p, "passes": bool(p <= float(alpha)), "ci95": e5.clopper_pearson(int(k), int(n))}


# ---------------------------------------------------------------------------
# Truth families and readout channels
# ---------------------------------------------------------------------------


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:16], 16)


def skew_weights(K: int, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    w = 1.0 + float(amplitude) * np.cos(2.0 * math.pi * np.arange(K) / K + float(phase))
    return w / w.sum()


def time_vector(K: int, tau: int) -> np.ndarray:
    return np.exp(-2j * math.pi * np.arange(K) * int(tau) / K) / math.sqrt(K)


def nonuniform_clock_static(ctx: Mapping[str, Any], i: int, amplitude: float) -> np.ndarray:
    """``sigma_C (x) rho_bar``: a time-diagonal clock state with weights ``(1 + a cos(2 pi tau / K + phi_i)) / K`` and the
    static history's conditional state; the conditional state is constant, the clock marginal is not uniform."""
    K, d_S = ctx["K"], ctx["d_S"]
    P = e5.family_density(ctx, i, "pure")
    rho_bar = np.einsum("kakb->ab", e5.pinch(P, K, d_S).reshape(K, d_S, K, d_S))
    phi = float(np.random.default_rng(_seed(f"{ctx['bank']['haar_seed']}:{ctx['bank']['bank_id']}:clock:{int(i)}")).uniform(0.0, 2.0 * math.pi))
    w = 1.0 + float(amplitude) * np.cos(2.0 * math.pi * np.arange(K) / K + phi)
    w = w / w.sum()
    sigma = sum(w[t] * np.outer(time_vector(K, t), time_vector(K, t).conj()) for t in range(K))
    return np.kron(sigma, rho_bar)


def parse_family(family: str) -> dict[str, Any]:
    """``base[_readout_eta][_clock_skew_a][_xor_e]`` (suffix order fixed)."""
    spec: dict[str, Any] = {"eta": 0.0, "skew": 0.0, "xor": 0.0}
    name = family
    for key in ("xor", "clock_skew"):
        tag = f"_{key}_"
        if tag in name:
            name, value = name.split(tag)
            spec["skew" if key == "clock_skew" else "xor"] = float(value)
    if name.startswith("readout_"):
        spec["eta"] = float(name.split("_", 1)[1])
        name = "pure"
    elif "_readout_" in name:
        name, value = name.split("_readout_")
        spec["eta"] = float(value)
    spec["base"] = name
    return spec


def family_state(ctx: Mapping[str, Any], i: int, base: str) -> np.ndarray:
    if base.startswith("nonuniform_clock_static"):
        amp = float(base.split("_")[-1]) if base.count("_") >= 3 else 0.5
        return nonuniform_clock_static(ctx, i, amp)
    return e5.family_density(ctx, i, base)


def apply_d_channels(table: Mapping[str, Mapping[str, float]], K: int, skew: float, xor: float) -> dict[str, dict[str, float]]:
    """Product clock skew (with probability ``skew`` the time-basis reading is redrawn from ``skew_weights``) and a joint
    error (with probability ``xor`` the system bit is XORed with the parity of the reported reading) on D contexts."""
    out = {c: dict(row) for c, row in table.items()}
    if skew == 0.0 and xor == 0.0:
        return out
    w = skew_weights(K)
    A = (1.0 - skew) * np.eye(K) + skew * np.outer(w, np.ones(K))          # column-stochastic: A[l', l]
    for cid, row in table.items():
        if not cid.startswith("D,"):
            continue
        q = np.asarray([[float(row[f"c{l}_s{s}"]) for s in (0, 1)] for l in range(K)])
        q = A @ q
        if xor:
            flipped = q[:, ::-1]
            odd = (np.arange(K) % 2 == 1)[:, None]
            q = np.where(odd, (1.0 - xor) * q + xor * flipped, q)
        out[cid] = {f"c{l}_s{s}": float(q[l, s]) for l in range(K) for s in (0, 1)}
    return out


def sample_counts(table: Mapping[str, Mapping[str, float]], context_ids: Sequence[str], n_shot: int, K: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Independent multinomial shots per context; counts as ``N[l, s]``."""
    out = {}
    for cid in sorted(context_ids):
        p = np.asarray([max(0.0, float(table[cid][f"c{l}_s{s}"])) for l in range(K) for s in (0, 1)])
        draw = rng.multinomial(int(n_shot), p / p.sum())
        out[cid] = draw.reshape(K, 2).astype(np.int64)
    return out


def support_from_counts(counts: Mapping[str, np.ndarray], n_shot: int, K: int) -> dict[str, dict[str, float]]:
    return {cid: {f"c{l}_s{s}": float(N[l, s]) / n_shot for l in range(K) for s in (0, 1)} for cid, N in counts.items()}


# ---------------------------------------------------------------------------
# Configuration and context
# ---------------------------------------------------------------------------


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load(config_dir: Path, stage: str) -> tuple[dict[str, Any], str]:
    path = Path(config_dir) / CONFIGS[stage]
    values = _yaml(path)
    if values.get("status") not in (STATUS[stage], "test_only"):
        raise ValueError(f"the {stage} stage requires status {STATUS[stage]}")
    test = values["test"]
    if not 0.0 < float(test["alpha_primary"]) < 0.5 or int(test["monte_carlo_draws"]) < 99:
        raise ValueError("alpha must lie in (0, 0.5) and B >= 99")
    bank = values["bank"]
    seeds = [int(bank["seed"]), int(bank["observation_seed"]), int(bank["haar_seed"]), int(test["monte_carlo_seed"])]
    if len(set(seeds)) != len(seeds):
        raise ValueError("bank, observation, haar and Monte Carlo seeds must be distinct")
    inside = [RESERVED[0] <= s <= RESERVED[1] for s in seeds]
    if values["status"] == "sealed_confirmatory":
        ns = values["reserved_future_confirmatory_namespace"]
        if not all(inside) or set(seeds) & {int(v) for v in ns["previously_allocated"]} or sorted(int(v) for v in ns["allocated_here"]) != sorted(seeds):
            raise ValueError("sealed seeds must be new and inside the reserved namespace, listed in allocated_here")
        for ep in values["endpoints"]["tests"]:
            fams = _endpoint_families(ep)
            if not fams or not set(fams) <= set(bank["families"]) or ep["stratum"] not in (*DEFICITS, "pooled") or ep["direction"] not in ("greater", "less"):
                raise ValueError(f"malformed endpoint {ep}")
    elif any(inside):
        raise ValueError("development or test seeds intrude on the reserved confirmatory namespace")
    for f in bank["families"]:
        parse_family(f)
    if not set(bank["null_families"]) <= set(bank["families"]):
        raise ValueError("null families must be evaluated")
    dev = _yaml(Path(config_dir) / values["design_source"]["development_config"])
    reference = dev["development_banks"]["evaluation"]
    if any(bank[k] != reference[k] for k in GENERATIVE_KEYS):
        raise ValueError("the bank must use the D1-v7 generative specification")
    return values, file_sha256(path)


def prepare(config_dir: Path, bank: Mapping[str, Any]) -> dict[str, Any]:
    """Construction bank, contexts, screen and frozen S3 bodies (no estimator: the test needs no fit)."""
    config_dir = Path(config_dir)
    p1 = load_p1_config(config_dir)
    dev = _yaml(config_dir / "p3_d1_v7.yaml")
    contexts = full_p1_context_list(p1.measurement)
    dims = p1.generator["dimensions"]
    K, d_S = int(dims["K"]), int(dims["d_S"])
    instances = seeded_construction_bank_p1(p1.generator, dict(bank))
    return {"bank": dict(bank), "p1": p1, "K": K, "d_S": d_S, "instances": instances,
            "instance_hash": hashlib.sha256("".join(physical_fingerprint(i) for i in instances).encode("ascii")).hexdigest()[:16],
            "contexts": contexts, "context_by_id": {c.context_id: c for c in contexts},
            "screen": SupportIdentifiabilityScreen(p1.generator["active_modes"], K, epsilon=float(dev["screen"]["disconnected_branch_null_tail_epsilon"])),
            "bodies": e5.frozen_s3_bodies(), "diag_keys": _keys(contexts, "diag", K)}


# ---------------------------------------------------------------------------
# One record
# ---------------------------------------------------------------------------


def fresh_record(ctx: Mapping[str, Any], i: int, deficit: str, family: str, B: int, mc_seed: int) -> dict[str, Any]:
    bank = ctx["bank"]
    K = ctx["K"]
    spec = parse_family(family)
    rho = family_state(ctx, i, spec["base"])
    meta = {"instance_index": int(i), "deficit": deficit, "family": family, "bank_id": bank["bank_id"]}
    ideal = {c: dict(row) for c, row in p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=K, eta=0.0, metadata=meta).copy_without_source().probabilities.items()}
    effective = ideal if spec["eta"] == 0.0 else {c: dict(row) for c, row in p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=K, eta=spec["eta"], metadata=meta).copy_without_source().probabilities.items()}
    effective = apply_d_channels(effective, K, spec["skew"], spec["xor"])
    n_shot = int(ctx["p1"].measurement["observation_deficit"][deficit]["n_shot"])
    observed = [c.context_id for c in support_contexts_for_deficit(ctx["p1"].measurement, deficit, seed_key=f"p3v2:{bank['bank_id']}:{deficit}:{int(i)}")]
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bank['bank_id']}:{deficit}:{int(i)}:{family}"))
    counts_all = sample_counts(effective, observed, n_shot, K, rng)
    support = support_from_counts(counts_all, n_shot, K)
    screen = ctx["screen"].evaluate(support, float(n_shot))
    counts = {c: N for c, N in counts_all.items() if c.startswith("D,")}
    tests = run_tests(counts, K, B, f"{mc_seed}:{bank['bank_id']}:{deficit}:{int(i)}:{family}")
    radius = ctx["bodies"]["radii"][deficit]["diag"]
    return {"index": int(i), "deficit": deficit, "family": family, "spec": spec, "observed": sorted(observed), "screen_passes": bool(screen["passes"]),
            "tests": tests,
            "truth": {"kappa": tau_flat_distance(ident.class_vector(ideal, ctx["contexts"], "diag", K), ctx["diag_keys"]) / radius,
                      "conditional_tau_spread": e5.conditional_tau_spread(rho, K, ctx["d_S"]),
                      "clock_marginal_spread": clock_marginal_spread(effective, K)}}


def clock_marginal_spread(table: Mapping[str, Mapping[str, float]], K: int) -> float:
    """Largest ``max_tau - min_tau`` of the reported clock marginal over D contexts (0 for a uniform marginal)."""
    worst = 0.0
    for cid, row in table.items():
        if cid.startswith("D,"):
            m = np.asarray([float(row[f"c{l}_s0"]) + float(row[f"c{l}_s1"]) for l in range(K)])
            worst = max(worst, float(m.max() - m.min()))
    return worst


def existing_record(ctx: Mapping[str, Any], i: int, deficit: str, family: str, role: str, B: int, mc_seed: int) -> dict[str, Any]:
    """Regenerate an E5 or G-guard support exactly as the observed-RDN v1 run did, then run both tests."""
    bank = ctx["bank"]
    inst = ctx["instances"][int(i)]
    rho, eta = orn.family_truth(ctx, i, family)
    meta = {"instance_index": int(i), "deficit": deficit, "family": family, "bank_id": bank["bank_id"]}
    effective = p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=inst.K, eta=eta, metadata=meta).copy_without_source()
    n_shot = int(ctx["p1"].measurement["observation_deficit"][deficit]["n_shot"])
    support_contexts = support_contexts_for_deficit(ctx["p1"].measurement, deficit, seed_key=f"p3v2:{bank['bank_id']}:{deficit}:{int(i)}")
    rng = np.random.default_rng(_seed(f"{bank['observation_seed']}:{bank['bank_id']}:{deficit}:{int(i)}"))
    support = sample_shot_batch(effective, support_contexts, n_shots=n_shot, rng=rng, shot_start=(e5.DEFICITS.index(deficit) * 1000 + int(i)) * 100_000_000, role=role).table.probabilities
    support = {c: dict(row) for c, row in support.items()}
    screen = ctx["screen"].evaluate(support, float(n_shot))
    counts = orn.d_counts(support, n_shot, ctx["K"])
    tests = run_tests(counts, ctx["K"], B, f"{mc_seed}:{bank['bank_id']}:{deficit}:{int(i)}:{family}")
    return {"index": int(i), "deficit": deficit, "family": family, "observed": sorted(support), "screen_passes": bool(screen["passes"]), "tests": tests,
            "v1_regenerated_matches": None}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _stage_bank(values: Mapping[str, Any], config_dir: Path, stage: str) -> dict[str, Any]:
    if stage in ("fresh", "sealed"):
        return dict(values["bank"])
    if stage == "e5":
        return dict(json.loads(Path(values["existing_banks"]["e5"]["artifact"]).read_text(encoding="utf-8"))["settings"]["bank"])
    if stage == "g_guard":
        return dict(_yaml(Path(config_dir) / values["existing_banks"]["g_guard"]["config"])["evaluation"]["bank"])
    raise ValueError(stage)


def _ctx(config_dir: Path, run_stage: str, bank_stage: str) -> dict[str, Any]:
    values, _ = _load(config_dir, run_stage)
    return prepare(config_dir, _stage_bank(values, config_dir, bank_stage))


def _job(ctx: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    if job["kind"] == "fresh":
        rows = [fresh_record(ctx, int(job["index"]), d, f, int(job["B"]), int(job["mc_seed"])) for d, f in job["cells"]]
    else:
        rows = [existing_record(ctx, int(job["index"]), d, f, job["role"], int(job["B"]), int(job["mc_seed"])) for d, f in job["cells"]]
    return {"key": (str(job["bank_stage"]), int(job["index"])), "rows": rows}


def _worker_init(root: str, config_dir: str, run_stage: str, bank_stage: str) -> None:
    os.chdir(root)
    _W.clear()
    _W.update(_ctx(Path(config_dir), run_stage, bank_stage))


def _worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    out = _job(_W, job)
    out["instance_hash"] = _W["instance_hash"]
    return out


def _run_jobs(config_dir: Path, run_stage: str, bank_stage: str, jobs: list[dict], workers: int, progress) -> list[dict]:
    ctx = _ctx(config_dir, run_stage, bank_stage)
    pool = None
    saved = {name: os.environ.get(name) for name in harness.BLAS_THREAD_VARIABLES}
    try:
        if int(workers) > 1:
            for name in harness.BLAS_THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(str(Path.cwd()), str(Path(config_dir).resolve()), run_stage, bank_stage))
        else:
            _W.clear()
            _W.update(ctx)
        results = _execute(pool, _worker_job, jobs, f"observed RDN v2 [{bank_stage}]", progress)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if any(x.get("instance_hash") != ctx["instance_hash"] for x in results.values()):
        raise RuntimeError("worker-rebuilt constructions differ from the parent constructions")
    return [row for key in sorted(results) for row in results[key]["rows"]]


# ---------------------------------------------------------------------------
# Analysis (pure)
# ---------------------------------------------------------------------------


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "ci95": e5.clopper_pearson(int(k), int(n)) if n else None}


def _cert(r: Mapping[str, Any], version: str, alpha: float) -> bool:
    return float(r["tests"][version]["p_monte_carlo"]) <= alpha


def analyse_bank(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    alphas = [float(values["test"]["alpha_primary"]), *[float(a) for a in values["test"].get("alpha_reported", [])]]
    a0 = alphas[0]
    bank = values["bank"]
    out: dict[str, Any] = {"by_family": {}}
    for fam in bank["families"]:
        block = {}
        for scope in ("medium", "low", "pooled"):
            sr = [r for r in rows if r["family"] == fam and (scope == "pooled" or r["deficit"] == scope)]
            b = {f"v2_certified_at_{a:g}": _rate(sum(_cert(r, "v2", a) for r in sr), len(sr)) for a in alphas}
            b.update({f"v1_certified_at_{a:g}": _rate(sum(_cert(r, "v1", a) for r in sr), len(sr)) for a in alphas})
            b["screen_pass"] = _rate(sum(r["screen_passes"] for r in sr), len(sr))
            b["v1_v2_decisions"] = {f"v1_{'c' if x else 'n'}|v2_{'c' if y else 'n'}": sum(_cert(r, "v1", a0) == x and _cert(r, "v2", a0) == y for r in sr)
                                    for x in (True, False) for y in (True, False)}
            b["v2_chi2_vs_mc_decision_differences"] = int(sum((r["tests"]["v2"]["p_chi2"] <= a0) != _cert(r, "v2", a0) for r in sr))
            b["truth_kappa"] = e5._q([r["truth"]["kappa"] for r in sr]) if sr and "truth" in sr[0] else None
            b["clock_marginal_spread"] = e5._q([r["truth"]["clock_marginal_spread"] for r in sr]) if sr and "truth" in sr[0] else None
            block[scope] = b
        out["by_family"][fam] = block
    out["type_one"] = {fam: {scope: {v: out["by_family"][fam][scope][f"{v}_certified_at_{a0:g}"] for v in ("v2", "v1")} for scope in ("medium", "low", "pooled")}
                       for fam in bank["null_families"]}
    bins = [float(b) for b in values["analysis"]["kappa_bins"]]
    nonnull = [r for r in rows if r["family"] not in bank["null_families"] and r["family"] not in bank.get("assumption_violating_families", [])]
    dose = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        for d in DEFICITS:
            sr = [r for r in nonnull if r["deficit"] == d and lo <= r["truth"]["kappa"] < hi]
            dose[f"{d}:[{lo:g},{hi:g})"] = {"v2": _rate(sum(_cert(r, "v2", a0) for r in sr), len(sr)), "v1": _rate(sum(_cert(r, "v1", a0) for r in sr), len(sr))}
    out["dose_response"] = dose
    return out


def analyse_existing(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], stored: Mapping[tuple, Mapping[str, Any]], bank: str) -> dict[str, Any]:
    """Join with the model route via the v1 analysis (read only), once per test version."""
    out = {}
    for version in ("v2", "v1"):
        rs = [{**r, "test": r["tests"][version]} for r in rows]
        out[version] = orn.analyse_existing({"test": {"alpha_primary": float(values["test"]["alpha_primary"])}}, rs, stored, bank)
    return out


def _endpoint_families(e: Mapping[str, Any]) -> list[str]:
    """An endpoint names one family or a list of families (pooled over them)."""
    f = e["family"]
    return [str(f)] if isinstance(f, str) else [str(x) for x in f]


def sealed_endpoints(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a0 = float(values["test"]["alpha_primary"])
    ep = values["endpoints"]
    results = {}
    for e in ep["tests"]:
        fams = set(_endpoint_families(e))
        sr = [r for r in rows if r["family"] in fams and (e["stratum"] == "pooled" or r["deficit"] == e["stratum"])]
        k = sum(_cert(r, "v2", a0) for r in sr)
        results[e["name"]] = binomial_endpoint(k, len(sr), float(e["p0"]), e["direction"], float(ep["alpha_per_test"]))
    passes = all(r["passes"] for r in results.values())
    return {"tests": results, "combination": ep["combination"], "claim_passes": bool(passes),
            "failed": [name for name, r in results.items() if not r["passes"]]}


# ---------------------------------------------------------------------------
# Retrospective S3-support analysis (post hoc; sealed stage)
# ---------------------------------------------------------------------------


def s3_retrospective(config_dir: Path, values: Mapping[str, Any], progress) -> dict[str, Any]:
    """Apply the fixed v2 rule to every S3 evaluation support; analyse the orbit of every S3 failure."""
    spec = values["s3_retrospective"]
    art_path = Path(spec["artifact"])
    if file_sha256(art_path) != spec["artifact_sha256"]:
        raise RuntimeError("the S3 artifact differs from its recorded SHA-256")
    stored = json.loads(art_path.read_text(encoding="utf-8"))
    ctx = s3._prepare_light(config_dir)
    s3._open_sealed_bank(ctx)
    a0 = float(values["test"]["alpha_primary"])
    B, mc = int(values["test"]["monte_carlo_draws"]), int(values["test"]["monte_carlo_seed"])
    K = ctx["K"]
    rows, mismatches = [], []
    for deficit in DEFICITS:
        for srow in stored["records"][deficit]:
            r = ctx["records"][int(srow["index"])]
            if r.deficit != deficit or int(r.instance_index) != int(srow["block"]):
                raise RuntimeError(f"S3 record {srow['index']} does not regenerate to block {srow['block']}")
            support = {c: dict(row) for c, row in r.support.table.probabilities.items()}
            n_shot = int(round(float(r.support.table.metadata["n_shots"])))
            screen = ctx["screens"][int(srow["index"])]
            if bool(screen["passes"]) != bool(srow["screen_passes"]):
                mismatches.append((deficit, int(srow["block"])))
            counts = orn.d_counts(support, n_shot, K)
            tests = run_tests(counts, K, B, f"{mc}:s3_retrospective:{deficit}:{int(srow['block'])}")
            rows.append({"deficit": deficit, "block": int(srow["block"]), "index": int(srow["index"]), "s3_success": bool(srow["success"]), "screen_passes": bool(srow["screen_passes"]),
                         "bi": srow["bi"]["status"], "rdn": srow["rdn"]["status"], "observed": sorted(support), "tests": tests})
    summary = {}
    for deficit in DEFICITS:
        sr = [r for r in rows if r["deficit"] == deficit]
        summary[deficit] = {"n": len(sr), "v2_certified": _rate(sum(_cert(r, "v2", a0) for r in sr), len(sr)), "v1_certified": _rate(sum(_cert(r, "v1", a0) for r in sr), len(sr)),
                            "s3_success": sum(r["s3_success"] for r in sr),
                            "cells": {f"s3_{'ok' if x else 'fail'}|v2_{'c' if y else 'n'}": sum(r["s3_success"] == x and _cert(r, "v2", a0) == y for r in sr) for x in (True, False) for y in (True, False)},
                            "v2_p_max": max(float(r["tests"]["v2"]["p_monte_carlo"]) for r in sr)}
    failures = [failure_analysis(ctx, r, stored, a0) for r in rows if not r["s3_success"]]
    return {"label": "post_hoc_retrospective_on_opened_sealed_bank", "regeneration_screen_mismatches": mismatches, "summary": summary, "failures": failures, "records": rows}


def failure_analysis(ctx: Mapping[str, Any], row: Mapping[str, Any], stored: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    """D commutant, exact class and fiber coverage of one S3 failure (refit with the frozen estimator)."""
    deficit, idx = row["deficit"], int(row["index"])
    r = ctx["records"][idx]
    inst = ctx["instances"][int(r.instance_index)]
    observed = row["observed"]
    Rs = dc.d_commutant(observed)
    out: dict[str, Any] = {"deficit": deficit, "block": row["block"], "screen_passes": row["screen_passes"], "bi": row["bi"], "rdn": row["rdn"],
                           "v2_certified": _cert(row, "v2", alpha), "v2_p": row["tests"]["v2"]["p_monte_carlo"], "d_commutant": Rs,
                           "unitary": inst.system_unitary_label, "observed": observed}
    U = np.asarray(getattr(inst.system_unitary, "data", inst.system_unitary), dtype=complex)
    n = len(ctx["p1"].generator["active_modes"])
    if len(Rs) == 1:
        cond = dc.branch_condition(U, n, Rs[0], observed)
        out["branch_condition"] = float(cond)
        out["class"] = "3a" if cond < 1.0e-10 else "3b"
    else:
        out["class"] = "no_D_commutant" if not Rs else "multiple_D_commutants"
    if not row["screen_passes"]:
        srow = next(s for s in stored["records"][deficit] if int(s["index"]) == idx)
        out["screen_reasons"] = srow["screen_reasons"]
        return out
    support = MeasurementTable(probabilities=r.support.table.probabilities, contexts=[ctx["context_by_id"][c] for c in sorted(r.support.table.probabilities)],
                               metadata={"n_shots": float(r.support.table.metadata["n_shots"])}, source_density=None)
    prediction = ctx["sbp"].predict_table(support, list(ctx["contexts"]))
    table = _table_dict(prediction)
    exact = {c: dict(row_) for c, row_ in p1_measurement_table(inst.density, ctx["contexts"], K=inst.K, eta=0.0, metadata={}).copy_without_source().probabilities.items()}
    radii = ctx["bodies"]["radii"][deficit]
    ids = [c.context_id for c in ctx["contexts"]]
    sub = {"contexts": ctx["contexts"], "K": ctx["K"], "ids": ids, "fw": ctx["sbp"].forward, "bodies": {"radii": radii}}
    scores = am.class_errors(sub, table, exact, ids)
    srow = next(s for s in stored["records"][deficit] if int(s["index"]) == idx)
    out["refit_reproduces_stored_scores"] = bool(abs(scores["diag"] - float(srow["diag_score"])) < 1.0e-6 and abs(scores["coh"] - float(srow["coh_score"])) < 1.0e-6)
    out["truth_ratio"] = {"diag": scores["diag"] / radii["diag"], "coh": scores["coh"] / radii["coh"]}
    out["fit_starts_within_near_optimal"] = int(prediction.metadata["shared_branch_pair_fit"]["starts_within_near_optimal"])
    if len(Rs) == 1:
        V = _true_branch_vectors(inst)
        out["fiber"] = ph.fiber_coverage(sub, table, V, Rs[0])
    return out


# ---------------------------------------------------------------------------
# Declaration, guard, run
# ---------------------------------------------------------------------------


def _folder(values: Mapping[str, Any], output_dir: str | Path) -> Path:
    return Path(output_dir) / Path(str(values["output_root"])).name


def _extra_files(config_dir: Path, values: Mapping[str, Any], stage: str) -> list[str]:
    names = [CONFIGS[stage], "p1_generator.yaml", "p1_measurement.yaml", "p1_seeds.yaml", "p1_ilgc_operational_r4.yaml", str(values["design_source"]["development_config"])]
    if "existing_banks" in values:
        names.append(str(values["existing_banks"]["g_guard"]["config"]))
    if "s3_retrospective" in values:
        names.append(str(values["s3_retrospective"]["config"]))
    return [str(Path(config_dir) / n) for n in dict.fromkeys(names)] + [*values["design_source"]["dev_inputs"], str(values["protocol_document"]), RUNNER]


def declare(stage: str, config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict:
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir, stage)
    folder = _folder(values, output_dir)
    target = folder / DECLARATIONS[stage]
    if target.exists():
        raise FileExistsError(f"declaration already exists and is immutable: {target}")
    dev_inputs = am._dev_inputs(root, values)
    p1 = load_p1_config(config_dir)
    prints = sorted(physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, values["bank"]))
    known = {"p1_construction_grid": {physical_fingerprint(i) for i in construction_bank_p1(p1.generator)}}
    unchecked = {}
    for path, spec in harness.declared_bank_specs(config_dir, exclude=CONFIGS[stage]):
        try:
            known[path] = {physical_fingerprint(i) for i in seeded_construction_bank_p1(p1.generator, spec)}
        except (ValueError, RuntimeError, KeyError) as error:
            unchecked[path] = repr(error)
    overlaps = {name: sorted(set(prints) & f) for name, f in known.items() if set(prints) & f}
    if overlaps:
        raise ValueError(f"the bank is not disjoint: {sorted(overlaps)}")
    snapshot = import_closure_snapshot(root, extra_files=_extra_files(config_dir, values, stage))
    declaration = {"artifact_type": f"observed_rdn_v2_{stage}_declaration", "status": values["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "config_sha256": config_sha, "protocol_document": str(values["protocol_document"]), "protocol_document_sha256": file_sha256(root / str(values["protocol_document"])),
                   "dev_inputs": dev_inputs, "test": values["test"], "endpoints": values.get("endpoints"),
                   "bank": {**values["bank"], "construction_fingerprint_hash": hashlib.sha256("".join(prints).encode("ascii")).hexdigest()[:16]},
                   "disjointness": {"checked_bank_count": len(known), "unchecked_bank_specs": unchecked, "overlaps": overlaps},
                   "source_snapshot": snapshot, "source_snapshot_hash": snapshot_hash(snapshot)}
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return declaration


def run(stage: str, config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, workers: int = 1, progress: Callable[[str], None] | None = print,
        existing: bool = True, retrospective: bool = True) -> dict:
    started = time.perf_counter()
    root = Path.cwd()
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir, stage)
    folder = _folder(values, output_dir)
    declaration = json.loads((folder / DECLARATIONS[stage]).read_text(encoding="utf-8"))
    if (folder / ARTIFACTS[stage]).exists():
        raise FileExistsError("the artifact already exists and is immutable")
    if declaration["config_sha256"] != config_sha or declaration["protocol_document_sha256"] != file_sha256(root / str(values["protocol_document"])):
        raise RuntimeError("config or protocol drifted after declaration")
    extra = [p for p in declaration["source_snapshot"] if not p.startswith("src/")]
    drift = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    if drift["changed"] or drift["missing"]:
        raise RuntimeError(f"source drift after declaration: {drift}")
    B, mc = int(values["test"]["monte_carlo_draws"]), int(values["test"]["monte_carlo_seed"])
    bank = values["bank"]
    cells = [(d, f) for d in DEFICITS for f in bank["families"]]
    bank_stage = "fresh" if stage == "dev" else "sealed"
    jobs = [{"kind": "fresh", "bank_stage": bank_stage, "index": i, "cells": cells, "B": B, "mc_seed": mc} for i in range(int(bank["block_count"]))]
    rows = _run_jobs(config_dir, stage, bank_stage, jobs, workers, progress)
    artifact: dict[str, Any] = {"artifact_type": f"observed_rdn_v2_{stage}", "status": values["status"], "bank_analysis": analyse_bank(values, rows), "records": rows}
    if stage == "sealed":
        artifact["endpoints"] = sealed_endpoints(values, rows)
        if retrospective and values.get("s3_retrospective", {}).get("enabled"):
            artifact["s3_retrospective"] = s3_retrospective(config_dir, values, progress)
    if stage == "dev" and existing:
        for name in ("e5", "g_guard"):
            spec = values["existing_banks"][name]
            stored_art = json.loads(Path(spec["artifact"]).read_text(encoding="utf-8"))
            stored = {(int(r["index"]), r["deficit"], r["family"]): r for r in stored_art["records"]}
            by_index: dict[int, list] = {}
            for (i, d, f) in stored:
                by_index.setdefault(i, []).append((d, f))
            ejobs = [{"kind": "existing", "bank_stage": name, "index": i, "cells": sorted(v), "role": spec["role"], "B": B, "mc_seed": mc} for i, v in sorted(by_index.items())]
            erows = _run_jobs(config_dir, stage, name, ejobs, workers, progress)
            artifact[f"existing_{name}"] = analyse_existing(values, erows, stored, name)
            artifact[f"existing_{name}_records"] = erows
    artifact["settings"] = {"test": values["test"], "bank": bank}
    artifact["claim_boundary"] = {"stage": stage, "model_free": True, "sealed_bank_opened": stage == "sealed",
                                  "exact_states_used_for": "truth families, true kappa and the S3 failure analysis; the test uses observed counts only"}
    artifact["execution"] = {"workers": int(workers), "records": len(rows), "wall_seconds": time.perf_counter() - started}
    artifact = m1._jsonable(artifact)
    artifact["declaration_sha256"] = file_sha256(folder / DECLARATIONS[stage])
    artifact["config_sha256"] = config_sha
    artifact["source_snapshot_hash_declared"] = declaration["source_snapshot_hash"]
    artifact["source_snapshot_post_run_drift"] = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=extra))
    (folder / ARTIFACTS[stage]).write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
