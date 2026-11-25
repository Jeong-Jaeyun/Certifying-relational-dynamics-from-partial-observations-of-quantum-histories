"""Design 2 exploratory pilot (documents/design2-pilot-design.md): one added context at finite shots.

Jobs are (design, family, record, N) for the truth families and (design, control, record, N) for the static controls
of the conditional test. Every random draw is a deterministic function of a SHA-256 seed text. Process pool of at most
six workers with one BLAS thread each.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import expm

from bi_qmn.p3.runs import design2_core as d2
from bi_qmn.p3.runs.observed_rdn_v2 import both_margin_test

DESIGN_DOC = "documents/design2-pilot-design.md"
SOURCES = ["src/bi_qmn/p3/runs/design2_core.py", "src/bi_qmn/p3/runs/design2_pilot.py",
           "src/bi_qmn/p3/models/p3_shared_branch_pair.py", "src/bi_qmn/p3/runs/observed_rdn_v2.py", DESIGN_DOC]
SEED_CONSTRUCT, SEED_OBS, SEED_PERTURB, SEED_MC, SEED_BOOT = 57101, 57102, 57103, 57104, 57105
DESIGNS = ("golomb", "collision")
FAMILIES = ("exact", "near_0.03", "near_0.1", "generic", "structured_equal", "structured_random")
CONTROLS = ("static_uniform", "static_nonuniform")
SHOTS = (512, 2048, 8192)
TGRID = np.unique(np.concatenate([np.linspace(0.0, math.pi, 25)[1:-1], np.logspace(-3, math.log10(0.5), 16)]))


def seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Truths
# ---------------------------------------------------------------------------------------------------------------


def construction(i: int) -> dict[str, Any]:
    """Exact type-3a basis (the condition does not depend on the active modes), weights, phases and a perturbation."""
    rng = np.random.default_rng(seed(f"{SEED_CONSTRUCT}:exact:{i}"))
    des = d2.Design.make("golomb")
    U, res = d2.project_exact(d2.haar_unitary(4, rng), des, rng)
    while True:
        p = rng.dirichlet(np.full(4, 8.0))
        if p.min() >= 0.1:
            break
    ph = np.concatenate([[0.0], rng.uniform(0.0, 2 * math.pi, 3)])
    hr = np.random.default_rng(seed(f"{SEED_PERTURB}:h:{i}"))
    h = hr.normal(size=(4, 4)) + 1j * hr.normal(size=(4, 4)); h = (h + h.conj().T) / 2; h /= np.linalg.norm(h)
    gr = np.random.default_rng(seed(f"{SEED_CONSTRUCT}:generic:{i}"))
    Ug = d2.haar_unitary(4, gr)
    while True:
        pg = gr.dirichlet(np.full(4, 8.0))
        if pg.min() >= 0.1:
            break
    phg = np.concatenate([[0.0], gr.uniform(0.0, 2 * math.pi, 3)])
    # structured exact basis exp(i alpha X) (x) I: the branch-basis matrix of IZ is diagonal
    sr = np.random.default_rng(seed(f"{SEED_CONSTRUCT}:structured:{i}"))
    alpha = float(sr.uniform(0.15, math.pi / 2 - 0.15))
    Us = np.kron(expm(1j * alpha * d2._P1["X"]), np.eye(2))
    while True:
        ps = sr.dirichlet(np.full(4, 8.0))
        if ps.min() >= 0.1:
            break
    phs = np.concatenate([[0.0], sr.uniform(0.0, 2 * math.pi, 3)])
    return {"U": U, "residual": res, "p": p, "ph": ph, "h": h, "Ug": Ug, "pg": pg, "phg": phg, "Us": Us, "alpha": alpha, "ps": ps, "phs": phs}


def truth(family: str, c: Mapping[str, Any]) -> np.ndarray:
    if family == "exact":
        return d2.branch_vectors(c["U"], c["p"], c["ph"])
    if family.startswith("near_"):
        eps = float(family.split("_")[1])
        return d2.branch_vectors(expm(1j * eps * c["h"]) @ c["U"], c["p"], c["ph"])
    if family == "generic":
        return d2.branch_vectors(c["Ug"], c["pg"], c["phg"])
    if family == "structured_equal":
        return d2.branch_vectors(c["Us"], [0.25] * 4, c["phs"])
    if family == "structured_random":
        return d2.branch_vectors(c["Us"], c["ps"], c["phs"])
    raise ValueError(family)


def sample(table: np.ndarray, n: int, text: str) -> np.ndarray:
    p = np.clip(np.asarray(table, float), 0.0, None)
    return np.random.default_rng(seed(text)).multinomial(int(n), p / p.sum())


def as_support(counts: Mapping[str, np.ndarray], n_ref: float, K: int) -> dict[str, dict[str, float]]:
    return {cid: {f"c{l}_s{s}": float(v[2 * l + s]) / n_ref for l in range(K) for s in (0, 1)} for cid, v in counts.items()}


def d_counts(counts: Mapping[str, np.ndarray], K: int) -> dict[str, np.ndarray]:
    return {cid: np.asarray(v, dtype=np.int64).reshape(K, 2) for cid, v in counts.items() if cid.startswith("D,")}


def fitted(est, support, n_ref) -> np.ndarray:
    fit = est.fit_support(support, n_ref)
    return np.asarray(fit["branch_vectors_real_imag"][0]) + 1j * np.asarray(fit["branch_vectors_real_imag"][1])


def l2_error(des, Vhat: np.ndarray, Vtrue: np.ndarray, ctx: Sequence[str]) -> float:
    A, B = d2.tables(des, Vhat, ctx), d2.tables(des, Vtrue, ctx)
    return float(math.sqrt(sum(float(np.sum((A[c] - B[c]) ** 2)) for c in ctx)))


def best_bound(des, V, shots: Mapping[str, float], ctx: Sequence[str]) -> dict[str, float]:
    best = {"bound": 0.0, "t": None, "d": 0.0}
    for t in TGRID:
        b = d2.two_point_bound(des, V, d2.rotate(V, des.R, float(t)), shots, ctx)
        if b["bound"] > best["bound"]:
            best = {"bound": b["bound"], "t": float(t), "d": b["d"]}
    return best


# ---------------------------------------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------------------------------------

_W: dict[str, Any] = {}


def _init() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    _W.clear()


def _est(design: str, starts: int):
    key = f"est:{design}:{starts}"
    if key not in _W:
        _W[key] = d2.estimator(d2.Design.make(design), starts=starts)
    return _W[key]


def truth_job(job: Mapping[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    design, family, i, N, starts = job["design"], job["family"], int(job["i"]), int(job["N"]), int(job["starts"])
    des = d2.Design.make(design)
    K = des.K
    c = construction(i)
    V = truth(family, c)
    tag = f"{SEED_OBS}:{design}:{family}:{i}:{N}"
    exact_tabs = d2.tables(des, V, des.support + des.candidates)
    sup = {cid: sample(exact_tabs[cid], N, f"{tag}:support:{cid}") for cid in des.support}
    extra = {cid: sample(exact_tabs[cid], N // 8, f"{tag}:extra:{cid}") for cid in des.support}
    cand = {cid: sample(exact_tabs[cid], N, f"{tag}:cand:{cid}") for cid in des.candidates}
    est = _est(design, starts)
    fits: dict[str, np.ndarray] = {}
    fits["A0"] = fitted(est, as_support(sup, N, K), N)
    fits["C"] = fitted(est, as_support({k: sup[k] + extra[k] for k in sup}, N, K), N)
    # rules at the A0 fit (support only) and the oracle at the truth
    sQ = {q: d2.s_Q(des, fits["A0"], q) for q in des.candidates}
    sig_fit = {q: d2.separation(des, fits["A0"], q)["sigma_min"] for q in des.candidates}
    sig_true = {q: d2.separation(des, V, q)["sigma_min"] for q in des.candidates}
    picks = {"A_s": max(sQ, key=sQ.get), "A_sig": max(sig_fit, key=sig_fit.get), "A_IZ": "D,IZ", "A_o": max(sig_true, key=sig_true.get)}
    by_ctx: dict[str, np.ndarray] = {}
    for arm, q in picks.items():
        if q not in by_ctx:
            by_ctx[q] = fitted(est, as_support({**sup, q: cand[q]}, N, K), N)
        fits[arm] = by_ctx[q]
    O = [cid for cid in des.unobserved_D if cid not in set(picks.values())]
    errors = {arm: l2_error(des, Vh, V, O) for arm, Vh in fits.items()}
    shots0 = {cid: float(N) for cid in des.support}
    bounds = {"A0": best_bound(des, V, shots0, O), "A_sig": best_bound(des, V, {**shots0, picks["A_sig"]: float(N)}, O),
              "C": best_bound(des, V, {cid: N * 9 / 8 for cid in des.support}, O)}
    rdn = both_margin_test(d_counts(sup, K), 10000, np.random.default_rng(seed(f"{SEED_MC}:{tag}")))
    return {"kind": "truth", "design": design, "family": family, "i": i, "N": N, "picks": picks, "O": O, "errors": errors,
            "rank1_picks": {arm: int(d2.separation(des, V, q)["rank"] < 2) for arm, q in picks.items()},
            "sigma_true_of_pick": {arm: sig_true[q] for arm, q in picks.items()}, "sigma_true": sig_true, "sQ_fit": sQ,
            "sigma_fit": sig_fit, "bounds": bounds, "rdn": {"p": rdn["p_monte_carlo"], "G": rdn["G"]},
            "residual": float(c["residual"]), "seconds": time.perf_counter() - t0}


def control_job(job: Mapping[str, Any]) -> dict[str, Any]:
    design, control, i, N = job["design"], job["control"], int(job["i"]), int(job["N"])
    des = d2.Design.make(design)
    K = des.K
    c = construction(i)
    V = truth("exact", c)
    # pinched history: conditional state rho_bar = sum_j |v_j><v_j| (unnormalized weights sum to 1), uniform clock marginal
    rho_bar = V @ V.conj().T
    if control == "static_uniform":
        w = np.full(K, 1.0 / K)
    else:
        phi = float(np.random.default_rng(seed(f"{SEED_CONSTRUCT}:phi:{i}")).uniform(0, 2 * math.pi))
        w = 1.0 + 0.5 * np.cos(2 * math.pi * np.arange(K) / K + phi); w /= w.sum()
    tag = f"{SEED_OBS}:{design}:{control}:{i}:{N}"
    counts = {}
    for P in des.observed_D:
        Pm = d2.pauli(P)
        q1 = float(np.real(np.trace(rho_bar @ (np.eye(4) + Pm) / 2)))
        tab = np.array([[w[l] * q1, w[l] * (1 - q1)] for l in range(K)]).reshape(-1)
        counts[f"D,{P}"] = sample(tab, N, f"{tag}:D,{P}")
    rdn = both_margin_test(d_counts(counts, K), 10000, np.random.default_rng(seed(f"{SEED_MC}:{tag}")))
    return {"kind": "control", "design": design, "control": control, "i": i, "N": N, "rdn": {"p": rdn["p_monte_carlo"], "G": rdn["G"]}}


def _run(job):
    return truth_job(job) if job["kind"] == "truth" else control_job(job)


def run(records: int = 48, shots: Sequence[int] = SHOTS, workers: int = 6, starts: int = 32, out: str | Path = "outputs/design2/pilot.json",
        families: Sequence[str] = FAMILIES, designs: Sequence[str] = DESIGNS, progress=print) -> dict[str, Any]:
    t0 = time.perf_counter()
    jobs = [{"kind": "truth", "design": d, "family": f, "i": i, "N": N, "starts": starts} for d in designs for f in families for i in range(records) for N in shots]
    jobs += [{"kind": "control", "design": d, "control": c, "i": i, "N": N} for d in designs for c in CONTROLS for i in range(records) for N in shots]
    jobs.sort(key=lambda j: (j["kind"] != "truth", -j["N"]))       # long jobs first
    rows = []
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_init) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if progress and k % 50 == 0:
                progress(f"{k}/{len(jobs)} ({time.perf_counter() - t0:.0f}s)")
    artifact = {"artifact_type": "design2_pilot", "status": "exploratory_pilot",
                "settings": {"records": records, "shots": list(shots), "starts": starts, "families": list(families), "designs": list(designs),
                             "seeds": {"construct": SEED_CONSTRUCT, "obs": SEED_OBS, "perturb": SEED_PERTURB, "mc": SEED_MC, "boot": SEED_BOOT}},
                "sources_sha256": {p: file_sha256(p) for p in SOURCES},
                "execution": {"workers": int(workers), "jobs": len(jobs), "wall_seconds": time.perf_counter() - t0},
                "rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(artifact, default=float) + "\n", encoding="utf-8")
    return artifact
