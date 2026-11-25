"""Design 2 sealed confirmatory run (documents/design2-confirmatory-preregistration.md).

``declare`` freezes the preregistration, the config and the import closure; ``run`` refuses on drift, generates the
bank from the reserved seeds, fits every arm, evaluates the eight endpoints and writes the artifact once.
Process pool of at most six workers with one BLAS thread each.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.linalg import expm
from scipy.stats import binomtest

from bi_qmn.core.provenance import file_sha256, import_closure_snapshot, snapshot_drift, snapshot_hash
from bi_qmn.p3.runs import design2_core as d2
from bi_qmn.p3.runs import design2_pilot as dp
from bi_qmn.p3.runs.observed_rdn_v2 import both_margin_test

RUNNER = "runners/p3/run_design2_confirm.py"
DECLARATION = "design2_confirmatory_declaration.json"
ARTIFACT = "design2_confirmatory.json"
RESERVED = (57200, 57299)
ARMS = ("A0", "C", "A_s", "A_sig", "A_IZ", "A_o")


def load(config: str | Path) -> dict[str, Any]:
    values = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    status = values.get("status")
    if status not in ("sealed_confirmatory", "test_only"):
        raise ValueError("status must be sealed_confirmatory or test_only")
    seeds = [int(v) for v in values["seeds"].values()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")
    inside = [RESERVED[0] <= s <= RESERVED[1] for s in seeds]
    if status == "sealed_confirmatory" and not all(inside):
        raise ValueError("sealed seeds must lie in the reserved block 57200-57299")
    if status == "test_only" and any(inside):
        raise ValueError("test seeds must not touch the reserved block")
    for f in values["families"]:
        if f not in dp.FAMILIES:
            raise ValueError(f"unknown family {f}")
    return values


# ---------------------------------------------------------------------------------------------------------------
# Bank (the pilot's constructions with the config's seeds)
# ---------------------------------------------------------------------------------------------------------------


def construction(i: int, seeds: Mapping[str, int]) -> dict[str, Any]:
    rng = np.random.default_rng(dp.seed(f"{seeds['construct']}:exact:{i}"))
    des = d2.Design.make("golomb")
    U, res = d2.project_exact(d2.haar_unitary(4, rng), des, rng)
    while True:
        p = rng.dirichlet(np.full(4, 8.0))
        if p.min() >= 0.1:
            break
    ph = np.concatenate([[0.0], rng.uniform(0.0, 2 * math.pi, 3)])
    hr = np.random.default_rng(dp.seed(f"{seeds['perturb']}:h:{i}"))
    h = hr.normal(size=(4, 4)) + 1j * hr.normal(size=(4, 4)); h = (h + h.conj().T) / 2; h /= np.linalg.norm(h)
    gr = np.random.default_rng(dp.seed(f"{seeds['construct']}:generic:{i}"))
    Ug = d2.haar_unitary(4, gr)
    while True:
        pg = gr.dirichlet(np.full(4, 8.0))
        if pg.min() >= 0.1:
            break
    phg = np.concatenate([[0.0], gr.uniform(0.0, 2 * math.pi, 3)])
    sr = np.random.default_rng(dp.seed(f"{seeds['construct']}:structured:{i}"))
    alpha = float(sr.uniform(0.15, math.pi / 2 - 0.15))
    Us = np.kron(expm(1j * alpha * d2._P1["X"]), np.eye(2))
    while True:
        ps = sr.dirichlet(np.full(4, 8.0))
        if ps.min() >= 0.1:
            break
    phs = np.concatenate([[0.0], sr.uniform(0.0, 2 * math.pi, 3)])
    return {"U": U, "residual": res, "p": p, "ph": ph, "h": h, "Ug": Ug, "pg": pg, "phg": phg, "Us": Us, "alpha": alpha, "ps": ps, "phs": phs}


_W: dict[str, Any] = {}


def _init() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    _W.clear()


def _est(design: str, starts: int):
    key = f"{design}:{starts}"
    if key not in _W:
        _W[key] = d2.estimator(d2.Design.make(design), starts=starts)
    return _W[key]


def truth_job(job: Mapping[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    seeds, design, family, i, N, starts = job["seeds"], job["design"], job["family"], int(job["i"]), int(job["N"]), int(job["starts"])
    des = d2.Design.make(design)
    K = des.K
    c = construction(i, seeds)
    V = dp.truth(family, c)
    tag = f"{seeds['obs']}:{design}:{family}:{i}:{N}"
    tabs = d2.tables(des, V, des.support + des.candidates)
    sup = {cid: dp.sample(tabs[cid], N, f"{tag}:support:{cid}") for cid in des.support}
    extra = {cid: dp.sample(tabs[cid], N // 8, f"{tag}:extra:{cid}") for cid in des.support}
    cand = {cid: dp.sample(tabs[cid], N, f"{tag}:cand:{cid}") for cid in des.candidates}
    est = _est(design, starts)
    fits = {"A0": dp.fitted(est, dp.as_support(sup, N, K), N),
            "C": dp.fitted(est, dp.as_support({k: sup[k] + extra[k] for k in sup}, N, K), N)}
    sQ = {q: d2.s_Q(des, fits["A0"], q) for q in des.candidates}
    sig_fit = {q: d2.separation(des, fits["A0"], q)["sigma_min"] for q in des.candidates}
    sig_true = {q: d2.separation(des, V, q)["sigma_min"] for q in des.candidates}
    picks = {"A_s": max(sQ, key=sQ.get), "A_sig": max(sig_fit, key=sig_fit.get), "A_IZ": "D,IZ", "A_o": max(sig_true, key=sig_true.get)}
    by_ctx: dict[str, np.ndarray] = {}
    for arm, q in picks.items():
        if q not in by_ctx:
            by_ctx[q] = dp.fitted(est, dp.as_support({**sup, q: cand[q]}, N, K), N)
        fits[arm] = by_ctx[q]
    O = [cid for cid in des.unobserved_D if cid not in set(picks.values())]
    errors = {arm: dp.l2_error(des, Vh, V, O) for arm, Vh in fits.items()}
    shots0 = {cid: float(N) for cid in des.support}
    bounds = {"A0": dp.best_bound(des, V, shots0, O), "A_sig": dp.best_bound(des, V, {**shots0, picks["A_sig"]: float(N)}, O),
              "C": dp.best_bound(des, V, {cid: N * 9 / 8 for cid in des.support}, O)}
    rdn = both_margin_test(dp.d_counts(sup, K), int(job["draws"]), np.random.default_rng(dp.seed(f"{seeds['mc']}:{tag}")))
    return {"kind": "truth", "design": design, "family": family, "i": i, "N": N, "picks": picks, "O": O, "errors": errors,
            "rank1_picks": {arm: int(d2.separation(des, V, q)["rank"] < 2) for arm, q in picks.items()},
            "sigma_true_of_pick": {arm: sig_true[q] for arm, q in picks.items()}, "bounds": bounds,
            "rdn": {"p": rdn["p_monte_carlo"], "G": rdn["G"]}, "residual": float(c["residual"]), "seconds": time.perf_counter() - t0}


def control_job(job: Mapping[str, Any]) -> dict[str, Any]:
    seeds, design, control, i, N = job["seeds"], job["design"], job["control"], int(job["i"]), int(job["N"])
    des = d2.Design.make(design)
    K = des.K
    V = dp.truth("exact", construction(i, seeds))
    rho_bar = V @ V.conj().T
    if control == "static_uniform":
        w = np.full(K, 1.0 / K)
    else:
        phi = float(np.random.default_rng(dp.seed(f"{seeds['construct']}:phi:{i}")).uniform(0, 2 * math.pi))
        w = 1.0 + 0.5 * np.cos(2 * math.pi * np.arange(K) / K + phi); w /= w.sum()
    tag = f"{seeds['obs']}:{design}:{control}:{i}:{N}"
    counts = {}
    for P in des.observed_D:
        q1 = float(np.real(np.trace(rho_bar @ (np.eye(4) + d2.pauli(P)) / 2)))
        tab = np.array([[w[l] * q1, w[l] * (1 - q1)] for l in range(K)]).reshape(-1)
        counts[f"D,{P}"] = dp.sample(tab, N, f"{tag}:D,{P}")
    rdn = both_margin_test(dp.d_counts(counts, K), int(job["draws"]), np.random.default_rng(dp.seed(f"{seeds['mc']}:{tag}")))
    return {"kind": "control", "design": design, "control": control, "i": i, "N": N, "rdn": {"p": rdn["p_monte_carlo"], "G": rdn["G"]}}


def _run(job):
    return truth_job(job) if job["kind"] == "truth" else control_job(job)


# ---------------------------------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------------------------------


def cvar(x: Sequence[float], level: float = 0.9) -> float:
    x = np.sort(np.asarray(x, float))[::-1]
    return float(np.mean(x[:max(1, math.ceil((1 - level) * len(x)))]))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], B: int, seed_text: str) -> dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0:
        return {"point": float("nan"), "q025": float("nan"), "q975": float("nan")}
    rng = np.random.default_rng(dp.seed(seed_text))
    idx = rng.integers(0, len(a), size=(int(B), len(a)))
    d = np.array([cvar(a[i]) - cvar(b[i]) for i in idx])
    return {"point": cvar(a) - cvar(b), "q025": float(np.quantile(d, 0.025)), "q975": float(np.quantile(d, 0.975))}


def analyse(values: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ep = values["endpoints"]
    B, boot = int(values["bootstrap_resamples"]), int(values["seeds"]["boot"])
    alpha = float(values["test"]["alpha"])
    truth = [r for r in rows if r["kind"] == "truth"]
    cell = lambda d, f: sorted([r for r in truth if r["design"] == d and r["family"] == f], key=lambda r: r["i"])  # noqa: E731
    tests: dict[str, Any] = {}
    gid = 0
    for d in values["designs"]:
        for f in ep["G"]["families"]:
            gid += 1
            rs = cell(d, f)
            x, y = ep["G"]["contrast"]
            res = paired_bootstrap([r["errors"][x] for r in rs], [r["errors"][y] for r in rs], B, f"{boot}:G{gid}")
            tests[f"G{gid}"] = {"design": d, "family": f, "n": len(rs), **res, "passes": bool(len(rs) > 0 and res["q025"] > float(ep["G"]["lower_bound_above"]))}
    zid = 0
    for d in values["designs"]:
        for f in ep["Z"]["families"]:
            zid += 1
            rs = cell(d, f)
            x, y = ep["Z"]["contrast"]
            res = paired_bootstrap([r["errors"][x] for r in rs], [r["errors"][y] for r in rs], B, f"{boot}:Z{zid}")
            tests[f"Z{zid}"] = {"design": d, "family": f, "n": len(rs), **res, "passes": bool(len(rs) > 0 and res["q975"] < float(ep["Z"]["upper_bound_below"]))}
    ctl = [r for r in rows if r["kind"] == "control"]
    k = int(sum(r["rdn"]["p"] <= alpha for r in ctl))
    p = float(binomtest(k, len(ctl), float(ep["V1"]["p0"]), alternative="less").pvalue) if ctl else 1.0
    tests["V1"] = {"k": k, "n": len(ctl), "rate": k / max(1, len(ctl)), "p_value": p, "passes": p < float(ep["V1"]["level"])}
    sens = [r for r in truth if r["family"] in ep["V2"]["families"]]
    k = int(sum(r["rdn"]["p"] <= alpha for r in sens))
    p = float(binomtest(k, len(sens), float(ep["V2"]["p0"]), alternative="greater").pvalue) if sens else 1.0
    tests["V2"] = {"k": k, "n": len(sens), "rate": k / max(1, len(sens)), "p_value": p, "passes": p < float(ep["V2"]["level"])}
    descriptive: dict[str, Any] = {}
    for d in values["designs"]:
        for f in values["families"]:
            rs = cell(d, f)
            if not rs:
                continue
            E = {a: [r["errors"][a] for r in rs] for a in ARMS}
            descriptive[f"{d}|{f}"] = {
                "n": len(rs), "cvar": {a: cvar(E[a]) for a in ARMS}, "median": {a: float(np.median(E[a])) for a in ARMS}, "max": {a: float(np.max(E[a])) for a in ARMS},
                "A_s_minus_A_sig": paired_bootstrap(E["A_s"], E["A_sig"], B, f"{boot}:desc:{d}:{f}:s-sig"),
                "A0_minus_C": paired_bootstrap(E["A0"], E["C"], B, f"{boot}:desc:{d}:{f}:A0-C"),
                "C_minus_A_sig": paired_bootstrap(E["C"], E["A_sig"], B, f"{boot}:desc:{d}:{f}:C-sig"),
                "rule_agreement": float(np.mean([r["picks"]["A_s"] == r["picks"]["A_sig"] for r in rs])),
                "rank1_picks": {a: int(sum(r["rank1_picks"][a] for r in rs)) for a in ("A_s", "A_sig", "A_IZ", "A_o")},
                "bound_median": {a: float(np.median([r["bounds"][a]["bound"] for r in rs])) for a in ("A0", "C", "A_sig")},
                "certified": int(sum(r["rdn"]["p"] <= alpha for r in rs))}
    for d in values["designs"]:
        for c_ in values["controls"]:
            rs = [r for r in ctl if r["design"] == d and r["control"] == c_]
            descriptive[f"{d}|{c_}"] = {"n": len(rs), "certified": int(sum(r["rdn"]["p"] <= alpha for r in rs))}
    return {"endpoints": tests, "claim_passes": all(t["passes"] for t in tests.values()), "descriptive": descriptive}


# ---------------------------------------------------------------------------------------------------------------
# Declaration and run
# ---------------------------------------------------------------------------------------------------------------


def _paths(config: str | Path, values: Mapping[str, Any], out_dir: str | Path | None):
    folder = Path(out_dir) if out_dir else Path(values["output_root"])
    return folder / DECLARATION, folder / ARTIFACT


def _extra(config: str | Path, values: Mapping[str, Any]) -> list[str]:
    return [Path(config).as_posix(), str(values["protocol_document"]), RUNNER]


def declare(config: str | Path = "config/design2_confirmatory.yaml", out_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path.cwd()
    values = load(config)
    decl_path, art_path = _paths(config, values, out_dir)
    if decl_path.exists() or art_path.exists():
        raise FileExistsError("declaration or artifact already exists; both are immutable")
    snap = import_closure_snapshot(root, extra_files=_extra(config, values))
    declaration = {"artifact_type": "design2_confirmatory_declaration", "status": values["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "config": Path(config).as_posix(), "config_sha256": file_sha256(Path(config)),
                   "protocol_document": values["protocol_document"], "protocol_document_sha256": file_sha256(root / values["protocol_document"]),
                   "seeds": values["seeds"], "source_snapshot": snap, "source_snapshot_hash": snapshot_hash(snap)}
    decl_path.parent.mkdir(parents=True, exist_ok=True)
    decl_path.write_text(json.dumps(declaration, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return declaration


def run(config: str | Path = "config/design2_confirmatory.yaml", out_dir: str | Path | None = None, workers: int = 6, progress=print) -> dict[str, Any]:
    t0 = time.perf_counter()
    root = Path.cwd()
    values = load(config)
    decl_path, art_path = _paths(config, values, out_dir)
    if not decl_path.exists():
        raise FileNotFoundError("declare first")
    if art_path.exists():
        raise FileExistsError("the confirmatory artifact already exists and is immutable")
    if not 1 <= int(workers) <= 6:
        raise ValueError("at most 6 workers")
    declaration = json.loads(decl_path.read_text(encoding="utf-8"))
    if declaration["config_sha256"] != file_sha256(Path(config)) or declaration["protocol_document_sha256"] != file_sha256(root / values["protocol_document"]):
        raise RuntimeError("config or protocol drifted after the declaration")
    drift = snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=_extra(config, values)))
    if drift["changed"] or drift["missing"]:
        raise RuntimeError(f"source drift after the declaration: {drift}")
    seeds = {k: int(v) for k, v in values["seeds"].items()}
    N, starts, draws = int(values["shots"]), int(values["estimator_starts"]), int(values["test"]["monte_carlo_draws"])
    jobs = [{"kind": "truth", "seeds": seeds, "design": d, "family": f, "i": i, "N": N, "starts": starts, "draws": draws}
            for d in values["designs"] for f in values["families"] for i in range(int(values["records_per_family"]))]
    jobs += [{"kind": "control", "seeds": seeds, "design": d, "control": c, "i": i, "N": N, "draws": draws}
             for d in values["designs"] for c in values["controls"] for i in range(int(values["records_per_control"]))]
    rows = []
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_init) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if progress and k % 100 == 0:
                progress(f"{k}/{len(jobs)} ({time.perf_counter() - t0:.0f}s)")
    rows.sort(key=lambda r: (r["kind"], r["design"], r.get("family", r.get("control")), r["i"]))
    artifact = {"artifact_type": "design2_confirmatory", "status": values["status"], "analysis": analyse(values, rows), "rows": rows,
                "declaration_sha256": file_sha256(decl_path), "config_sha256": file_sha256(Path(config)),
                "source_snapshot_hash_declared": declaration["source_snapshot_hash"],
                "source_snapshot_post_run_drift": snapshot_drift(declaration["source_snapshot"], import_closure_snapshot(root, extra_files=_extra(config, values))),
                "execution": {"workers": int(workers), "jobs": len(jobs), "wall_seconds": time.perf_counter() - t0},
                "claim_boundary": {"exact_states_used_for": "generation, the oracle arm, errors, bounds and the static controls",
                                   "sealed_bank_opened_once": values["status"] == "sealed_confirmatory"}}
    art_path.write_text(json.dumps(artifact, default=float) + "\n", encoding="utf-8")
    return artifact
