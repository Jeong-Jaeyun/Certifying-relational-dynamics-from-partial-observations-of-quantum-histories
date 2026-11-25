"""M1 post-hoc check: the third support non-identifiability is a D-commutant symmetry that F cannot break.

Added after the M1 structural map (`m1_support_identifiability_map`). The map showed that the same 110
low masks are deficient for every unitary and every draw. This module tests the explanation read off
those masks. It reads the map artifact, opens no observation bank and samples no shots.

Statement tested.
* **First order.** The SBP support Jacobian has a null direction beyond gauge exactly when the Paulis
  of the observed D (time-basis) contexts have a non-trivial Pauli commutant ``R``. The F contexts
  and the unitary do not matter. Screen v1 tests the commutant of the D and F Paulis together, so it
  misses the case where only an F context breaks ``R``.
* **Exact family.** ``exp(itR)`` leaves every D table invariant, since ``R`` commutes with each observed
  D Pauli. An F context reads only the branch diagonals ``p_j <u_j|P|u_j>``. For an F Pauli ``P`` that
  anticommutes with ``R``, ``<u_j|e^{-itR} P e^{itR}|u_j> = cos(2t) <u_j|P|u_j> + i sin(2t) <u_j|PR|u_j>``.
  So the whole orbit is support-equivalent exactly when ``<u_j|P|u_j> = <u_j|PR|u_j> = 0`` for every
  such ``P`` and every branch ``j``. The condition involves the branch directions only, not the branch
  probabilities or phases, so it is fixed by (unitary, mask). The orbit is a circle of period pi in ``t``.
  It moves the unobserved D contexts whose Pauli anticommutes with ``R``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import expm

from bi_qmn.p3.runs import m1_support_identifiability_map as m1


METHOD = "m1_d_commutant_check_v1"
ARTIFACT = "m1_d_commutant_check.json"
MAP_ARTIFACT = Path("outputs") / m1.OUTPUT_SUBDIR / m1.ARTIFACT
ORBIT_T = tuple(float(t) for t in np.linspace(0.0, math.pi, 25)[1:-1])
EXACT_TOL = 1.0e-10
FAMILY_STATE_DISTANCE_MIN = 1.0e-3
PAULIS = tuple(x + y for x in "IXYZ" for y in "IXYZ" if x + y != "II")


def paulis_commute(p: str, q: str) -> bool:
    return sum(1 for x, y in zip(p, q) if x != "I" and y != "I" and x != y) % 2 == 0


def d_commutant(mask: Sequence[str]) -> list[str]:
    """Non-identity Paulis that commute with every Pauli of the observed D contexts."""
    d = [c.split(",")[1] for c in mask if c.startswith("D,")]
    return [r for r in PAULIS if all(paulis_commute(r, p) for p in d)]


def branch_condition(U: np.ndarray, n: int, R: str, mask: Sequence[str]) -> float:
    """Max of ``|<u_j|P|u_j>|`` and ``|<u_j|PR|u_j>|`` over F Paulis ``P`` anticommuting with ``R`` and branches ``j``."""
    u = np.asarray(U, dtype=complex)[:, :n]
    Rm = m1.pauli_matrix(R)
    worst = 0.0
    for cid in mask:
        clock, P = cid.split(",")
        if clock != "F" or paulis_commute(P, R):
            continue
        Pm = m1.pauli_matrix(P)
        for j in range(n):
            worst = max(worst, abs(u[:, j].conj() @ Pm @ u[:, j]), abs(u[:, j].conj() @ Pm @ Rm @ u[:, j]))
    return float(worst)


def r_orbit(ctx: Mapping[str, Any], V: np.ndarray, R: str, mask: Sequence[str]) -> dict[str, float]:
    """Support residual, unobserved-table deviation and trace distance along ``exp(itR) V``, t in (0, pi)."""
    fw = ctx["fw"]
    G = m1.pauli_matrix(R)
    T0 = fw.tables(V, ctx["ids"])
    unobs = [c for c in ctx["ids"] if c not in mask]
    x0 = fw.vector(V)
    res = dev = dist = 0.0
    for t in ORBIT_T:
        Vt = expm(1j * t * G) @ V
        T = fw.tables(Vt, ctx["ids"])
        res = max(res, max(float(np.max(np.abs(T[c] - T0[c]))) for c in mask))
        dev = max(dev, max(float(np.max(np.abs(T[c] - T0[c]))) for c in unobs))
        dist = max(dist, m1._state_distance(x0, fw.vector(Vt)))
    return {"support_residual": res, "unobserved_max_abs": dev, "state_distance": dist}


def _family_exact(row: Mapping[str, Any]) -> bool:
    fam = row.get("family", [])
    return bool(fam) and all(f["stop"] != "left_family" for f in fam) and max(f["max_state_distance"] for f in fam) > FAMILY_STATE_DISTANCE_MIN


def run_m1_d_commutant_check(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, write: bool = True) -> dict[str, Any]:
    map_path = Path(output_dir) / m1.OUTPUT_SUBDIR / m1.ARTIFACT
    amap = json.loads(map_path.read_text(encoding="utf-8"))
    ctx = m1.model_context(config_dir, int(amap["settings"]["draws"]))
    n = len(ctx["p1"].generator["active_modes"])
    labels = list(amap["settings"]["unitaries"])

    # (a) the first-order rule against the structural map
    deficient = {(c["deficit"], tuple(c["mask"]), c["unitary"]) for c in amap["e1_structural_map"]["combinations_not_identifiable"]}
    confusion: Counter = Counter()
    for deficit in m1.DEFICITS:
        for mask in ctx["masks"][deficit]:
            has_r = bool(d_commutant(mask))
            for u in labels:
                confusion[f"d_commutant={has_r},deficient={(deficit, tuple(mask), u) in deficient}"] += 1

    # (b) exact orbits over every deficient (unitary, mask), with the oracle screen
    combos = []
    for c in amap["e1_structural_map"]["combinations_not_identifiable"]:
        Rs = d_commutant(c["mask"])
        U = ctx["unitaries"][c["unitary"]]
        cond = min(branch_condition(U, n, R, c["mask"]) for R in Rs) if Rs else math.inf
        exact = cond < EXACT_TOL
        devs = []
        res_max = 0.0
        if exact:
            R = next(R for R in Rs if branch_condition(U, n, R, c["mask"]) < EXACT_TOL)
            for p, ph in ctx["draws"]:
                o = r_orbit(ctx, m1.truth_branch_vectors(U, p, ph), R, c["mask"])
                devs.append(o["unobserved_max_abs"])
                res_max = max(res_max, o["support_residual"])
        combos.append({
            "deficit": c["deficit"], "unitary": c["unitary"], "mask": c["mask"], "d_commutant": Rs,
            "screened_in": bool(c["screened_in"]), "branch_condition": cond, "exact_orbit": exact,
            "orbit_support_residual_max": res_max if exact else None,
            "orbit_unobserved_deviation_by_draw": devs if exact else None,
        })
    screened_in_total = amap["e1_structural_map"]["predicted_incidence"]["low"]["screened_in"]
    exact_in = [c for c in combos if c["exact_orbit"] and c["screened_in"]]
    exact_devs = [v for c in combos if c["exact_orbit"] for v in c["orbit_unobserved_deviation_by_draw"]]
    incidence = {
        "first_order_screened_in": sum(1 for c in combos if c["screened_in"] and c["deficit"] == "low"),
        "exact_screened_in": sum(1 for c in exact_in if c["deficit"] == "low"),
        "screened_in": screened_in_total,
    }
    incidence["first_order_rate"] = incidence["first_order_screened_in"] / screened_in_total
    incidence["exact_rate"] = incidence["exact_screened_in"] / screened_in_total
    by_unitary = {u: {"deficient": sum(1 for c in combos if c["unitary"] == u),
                      "exact": sum(1 for c in combos if c["unitary"] == u and c["exact_orbit"]),
                      "exact_screened_in": sum(1 for c in combos if c["unitary"] == u and c["exact_orbit"] and c["screened_in"]),
                      "first_order_screened_in": sum(1 for c in combos if c["unitary"] == u and c["screened_in"])} for u in labels}

    # (c) agreement with the E2 predictor-corrector families
    agree: Counter = Counter()
    for r in amap["e2_mechanism"]:
        if r["extra_null"] < 1:
            continue
        Rs = d_commutant(r["mask"])
        cond = min(branch_condition(ctx["unitaries"][r["unitary"]], n, R, r["mask"]) for R in Rs) if Rs else math.inf
        agree[f"{r['kind']}:family_exact={_family_exact(r)},orbit_condition={cond < EXACT_TOL}"] += 1

    # (d) every deficient record of D1-v7, T1 run 2 and A1
    records = []
    for r in amap["e3_retrospective_rows"]:
        if r["extra_null"] < 1:
            continue
        Rs = d_commutant(r["mask"])
        cond = min(branch_condition(ctx["unitaries"][r["unitary"]], n, R, r["mask"]) for R in Rs) if Rs else math.inf
        art = r.get("artifact") or {}
        records.append({k: r[k] for k in ("source", "role", "deficit", "block", "unitary", "mask")} | {
            "d_commutant": Rs, "exact_orbit": cond < EXACT_TOL, "screen_passes": art.get("screen_passes"),
            "diag_score": art.get("diag_score"), "coef_error_rms": art.get("coef_error_rms"), "bi_status": art.get("bi_status"),
        })
    v7_ok = [float(r["artifact"]["diag_score"]) for r in amap["e3_retrospective_rows"]
             if r["source"] == "d1_v7" and r["deficit"] == "low" and r["extra_null"] == 0 and (r.get("artifact") or {}).get("screen_passes")]
    first_only = [r for r in records if r["source"] == "d1_v7" and r["screen_passes"] and not r["exact_orbit"]]

    artifact = {
        "artifact_type": METHOD,
        "status": "development_diagnostic_post_hoc",
        "not_declared": "post-hoc test of the mechanism read off the M1 structural map; opens no observation bank",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "map_artifact": str(map_path).replace("\\", "/"),
        "map_artifact_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {"exact_tol": EXACT_TOL, "orbit_t": list(ORBIT_T), "family_state_distance_min": FAMILY_STATE_DISTANCE_MIN},
        "first_order_rule_confusion": dict(confusion),
        "d_commutant_paulis_of_deficient_masks": dict(Counter(R for c in combos if c["unitary"] == labels[0] for R in c["d_commutant"])),
        "exact_orbit_incidence_low": incidence,
        "exact_orbit_by_unitary": by_unitary,
        "exact_orbit_support_residual_max": max((c["orbit_support_residual_max"] for c in combos if c["exact_orbit"]), default=None),
        "exact_orbit_unobserved_deviation_quantiles": {q: float(np.quantile(exact_devs, float(q))) for q in ("0.1", "0.5", "0.9", "1.0")} if exact_devs else None,
        "e2_family_vs_orbit_condition": dict(agree),
        "deficient_records": records,
        "d1_v7_low_screened_in": {
            "exact_orbit": [f"{r['role']}:{r['block']}" for r in records if r["source"] == "d1_v7" and r["screen_passes"] and r["exact_orbit"]],
            "first_order_only": [f"{r['role']}:{r['block']}" for r in first_only],
            "first_order_only_diag_scores": sorted(float(r["diag_score"]) for r in first_only),
            "identifiable_diag_score_p50_p99": [float(np.quantile(v7_ok, 0.5)), float(np.quantile(v7_ok, 0.99))],
        },
        "combinations": combos,
    }
    if write:
        out = Path(output_dir) / m1.OUTPUT_SUBDIR / ARTIFACT
        out.write_text(json.dumps(m1._jsonable(artifact), indent=1), encoding="utf-8")
    return m1._jsonable(artifact)
