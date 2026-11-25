"""M1 post-hoc check: one added D context breaks the D-commutant symmetry.

Added after the M1 structural map and the D-commutant check (`m1_d_commutant_check`), when the M1
block of the roadmap was revised (2026-11-22). It reads those two artifacts, opens no observation
bank and samples no shots. Everything is computed from exact tables at the map's development draws.

Statements tested, for every deficient (unitary, mask) of the map:
* **Candidates.** The unobserved input D contexts whose Pauli anticommutes with the D-commutant ``R``.
  Adding any one of them removes ``R`` from the commutant of the observed D Paulis.
* **Breaking.** Along the exact orbit ``exp(itR) V`` (type 3a), the added context's table moves.
  For a Pauli ``Q`` that anticommutes with ``R``,
  ``<u_j|e^{-itR} Q e^{itR}|u_k> = cos(2t) <u_j|Q|u_k> + sin(2t) <u_j|iQR|u_k>``.
* **Local identifiability.** With the added context the support Jacobian has no null direction beyond
  gauge (types 3a and 3b).
* **Orbit-invariant sensitivity.** Write ``A_jk = <v_j|Q|v_k>`` and ``B_jk = <v_j|iQR|v_k>`` for the
  branch vectors ``v_j = c_j u_j``. Along the orbit ``(A, B)`` rotates by ``2t``, so
  ``s_Q^2 = sum_{j<k} (|A_jk|^2 + |B_jk|^2) + |sum_j A_jj|^2 + |sum_j B_jj|^2`` is the same at every
  orbit point. A fit that lands anywhere on the circle gives the same ``s_Q``. It is a candidate
  support-only score for choosing the added context; this module only reports how it would choose.
* **What the orbit preserves.** The orbit acts as ``I (x) exp(itR)`` on the history, so every
  conditional state ``rho_{S|tau}`` is conjugated by the same system unitary and ``p(tau)`` is unchanged.
  Distances between conditional states, hence whether they change with ``tau``, are the same at every
  orbit point.
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

from bi_qmn.p3.runs import m1_d_commutant_check as dc
from bi_qmn.p3.runs import m1_support_identifiability_map as m1


METHOD = "m1_added_context_check_v1"
ARTIFACT = "m1_added_context_check.json"
BREAK_MIN = 1.0e-6          # smallest orbit movement of the added table counted as breaking
INVARIANCE_TOL = 1.0e-10


def input_d_pool(ctx: Mapping[str, Any]) -> list[str]:
    """Input D contexts that a mask can draw (the comp/op contexts are not candidates)."""
    return sorted({cid for mask in ctx["masks"]["low"] for cid in mask if cid.startswith("D,")})


def candidates(mask: Sequence[str], R: str, pool: Sequence[str]) -> list[str]:
    """Unobserved input D contexts whose Pauli anticommutes with ``R``."""
    return [q for q in pool if q not in mask and not dc.paulis_commute(q.split(",")[1], R)]


def orbit_invariant_sensitivity(V: np.ndarray, Q: str, R: str) -> float:
    """``s_Q`` of the module docstring at branch vectors ``V`` (columns ``c_j u_j``); ``Q`` is a Pauli or a ``D,<Pauli>`` id."""
    Qm = m1.pauli_matrix(Q.split(",")[-1])
    A = V.conj().T @ Qm @ V
    B = V.conj().T @ (1j * Qm @ m1.pauli_matrix(R)) @ V
    iu = np.triu_indices(V.shape[1], k=1)
    s2 = np.sum(np.abs(A[iu]) ** 2 + np.abs(B[iu]) ** 2) + abs(np.trace(A)) ** 2 + abs(np.trace(B)) ** 2
    return float(math.sqrt(float(s2)))


def conditional_states(V: np.ndarray, modes: Sequence[int], K: int) -> tuple[np.ndarray, np.ndarray]:
    """``p(tau)`` and ``rho_{S|tau}`` in the Fourier clock basis, ``<tau|m> = exp(-2 pi i tau m / K) / sqrt(K)``."""
    F = np.exp(-2j * math.pi * np.outer(np.arange(K), np.asarray(modes)) / K) / math.sqrt(K)
    W = F @ V.T                                     # (K, d_S): unnormalized system vector per tau
    p = np.sum(np.abs(W) ** 2, axis=1)
    rho = np.einsum("ti,tj->tij", W, W.conj()) / p[:, None, None]
    return p, rho


def _trace_distances(rho: np.ndarray) -> np.ndarray:
    K = rho.shape[0]
    out = np.zeros((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            out[a, b] = out[b, a] = 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(rho[a] - rho[b]))))
    return out


def orbit_preservation(V: np.ndarray, R: str, modes: Sequence[int], K: int, ts: Sequence[float]) -> dict[str, float]:
    """Largest change along the orbit of ``p(tau)`` and of the pairwise trace distances, and the conjugation error."""
    p0, rho0 = conditional_states(V, modes, K)
    D0 = _trace_distances(rho0)
    G = m1.pauli_matrix(R)
    dp = dD = conj = 0.0
    for t in ts:
        W = expm(1j * t * G)
        p1, rho1 = conditional_states(W @ V, modes, K)
        dp = max(dp, float(np.max(np.abs(p1 - p0))))
        dD = max(dD, float(np.max(np.abs(_trace_distances(rho1) - D0))))
        conj = max(conj, float(np.max(np.abs(np.einsum("ij,tjk,lk->til", W, rho0, W.conj()) - rho1))))
    return {"p_tau_change": dp, "trace_distance_change": dD, "conjugation_error": conj, "trace_distance_max": float(D0.max())}


def candidate_readout(ctx: Mapping[str, Any], tp: Mapping[str, Any], V: np.ndarray, mask: Sequence[str], R: str,
                      cands: Sequence[str], exact: bool) -> dict[str, dict[str, Any]]:
    """Per candidate: rank with the context added, ``smin_rel``, ``s_Q``, and (exact orbits) orbit movement and ``s_Q`` drift."""
    out: dict[str, dict[str, Any]] = {}
    for q in cands:
        r = m1.mask_readout(tp, list(mask) + [q])
        out[q] = {"extra_null": r["extra_null"], "smin_rel": r["smin_rel"], "s_Q": orbit_invariant_sensitivity(V, q, R)}
    if exact:
        G = m1.pauli_matrix(R)
        T0 = ctx["fw"].tables(V, list(cands))
        for q in cands:
            out[q]["orbit_movement"] = 0.0
            out[q]["s_Q_drift"] = 0.0
        for t in dc.ORBIT_T:
            Vt = expm(1j * t * G) @ V
            T = ctx["fw"].tables(Vt, list(cands))
            for q in cands:
                out[q]["orbit_movement"] = max(out[q]["orbit_movement"], float(np.max(np.abs(T[q] - T0[q]))))
                out[q]["s_Q_drift"] = max(out[q]["s_Q_drift"], abs(orbit_invariant_sensitivity(Vt, q, R) - out[q]["s_Q"]))
    return out


def _quantiles(v: Sequence[float]) -> dict[str, float] | None:
    return {q: float(np.quantile(v, float(q))) for q in ("0.0", "0.1", "0.5", "0.9")} if len(v) else None


def run_m1_added_context_check(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, write: bool = True) -> dict[str, Any]:
    base = Path(output_dir) / m1.OUTPUT_SUBDIR
    map_path, dc_path = base / m1.ARTIFACT, base / dc.ARTIFACT
    amap = json.loads(map_path.read_text(encoding="utf-8"))
    check = json.loads(dc_path.read_text(encoding="utf-8"))
    ctx = m1.model_context(config_dir, int(amap["settings"]["draws"]))
    modes = list(ctx["p1"].generator["active_modes"])
    n, K = len(modes), int(ctx["p1"].generator["dimensions"]["K"])
    pool = input_d_pool(ctx)

    points: dict[tuple[str, int], tuple[np.ndarray, dict[str, Any]]] = {}

    def point(u: str, k: int):
        if (u, k) not in points:
            p, ph = ctx["draws"][k]
            V = m1.truth_branch_vectors(ctx["unitaries"][u], p, ph)
            points[(u, k)] = (V, m1.truth_point(ctx, V))
        return points[(u, k)]

    combos = []
    for c in check["combinations"]:
        mask, U = tuple(c["mask"]), ctx["unitaries"][c["unitary"]]
        if len(c["d_commutant"]) != 1:
            raise ValueError(f"expected one D-commutant Pauli, got {c['d_commutant']} for {mask}")
        R = c["d_commutant"][0]
        if c["exact_orbit"] and dc.branch_condition(U, n, R, mask) >= dc.EXACT_TOL:
            raise ValueError("exact orbit without the branch condition")
        cands = candidates(mask, R, pool)
        draws = []
        for k in range(len(ctx["draws"])):
            V, tp = point(c["unitary"], k)
            row = {"draw": k, "candidates": candidate_readout(ctx, tp, V, mask, R, cands, bool(c["exact_orbit"]))}
            if c["exact_orbit"]:
                row["orbit_preservation"] = orbit_preservation(V, R, modes, K, dc.ORBIT_T)
            draws.append(row)
        combos.append({k: c[k] for k in ("deficit", "unitary", "mask", "screened_in", "exact_orbit")} | {"R": R, "candidates": cands, "draws": draws})

    # summaries
    cand_by_R = Counter((c["R"], len(c["candidates"])) for c in combos if c["unitary"] == combos[0]["unitary"])
    restore: Counter = Counter()
    breaks: Counter = Counter()
    for c in combos:
        kind = "3a" if c["exact_orbit"] else "3b"
        every = all(d["candidates"][q]["extra_null"] == 0 for d in c["draws"] for q in c["candidates"])
        restore[f"{c['deficit']}:{kind}:screened_in={c['screened_in']}:every_candidate_restores_rank={every}"] += 1
        if c["exact_orbit"]:
            ok = all(d["candidates"][q]["orbit_movement"] > BREAK_MIN for d in c["draws"] for q in c["candidates"])
            breaks[f"every_candidate_breaks_the_orbit_on_every_draw={ok}"] += 1

    def rule_table(select: list[dict[str, Any]]) -> dict[str, Any]:
        """``smin_rel`` with the added context under three choice rules, per (combination, draw)."""
        best, first, by_s, ratio_first, ratio_s = [], [], [], [], []
        for c in select:
            for d in c["draws"]:
                rows = d["candidates"]
                b = max(r["smin_rel"] for r in rows.values())
                f = rows[c["candidates"][0]]["smin_rel"]
                s = rows[max(c["candidates"], key=lambda q: rows[q]["s_Q"])]["smin_rel"]
                best.append(b); first.append(f); by_s.append(s)
                ratio_first.append(f / b); ratio_s.append(s / b)
        return {"n": len(best), "smin_rel_best_candidate": _quantiles(best), "smin_rel_first_in_order": _quantiles(first),
                "smin_rel_max_s_Q": _quantiles(by_s), "ratio_first_to_best": _quantiles(ratio_first), "ratio_max_s_Q_to_best": _quantiles(ratio_s)}

    low_in = [c for c in combos if c["deficit"] == "low" and c["screened_in"]]
    exact_in = [c for c in low_in if c["exact_orbit"]]
    pres = [d["orbit_preservation"] for c in combos if c["exact_orbit"] for d in c["draws"]]
    drift = max((d["candidates"][q]["s_Q_drift"] / max(d["candidates"][q]["s_Q"], 1e-300)
                 for c in combos if c["exact_orbit"] for d in c["draws"] for q in c["candidates"]), default=None)
    movement = [min(d["candidates"][q]["orbit_movement"] for d in c["draws"]) for c in exact_in for q in c["candidates"]]
    identifiable_smin = amap["e1_structural_map"]["cells"]

    artifact = {
        "artifact_type": METHOD,
        "status": "development_diagnostic_post_hoc",
        "not_declared": "post-hoc check for the M1 revision; candidates were read from the development map; opens no observation bank",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in (map_path, dc_path)},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {"break_min": BREAK_MIN, "orbit_t": list(dc.ORBIT_T), "draws": len(ctx["draws"]), "input_d_pool": pool},
        "deficient_low_masks": len({tuple(c["mask"]) for c in combos if c["deficit"] == "low"}),
        "candidates_by_R": {f"{R}:{m}": v for (R, m), v in sorted(cand_by_R.items())},
        "exact_screened_in_low": {"combinations": len(exact_in), "distinct_masks": len({tuple(c["mask"]) for c in exact_in}),
                                  "min_orbit_movement_per_candidate": _quantiles(movement)},
        "rank_restoration": dict(restore),
        "orbit_breaking": dict(breaks),
        "choice_rules": {"exact_screened_in_low_3a": rule_table(exact_in),
                         "first_order_screened_in_low_3b": rule_table([c for c in low_in if not c["exact_orbit"]])},
        "identifiable_reference_min_smin_rel": {k: v.get("screened_in_identifiable_min_smin_rel_quantiles") for k, v in identifiable_smin.items()},
        "s_Q_max_relative_drift_along_orbit": drift,
        "orbit_preservation_max": {k: max(p[k] for p in pres) for k in ("p_tau_change", "trace_distance_change", "conjugation_error")} if pres else None,
        "orbit_conditional_state_trace_distance_max": _quantiles([p["trace_distance_max"] for p in pres]),
        "combinations": combos,
    }
    artifact = m1._jsonable(artifact)
    if write:
        (base / ARTIFACT).write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    return artifact
