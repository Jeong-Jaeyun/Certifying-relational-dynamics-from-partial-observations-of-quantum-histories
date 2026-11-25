"""D1-v6 post-run tail diagnostics (report-only, added after the run; not part of the D1-v6 declaration).

The D1-v6 run (``p3_d1_v6_development``) found that the shared branch-pair estimator certifies
almost every low-stratum block under two calibration draws, and none under the other two. With
alpha = 0.04 and 48 calibration blocks the split-conformal radius is the block maximum, so one bad
calibration record decides the draw. This module reads the immutable run artifact and explains
those records. It writes a sibling JSON file and never modifies the run artifact.

Two support-identifiability failures are distinguished.

* **Pauli-commutant degeneracy (support-only).** Every observed context measures a two-qubit Pauli
  ``P`` on the system. If all observed Paulis commute with one non-identity Pauli ``R``, then
  ``u_j -> exp(i theta R) u_j`` leaves every support table unchanged, for every state. Unobserved
  contexts are not protected. Whether this happens depends only on the context list.
* **Disconnected branch (state-dependent).** ``<u_j|P|u_k> = 0`` for every observed ``P`` and both
  pairs touching branch ``j``. This leaves the branch phase unidentified (protocol §Background).

The oracle-screened radii below use the true basis to flag disconnected branches. They are a
what-if readout, not an estimator result.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize

from bi_qmn.core.provenance import file_sha256
from bi_qmn.p1.model.p1_generator import build_p1_instance, system_unitary_from_config
from bi_qmn.p1.model.p1_measurement import p1_measurement_table
from bi_qmn.p3.runs.p3_d1_v6_development import ARTIFACT, CONFIG_FILE, TRUE_MASK_THRESHOLD, _load, _prepare, _true_branch_vectors


OUTPUT = "p3_d1_v6_tail_diagnostics.json"
TAIL_SCORE = 0.1  # used only to list calibration-tail records; every non-tail low score in the run is <= 0.052
EXACT_SHOTS = 1.0e6
TWO_QUBIT_PAULIS = tuple(a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II")
CALIBRATION_ROLES = ("calibration_0", "calibration_1", "calibration_2", "calibration_3")


# ---------------------------------------------------------------------------
# Pauli commutant
# ---------------------------------------------------------------------------


def paulis_commute(a: str, b: str) -> bool:
    """Two Pauli strings commute iff they anticommute on an even number of tensor factors."""
    return sum(1 for x, y in zip(a, b, strict=True) if x != "I" and y != "I" and x != y) % 2 == 0


def pauli_commutant(paulis: Iterable[str]) -> list[str]:
    """Non-identity two-qubit Paulis that commute with every Pauli in ``paulis``."""
    paulis = list(paulis)
    return [r for r in TWO_QUBIT_PAULIS if all(paulis_commute(r, p) for p in paulis)]


def support_paulis(context_ids: Iterable[str]) -> list[str]:
    return sorted({cid.split(",")[1] for cid in context_ids})


def commutant_census(d_inputs: list[str], f_inputs: list[str], n_d: int, n_f: int) -> dict[str, Any]:
    by: Counter = Counter()
    total = 0
    for ds in itertools.combinations(d_inputs, n_d):
        for fs in itertools.combinations(f_inputs, n_f):
            total += 1
            com = pauli_commutant(support_paulis(ds + fs))
            if com:
                by[",".join(com)] += 1
    degenerate = sum(by.values())
    return {"masks": total, "degenerate": degenerate, "fraction": degenerate / total, "by_commutant": dict(sorted(by.items()))}


_SIGMA = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}


def pauli_matrix(label: str) -> np.ndarray:
    return np.kron(_SIGMA[label[0]], _SIGMA[label[1]]).astype(complex)


def normalized_pauli(forward, label: str) -> np.ndarray:
    """System matrix of Pauli ``label``, checked against the forward model's convention on every D context."""
    for cid, P in forward.system_observable.items():
        P = np.asarray(P, dtype=complex)
        P = P / math.sqrt(np.trace(P @ P).real / P.shape[0])
        if np.max(np.abs(P - pauli_matrix(cid.split(",")[1]))) > 1e-12:
            raise RuntimeError(f"Pauli convention mismatch on {cid}")
    return pauli_matrix(label)


def symmetry_check(forward, V: np.ndarray, support_ids: list[str], R: np.ndarray, thetas: Iterable[float]) -> list[dict[str, float]]:
    """Apply ``exp(i theta R)`` to every branch; report table changes on and off the support."""
    all_ids = sorted(forward.A)
    base = forward.tables(V, all_ids)
    rows = []
    for theta in thetas:
        Vt = expm(1j * theta * R) @ V
        moved = forward.tables(Vt, all_ids)
        on = max(float(np.max(np.abs(moved[c] - base[c]))) for c in support_ids)
        off = max(float(np.max(np.abs(moved[c] - base[c]))) for c in all_ids if c not in support_ids)
        rows.append({"theta": float(theta), "max_abs_change_support": on, "max_abs_change_unobserved": off})
    return rows


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def theta_from_branch_vectors(sbp, V: np.ndarray) -> np.ndarray:
    """A parameter vector whose branch vectors equal ``V`` up to a global phase."""
    d, n = sbp.forward.d_S, sbp.forward.n
    norms = np.linalg.norm(V, axis=0)
    Q = V / norms[None, :]
    filler = np.random.default_rng(0).normal(size=(d, d - n)) + 0j
    filler -= Q @ (Q.conj().T @ filler)
    U = np.concatenate([Q, np.linalg.qr(filler)[0]], axis=1)
    X = logm(U)
    X = 0.5 * (X - X.conj().T)
    B = sbp.basis.reshape(sbp.basis.shape[0], -1)
    coef, *_ = np.linalg.lstsq(np.concatenate([B.real.T, B.imag.T]), np.concatenate([X.real.reshape(-1), X.imag.reshape(-1)]), rcond=None)
    p = norms**2 / float(np.sum(norms**2))
    theta = np.concatenate([coef, np.log(p[1:] / p[0]), np.zeros(n - 1)])
    ids = sorted(sbp.forward.A)
    got, want = sbp.forward.tables(sbp.branch_vectors(theta), ids), sbp.forward.tables(V, ids)
    if max(float(np.max(np.abs(got[c] - want[c]))) for c in ids) > 1e-9:
        raise RuntimeError("branch-vector parametrization did not reproduce the target tables")
    return theta


def nll_of(sbp, V: np.ndarray, A: np.ndarray, counts: np.ndarray) -> float:
    x = V.T.reshape(-1)
    q = np.einsum("i,oi->o", x.conj(), A @ x).real
    return -float(counts @ np.log(q + sbp.likelihood_floor))


def polish(sbp, theta0: np.ndarray, A: np.ndarray, counts: np.ndarray, *, tight: bool = False):
    options = {"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-11} if tight else {"maxiter": sbp.max_iterations, "ftol": 1e-13, "gtol": 1e-9}
    return minimize(sbp._objective, theta0, args=(A, counts), jac=True, method="L-BFGS-B", options=options)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _record(ctx, role: str, block: int, deficit: str) -> tuple[int, Any]:
    hits = [(i, r) for i, r in enumerate(ctx["records"][role]) if r.instance_index == block and r.deficit == deficit]
    if len(hits) != 1:
        raise KeyError((role, block, deficit))
    return hits[0]


def _tail_record_diagnostic(ctx, art: Mapping[str, Any], role: str, block: int, metric: Mapping[str, Any]) -> dict[str, Any]:
    sbp, fw = ctx["sbp"], ctx["sbp"].forward
    index, record = _record(ctx, role, block, "low")
    instance = ctx["instances"][role][block]
    support = record.support.table.probabilities
    n_shots = float(record.support.table.metadata.get("n_shots", 1.0))
    ids, A, counts = sbp._support_arrays(support, n_shots)
    V_true = _true_branch_vectors(instance)
    key = f"{role}:{index}"
    declared_objective = float(art["shared_branch_pair_fits"][key]["objective_nats"])
    from_truth = polish(sbp, theta_from_branch_vectors(sbp, V_true), A, counts)
    V_from_truth = sbp.branch_vectors(from_truth.x)
    observability = fw.pair_observability(V_true, ids, TRUE_MASK_THRESHOLD)
    commutant = pauli_commutant(support_paulis(ids))
    out: dict[str, Any] = {
        "role": role, "block": block, "unitary": metric["unitary"], "support_context_ids": ids,
        "true_mask_class": observability["mask_class"], "true_hidden_pairs": observability["hidden_pairs"],
        "support_pauli_commutant": commutant,
        "diag_score": metric["diag_score"], "coh_score": metric["coh_score"],
        "coef_error_rms": metric["coef_error_rms"], "coef_amplitude_error_rms": metric["coef_amplitude_error_rms"],
        "nll": {"truth": nll_of(sbp, V_true, A, counts), "declared_fit": declared_objective, "refit_from_truth": float(from_truth.fun)},
        "declared_fit_minus_refit_from_truth_nats": declared_objective - float(from_truth.fun),
    }
    unobs = [c for c in fw.system_observable if c not in ids]

    def coefficients(V):
        return np.array([np.conj(V[:, j]) @ fw.system_observable[c] @ V[:, k] for c in unobs for j, k in fw.pairs])

    out["refit_from_truth_coef_error_rms"] = float(np.sqrt(np.mean(np.abs(coefficients(V_from_truth) - coefficients(V_true)) ** 2)))
    if commutant:
        # Rerun the declared fit (deterministic in the support), then polish it on the exact support.
        fit = sbp.fit_support(support, n_shots)
        out["declared_fit_reproduced_objective_difference"] = float(fit["objective_nats"]) - declared_objective
        V_fit = np.asarray(fit["branch_vectors_real_imag"][0]) + 1j * np.asarray(fit["branch_vectors_real_imag"][1])
        exact = p1_measurement_table(instance.density, ctx["contexts"], K=ctx["K"], eta=0.0).probabilities
        _, A_x, counts_x = sbp._support_arrays({c: exact[c] for c in ids}, EXACT_SHOTS)
        alt = polish(sbp, theta_from_branch_vectors(sbp, V_fit), A_x, counts_x, tight=True)
        V_alt = sbp.branch_vectors(alt.x)
        all_ids = sorted(fw.A)
        T_alt, T_true = fw.tables(V_alt, all_ids), fw.tables(V_true, all_ids)
        R = normalized_pauli(fw, commutant[0])
        overlaps = [abs(np.vdot(V_true[:, j], V_alt[:, j])) / (np.linalg.norm(V_true[:, j]) * np.linalg.norm(V_alt[:, j])) for j in range(fw.n)]
        angle = math.acos(min(1.0, float(np.mean(overlaps))))
        image = {}
        for sign in (1.0, -1.0):
            T_img = fw.tables(expm(1j * sign * angle * R) @ V_true, all_ids)
            image[f"{sign:+.0f}"] = max(float(np.max(np.abs(T_img[c] - T_alt[c]))) for c in all_ids)
        out["exact_support_alternative"] = {
            "shots": EXACT_SHOTS,
            "nll_excess_over_truth_nats": float(alt.fun) - nll_of(sbp, V_true, A_x, counts_x),
            "max_abs_table_difference_support": max(float(np.max(np.abs(T_alt[c] - T_true[c]))) for c in ids),
            "max_abs_table_difference_unobserved": {c: float(np.max(np.abs(T_alt[c] - T_true[c]))) for c in all_ids if c not in ids},
            "branch_overlaps_with_truth": [float(o) for o in overlaps],
            "true_expectation_of_R_per_branch": [float(np.real(np.vdot(V_true[:, j], R @ V_true[:, j])) / np.linalg.norm(V_true[:, j]) ** 2) for j in range(fw.n)],
            "symmetry_angle_from_overlap": angle,
            "max_abs_table_difference_alternative_vs_symmetry_image": image,
        }
        out["symmetry_check_true_state"] = symmetry_check(fw, V_true, ids, R, (math.pi / 8, math.pi / 4, 3 * math.pi / 8, angle))
    return out


def _disconnect_fraction_by_unitary(ctx, n_d: int) -> dict[str, Any]:
    """Fraction of ``n_d``-of-11 D masks that disconnect a branch, per registered unitary (generic populations and phases)."""
    p1, fw = ctx["p1"], ctx["sbp"].forward
    d_inputs = [c.context_id for c in ctx["input_contexts"] if c.clock == "D"]
    out = {}
    for item in p1.generator["construction_grid"]["system_unitaries"]:
        label, U = system_unitary_from_config(item)
        inst = build_p1_instance(coefficient_label="g", probabilities=np.array([0.3, 0.33, 0.37]), phase_label="g", phases=np.array([0.0, 0.7, 1.9]),
                                 system_unitary_label=label, system_unitary=U, active_modes=tuple(p1.generator["active_modes"]),
                                 omega=p1.generator["omega"], d_C=ctx["d_C"], d_S=ctx["d_S"], K=ctx["K"])
        V = _true_branch_vectors(inst)
        classes = Counter(fw.pair_observability(V, list(mask), TRUE_MASK_THRESHOLD)["mask_class"] for mask in itertools.combinations(d_inputs, n_d))
        total = sum(classes.values())
        out[label] = {"masks": total, "classes": dict(classes), "disconnected_fraction": classes.get("branch_disconnected", 0) / total}
    return out


def _binomial_tail(n: int, p: float, k: int) -> float:
    return float(sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def tail_risk(p: float, alpha: float, sizes: Iterable[int]) -> list[dict[str, Any]]:
    """P(the split-conformal radius lands on a bad record) when bad records occur independently at rate p."""
    rows = []
    for n in sizes:
        index = math.ceil((n + 1) * (1 - alpha))
        if index > n:
            rows.append({"n": n, "order_index": None, "probability": 1.0})
            continue
        rows.append({"n": n, "order_index": index, "bad_records_needed": n - index + 1, "probability": _binomial_tail(n, p, n - index + 1)})
    return rows


def _radius(scores: list[float], alpha: float) -> tuple[float, int, int]:
    s = sorted(scores)
    index = math.ceil((len(s) + 1) * (1 - alpha))
    if index > len(s):
        return math.inf, len(s), index
    return float(s[index - 1]), len(s), index


def run_tail_diagnostics(config_dir: str | Path = "config", output_dir: str | Path = "outputs") -> dict[str, Any]:
    config_dir = Path(config_dir)
    values, config_sha = _load(config_dir)
    folder = Path(output_dir) / Path(str(values["output_root"])).name
    target = folder / OUTPUT
    if target.exists():
        raise FileExistsError(f"diagnostic already exists and is immutable: {target}")
    art_path = folder / ARTIFACT
    art = json.loads(art_path.read_text(encoding="utf-8"))
    if art["config_sha256"] != config_sha:
        raise RuntimeError("config differs from the one the D1-v6 artifact was produced with")
    alpha = 0.04  # the R4 rule used by the run (protocol §Radii)
    ctx = _prepare(config_dir)
    metrics = art["record_metrics"]["shared_branch_pair"]

    # 1. Record-level join: commutant (support-only) and disconnected branch (true basis).
    join: dict[str, Any] = {}
    flagged: set[tuple[str, int, str]] = set()
    for role in (*CALIBRATION_ROLES, "evaluation"):
        by_block = {(m["block"], m["deficit"]): m for m in metrics[role]}
        for deficit in ("medium", "low"):
            rows = []
            for record in ctx["records"][role]:
                if record.deficit != deficit:
                    continue
                m = by_block[(record.instance_index, deficit)]
                com = pauli_commutant(support_paulis(record.support.table.probabilities))
                disconnected = m["true_mask_class"] == "branch_disconnected"
                if com or disconnected:
                    flagged.add((role, record.instance_index, deficit))
                rows.append({"block": record.instance_index, "unitary": m["unitary"], "commutant": com, "disconnected": disconnected, "diag_score": m["diag_score"], "coh_score": m["coh_score"]})
            clean = [r["diag_score"] for r in rows if not r["commutant"] and not r["disconnected"]]
            join[f"{role}:{deficit}"] = {
                "records": len(rows),
                "commutant_degenerate": sum(bool(r["commutant"]) for r in rows),
                "branch_disconnected": sum(r["disconnected"] for r in rows),
                "tail_records": [r for r in rows if r["diag_score"] > TAIL_SCORE],
                "max_diag_score_unflagged": max(clean) if clean else None,
            }

    # 2. Per tail record: is the alternative fit an exact support ambiguity?
    tail = []
    for role in CALIBRATION_ROLES:
        for m in sorted(metrics[role], key=lambda r: -r["diag_score"]):
            if m["deficit"] == "low" and m["diag_score"] > TAIL_SCORE:
                tail.append(_tail_record_diagnostic(ctx, art, role, int(m["block"]), m))

    # 3. Mask-level rates.
    d_inputs = [c.context_id for c in ctx["input_contexts"] if c.clock == "D"]
    f_inputs = [c.context_id for c in ctx["input_contexts"] if c.clock == "F"]
    sizes = Counter((r.deficit, sum(c.startswith("D,") for c in r.support.table.probabilities), sum(c.startswith("F,") for c in r.support.table.probabilities))
                    for role in (*CALIBRATION_ROLES, "evaluation") for r in ctx["records"][role])
    census = {f"{d}:{nd}D+{nf}F": commutant_census(d_inputs, f_inputs, nd, nf) for (d, nd, nf) in sorted(sizes)}
    low_sizes = {(nd, nf) for (d, nd, nf) in sizes if d == "low"}
    if len(low_sizes) != 1:
        raise RuntimeError(f"low supports do not have one size: {low_sizes}")
    disconnect = _disconnect_fraction_by_unitary(ctx, next(iter(low_sizes))[0])
    p_commutant = census[next(k for k in census if k.startswith("low:"))]["fraction"]
    p_disconnect = float(np.mean([v["disconnected_fraction"] for v in disconnect.values()]))
    low_records = sum(join[f"{role}:low"]["records"] for role in (*CALIBRATION_ROLES, "evaluation"))
    low_flagged = sum(1 for key in flagged if key[2] == "low")
    rates = {
        "commutant_fraction_low_masks": p_commutant,
        "disconnected_fraction_low_masks_mean_over_unitaries": p_disconnect,
        "expected_unidentifiable_fraction_low_if_independent": 1 - (1 - p_commutant) * (1 - p_disconnect),
        "observed_unidentifiable_low_records": {"flagged": low_flagged, "records": low_records, "fraction": low_flagged / low_records},
    }
    risk = {name: tail_risk(p, alpha, (48, 96, 160, 192, 256)) for name, p in (
        ("expected_mask_rate", rates["expected_unidentifiable_fraction_low_if_independent"]),
        ("observed_rate", rates["observed_unidentifiable_low_records"]["fraction"]),
        ("disconnected_only", p_disconnect),
    )}

    # 4. Oracle-screened radii (what-if; the disconnected flag uses the true basis) and BI/coverage cross-check.
    summary = art["summary"]["shared_branch_pair"]["low"]["by_calibration"]
    r0 = summary["calibration_0"]["radius"]["diag"]
    cert0 = [x for x in art["certificate_records"]["shared_branch_pair"]["calibration_0"] if x["deficit"] == "low"]
    distances = [float(x["kappa"]) * r0 for x in cert0]  # kappa = tau-flat distance / r_diag, fixed per evaluation centre
    eval_low = [m for m in metrics["evaluation"] if m["deficit"] == "low"]
    screened = {}
    pooled: dict[str, list[float]] = {"diag": [], "coh": []}
    for role in CALIBRATION_ROLES:
        kept = [m for m in metrics[role] if m["deficit"] == "low" and (role, m["block"], "low") not in flagged]
        for cls in pooled:
            pooled[cls] += [m[f"{cls}_score"] for m in kept]
        rd, n, k = _radius([m["diag_score"] for m in kept], alpha)
        rc, _, _ = _radius([m["coh_score"] for m in kept], alpha)
        screened[role] = {"kept": n, "order_index": k, "r_diag": rd, "r_coh": rc, "as_run_r_diag": summary[role]["radius"]["diag"],
                          "rdn_certified_implied": sum(d > rd for d in distances),
                          "evaluation_coverage": {"diag": sum(m["diag_score"] <= rd for m in eval_low), "coh": sum(m["coh_score"] <= rc for m in eval_low), "n": len(eval_low)}}
    rd, n, k = _radius(pooled["diag"], alpha)
    rc, _, _ = _radius(pooled["coh"], alpha)
    screened["pooled"] = {"kept": n, "order_index": k, "r_diag": rd, "r_coh": rc, "as_run_r_diag": summary["pooled"]["radius"]["diag"]}
    bi_unresolved = []
    for draw, rows in art["certificate_records"]["shared_branch_pair"].items():
        for x in rows:
            if x["bi_status"] not in ("primal_verified", "not_run"):
                m = next(r for r in metrics["evaluation"] if r["block"] == x["block"] and r["deficit"] == x["deficit"])
                radii = art["summary"]["shared_branch_pair"][x["deficit"]]["by_calibration"][draw]["radius"]
                bi_unresolved.append({"draw": draw, "deficit": x["deficit"], "block": x["block"], "bi_status": x["bi_status"], "bi_verified_upper_bound": x["bi_verified_upper_bound"],
                                      "diag_score_over_radius": m["diag_score"] / radii["diag"], "coh_score_over_radius": m["coh_score"] / radii["coh"]})

    payload = {
        "artifact_type": "p3_d1_v6_tail_diagnostics",
        "status": "development_only_report_only_post_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_artifact": str(art_path).replace("\\", "/"),
        "input_artifact_sha256": file_sha256(art_path),
        "config_sha256": config_sha,
        "not_declared": "added after the D1-v6 run; it re-reads the artifact and refits a few records, and changes no D1-v6 result",
        "oracle_note": "the disconnected-branch flag and the screened radii use the true basis; the commutant flag uses the support context list only",
        "tail_score_listing_threshold": TAIL_SCORE,
        "record_join": join,
        "tail_records": tail,
        "mask_census": {"commutant": census, "disconnected_by_unitary": disconnect},
        "rates": rates,
        "radius_tail_risk_alpha_0_04": risk,
        "oracle_screened_low_radii": screened,
        "shared_branch_pair_bi_unresolved": bi_unresolved,
    }
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    return payload
