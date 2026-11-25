"""M1 post-hoc check: the RDN margin kappa is not invariant along an exact 3a orbit.

Added before the Q1 verification design was declared (2026-11-22), to separate two statements:

* **physical tau-dependence is invariant.** The orbit conjugates every ``rho_{S|tau}`` by one system
  unitary and keeps ``p(tau)`` (`m1_added_context_check`, orbit preservation).
* **the registered RDN margin need not be.** ``rdn_v1`` computes the tau-flat distance of the diag class,
  ``sqrt(sum over (D context, system bit) of the squared deviation of q(tau, bit) from its tau-mean)``,
  over the 12 D contexts of the measurement design. These are 12 of the 15 Paulis, so a unitary that
  moves tau-variation between Paulis can change the distance. ``kappa`` is that distance over the frozen
  diag radius.

The check evaluates ``kappa`` from exact tables at grid points of every exact, screened-in low orbit of
the M1 map (41 combinations x 16 development draws), against the frozen S3 low diag radius. It opens no
observation bank and samples no shots.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm

from bi_qmn.core import identifiability as ident
from bi_qmn.ilgc.relational_dynamics import tau_flat_distance
from bi_qmn.p1.model.p1_measurement import full_p1_context_list
from bi_qmn.p3.runs import m1_d_commutant_check as dc
from bi_qmn.p3.runs import m1_support_identifiability_map as m1
from bi_qmn.p3.runs.p3_d1_v6_development import _keys


METHOD = "m1_orbit_kappa_check_v1"
ARTIFACT = "m1_orbit_kappa_check.json"
S3_CALIBRATION = Path("outputs") / "p3_s3_calibration" / "p3_s3_calibration.json"
ORBIT_T = tuple(float(t) for t in np.linspace(0.0, np.pi, 49)[:-1])   # includes t = 0 (the truth)


def frozen_low_diag_radius(path: Path = S3_CALIBRATION) -> tuple[float, str, str]:
    freeze = json.loads(path.read_text(encoding="utf-8"))["freeze"]
    if not freeze.get("frozen"):
        raise RuntimeError("the S3 calibration is not frozen")
    return float(freeze["radii"]["low"]["diag"]), str(freeze["calibration_hash"]), hashlib.sha256(path.read_bytes()).hexdigest()


def orbit_tau_flat_distances(ctx, V: np.ndarray, R: str, contexts, keys) -> np.ndarray:
    fw, K = ctx["fw"], int(ctx["p1"].generator["dimensions"]["K"])
    ids = [c.context_id for c in contexts if c.context_class == "diag"]
    G = m1.pauli_matrix(R)
    out = []
    for t in ORBIT_T:
        T = fw.tables(expm(1j * t * G) @ V, ids)
        table = {cid: dict(zip(fw.outcomes, T[cid].tolist(), strict=True)) for cid in ids}
        out.append(tau_flat_distance(ident.class_vector(table, contexts, "diag", K), keys))
    return np.asarray(out)


def run_m1_orbit_kappa_check(config_dir: str | Path = "config", output_dir: str | Path = "outputs", *, write: bool = True) -> dict[str, Any]:
    base = Path(output_dir) / m1.OUTPUT_SUBDIR
    dc_path = base / dc.ARTIFACT
    check = json.loads(dc_path.read_text(encoding="utf-8"))
    ctx = m1.model_context(config_dir, m1.DRAWS)
    contexts = full_p1_context_list(ctx["p1"].measurement)
    K = int(ctx["p1"].generator["dimensions"]["K"])
    keys = _keys(contexts, "diag", K)
    r_diag, cal_hash, cal_sha = frozen_low_diag_radius()
    n = len(ctx["p1"].generator["active_modes"])
    orbits = []
    for c in check["combinations"]:
        if c["deficit"] != "low" or not c["exact_orbit"] or not c["screened_in"]:
            continue
        mask, U = tuple(c["mask"]), ctx["unitaries"][c["unitary"]]
        R = next(R for R in c["d_commutant"] if dc.branch_condition(U, n, R, mask) < dc.EXACT_TOL)
        for k, (p, ph) in enumerate(ctx["draws"]):
            d = orbit_tau_flat_distances(ctx, m1.truth_branch_vectors(U, p, ph), R, contexts, keys)
            kap = d / r_diag
            orbits.append({"unitary": c["unitary"], "mask": list(mask), "R": R, "draw": k, "kappa_truth": float(kap[0]),
                           "kappa_min": float(kap.min()), "kappa_max": float(kap.max()), "relative_range": float((d.max() - d.min()) / d.mean())})
    rel = [o["relative_range"] for o in orbits]
    kmin = [o["kappa_min"] for o in orbits]
    artifact = {
        "artifact_type": METHOD,
        "status": "development_diagnostic_post_hoc",
        "not_declared": "post-hoc check before the Q1 verification declaration; exact tables only; opens no observation bank",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {str(dc_path).replace("\\", "/"): hashlib.sha256(dc_path.read_bytes()).hexdigest(), str(S3_CALIBRATION).replace("\\", "/"): cal_sha},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {"orbit_t_points": len(ORBIT_T), "orbit_t_range": [ORBIT_T[0], ORBIT_T[-1]], "frozen_low_diag_radius": r_diag, "s3_calibration_hash": cal_hash,
                     "diag_contexts": [c.context_id for c in contexts if c.context_class == "diag"]},
        "orbits": len(orbits),
        "relative_range_of_tau_flat_distance_quantiles": {q: float(np.quantile(rel, float(q))) for q in ("0.0", "0.1", "0.5", "0.9", "1.0")},
        "orbits_with_relative_range_above_1e-6": int(sum(r > 1e-6 for r in rel)),
        "kappa_min_over_grid_quantiles": {q: float(np.quantile(kmin, float(q))) for q in ("0.0", "0.1", "0.5")},
        "orbits_with_kappa_at_or_below_1_on_grid": int(sum(k <= 1.0 for k in kmin)),
        "orbit_rows": orbits,
    }
    artifact = m1._jsonable(artifact)
    if write:
        (base / ARTIFACT).write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    return artifact
