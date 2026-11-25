"""M1 post-hoc check: D-commutant rule and exact R-orbits (reads the M1 structural-map artifact).

Run from the repository root with PYTHONPATH=src, after run_m1_support_identifiability_map.py:
    python runners/p3/run_m1_d_commutant_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    from bi_qmn.p3.runs.m1_d_commutant_check import run_m1_d_commutant_check

    a = run_m1_d_commutant_check()
    for key in ("first_order_rule_confusion", "d_commutant_paulis_of_deficient_masks", "exact_orbit_incidence_low", "exact_orbit_by_unitary",
                "exact_orbit_support_residual_max", "exact_orbit_unobserved_deviation_quantiles", "e2_family_vs_orbit_condition", "d1_v7_low_screened_in"):
        print(f"{key}: {json.dumps(a[key])}", flush=True)
    for r in a["deficient_records"]:
        print("record", json.dumps({k: r[k] for k in ("source", "role", "block", "unitary", "d_commutant", "exact_orbit", "screen_passes", "diag_score", "bi_status")}), flush=True)


if __name__ == "__main__":
    main()
