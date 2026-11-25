"""M1 post-hoc check: one added D context breaks the D-commutant symmetry (reads the M1 map and D-commutant artifacts).

Run from the repository root with PYTHONPATH=src, after run_m1_d_commutant_check.py:
    python runners/p3/run_m1_added_context_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    from bi_qmn.p3.runs.m1_added_context_check import run_m1_added_context_check

    a = run_m1_added_context_check()
    for key in ("deficient_low_masks", "candidates_by_R", "exact_screened_in_low", "rank_restoration", "orbit_breaking", "choice_rules",
                "identifiable_reference_min_smin_rel", "s_Q_max_relative_drift_along_orbit", "orbit_preservation_max",
                "orbit_conditional_state_trace_distance_max"):
        print(f"{key}: {json.dumps(a[key])}", flush=True)


if __name__ == "__main__":
    main()
