"""M1 post-hoc check: the RDN margin kappa along exact 3a orbits (reads the D-commutant artifact and the frozen S3 calibration).

Run from the repository root with PYTHONPATH=src, after run_m1_d_commutant_check.py:
    python runners/p3/run_m1_orbit_kappa_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    from bi_qmn.p3.runs.m1_orbit_kappa_check import run_m1_orbit_kappa_check

    a = run_m1_orbit_kappa_check()
    for key in ("settings", "orbits", "relative_range_of_tau_flat_distance_quantiles", "orbits_with_relative_range_above_1e-6",
                "kappa_min_over_grid_quantiles", "orbits_with_kappa_at_or_below_1_on_grid"):
        print(f"{key}: {json.dumps(a[key])}", flush=True)


if __name__ == "__main__":
    main()
