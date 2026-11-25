"""M1 Q1 post-hoc analyses on the declared development bank (report only).

See src/bi_qmn/p3/runs/m1_added_measurement_posthoc.py. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_m1_added_measurement_posthoc.py posthoc --workers 6
    python runners/p3/run_m1_added_measurement_posthoc.py h_known --workers 6 [--certify]
Each action writes one artifact under outputs/m1_added_measurement_posthoc/ and refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("posthoc", "h_known"))
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--certify", action="store_true", help="h_known: also certify the A0_H tables")
    args = parser.parse_args()
    if args.workers > 6:
        raise SystemExit("at most 6 worker processes")

    from bi_qmn.p3.runs.m1_added_measurement_posthoc import run_h_known, run_posthoc

    log = lambda line: print(line, flush=True)   # noqa: E731
    if args.action == "posthoc":
        a = run_posthoc(args.config_dir, args.output_dir, workers=args.workers, progress=log)
        print("reproduction:", json.dumps(a["reproduction"]), flush=True)
        s = a["summary_3a"]
        print("fiber coverage (3a):", json.dumps(s["fiber_coverage"]), flush=True)
        print("orbit coefficients (3a):", json.dumps({k: v for k, v in s["orbit_coefficients"].items() if k != "A_f_tail_records"}), flush=True)
        print("A_f tail:", json.dumps(s["orbit_coefficients"]["A_f_tail_records"]), flush=True)
        print("fitted orbit point:", json.dumps(s["fitted_orbit_point"]), flush=True)
        print("energy:", json.dumps(s["energy"]), flush=True)
        print("recertification:", json.dumps(a["recertification"]), flush=True)
        print("execution:", json.dumps(a["execution"]), flush=True)
        return
    a = run_h_known(args.config_dir, args.output_dir, workers=args.workers, progress=log, certify=args.certify)
    print("strata:", json.dumps(a["strata"]), flush=True)
    print("execution:", json.dumps(a["execution"]), flush=True)


if __name__ == "__main__":
    main()
