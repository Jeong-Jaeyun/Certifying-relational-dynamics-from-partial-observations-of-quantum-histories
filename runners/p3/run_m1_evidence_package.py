"""M1.1 / M1.2 / E6 evidence package (development only).

See src/bi_qmn/p3/runs/m1_evidence_package.py. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_m1_evidence_package.py [--starts 128]
Writes outputs/m1_evidence_package/m1_evidence_package.json and m1_reproduction_kit_3a.json; refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--starts", type=int, default=128)
    args = parser.parse_args()

    from bi_qmn.p3.runs.m1_evidence_package import run_m1_evidence_package

    a = run_m1_evidence_package(args.config_dir, args.output_dir, starts=args.starts, progress=lambda line: print(line, flush=True))
    for name, case in a["cases"].items():
        fam = case.get("family", {})
        print(name, json.dumps({"R": case.get("R"), "jacobian": case["jacobian"], "max_support_residual": fam.get("max_support_residual"),
                                "max_heldout_difference": fam.get("max_heldout_difference"), "order": (case.get("response_order") or {}).get("log_slope"),
                                "global": {k: v for k, v in case["global_search"].items() if k != "orbit_t_of_exact_fits"}}), flush=True)
    print("m1_2:", json.dumps({k: v for k, v in a["m1_2"].items()}), flush=True)
    print("execution:", json.dumps(a["execution"]), flush=True)


if __name__ == "__main__":
    main()
