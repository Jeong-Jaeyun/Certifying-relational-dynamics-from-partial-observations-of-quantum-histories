"""M1 cause-identification experiment: structural map of exact-support identifiability (SBP model).

Development diagnostic for roadmap milestone M1. See documents/M1-third-mechanism-design.md.
It opens no observation bank and samples no shots. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_m1_support_identifiability_map.py --workers 6
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="overwrite an existing artifact")
    args = parser.parse_args()

    from bi_qmn.p3.runs.m1_support_identifiability_map import DEFICITS, run_m1_support_identifiability_map

    artifact = run_m1_support_identifiability_map(args.config_dir, args.output_dir, workers=args.workers, force=args.force,
                                                  progress=lambda line: print(line, flush=True))
    agg = artifact["e1_structural_map"]
    for deficit in DEFICITS:
        print(f"E1 {deficit}: {json.dumps(agg['predicted_incidence'][deficit])}", flush=True)
    for key, cell in agg["cells"].items():
        print(f"E1 {key}: counts={cell['counts']} screened_in={cell['counts_screened_in']}", flush=True)
    print("E1 separation:", json.dumps(agg["separation"]), "gauge dims:", agg["gauge_dims_seen"], flush=True)
    print("E2:", json.dumps(artifact["e2_mechanism_summary"], indent=1), flush=True)
    summary = artifact["e3_retrospective_summary"] or {}
    for key, value in summary.items():
        brief = {k: v for k, v in value.items() if not isinstance(v, list)}
        print(f"E3 {key}: {json.dumps(brief)}", flush=True)
    print("E3 rebuild mismatches:", len(artifact["e3_retrospective_join"].get("rebuild_mismatches", [])), flush=True)
    print(f"total seconds: {artifact['execution']['total_seconds']:.0f}", flush=True)


if __name__ == "__main__":
    main()
