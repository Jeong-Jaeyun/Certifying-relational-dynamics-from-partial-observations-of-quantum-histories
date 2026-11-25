"""P3 S3 calibration freeze: declare, then run (see documents/P3-s3-calibration-protocol.md).

Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_p3_s3_calibration.py declare
    python runners/p3/run_p3_s3_calibration.py run --workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("declare", "run"))
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    from bi_qmn.p3.runs.p3_s3_calibration import declare_p3_s3_calibration, run_p3_s3_calibration

    if args.action == "declare":
        declaration = declare_p3_s3_calibration(args.config_dir, args.output_dir)
        print(f"declared: snapshot={declaration['source_snapshot_hash']} bank={declaration['calibration_bank']['bank_id']} checked_banks={declaration['disjointness']['checked_bank_count']}")
        return
    artifact = run_p3_s3_calibration(args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    for deficit, cell in artifact["calibration"].items():
        if not cell["available"]:
            print(f"{deficit}: unavailable (scanned {cell['scanned_through']})", flush=True)
            continue
        c = cell["classes"]
        print(f"{deficit}: r_diag={c['diag']['radius']:.4f} (norm {c['diag']['normalized_radius']:.4f} <= {c['diag']['anti_vacuity_ceiling']}) "
              f"r_coh={c['coh']['radius']:.4f} screened_out_before={cell['screened_out_before_last_selected']} wide_tail={cell['report_only']['wide_tail_records']}", flush=True)
    print("freeze:", json.dumps(artifact["freeze"], sort_keys=True), flush=True)
    print("screen audit:", artifact["screen_audit"]["screen_vs_oracle"], flush=True)
    print(f"post-run drift: {artifact['source_snapshot_post_run_drift']}")


if __name__ == "__main__":
    main()
