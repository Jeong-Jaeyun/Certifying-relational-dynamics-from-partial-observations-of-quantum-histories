"""P3 S3 confirmatory (F_RDN on the sealed evaluation bank): declare, then run once.

See documents/P3-S3-preregistration.md. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_p3_s3_confirmatory.py declare
    python runners/p3/run_p3_s3_confirmatory.py run --workers 6
The run opens the sealed bank and performs the single final analysis; it refuses to run twice.
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

    from bi_qmn.p3.runs.p3_s3_confirmatory import declare_p3_s3_confirmatory, run_p3_s3_confirmatory

    if args.action == "declare":
        declaration = declare_p3_s3_confirmatory(args.config_dir, args.output_dir)
        power = declaration["planned_power"]
        print(f"declared: snapshot={declaration['source_snapshot_hash']} checked={declaration['disjointness']['checked_bank_count']} "
              f"statistical_protocol={declaration['statistical_protocol']['sha256'][:12]} critical_count={power['critical_count']} "
              f"joint_power_lower_bound={power['joint_power_lower_bound']:.3f}")
        return
    artifact = run_p3_s3_confirmatory(args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    for stratum, d in artifact["decision"]["strata"].items():
        print(f"{stratum}: {d['successes']}/{d['n']} lower={d['one_sided_lower_bound']:.4f} critical={d['critical_count']} p={d['p_value']:.3g} passes={d['passes']}", flush=True)
    print("status:", artifact["status"], "family p-value:", artifact["decision"]["p_value"], flush=True)
    print("report-only:", json.dumps({s: {k: v[k] for k in ("screened_out", "bi_status_counts", "rdn_status_counts", "coverage")} for s, v in artifact["report_only"].items()}), flush=True)
    print(f"post-run drift: {artifact['source_snapshot_post_run_drift']}")


if __name__ == "__main__":
    main()
