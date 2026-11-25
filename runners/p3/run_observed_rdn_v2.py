"""Observed RDN v2 (both-margin conditional test): declare, then run once per stage.

See documents/observed-RDN-v2-protocol.md (dev) and documents/observed-RDN-v2-sealed-preregistration.md (sealed).
Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_observed_rdn_v2.py declare --stage dev
    python runners/p3/run_observed_rdn_v2.py run --stage dev --workers 6
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
    parser.add_argument("--stage", choices=("dev", "sealed"), required=True)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers > 6:
        raise SystemExit("at most 6 worker processes")

    from bi_qmn.p3.runs.observed_rdn_v2 import declare, run

    if args.action == "declare":
        d = declare(args.stage, args.config_dir, args.output_dir)
        print(f"declared {args.stage}: snapshot={d['source_snapshot_hash']} checked_banks={d['disjointness']['checked_bank_count']} "
              f"unchecked={len(d['disjointness']['unchecked_bank_specs'])} bank={d['bank']['construction_fingerprint_hash']}", flush=True)
        return
    a = run(args.stage, args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    ba = a["bank_analysis"]
    print("type I:", json.dumps(ba["type_one"]), flush=True)
    for fam, block in ba["by_family"].items():
        print(fam, json.dumps({s: {k: block[s][k]["rate"] for k in block[s] if k.endswith("_at_0.01")} for s in ("medium", "low")}), flush=True)
    if "endpoints" in a:
        print("endpoints:", json.dumps(a["endpoints"]), flush=True)
    if "s3_retrospective" in a:
        print("S3 retrospective:", json.dumps(a["s3_retrospective"]["summary"]), flush=True)
        print("S3 failures:", json.dumps(a["s3_retrospective"]["failures"]), flush=True)
    print("execution:", json.dumps(a["execution"]), "drift:", a["source_snapshot_post_run_drift"], flush=True)


if __name__ == "__main__":
    main()
