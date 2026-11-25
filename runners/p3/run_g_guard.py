"""G guard (support goodness-of-fit abstention): declare, calibrate, then evaluate on an independent bank.

See documents/G-guard-protocol.md. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_g_guard.py declare
    python runners/p3/run_g_guard.py calibrate --workers 6
    python runners/p3/run_g_guard.py evaluate --workers 6
Each stage writes one artifact under outputs/g_guard/ and refuses to run twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("declare", "calibrate", "evaluate"))
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers > 6:
        raise SystemExit("at most 6 worker processes")

    from bi_qmn.p3.runs.g_guard import calibrate, declare_g_guard, evaluate

    log = lambda line: print(line, flush=True)   # noqa: E731
    if args.action == "declare":
        d = declare_g_guard(args.config_dir, args.output_dir)
        print(f"declared: snapshot={d['source_snapshot_hash']} checked_banks={d['disjointness']['checked_bank_count']} unchecked={len(d['disjointness']['unchecked_bank_specs'])} "
              f"calibration={d['banks']['calibration']['construction_fingerprint_hash']} evaluation={d['banks']['evaluation']['construction_fingerprint_hash']}", flush=True)
        return
    if args.action == "calibrate":
        a = calibrate(args.config_dir, args.output_dir, workers=args.workers, progress=log)
        print("thresholds:", json.dumps(a["thresholds"]), flush=True)
        print("strata:", json.dumps({d: {k: v for k, v in s.items() if k != "kept_indices"} for d, s in a["strata"].items()}), flush=True)
        print("execution:", json.dumps(a["execution"]), "drift:", a["source_snapshot_post_run_drift"], flush=True)
        return
    a = evaluate(args.config_dir, args.output_dir, workers=args.workers, progress=log)
    print("thresholds:", json.dumps(a["thresholds"]), flush=True)
    print("endpoints:", json.dumps(a["endpoints"]), flush=True)
    print("decision:", json.dumps(a["decision"]), flush=True)
    for fam, block in a["by_family"].items():
        print(fam, json.dumps(block["pooled"]), flush=True)
    print("execution:", json.dumps(a["execution"]), "drift:", a["source_snapshot_post_run_drift"], flush=True)


if __name__ == "__main__":
    main()
