"""M1 Q1 verification (risk detection on a fixed support, recovery by one added D context): declare, then run once.

See documents/M1-added-measurement-design.md. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_m1_added_measurement.py declare
    python runners/p3/run_m1_added_measurement.py run --workers 6
The run builds and screens the new development bank; it refuses to run twice.
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
    if args.workers > 6:
        raise SystemExit("at most 6 worker processes")

    from bi_qmn.p3.runs.m1_added_measurement import declare_m1_added_measurement, run_m1_added_measurement

    if args.action == "declare":
        d = declare_m1_added_measurement(args.config_dir, args.output_dir)
        print(f"declared: snapshot={d['source_snapshot_hash']} checked_banks={d['disjointness']['checked_bank_count']} unchecked={len(d['disjointness']['unchecked_bank_specs'])} "
              f"flagged_masks={d['bank']['flagged_mask_count']} bank_fingerprint={d['bank']['construction_fingerprint_hash']}", flush=True)
        return
    a = run_m1_added_measurement(args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    print("ledger:", json.dumps(a["bank_ledger"]), flush=True)
    print("decision:", json.dumps(a["decision"]), flush=True)
    for s, block in a["strata"].items():
        if not block.get("n"):
            continue
        print(s, "n", block["n"], "picks", json.dumps(block["picks"]), flush=True)
        for set_name, comps in block["comparisons_primary_class"].items():
            for name, c in comps.items():
                print(f"  {s} {set_name} {name}: point {c['point']:.4f} [{c['lower']:.4f}, {c['upper']:.4f}] cvar {c['cvar_first']:.4f} vs {c['cvar_second']:.4f}", flush=True)
        print("  rank:", json.dumps(block["rank_stability"]), flush=True)
        print("  certification:", json.dumps(block["certification_a0"]), flush=True)
        print("  kappa:", json.dumps(block["kappa"]), flush=True)
        print("  structural:", json.dumps(block["structural"]), flush=True)
    print("execution:", json.dumps(a["execution"]), flush=True)
    print(f"post-run drift: {a['source_snapshot_post_run_drift']}", flush=True)


if __name__ == "__main__":
    main()
