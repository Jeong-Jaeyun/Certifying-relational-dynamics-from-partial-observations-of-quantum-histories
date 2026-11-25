"""E1 sealed confirmation (same-budget recovery by one added D context): declare, then run once.

See documents/E1-preregistration.md. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_m1_e1_confirmatory.py declare
    python runners/p3/run_m1_e1_confirmatory.py run --workers 6
The run builds and opens the two sealed banks once; it refuses to run twice.
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

    from bi_qmn.p3.runs.m1_e1_confirmatory import declare_m1_e1, run_m1_e1

    if args.action == "declare":
        d = declare_m1_e1(args.config_dir, args.output_dir)
        print(f"declared: status={d['status']} snapshot={d['source_snapshot_hash']} checked_banks={d['disjointness']['checked_bank_count']} "
              f"unchecked={len(d['disjointness']['unchecked_bank_specs'])} delta_min={d['delta_min']} "
              f"H1={d['banks']['H1']['construction_fingerprint_hash']} H2={d['banks']['H2']['construction_fingerprint_hash']}", flush=True)
        return
    a = run_m1_e1(args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    print("decision:", json.dumps(a["decision"]), flush=True)
    for b, block in a["banks"].items():
        print(f"[{b}] ledger:", json.dumps(block["ledger"]), flush=True)
        if not block.get("n"):
            continue
        print(f"[{b}] primary:", json.dumps(block["primary"]), flush=True)
        for name, d in {**block["secondary_E"], **block["oracle_E_o"]}.items():
            print(f"[{b}]   {name}: point {d['point']:.4f} [{d['lower']:.4f}, {d['upper']:.4f}]", flush=True)
        print(f"[{b}] leave-one-out:", json.dumps(block["leave_one_out_primary"]), flush=True)
        print(f"[{b}] by class:", json.dumps(block["by_class"]), flush=True)
        print(f"[{b}] execution:", json.dumps(block["execution"]), flush=True)
    print("post-run drift:", a["source_snapshot_post_run_drift"], flush=True)


if __name__ == "__main__":
    main()
