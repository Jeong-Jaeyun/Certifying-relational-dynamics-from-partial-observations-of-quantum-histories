"""E3 support-only rule comparison for the single added D context: declare, then run once.

See documents/M1-E3-rule-comparison-protocol.md. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_m1_e3_rules.py declare
    python runners/p3/run_m1_e3_rules.py run --workers 6
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

    from bi_qmn.p3.runs.m1_e3_rules import declare_m1_e3, run_m1_e3

    if args.action == "declare":
        d = declare_m1_e3(args.config_dir, args.output_dir)
        print(f"declared: snapshot={d['source_snapshot_hash']} checked_banks={d['disjointness']['checked_bank_count']} "
              f"unchecked={len(d['disjointness']['unchecked_bank_specs'])} bank_fingerprint={d['bank']['construction_fingerprint_hash']}", flush=True)
        return
    a = run_m1_e3(args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    print("ledger:", json.dumps(a["bank_ledger"]), flush=True)
    print("decision:", json.dumps(a["decision"]), flush=True)
    for s, block in a["strata"].items():
        if not block.get("n"):
            continue
        print(f"[{s}] n={block['n']}", flush=True)
        for k, d in block["comparisons_pairwise"].items():
            print(f"[{s}]   {k}: point {d['point']:.4f} [{d['lower']:.4f}, {d['upper']:.4f}] cvar {d['cvar_first']:.4f} vs {d['cvar_second']:.4f}", flush=True)
        print(f"[{s}] large failures:", json.dumps(block["large_failures"]), flush=True)
        print(f"[{s}] near-degenerate:", json.dumps(block["near_degenerate"]), flush=True)
        print(f"[{s}] agreement with truth:", json.dumps(block["agreement_with_truth"]), flush=True)
        print(f"[{s}] pick agreement:", json.dumps(block["pick_agreement"]), flush=True)
    print("detector:", json.dumps(a["detector_3a_vs_3b"]), flush=True)
    print("execution:", json.dumps(a["execution"]), flush=True)
    print("post-run drift:", a["source_snapshot_post_run_drift"], flush=True)


if __name__ == "__main__":
    main()
