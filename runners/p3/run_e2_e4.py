"""E2 (scope extension) and E4 (Hamiltonian information): declare, then run once per stage.

See documents/E2-scope-extension-protocol.md and documents/E4-hamiltonian-information-protocol.md.
Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_e2_e4.py declare --stage e2
    python runners/p3/run_e2_e4.py run --stage e2 --workers 6
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
    parser.add_argument("--stage", choices=("e2", "e4"), required=True)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers > 6:
        raise SystemExit("at most 6 worker processes")

    from bi_qmn.p3.runs.e2_e4_extensions import declare, run

    if args.action == "declare":
        d = declare(args.stage, args.config_dir, args.output_dir)
        print(f"declared {args.stage}: snapshot={d['source_snapshot_hash']} checked_banks={d['disjointness']['checked_bank_count']} "
              f"unchecked={len(d['disjointness']['unchecked_bank_specs'])}", flush=True)
        return
    a = run(args.stage, args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    an = a["analysis"]
    print("ledger:", json.dumps(an["ledger"]), flush=True)
    if args.stage == "e2":
        for fam, block in an["by_family"].items():
            print(fam, json.dumps({k: block.get(k) for k in ("screened_in", "classes", "cvar", "same_budget_gain_above_delta_min", "coverage_joint", "certificates", "fiber_A0")}), flush=True)
            if "contrasts" in block:
                print("  contrasts", json.dumps({k: (v["point"], v["lower"], v["upper"]) for k, v in block["contrasts"].items() if v}), flush=True)
    else:
        for arm, block in an["by_arm"].items():
            print(arm, json.dumps({k: block.get(k) for k in ("feasible", "cvar", "coverage_joint", "orbit_angle", "nll_minus_A0", "certificate")}), flush=True)
        print("registries:", json.dumps(an["registries"]), flush=True)
    print("execution:", json.dumps(a["execution"]), "drift:", a["source_snapshot_post_run_drift"], flush=True)


if __name__ == "__main__":
    main()
