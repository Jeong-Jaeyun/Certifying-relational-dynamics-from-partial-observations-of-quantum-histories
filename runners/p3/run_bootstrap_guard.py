"""Bootstrap guard (record-specific parametric-bootstrap goodness of fit): declare, then run once.

See documents/bootstrap-guard-protocol.md. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_bootstrap_guard.py declare
    python runners/p3/run_bootstrap_guard.py run --workers 6
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

    from bi_qmn.p3.runs.bootstrap_guard import declare_bootstrap_guard, run_bootstrap_guard

    if args.action == "declare":
        d = declare_bootstrap_guard(args.config_dir, args.output_dir)
        print(f"declared: snapshot={d['source_snapshot_hash']} checked_banks={d['disjointness']['checked_bank_count']} "
              f"unchecked={len(d['disjointness']['unchecked_bank_specs'])} bank={d['bank']['construction_fingerprint_hash']}", flush=True)
        return
    a = run_bootstrap_guard(args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    an = a["analysis"]
    print("endpoints:", json.dumps(an["endpoints"]), flush=True)
    print("decision:", json.dumps(an["decision"]), flush=True)
    for fam, block in an["by_family"].items():
        p = block["pooled"]
        print(fam, json.dumps({"screened_in": p["screened_in"], "bootstrap": p["bootstrap"], "conformal": p["conformal"]}), flush=True)
    print("execution:", json.dumps(a["execution"]), "drift:", a["source_snapshot_post_run_drift"], flush=True)


if __name__ == "__main__":
    main()
