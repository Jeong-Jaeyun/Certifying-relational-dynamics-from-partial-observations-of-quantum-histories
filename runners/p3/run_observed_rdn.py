"""Observed RDN (model-free companion certificate of relational dynamics): declare, then run once.

See documents/observed-RDN-protocol.md. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_observed_rdn.py declare
    python runners/p3/run_observed_rdn.py run --workers 6
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

    from bi_qmn.p3.runs.observed_rdn import declare_observed_rdn, run_observed_rdn

    if args.action == "declare":
        d = declare_observed_rdn(args.config_dir, args.output_dir)
        print(f"declared: snapshot={d['source_snapshot_hash']} checked_banks={d['disjointness']['checked_bank_count']} unchecked={len(d['disjointness']['unchecked_bank_specs'])} "
              f"fresh={d['fresh_bank']['construction_fingerprint_hash']}", flush=True)
        return
    a = run_observed_rdn(args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    print("O1 type I:", json.dumps(a["fresh"]["O1_type_one"]), flush=True)
    for fam, block in a["fresh"]["by_family"].items():
        print(fam, json.dumps({k: block["pooled"][k] for k in block["pooled"] if k.startswith("certified") or k == "screen_pass"}), flush=True)
    print("O3 dose:", json.dumps(a["fresh"]["O3_dose_response"]), flush=True)
    print("O5 screen:", json.dumps(a["fresh"]["O5_screen_agreement"]), flush=True)
    for name in ("e5", "g_guard"):
        if f"existing_{name}" in a:
            e = a[f"existing_{name}"]
            print(name, "mismatches", len(e["regeneration_mismatches"]), json.dumps(e["by_family"]), flush=True)
    print("execution:", json.dumps(a["execution"]), "drift:", a["source_snapshot_post_run_drift"], flush=True)


if __name__ == "__main__":
    main()
