"""E5 development pilot: specificity of the S3 certification route under controlled truths.

See src/bi_qmn/p3/runs/e5_rdn_specificity.py. Run from the repository root with PYTHONPATH=src:
    python runners/p3/run_e5_rdn_specificity.py --workers 6
Writes outputs/e5_rdn_specificity/e5_rdn_specificity.json; refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers > 6:
        raise SystemExit("at most 6 worker processes")

    from bi_qmn.p3.runs.e5_rdn_specificity import run_e5

    a = run_e5(args.config_dir, args.output_dir, workers=args.workers, progress=lambda line: print(line, flush=True))
    for key, block in a["summary"].items():
        print(key, json.dumps(block), flush=True)
    print("execution:", json.dumps(a["execution"]), flush=True)


if __name__ == "__main__":
    main()
