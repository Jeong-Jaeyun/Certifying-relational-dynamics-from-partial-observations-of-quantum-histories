"""Design 2 sealed confirmatory run (documents/design2-confirmatory-preregistration.md).

Usage (repository root, PYTHONPATH=src):
  python runners/p3/run_design2_confirm.py declare
  python runners/p3/run_design2_confirm.py run --workers 6
"""
import argparse
import json

from bi_qmn.p3.runs import design2_confirm as dc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["declare", "run"])
    ap.add_argument("--config", default="config/design2_confirmatory.yaml")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.action == "declare":
        d = dc.declare(a.config, a.out_dir)
        print("declared", d["source_snapshot_hash"], len(d["source_snapshot"]), "files")
    else:
        art = dc.run(a.config, a.out_dir, workers=a.workers, progress=lambda s: print(s, flush=True))
        print(json.dumps({k: {kk: v[kk] for kk in ("passes", "point", "q025", "q975", "k", "n", "p_value") if kk in v} for k, v in art["analysis"]["endpoints"].items()}, indent=1))
        print("claim passes:", art["analysis"]["claim_passes"], "| drift:", art["source_snapshot_post_run_drift"], "|", art["execution"])
