"""Run the design 2 exploratory pilot (documents/design2-pilot-design.md).

Usage (repository root, PYTHONPATH=src):  python runners/p3/run_design2_pilot.py --workers 6
"""
import argparse

from bi_qmn.p3.runs import design2_pilot as dp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--records", type=int, default=48)
    ap.add_argument("--out", default="outputs/design2/pilot.json")
    a = ap.parse_args()
    if not 1 <= a.workers <= 6:
        raise SystemExit("at most 6 workers")
    art = dp.run(records=a.records, workers=a.workers, out=a.out, progress=lambda s: print(s, flush=True))
    print("done", art["execution"], flush=True)
