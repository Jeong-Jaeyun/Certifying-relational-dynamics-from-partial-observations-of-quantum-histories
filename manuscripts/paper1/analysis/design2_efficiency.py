"""Design 2 pilot, post hoc: is the gap between estimator error and the two-point bound an estimator limit?

For every exact and nearly exact pilot record (support only, arm A0) compute the local efficient-estimator
prediction along the circle: Fisher information I_t of the support at N shots per context (from N * KL at a small
angle), the rate g at which the record's evaluation tables O change along the circle, and the predicted mean L2
error sqrt(2/pi) * g / sqrt(I_t) of an efficient (Gaussian) estimate of the position, capped at the error of the
farthest circle point. Compare with the estimator's actual A0 error and the two-point bound of the same record.
Usage: python design2_efficiency.py outputs/design2/pilot.json out.json
"""
import json
import math
import sys

import numpy as np

from bi_qmn.p3.runs import design2_core as d2
from bi_qmn.p3.runs import design2_pilot as dp


def main():
    art = json.load(open(sys.argv[1]))
    rows = [r for r in art["rows"] if r["kind"] == "truth" and r["family"] in ("exact", "near_0.03", "near_0.1")]
    out = []
    cache = {}
    for r in rows:
        des = d2.Design.make(r["design"])
        if r["i"] not in cache:
            cache[r["i"]] = dp.construction(r["i"])
        V = dp.truth(r["family"], cache[r["i"]])
        N, O = r["N"], r["O"]
        delta = 1e-3
        Vd = d2.rotate(V, des.R, delta)
        T0, Td = d2.tables(des, V, des.support + O), d2.tables(des, Vd, des.support + O)
        kl = 0.0
        for c in des.support:
            p, q = np.clip(T0[c], 1e-15, None), np.clip(Td[c], 1e-15, None)
            kl += float(np.sum(p * np.log(p / q)))
        I_t = 2.0 * N * kl / delta ** 2
        g = math.sqrt(sum(float(np.sum((Td[c] - T0[c]) ** 2)) for c in O)) / delta
        far = max(math.sqrt(sum(float(np.sum((d2.tables(des, d2.rotate(V, des.R, t), [c])[c] - T0[c]) ** 2)) for c in O)) for t in np.linspace(0, math.pi, 25)[1:])
        pred = min(math.sqrt(2 / math.pi) * g / math.sqrt(I_t), far) if I_t > 0 else far
        out.append({"design": r["design"], "family": r["family"], "N": N, "i": r["i"], "error_A0": r["errors"]["A0"], "bound_A0": r["bounds"]["A0"]["bound"],
                    "efficient_pred": pred, "I_t": I_t, "g": g, "far": far})
    json.dump(out, open(sys.argv[2], "w"))
    for design in ("golomb", "collision"):
        for fam in ("exact", "near_0.03", "near_0.1"):
            for N in (512, 2048, 8192):
                s = [x for x in out if x["design"] == design and x["family"] == fam and x["N"] == N]
                e = np.median([x["error_A0"] for x in s]); b = np.median([x["bound_A0"] for x in s]); p = np.median([x["efficient_pred"] for x in s])
                ratio = np.median([x["error_A0"] / x["efficient_pred"] for x in s])
                print(f"{design:9s} {fam:9s} N={N:5d}: median error {e:.4f} | two-point bound {b:.4f} | efficient prediction {p:.4f} | median error/prediction {ratio:.2f}")


if __name__ == "__main__":
    main()
