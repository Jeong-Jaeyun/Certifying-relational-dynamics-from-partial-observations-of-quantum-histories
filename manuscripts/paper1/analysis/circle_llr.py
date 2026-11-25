"""Post hoc: expected log-likelihood ratio N*KL(q_truth || q_t) along the R circle, E2 records.

For each record the support distribution at the truth (t = 0) is compared with every point of the
circle (I x exp(itR)) on a 64-point grid; D(t) = N * sum_ctx KL(q_0 || q_t).  Large D(t) means the
support can tell the truth from that point.  Also reports the max-entry z-score at the point of
largest movement (z = |dq| / sqrt(q (1 - q) / N)).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import bi_qmn.p3.runs.e2_e4_extensions as m

ctx = m.prepare(Path("config"), "e2")
fw = ctx["fw"]
N = float(ctx["n_shot"])
a = json.load(open("outputs/e2_scope_extension/e2_scope_extension.json"))
stored = {(r["family"], r["index"]): r for r in a["records"]}
regen, _ = m.generate_e2(ctx)
print("regenerated", len(regen), "stored", len(stored))
grid = np.linspace(0.0, math.pi, 65)[1:-1]
out = {}
for fam in ["constructed_3a", "perturbed_0.01", "perturbed_0.03", "perturbed_0.1", "haar"]:
    rows = []
    for r in regen:
        if r["family"] != fam or not r["screen_passes"]:
            continue
        st = stored[(r["family"], r["index"])]
        U = m._cplx(r["U"])
        c = np.sqrt(np.asarray(r["probs"])) * np.exp(1j * np.asarray(r["phases"]))
        V = U[:, : ctx["n"]] * c[None, :]
        mask = list(r["mask"])
        G = m.m1.pauli_matrix(r["R"])
        T0 = fw.tables(V, mask)
        Dmax, zmax, dev = 0.0, 0.0, 0.0
        for t in grid:
            Tt = fw.tables(expm(1j * t * G) @ V, mask)
            D = 0.0
            for cid in mask:
                q0 = np.clip(np.asarray(T0[cid], float), 1e-15, None)
                qt = np.clip(np.asarray(Tt[cid], float), 1e-15, None)
                D += N * float(np.sum(q0 * np.log(q0 / qt)))
                d = np.abs(qt - q0)
                k = int(np.argmax(d))
                if d[k] > dev:
                    dev = float(d[k])
                    zmax = float(d[k] / math.sqrt(max(q0[k] * (1 - q0[k]), 1e-30) / N))
            Dmax = max(Dmax, D)
        rows.append({"Dmax": Dmax, "dev": dev, "z_at_maxdev": zmax, "stored_dev": st["support_circle_deviation"],
                     "A0_diag": st["errors"]["A0"]["diag"], "covered_A0": st["coverage"]["A0"]["joint"]})
    Dm = np.array([x["Dmax"] for x in rows]); z = np.array([x["z_at_maxdev"] for x in rows])
    print(f"{fam:16s} n={len(rows)}  max_t N*KL: median {np.median(Dm):.3g}  90% {np.quantile(Dm, 0.9):.3g}  max {Dm.max():.3g} | "
          f"z at max dev: median {np.median(z):.2f} max {z.max():.2f} | dev check {max(abs(x['dev'] - x['stored_dev']) for x in rows):.1e}")
    out[fam] = rows
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "circle_llr.json", "w"))
