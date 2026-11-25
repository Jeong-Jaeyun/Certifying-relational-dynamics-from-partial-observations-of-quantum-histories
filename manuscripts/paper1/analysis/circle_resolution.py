"""Post hoc: how well does the support resolve the position on a nearly exact circle, and is that enough?

For each screened-in E2 record (regenerated from the declared seeds):
  I_t  = Fisher information of the support (N shots per context) along the circle at the truth,
         from N*KL(q_0 || q_delta) ~ I_t delta^2 / 2;
  g    = rate of change of the unobserved D tables along the circle at the truth (L2 over all
         outcomes of the D contexts outside the mask), so that their error is ~ g |t|;
  I_req= (1.96 g / r_D)^2, the information needed for |t_hat| * g <= r_D at about 95%.
Reports N_req = N * I_req / I_t (shots per context needed at this epsilon) and, using I_t ~ eps^2,
the critical eps* = eps * sqrt(I_req / I_t) at the actual N.
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
rD = float(ctx["bodies"]["radii"]["diag"])
regen, _ = m.generate_e2(ctx)
d_all = [c for c in ctx["ids"] if c.startswith("D,")]
delta = 1e-3
res = {}
for fam in ["constructed_3a", "perturbed_0.01", "perturbed_0.03", "perturbed_0.1"]:
    eps = 0.0 if fam == "constructed_3a" else float(fam.split("_")[1])
    rows = []
    for r in regen:
        if r["family"] != fam or not r["screen_passes"]:
            continue
        U = m._cplx(r["U"])
        c = np.sqrt(np.asarray(r["probs"])) * np.exp(1j * np.asarray(r["phases"]))
        V = U[:, : ctx["n"]] * c[None, :]
        mask = list(r["mask"])
        unobs = [x for x in d_all if x not in mask]
        G = m.m1.pauli_matrix(r["R"])
        Vd = expm(1j * delta * G) @ V
        T0, Td = fw.tables(V, mask + unobs), fw.tables(Vd, mask + unobs)
        kl = 0.0
        for cid in mask:
            q0 = np.clip(np.asarray(T0[cid], float), 1e-15, None)
            qd = np.clip(np.asarray(Td[cid], float), 1e-15, None)
            kl += float(np.sum(q0 * np.log(q0 / qd)))
        I_t = 2.0 * N * kl / delta ** 2
        g = float(np.sqrt(sum(np.sum((np.asarray(Td[u], float) - np.asarray(T0[u], float)) ** 2) for u in unobs))) / delta
        I_req = (1.96 * g / rD) ** 2
        rows.append({"I_t": I_t, "g": g, "I_req": I_req, "N_req": N * I_req / I_t if I_t > 0 else math.inf,
                     "eps_star": eps * math.sqrt(I_req / I_t) if I_t > 0 else math.inf, "sd_t": 1 / math.sqrt(I_t) if I_t > 0 else math.inf})
    q = lambda k: np.quantile([x[k] for x in rows], [0.1, 0.5, 0.9])
    print(f"{fam:16s} n={len(rows)} sd(t) {np.round(q('sd_t'), 3)}  g {np.round(q('g'), 3)}  t_needed(r/1.96g) {np.round(rD / 1.96 / q('g')[::-1], 4)}"
          f"  N_req {np.round(q('N_req'), 0)}  eps* {np.round(q('eps_star'), 3)}")
    res[fam] = rows
json.dump(res, open(sys.argv[1], "w"))
