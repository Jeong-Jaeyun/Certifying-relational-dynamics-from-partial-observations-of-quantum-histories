"""Design 2, two-point (Le Cam) bound with every measured context held out of the evaluation set.

Corrects design2_bound.py, whose evaluation set (unobserved D minus {IZ, ZZ}) kept the fixed added context
"+best" when that context was one of the seven. Here, for each history, the evaluation set is the unobserved D
contexts minus {IZ, ZZ, best_q}, common to all arms of that history (six or seven contexts). Everything else
(histories, rotations, arms, t grid, shot grid, bound formula) is identical to design2_bound.py, and the script
checks that its histories reproduce the weights and best candidates recorded in the earlier output.

For a history V and a point exp(itR) V of its circle, any estimator of the evaluation tables has, at one of the
two histories, expected L2 error at least (d/2) max(1 - sqrt(1 - BC^2), BC^2/2), with d the L2 distance of the
two histories' evaluation tables and BC = prod_c BC_c^{N_c} over the measured contexts. Maximized over t.

Arms (budget N per support context): A0 support only; C the extra N shots spread over the 8 support contexts;
+IZ, +ZZ, +best support plus N shots on one fixed added context (+best: the candidate with the largest TRUE
sigma_min, i.e. chosen with the true history, not from data).
Usage: python design2_bound_heldout.py out.json [earlier_bound.json]
"""
import hashlib
import json
import math
import sys

import numpy as np
from scipy.linalg import expm

from bi_qmn.p3.runs import design2_core as d2

EPS = [0.0, 0.01, 0.03, 0.1, 0.3]
SHOTS = [128, 512, 2048, 8192, 32768, 131072, 524288]
TGRID = np.unique(np.concatenate([np.linspace(0.0, math.pi, 49)[1:-1], np.logspace(-4, math.log10(0.5), 40)]))
N_CONSTRUCTED = 16


def seed_of(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def histories(des, rng):
    U = np.kron(expm(1j * math.pi / 7 * d2._P1["X"]), np.eye(2))
    out = [("specified-given", U, [0.38, 0.17, 0.27, 0.18], [0, 0.3, 0.7, 1.1]),
           ("specified-equal", U, [0.25] * 4, [0, 0.3, 0.7, 1.1])]
    for i in range(N_CONSTRUCTED):
        Ui, r = d2.project_exact(d2.haar_unitary(4, rng), des, rng)
        if r > 1e-12:
            continue
        while True:
            p = rng.dirichlet(np.full(4, 8.0))
            if p.min() >= 0.1:
                break
        ph = np.concatenate([[0.0], rng.uniform(0, 2 * math.pi, 3)])
        out.append((f"constructed-{i}", Ui, p.tolist(), ph.tolist()))
    return out


def per_t(des, V, contexts, eval_ctx):
    T0 = d2.tables(des, V, contexts + eval_ctx)
    rows = []
    for t in TGRID:
        T1 = d2.tables(des, d2.rotate(V, des.R, t), contexts + eval_ctx)
        lbc = {c: math.log(max(d2.bhattacharyya(T0[c], T1[c]), 1e-300)) for c in contexts}
        d = math.sqrt(sum(float(np.sum((T0[c] - T1[c]) ** 2)) for c in eval_ctx))
        rows.append((float(t), lbc, d))
    return rows


def curve(rows, shots_of):
    res = []
    for N in SHOTS:
        sh = shots_of(N)
        best = {"bound": 0.0, "t": None, "d": 0.0, "bc": 1.0}
        for t, lbc, d in rows:
            bc = math.exp(sum(n * lbc[c] for c, n in sh.items()))
            b = 0.5 * d * max(1.0 - math.sqrt(max(0.0, 1.0 - bc * bc)), 0.5 * bc * bc)
            if b > best["bound"]:
                best = {"bound": b, "t": t, "d": d, "bc": bc}
        res.append({"N": N, **best})
    return res


def main():
    earlier = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    rng = np.random.default_rng(20261124)
    out = {}
    for name in ("golomb", "collision"):
        des = d2.Design.make(name)
        for label, U, p, ph in histories(des, rng):
            h_rng = np.random.default_rng(seed_of(f"{name}|{label}"))
            h = h_rng.normal(size=(4, 4)) + 1j * h_rng.normal(size=(4, 4)); h = (h + h.conj().T) / 2; h /= np.linalg.norm(h)
            for eps in EPS:
                V = d2.branch_vectors(expm(1j * eps * h) @ U, p, ph)
                sig = {c: d2.separation(des, V, c)["sigma_min"] for c in des.candidates}
                best_q = max(sig, key=sig.get)
                key = f"{name}|{label}|{eps}"
                if earlier is not None:
                    e = earlier[key]
                    assert e["best_candidate"] == best_q and np.allclose(e["weights"], p), key
                eval_ctx = [c for c in des.unobserved_D if c not in ("D,IZ", "D,ZZ", best_q)]
                measured = list(dict.fromkeys(des.support + ["D,IZ", "D,ZZ", best_q]))
                rows = per_t(des, V, measured, eval_ctx)
                arms = {"A0": lambda N: {c: N for c in des.support},
                        "C": lambda N: {c: N * 9 / 8 for c in des.support},
                        "+IZ": lambda N: {**{c: N for c in des.support}, "D,IZ": N},
                        "+ZZ": lambda N: {**{c: N for c in des.support}, "D,ZZ": N},
                        "+best": lambda N, q=best_q: {**{c: N for c in des.support}, q: N}}
                out[key] = {"weights": p, "sigma_min": sig, "best_candidate": best_q, "eval_contexts": eval_ctx,
                            "arms": {a: curve(rows, f) for a, f in arms.items()}}
                g = lambda a, N: [r for r in out[key]["arms"][a] if r["N"] == N][0]["bound"]  # noqa: E731
                print(f"{key:34s} n_eval={len(eval_ctx)} N=2048: A0 {g('A0', 2048):.4f}  C {g('C', 2048):.4f}  "
                      f"+IZ {g('+IZ', 2048):.4f}  +ZZ {g('+ZZ', 2048):.4f}  +best({best_q[2:]}) {g('+best', 2048):.4f}",
                      flush=True)
    json.dump(out, open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
