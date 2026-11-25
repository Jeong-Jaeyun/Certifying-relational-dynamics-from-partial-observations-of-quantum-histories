"""Post hoc numerical bound: can ANY support-only check detect mild dephasing?

For each record of the declared bootstrap-guard development bank (regenerated exactly from its seeds),
the frozen level-i estimator is fitted to the EXACT support probabilities of the truth with weight
n_shot.  The fit minimizes sum_ctx KL(q_true || q_theta) over pure histories (the KL projection P*).
With D = n_shot * sum_ctx KL(q_true || q_P*), every test of the pure-history model at level alpha has
power at most alpha + TV(P*^n, Q^n) <= alpha + min(sqrt(D/2), sqrt(1 - exp(-D)))  (Pinsker,
Bretagnolle-Huber), because it must hold its level at the pure history P*.  The script also records
how far P* lies from the truth on the D and F classes relative to the frozen body radii.
Workers: process pool, at most 6, one BLAS thread each.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path("D:/BI-QMN")
FAMILIES = ["pure", "dephased_0.05", "dephased_0.1", "dephased_0.25"]
_W = {}


def _init():
    os.chdir(ROOT)
    import bi_qmn.p3.runs.bootstrap_guard as bg
    _W["bg"] = bg
    _W["ctx"] = bg._ctx(Path("config"))


def job(spec):
    bg = _W["bg"]
    ctx = _W["ctx"]
    from qiskit.quantum_info import DensityMatrix
    i, deficit, family = spec
    bank = ctx["bank"]
    inst = ctx["instances"][i]
    rho, eta = bg.gg.family_truth(ctx, i, family)
    meta = {"instance_index": i, "deficit": deficit, "family": family, "bank_id": bank["bank_id"]}
    exact = bg.p1_measurement_table(DensityMatrix(rho), ctx["contexts"], K=inst.K, eta=eta, metadata=meta).copy_without_source()
    tables = {c: dict(row) for c, row in exact.probabilities.items()}
    n_shot = int(ctx["p1"].measurement["observation_deficit"][deficit]["n_shot"])
    sup_ctx = bg.support_contexts_for_deficit(ctx["p1"].measurement, deficit, seed_key=f"p3v2:{bank['bank_id']}:{deficit}:{i}")
    rng = np.random.default_rng(bg._seed(f"{bank['observation_seed']}:{bank['bank_id']}:{deficit}:{i}"))
    sampled = bg.sample_shot_batch(exact, sup_ctx, n_shots=n_shot, rng=rng, shot_start=(bg.e5.DEFICITS.index(deficit) * 1000 + i) * 100_000_000,
                                   role="bootstrap_guard_support").table.probabilities
    sampled = {c: dict(row) for c, row in sampled.items()}
    screen = ctx["screen"].evaluate(sampled, float(n_shot))
    out = {"index": i, "deficit": deficit, "family": family, "screen_passes": bool(screen["passes"])}
    if not out["screen_passes"]:
        return out
    ids = sorted(sampled)
    exact_support = {c: tables[c] for c in ids}
    pred = ctx["sbp"].predict_table(bg.MeasurementTable(probabilities=exact_support, contexts=[ctx["context_by_id"][c] for c in ids], metadata={"n_shots": float(n_shot)}, source_density=None),
                                    list(ctx["contexts"]))
    fit_tables = bg._table_dict(pred)
    D = 0.0
    for c in ids:
        for o, q in exact_support[c].items():
            if q > 0:
                D += n_shot * q * math.log(q / max(fit_tables[c][o], 1e-300))
    radii = ctx["bodies"]["radii"][deficit]
    s = {cls: float(np.linalg.norm(bg.ident.class_vector(fit_tables, ctx["contexts"], cls, ctx["K"]) - bg.ident.class_vector(tables, ctx["contexts"], cls, ctx["K"])))
         for cls in bg.CLASSES}
    tv = min(math.sqrt(D / 2.0), math.sqrt(1.0 - math.exp(-D)))
    out.update(D=D, tv_bound=tv, power_bound_005=min(1.0, 0.05 + tv), score=s, ratio={cls: s[cls] / radii[cls] for cls in bg.CLASSES},
               outside=any(s[cls] > radii[cls] for cls in bg.CLASSES))
    return out


def main():
    jobs = [(i, d, f) for f in FAMILIES for d in ("medium", "low") for i in range(64)]
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=5, initializer=_init) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if k % 64 == 0:
                print(f"{k}/{len(jobs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    json.dump(rows, open(sys.argv[1], "w"))
    print("done", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
