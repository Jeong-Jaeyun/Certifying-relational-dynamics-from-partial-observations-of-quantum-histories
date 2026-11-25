"""Design 2, structural analysis (numerical; no sampled data).

For the specified history in the Golomb and collision designs, with the given and with equal weights:
1. first-order identifiability: rank of the support Jacobian versus the all-context Jacobian in a local
   parametrization dV = iHV + V diag(delta c / c); deficiency = rank_all - rank_support; is the R rotation the
   only missing direction?
2. a multi-start search (estimator parametrization, 128 starts) on the exact support probabilities: which fits
   reach the exact optimum, and do they lie on the R circle or elsewhere?
Usage: python design2_structure.py out.json
"""
import json
import math
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from bi_qmn.p3.runs import design2_core as d2

U = np.kron(expm(1j * math.pi / 7 * d2._P1["X"]), np.eye(2))
PHASES = [0.0, 0.3, 0.7, 1.1]
CASES = {"given": [0.38, 0.17, 0.27, 0.18], "equal": [0.25] * 4}


def local_params(V):
    """Real local coordinates: 16 for a Hermitian H (dV = iHV) and 8 for the coefficients (dc/c = dr + i dphi)."""
    d, n = V.shape
    herm = []
    for k in range(d):
        E = np.zeros((d, d), complex); E[k, k] = 1; herm.append(E)
    for k in range(d):
        for l in range(k + 1, d):
            E = np.zeros((d, d), complex); E[k, l] = E[l, k] = 1; herm.append(E)
            E = np.zeros((d, d), complex); E[k, l] = -1j; E[l, k] = 1j; herm.append(E)
    dirs = [1j * H @ V for H in herm]
    for j in range(n):
        for z in (1.0, 1j):
            D = np.zeros_like(V); D[:, j] = z * V[:, j]; dirs.append(D)
    return dirs


def jac(des, V, contexts, dirs, h=1e-5):
    cols = []
    for Dv in dirs:
        Tp, Tm = d2.tables(des, V + h * Dv, contexts), d2.tables(des, V - h * Dv, contexts)
        cols.append(np.concatenate([(Tp[c] - Tm[c]) / (2 * h) for c in contexts]))
    return np.stack(cols, 1)


def rank(J, tol=1e-8):
    s = np.linalg.svd(J, compute_uv=False)
    return int(np.sum(s > tol * s[0])), s


def first_order(des, V):
    dirs = local_params(V)
    Ja, Js = jac(des, V, des.all_contexts, dirs), jac(des, V, des.support, dirs)
    ra, _ = rank(Ja); rs, _ = rank(Js)
    # is the R rotation in the support null space, and does the deficiency equal 1?
    vR = 1j * d2.pauli(des.R) @ V
    Jr = jac(des, V, des.support, [vR])[:, 0]
    # null space of Js modulo null space of Ja
    _, _, vh = np.linalg.svd(Js); null_s = vh[rs:].conj().T
    _, _, vh = np.linalg.svd(Ja); null_a = vh[ra:].conj().T
    # components of null_s outside null_a
    P = null_a @ np.linalg.pinv(null_a) if null_a.size else np.zeros((Js.shape[1], Js.shape[1]))
    extra = null_s - P @ null_s
    ext_rank = int(np.sum(np.linalg.svd(extra, compute_uv=False) > 1e-6)) if extra.size else 0
    return {"rank_all": ra, "rank_support": rs, "deficiency": ra - rs, "extra_null_dim": ext_rank,
            "R_rotation_support_derivative": float(np.max(np.abs(Jr)))}


def circle_distance(des, Vfit, Vtrue):
    """min_t of the largest table difference (all contexts) between the fit and exp(itR) Vtrue (grid, then refined)."""
    from scipy.optimize import minimize_scalar
    f = lambda t: d2.max_table_difference(des, Vfit, d2.rotate(Vtrue, des.R, t), des.all_contexts)
    grid = np.linspace(0, math.pi, 181)
    vals = [f(t) for t in grid]
    k = int(np.argmin(vals))
    step = grid[1] - grid[0]
    r = minimize_scalar(f, bounds=(grid[k] - step, grid[k] + step), method="bounded", options={"xatol": 1e-10})
    return (float(r.fun), float(r.x)) if r.fun < vals[k] else (float(vals[k]), float(grid[k]))


def exact_search(des, V, starts=128, N=1e5, seed=7):
    est = d2.estimator(des, starts=starts, seed=seed)
    T = d2.tables(des, V, des.support)
    support = {c: {f"c{l}_s{s}": float(T[c][2 * l + s]) for l in range(des.K) for s in (0, 1)} for c in des.support}
    ids, A, counts = est._support_arrays(support, N)
    xt = V.T.reshape(-1)
    q_true = np.einsum("i,oij,j->o", xt.conj(), A, xt).real
    f_true = -float(counts @ np.log(q_true + est.likelihood_floor))
    rng = np.random.default_rng(seed)
    d, n = des.d_S, len(des.modes)
    rows = []
    for _ in range(starts):
        th0 = np.concatenate([rng.normal(0, 1, d * d), np.zeros(n - 1), rng.uniform(0, 2 * math.pi, n - 1)])
        r = minimize(est._objective, th0, args=(A, counts), jac=True, method="L-BFGS-B", options={"maxiter": 6000, "ftol": 1e-15, "gtol": 1e-11})
        rows.append((float(r.fun), r.x))
    rows.sort(key=lambda x: x[0])
    exact = []
    for f, th in rows:
        Vf = est.branch_vectors(th)
        if d2.max_table_difference(des, Vf, V, des.support) < 1e-6:     # reproduces every observed probability
            exact.append((f, th))
    out = {"f_true": f_true, "best_minus_true": rows[0][0] - f_true, "n_exact": len(exact), "fits": []}
    for f, th in exact:
        Vf = est.branch_vectors(th)
        dist, t = circle_distance(des, Vf, V)
        unobs = d2.max_table_difference(des, Vf, V, des.unobserved_D)
        out["fits"].append({"obj_minus_true": f - f_true, "on_circle_maxdiff": dist, "t": t, "unobserved_maxdiff_vs_truth": unobs})
    return out


def main():
    res = {}
    for name in ("golomb", "collision"):
        des = d2.Design.make(name)
        for case, p in CASES.items():
            V = d2.branch_vectors(U, p, PHASES)
            fo = first_order(des, V)
            ex = exact_search(des, V)
            off = [f for f in ex["fits"] if f["on_circle_maxdiff"] > 1e-5]
            print(f"{name:9s} {case:5s} | Jacobian rank all {fo['rank_all']} support {fo['rank_support']} deficiency {fo['deficiency']} "
                  f"(extra null dim {fo['extra_null_dim']}; d support/dt along R = {fo['R_rotation_support_derivative']:.1e}) | "
                  f"exact fits {ex['n_exact']}/128, off-circle {len(off)}, max unobserved diff among exact fits "
                  f"{max([f['unobserved_maxdiff_vs_truth'] for f in ex['fits']] or [0]):.3f}", flush=True)
            res[f"{name}:{case}"] = {"first_order": fo, "exact_search": ex}
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
