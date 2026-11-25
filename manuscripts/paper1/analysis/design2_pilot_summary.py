"""Summarize the design 2 pilot artifact (descriptive; paired bootstrap seed 57105).

Usage: python design2_pilot_summary.py outputs/design2/pilot.json outputs/design2/pilot_summary.json
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import beta

ARMS = ["A0", "C", "A_s", "A_sig", "A_IZ", "A_o"]
CONTRASTS = [("C", "A_s"), ("C", "A_sig"), ("C", "A_IZ"), ("A0", "C"), ("A_s", "A_sig")]


def cvar(x, level=0.9):
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, math.ceil((1 - level) * len(x)))
    return float(np.mean(x[:k]))


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def paired(a, b, B=10000, seed=57105):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    diffs = np.array([cvar(a[i]) - cvar(b[i]) for i in idx])
    return {"point": cvar(a) - cvar(b), "lo": float(np.quantile(diffs, 0.025)), "hi": float(np.quantile(diffs, 0.975))}


def main():
    art = json.load(open(sys.argv[1]))
    rows = art["rows"]
    groups = defaultdict(list)
    for r in rows:
        if r["kind"] == "truth":
            groups[(r["design"], r["family"], r["N"])].append(r)
    out = {"execution": art["execution"], "cells": {}, "controls": {}}
    for (des, fam, N), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["i"])
        E = {a: [r["errors"][a] for r in rs] for a in ARMS}
        cell = {"n": len(rs),
                "arms": {a: {"mean": float(np.mean(E[a])), "median": float(np.median(E[a])), "max": float(np.max(E[a])), "cvar": cvar(E[a])} for a in ARMS},
                "contrasts": {f"{x}-{y}": paired(E[x], E[y]) for x, y in CONTRASTS},
                "rules": {"agree_s_sig": float(np.mean([r["picks"]["A_s"] == r["picks"]["A_sig"] for r in rs])),
                          "rank1_A_s": int(sum(r["rank1_picks"]["A_s"] for r in rs)), "rank1_A_sig": int(sum(r["rank1_picks"]["A_sig"] for r in rs)),
                          "rank1_A_IZ": int(sum(r["rank1_picks"]["A_IZ"] for r in rs)),
                          "sigma_true_pick_median": {a: float(np.median([r["sigma_true_of_pick"][a] for r in rs])) for a in ("A_s", "A_sig", "A_IZ", "A_o")}},
                "bound": {a: {"median_bound": float(np.median([r["bounds"][a]["bound"] for r in rs])),
                              "median_ratio": float(np.median([r["errors"][a] / max(r["bounds"][a]["bound"], 1e-12) for r in rs]))} for a in ("A0", "C", "A_sig")},
                "rdn": {"k": int(sum(r["rdn"]["p"] <= 0.01 for r in rs)), "n": len(rs)}}
        cell["rdn"]["ci95"] = cp(cell["rdn"]["k"], cell["rdn"]["n"])
        out["cells"][f"{des}|{fam}|{N}"] = cell
        a = cell["arms"]; c = cell["contrasts"]; b = cell["bound"]
        print(f"{des:9s} {fam:17s} N={N:5d} n={len(rs):2d} | CVaR A0 {a['A0']['cvar']:.4f} C {a['C']['cvar']:.4f} As {a['A_s']['cvar']:.4f} "
              f"Asig {a['A_sig']['cvar']:.4f} AIZ {a['A_IZ']['cvar']:.4f} Ao {a['A_o']['cvar']:.4f} | C-Asig {c['C-A_sig']['point']:.3f} "
              f"[{c['C-A_sig']['lo']:.3f},{c['C-A_sig']['hi']:.3f}] | agree {cell['rules']['agree_s_sig']:.2f} rank1(s/sig/IZ) "
              f"{cell['rules']['rank1_A_s']}/{cell['rules']['rank1_A_sig']}/{cell['rules']['rank1_A_IZ']} | err/bound A0 {b['A0']['median_ratio']:.2f} "
              f"Asig {b['A_sig']['median_ratio']:.1f} | RDN {cell['rdn']['k']}/{cell['rdn']['n']}", flush=True)
    cg = defaultdict(list)
    for r in rows:
        if r["kind"] == "control":
            cg[(r["design"], r["control"], r["N"])].append(r)
    for (des, ctl, N), rs in sorted(cg.items()):
        k = int(sum(r["rdn"]["p"] <= 0.01 for r in rs))
        out["controls"][f"{des}|{ctl}|{N}"] = {"k": k, "n": len(rs), "ci95": cp(k, len(rs))}
        print(f"control {des:9s} {ctl:18s} N={N:5d}: {k}/{len(rs)} certified", flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
