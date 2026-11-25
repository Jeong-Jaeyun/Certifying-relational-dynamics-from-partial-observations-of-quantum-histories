"""Design-wide check: does a disconnected branch ever occur without a D-mask commutant? (M1 evidence package §2.5)

Exhaustive over the low and medium masks and the four registered unitaries, at the true branch basis
(branch weights fixed; disconnection depends on the basis and the mask only). Run from the repository root:
    python runners/p3/check_m1_disconnected_branch_masks.py
Result recorded on 2026-11-23: every disconnected branch lies on a D-commutant low mask (cnot_h0 70, h0 20,
identity 20, cnot_ry 0); none without a D commutant; medium none.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    import numpy as np

    from bi_qmn.p3.runs import m1_d_commutant_check as dc
    from bi_qmn.p3.runs import m1_support_identifiability_map as m1

    ctx = m1.model_context("config", draws=1)
    fw = ctx["fw"]
    counts: dict[tuple, int] = {}
    for deficit in ("low", "medium"):
        for name, U in ctx["unitaries"].items():
            V = np.asarray(U)[:, :3] * np.sqrt(np.array([0.4, 0.35, 0.25]))[None, :]
            for mask in ctx["masks"][deficit]:
                disconnected = fw.pair_observability(V, [c for c in mask if c.startswith("D,")], 1e-9)["mask_class"] == "branch_disconnected"
                key = (deficit, name, "disconnected" if disconnected else "connected", "D_commutant" if dc.d_commutant(tuple(mask)) else "no_D_commutant")
                counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(*key, counts[key])
    bad = sum(v for k, v in counts.items() if k[2] == "disconnected" and k[3] == "no_D_commutant")
    print("disconnected without a D commutant:", bad)


if __name__ == "__main__":
    main()
