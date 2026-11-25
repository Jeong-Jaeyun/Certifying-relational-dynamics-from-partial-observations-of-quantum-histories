# Design 2 pilot: one added context at finite shots in a second clock and observation design

## Status

**Exploratory pilot (expl.), designed in advance; not a declared run.** This document is written before any data of the pilot exist.
- The runner records its SHA-256 and the source digests in the artifact.
- The pilot estimates effect sizes and tail variances. A confirmatory run with pre-fixed contrasts, margins and sample sizes on new seeds (reserved block 57200–57299) needs the user's decision and is not part of this pilot.

**Question.**
- Where does "dynamics visible, history not identified" occur outside the registered design?
- When does one added measurement resolve it at finite shots?
- How much of a finite-shot failure is an information limit?

The deterministic parts (exact counterexamples, controls, structure, two-point bound) are in [design2-analysis-note.md](design2-analysis-note.md).

## 1. Design (fixed)

| item | value |
|---|---|
| clock | `K = 16`, `omega_r = r`, time basis of main-text Eq. (1) |
| system | two qubits |
| active modes | `golomb` {0, 1, 4, 6} (all pair differences distinct); `collision` {0, 1, 2, 3} (bins 1, 2, 3 hold 3, 2, 1 pairs) |
| observed contexts | `D = {IX, XI, XX, YI, YX, ZI}`, `F = {XZ, XY}`; shared `D` commutant `R = IX` |
| candidates | unobserved `D` Paulis anticommuting with `R`: IY, IZ, XY, XZ, YY, YZ, ZY, ZZ |
| estimator | level-i pure-history MLE (`shared_branch_pair_v1`, 32 starts), effects of this design |

## 2. Truth families (per design, 48 records each; seeds 57101 constructions, 57103 perturbations)

| family | construction | role |
|---|---|---|
| `exact` | Haar `U_0` projected on `<u_j|P|u_j> = <u_j|iPR|u_j> = 0` for `P` in {XZ, XY} (residual < 1e-12); weights Dirichlet(8), rejected below 0.1; phases uniform | primary |
| `near_0.03`, `near_0.1` | `exp(i eps h) U` of the `exact` bases, `h` GUE with unit Frobenius norm; same weights and phases | primary |
| `generic` | Haar `U`; weights and phases as above | control: where the gain shrinks |
| `structured_equal` | `U = exp(i alpha X) (x) I`, `alpha` uniform in (0.15, pi/2 - 0.15); weights 1/4; phases uniform | control: D:IZ has rank 1 |
| `structured_random` | the same bases with Dirichlet weights | D:IZ separates only through the population imbalance |

**Pre-pilot correction (2026-11-24).** A one-record smoke test (no analysis) showed that the first version of the control, the `exact` bases with weights 1/4, does not reproduce the rank-1 failure of D:IZ. That failure requires the branch-basis matrix of IZ to be diagonal, which holds for the structured bases and not for projected Haar bases. The control was therefore replaced by the two `structured` families before any pilot data were generated.

## 3. Shots and arms (seed 57102 for counts)

Shot levels `N` = 512, 2048, 8192 per support context, with an independent sample at each level. All arms at a level share the support sample.

| arm | measurement |
|---|---|
| `A0` | support only |
| `C` | support plus `N` extra shots, `N/8` per support context (own extra sample, pooled) |
| `A_s` | support plus `N` shots on the candidate with the largest `s_Q` at the `A0` fit |
| `A_sig` | support plus `N` shots on the candidate with the largest `sigma_min(M_Q)` at the `A0` fit (actual tables) |
| `A_IZ` | support plus `N` shots on `D:IZ` |
| `A_o` | support plus `N` shots on the candidate with the largest true `sigma_min(M_Q)` (diagnostic) |

- **Candidate samples.** Every candidate has its own sample at each level, drawn before any fit. Arms that choose the same candidate share it.
- **Evaluation set.** Per record and level, `O` is the set of unobserved `D` contexts minus `D:IZ` and minus every arm's choice; it is common to all arms.
- **Error.** The L2 error of the fitted D tables on `O`.

## 4. Readouts (descriptive)

- **Error summaries.** Per design, family, level and arm: mean, median, maximum and `CVaR_0.9`. Paired-bootstrap intervals (10,000 resamples, seed 57105) for these contrasts:
  - `CVaR(C) - CVaR(A_s)`
  - `CVaR(C) - CVaR(A_sig)`
  - `CVaR(C) - CVaR(A_IZ)`
  - `CVaR(A0) - CVaR(C)`
  - `CVaR(A_s) - CVaR(A_sig)`
- **Rules.**
  - Agreement of `s_Q` and `sigma_min` choices.
  - True `sigma_min` of each rule's pick.
  - Picks with true rank 1.
- **Two-point bound.** For `A0` and for `A_sig`, computed on the record's own `O` at its level. This allows the comparison: error near the bound means an information limit; error far above it means an estimator limit.
- **Conditional test.** Certification at alpha = 0.01 on the support D counts (L = 10,000 draws, seed 57104) for every truth record. Static controls per design and level, 48 records each:
  - the pinched `exact` history (uniform clock marginal);
  - `sigma_C (x) rho_bar` with time-diagonal clock weights `(1 + 0.5 cos(2 pi tau/K + phi))/K`, where `rho_bar` is the constant conditional state of the pinched history.

## 5. Use

- The pilot fixes the primary contrast, the margin and the sample size of a confirmatory run.
- Expected primary contrast: `CVaR(C) - CVaR(A_sig)` or `CVaR(C) - CVaR(A_s)`, in `exact` and `near` families at 2048 shots.
- The margin cannot reuse the registered design's body radius. It will be set from the pilot's noise scale and the two-point bound.

## 6. Integrity

- **Execution.** A process pool of at most 6 workers, 1 BLAS thread each.
- **Determinism.** Records and samples are deterministic functions of seeds and labels (SHA-256). Exact states are used only for generation, labels, the oracle arm, errors, bounds and the static controls; no exact quantity enters `A0`, `C`, `A_s`, `A_sig` or `A_IZ`.
- **Isolation.** No sealed or reserved bank is opened.
