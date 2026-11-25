# E2: does the exact orbit and its repair extend beyond the registered bases? (development protocol)

## Status

**Development protocol, declared before generation.** The declaration freezes this file, `config/e2_scope_extension.yaml` and the import closure of `src/bi_qmn/p3/runs/e2_e4_extensions.py`.

**Question.** Propositions 1–4 of the manuscript (D-commutant invariance, exact circle, invariance of relational dynamics, separation bound) hold for any system basis. The finite-shot evidence, however, comes from the registered law, where exact circles sit mostly on one of four unitaries (h0). E2 asks whether three things carry over to bases outside that set:
- the support-only failure;
- the same-budget gain of one support-chosen context;
- the re-certification.

E2 is the first of the scope extensions in the remaining-experiments plan (§E2): a new basis that satisfies the 3a condition, perturbations around it, and generic rotations.

## 1. Generation (seeds 56801 / 56802 / 56803, outside the reserved namespace)

Weights follow Dirichlet(8) with floor 0.15, phases are uniform, and the active modes are {0, 1, 3}, as in the D1-v7 law. Each construction draws one of the 110 flagged low masks uniformly. `R` is its D commutant.

| family | system basis | acceptance |
|---|---|---|
| `constructed_3a` | Haar `U_0`, projected by least squares on U(4) onto `⟨u_j|P|u_j⟩ = ⟨u_j|iPR|u_j⟩ = 0` for every observed F Pauli `P` that anticommutes with `R` (residual < 1e-12; up to 8 restarts) | screened in and exact class 3a; quota 128, cap 1,024. Masks with no anticommuting observed F (type 1) are skipped and counted |
| `perturbed_ε`, ε = 0.01, 0.03, 0.1 | `exp(iεG) U` for the first 64 constructed bases; `G` Hermitian, unit Frobenius norm; same weights, phases and mask | all 64 kept; screen and class recorded |
| `haar` | unconstrained Haar `U` | screened in; quota 128, cap 1,024; classes counted |

**Classes.** A record is **3a** if `max |⟨P⟩_j| < 1e-10` and `max |⟨iPR⟩_j| < 1e-10`. It is **3b** if only the second holds, and **regular** otherwise.

**Recorded per record.**
- The support's largest change along the truth's `R` circle (16 points).
- The distance of the branch basis from the four registered unitaries.

**Labels.** The estimator and the selection rule never receive `U`, the class or `R`. `R` is a function of the observed mask, as in the M1 run.

## 2. Arms (as in the M1 development run)

- **Arms.** `A0` uses the support (8 contexts × 2,048 shots). `C` adds 256 extra shots per support context. `A_s` adds the max-`s_Q` candidate at the A0 fit, with 2,048 shots. `A_f` adds the first candidate in sorted order.
- **Samples.** All samples are drawn before any fit. Every candidate has its own sample.
- **Estimator.** `shared_branch_pair_v1` with the D1-v7 settings.
- **Error.** The L2 diag error on `E = all − support − {Q_s, Q_f}`.
- **Coverage.** Joint truth coverage of each arm's table by the frozen S3 low bodies.
- **Certificates** (constructed_3a only). BI existence with the record's own registry and `rdn_v1`, for the A0 and A_s tables.
- **Fiber coverage.** For 3a records, the A0 body's fiber coverage along the truth's circle.

## 3. Endpoints (development; descriptive intervals)

- **X1, mechanism.** In `constructed_3a`, the support is invariant along the circle (deviation below 1e-10), and spending the extra shots on the support does not lower the tail. The `A0 − C` interval contains zero or lies below it.
- **X2, same budget.** `CVaR_0.9(C) − CVaR_0.9(A_s)` in `constructed_3a`, with a paired bootstrap (10,000 resamples, seed 56804). *Reproduced* means the 2.5% bound exceeds the frozen low diag radius (0.0403), the E1 margin.
- **X3, near exact.** For each ε: per-arm CVaR, the contrasts and the classes. Expectation: the support begins to resolve the circle as ε grows, so the A0 tail falls.
- **X4, generic.** In `haar`: the number of exact 3a records (expected 0), per-arm CVaR and the contrasts.
- **X5, re-certification.** In `constructed_3a`: A0 and A_s success counts (BI `primal_verified` ∧ RDN `certified`) and fiber coverage.

## 4. Use of the result

- **Reproduced.** If X1 and X2 reproduce, the manuscript's finite-shot statements extend from the registered bases to "constructed 3a bases of the same mask design" as development evidence.
- **Not reproduced.** If they do not, the manuscript limits them to the registered law and reports E2.

**Not covered.** Other mode sets, clock sizes, measurement sets and readout models are outside E2 and remain stated limitations.

## 5. Integrity

- **Freezing and drift.** As in the other declared development runs: the declaration snapshot, drift refusal, and post-run drift recorded.
- **Execution.** At most 6 workers with 1 BLAS thread each. Jobs carry each record's construction data.
- **Disjointness.** E2 constructions use new system bases, so they cannot coincide with any registered-law bank. The declaration records the check of the registered-law banks for completeness.
