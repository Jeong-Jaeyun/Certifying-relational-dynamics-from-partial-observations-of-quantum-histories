# Design 2: sealed confirmatory preregistration

## Status

**Preregistration of a sealed confirmatory run.** The user decided on 2026-11-24:
- the primary rule: `sigma_min` at the support-only fit;
- `Delta_min = 0.1`;
- the rank-1 control as confirmatory endpoints;
- n = 96 per cell.

The declaration freezes:
- this document;
- `config/design2_confirmatory.yaml`;
- the import closure of `src/bi_qmn/p3/runs/design2_confirm.py`, which includes `design2_core.py`, `design2_pilot.py`, the estimator and the conditional test;
- the runner.

The bank is generated from seeds in the reserved block 57200–57299 after the declaration, and analysed once. Basis: the [pilot](design2-pilot-run.md) and the [analysis note](design2-analysis-note.md).

## 1. Claim

In a second finite Page–Wootters design, with either mode set (`golomb` {0, 1, 4, 6}, `collision` {0, 1, 2, 3}), the following hold at 2048 shots per context:
- **(a) Gain.** One context chosen from the support alone lowers the worst-decile error on unobserved tables relative to spending the same shots on the observed contexts. This holds by more than 0.1 for exact and nearly exact (`eps = 0.03`) circles.
- **(b) No gain.** An added context that anticommutes with the commutant but has rank 1 on the circle gives no gain larger than 0.05.
- **(c) Validity and sensitivity.** The conditional test keeps its validity and sensitivity in this design.

The design: `K = 16`, two qubits, observed `D = {IX, XI, XX, YI, YX, ZI}`, `F = {XZ, XY}`, `R = IX`.

## 2. Bank (seeds 57201 construction, 57202 observation, 57203 perturbation, 57204 Monte Carlo, 57205 bootstrap)

| family | construction | n per design | role |
|---|---|---|---|
| `exact` | Haar `U_0` projected on Eq. (5) for `P` in {XZ, XY} (residual < 1e-12); weights Dirichlet(8), rejected below 0.1; phases uniform | 96 | primary |
| `near_0.03` | `exp(i 0.03 h) U` of the `exact` bases (`h` GUE, unit Frobenius norm) | 96 | primary |
| `structured_equal` | `exp(i alpha X) (x) I`, `alpha` uniform in (0.15, pi/2 − 0.15); weights 1/4 | 96 | primary (rank-1 control) |
| `generic` | Haar `U` | 96 | descriptive |
| `structured_random` | structured bases with Dirichlet weights | 96 | descriptive |
| static controls | pinched `exact` history (uniform clock marginal); `sigma_C (x) rho_bar` with time-diagonal clock weights `(1 + 0.5 cos(2 pi tau/K + phi))/K` | 512 per control and design | validity |

**Constructions** are those of the pilot's code with these seeds: record `i` uses the same `i` in both designs.

## 3. Arms (N = 2048; every arm shares the support sample)

| arm | measurement |
|---|---|
| `A0` | support only |
| `C` | support plus 2048 extra shots, 256 per support context (own sample, pooled) |
| `A_sig` (primary) | support plus 2048 shots on the candidate with the largest `sigma_min(M_Q)` at the `A0` fit |
| `A_s` (secondary) | the same with the largest `s_Q` |
| `A_IZ` | support plus 2048 shots on `D:IZ` |
| `A_o` | the candidate with the largest true `sigma_min` (diagnostic) |

- **Estimator.** `shared_branch_pair_v1`, 32 starts, effects of this design.
- **Error.** The L2 error of the fitted D tables on `O`. `O` is the set of unobserved D contexts minus `D:IZ` and minus every arm's choice; it is common to all arms of a record.

## 4. Endpoints (intersection–union; the claim requires all eight)

Bootstrap: 10,000 paired resamples of records, seed text `57205:<endpoint>`.

| id | cell | statistic | pass |
|---|---|---|---|
| G1 | golomb, exact | `CVaR_0.9(C) − CVaR_0.9(A_sig)` | 2.5% bound > 0.1 |
| G2 | golomb, near 0.03 | same | same |
| G3 | collision, exact | same | same |
| G4 | collision, near 0.03 | same | same |
| Z1 | golomb, structured equal | `CVaR_0.9(A0) − CVaR_0.9(A_IZ)` | 97.5% bound < 0.05 |
| Z2 | collision, structured equal | same | same |
| V1 | static controls, both designs and both controls (2048 records) | certification at alpha = 0.01 (L = 10,000) | exact one-sided binomial test of `p ≥ 0.02` rejected at 0.05 |
| V2 | `exact` and `near_0.03` records, both designs (384 records) | certification at alpha = 0.01 | exact one-sided binomial test of `p ≤ 0.97` rejected at 0.05 |

- `CVaR_0.9` is the mean of the `ceil(0.1 n)` largest errors.
- **Power** (pilot resampling): ≈ 1 for G and Z when no record loses the gain.
  - A CVaR_0.9 contrast fails once about 10% of the records lose the gain (power 0.15–0.45 at 10% in the pilot simulation). The endpoint therefore asserts the tail.
  - V1: at the pilot rate (0.5%), the probability that the binomial test rejects is high. V2: the pilot detected 100%.

## 5. Declared descriptive readouts (no confirmatory claim)

- `A_s` against `A_sig` (contrast, agreement, true `sigma_min` of the picks).
- `CVaR(A0) − CVaR(C)`.
- The generic and structured-random families: all arms, and the dose–response of `D:IZ`.
- Per-record two-point bounds for `A0`, `C` and `A_sig`.
- Certification rates by family.

## 6. Decision table

| outcome | reading |
|---|---|
| all 8 pass | sealed confirmation (S) of (a)–(c) in the second design |
| a G endpoint fails | the gain is not confirmed in that cell; G intervals and failing records are reported |
| a Z endpoint fails | an anticommuting rank-1 context showed a gain above 0.05; reported, with its records |
| V1 or V2 fails | the conditional test's validity or sensitivity is not confirmed in this design |

## 7. Integrity

- **Declaration.** It records the SHA-256 of this document and the config, an import-closure snapshot and the seeds. It refuses if a declaration or artifact already exists.
- **Run.** It refuses on drift and records the post-run drift.
- **Execution.** At most 6 workers with 1 BLAS thread each.
- **Use of exact states.** Only for generation, the oracle arm, errors, bounds and the static controls. No exact quantity enters `A0`, `C`, `A_sig`, `A_s` or `A_IZ`.
- **Seeds.** No seed of this block has been used before.
