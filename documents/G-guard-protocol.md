# G guard protocol: a support goodness-of-fit abstention guard for the certification route (development)

## Status

**Development protocol, frozen by its declaration.** The user decided on 2026-11-23 to design this guard after the E5 pilot, and to finish its independent evaluation before drafting resumes.

**Two declared stages on two new development banks.**
1. Calibrate the threshold on pure histories.
2. Evaluate the frozen guard on independent controlled truths.

The evaluation reads the calibration artifact produced by the declared code. Its SHA-256 is recorded, and the artifact is never recomputed. Nothing sealed is opened, and the frozen S3 bodies are read only.

```text
config:   config/g_guard.yaml
code:     src/bi_qmn/p3/runs/g_guard.py, runners/p3/run_g_guard.py (actions: declare, calibrate, evaluate)
tests:    tests/p3/test_g_guard.py
inputs:   E5 pilot artifact, S3 calibration artifact (SHA-256 in the config)
```

## 1. Why

In the E5 pilot, decohered truths (dephasing ν = 0.5) were RDN-certified correctly, but no body covered the true table (0/96). The certificate's statement about the truth was true but not licensed by the calibration, because the calibration assumes the pure-history model.

The support misfit `G` separated every decohered record from every pure one. A guard that abstains when `G` is too large should therefore restore this property: **when the route certifies, its calibrated justification holds**. The cost is some false abstention on pure histories.

## 2. Statistic and rule

**Statistic.**
- `G = 2 Σ n p̂ log(p̂ / q̂)`, summed over the outcomes of the observed (support) contexts.
- `p̂` is the observed frequency, `n` the per-context shot count, and `q̂` the SBP-fitted table in the estimator's output convention.

**Rule.** The guard **abstains** when `G > g_α(stratum)`. The threshold is a split-conformal order statistic on pure calibration records:
- `k = ceil((n + 1)(1 − α))`, with `α = 0.05` and `n = 320` per stratum, so `k = 305`;
- `g_α` is the k-th smallest calibration `G`.

**Guarded route.** The record must be screened in, the guard must pass, then BI existence and RDN run as in S3. An abstention counts as no claim.

## 3. Stage 1: calibration

**Bank.** `g_guard_calibration_v1`, with seeds 56301 / 56302. It is a new bank under the D1-v7 law, checked to be disjoint from every declared bank.

**Records.**
- Constructions are taken in index order, and each is observed once per stratum (medium, low) with the S3 mask rule.
- The truth is the pure history.
- The first 320 screened-in records of each stratum are kept, from a cap of 340 constructions. A shortfall is reported, and `k` is recomputed from the `n` actually kept.
- One SBP fit per record. No certificate.

**Output.** `g_α(medium)` and `g_α(low)`, and the quantiles of `G`.

## 4. Stage 2: independent evaluation

**Bank.** `g_guard_evaluation_v1`, with seeds 56311 / 56312 / Haar 56313. It is new and disjoint from the calibration bank and from every declared bank.

**Truth families.** 64 constructions × 2 strata = 128 records per family. Every family uses the same construction, the same mask and the same shot random numbers.

| family | truth | in the model class? |
|---|---|---|
| pure | the PW history | yes |
| dephased ν = 0.1, 0.25, 0.5 | `(1−ν)|Ψ><Ψ| + ν E(|Ψ><Ψ|)` | no (mixed) |
| readout η = 0.02, 0.05 | pure history, observed through symmetric clock-label and system-bit flips of rate η; the estimator assumes η = 0 | no (readout) |
| static | `E(|Ψ><Ψ|)` | no (τ-flat) |
| registry violating | `(I ⊗ W)|Ψ>`, `W` Haar random | yes for the estimator (it does not know H); no for the registry |

**Per record.**
- screen, fit, `G`, and the guard decision;
- the distance-stage κ of the fit (what the unguarded route would certify at stage 1);
- coverage of the **ideal** true table (the state's Born table, η = 0) and of the effective table;
- the full S3 certificates (BI existence and `rdn_v1`) for every record that is screened in and passes the guard.

## 5. Endpoints (development; declared before the evaluation bank is built)

| | statistic | reading |
|---|---|---|
| **G1** false abstention | guard abstentions among screened-in pure records, with a Clopper–Pearson 95% interval | consistent with the calibration if the lower bound is at most α = 0.05 |
| **G2** detection | abstentions among screened-in records, per misspecified family | reported with intervals |
| **G3** validity | among records the guarded route RDN-certifies, the fraction whose bodies cover the ideal truth jointly | per family with at least 20 such records; target 0.90 (two class bodies at α = 0.04 each) |
| **G4** static | certified static records | expected 0 |
| **G5** registry | BI refutations of registry-violating records | reported; `G` is not expected to detect these |
| **G6** cost | guarded S3 success on pure records (screened in, guard pass, BI primal, RDN certified) | reported against the unguarded rate |

**Readings.**
- **Guard adopted for the paper.** G1 is consistent and G3 meets its target in every evaluable misspecified family. The paper then states the guarded route's validity under the tested misspecifications.
- **G3 fails for a family.** That family is reported as a limit of the guard.
- **G1 is inconsistent.** The calibration is reported as failed, and the guard is not adopted.

## 6. Integrity

- **Declaration.** It records the config, this document, the dev-input hashes, both banks' fingerprint hashes and an import-closure snapshot. Both stages refuse on drift.
- **Calibration artifact.** Written once and never overwritten. The evaluation refuses to start without it and records its SHA-256.
- **Execution.** A process pool of at most 6 workers, with 1 BLAS thread each. A test requires serial and parallel runs to agree on test-only banks.
- **Use of the truth.** The exact state defines the families and is used for coverage, true κ and the energy-pinched exact-state candidate (as in S3, which can only refute). It never enters the screen, the fit, `G` or the guard.

## 7. Cost

- Calibration: about 680 SBP fits, about 20 minutes on 6 workers.
- Evaluation: about 1,024 fits and about 500 certificates (those that pass the screen and the guard), about 1.5–2 hours on 6 workers.
