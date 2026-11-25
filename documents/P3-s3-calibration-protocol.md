# P3 S3 calibration freeze: protocol

Written and hashed before the declaration. Config: `config/p3_s3_calibration.yaml`. Code: `src/bi_qmn/p3/runs/p3_s3_calibration.py`, runner `runners/p3/run_p3_s3_calibration.py`.

## Purpose

S3 is the P3 confirmatory analysis. Its calibration is frozen **once**, before the sealed evaluation bank is built, and the final analysis is run **once** (formulation patch v3 §10.4). This protocol produces that frozen calibration.

## Decisions it implements (user, 2026-11-21)

- **A fresh calibration draw.** None of the four D1-v7 draws is reused, so the frozen radii were not chosen among draws already seen ([P3-d1-v7-run.md](P3-d1-v7-run.md) §7, Option A).
- **The screen is frozen as it is.** `support_identifiability_screen_v1` (Pauli commutant + disconnected-branch test, ε = 1e-6) is used unchanged. The residual support non-identifiability inside the screened-in population (run report §5, about 0.3% of low records) is a known limitation of the coverage claim. Its characterization is a separate task and is not an input to S3.
- **The design read off the predeclared D1-v7 rule** (`config/p3_statistical_protocol_v2.yaml`, `rdn_primary_claim.selected_design`): per stratum, the first **160 screened-in** blocks in ascending block index.

## Fixed in advance

| item | value |
|---|---|
| estimator | `shared_branch_pair_v1`, settings identical to `config/p3_d1_v7.yaml` (checked at load) |
| screen | `support_identifiability_screen_v1`, identical to D1-v7 (checked at load) |
| bank | `p3_s3_calibration_v1`: seed 54126, observation seed 54226, 192 blocks, Dirichlet 8.0, minimum mode probability 0.15, phase range [0, 2π) |
| strata | medium and low (deficits), as in D1-v7 and the S3 claim |
| selection | per stratum, the first 160 blocks that pass the screen, scanning blocks in ascending index; only those 320 records are fitted |
| radius | split-conformal per context class (diag, coh) with the ILGC alpha of `config/p1_ilgc_operational_r4.yaml` (0.04 each: order index 155 of 160), identity whitening, score = ‖q_exact − q̂‖₂ over the class |
| anti-vacuity | normalized radius r/√d_c must not exceed the stratum ceiling of the same operational config |

**Bank choice.**
- The generative specification equals the D1-v7 banks (checked at load), so calibration and evaluation blocks are exchangeable.
- The seeds lie outside the reserved confirmatory namespace 53500–53999, which stays sealed for the S3 evaluation bank.
- The declaration checks construction-level disjointness against the P1 construction grid and every declared bank in `config/`.
- 192 blocks leave a wide margin: at the D1-v7 low exclusion rate (1.09%), the chance that fewer than 160 of 192 pass is negligible.

## Outcome rule (fixed in advance)

- **Frozen** if both strata have 160 screened-in blocks and every class radius is within its anti-vacuity ceiling. The artifact then records the four radii and a `calibration_hash` over the bank, estimator, screen, alpha, exact radii and selected blocks. S3 cites the artifact SHA-256 and this hash.
- **Not frozen** otherwise: the bank is **not** extended and no radius is cited. The user decides how to proceed.
- Nothing else is decided here. The radii are not compared with any target, and no result can change the S3 design (n = 160 per stratum, pass count 137).

## Report-only readouts

- Radius-determining block, the five largest scores per class, normalized radii.
- Screen against the oracle class (commutant, or a disconnected branch in the true basis) on all 384 records.
- Selected records with diag score above 0.15, the threshold of the D1-v7 wide-tail diagnostic.
- Unobserved-D coefficient error, fit adequacy (fitted NLL minus NLL at the true parameters), near-optimal start counts.
- For context only: in D1-v7, r_diag at 160 blocks was 0.037–0.041 (low) and 0.019–0.021 (medium).

## What is not done

- No evaluation bank is built or opened, and no certificate (BI, RDN) is computed.
- The reserved namespace is not touched.
- Exact tables are used only for the calibration scores, as in every ILGC body calibration, and for the report-only oracle audit.

## Execution

- Declare: `python runners/p3/run_p3_s3_calibration.py declare`. The declaration snapshots the import closure, the config files (including `p3_d1_v7.yaml` and `p3_statistical_protocol_v2.yaml`), this document and the runner.
- Run: `python runners/p3/run_p3_s3_calibration.py run --workers 6`. This uses a process pool of 6 workers with 1 BLAS thread each. The run refuses to start on any drift, and the artifact is immutable.
