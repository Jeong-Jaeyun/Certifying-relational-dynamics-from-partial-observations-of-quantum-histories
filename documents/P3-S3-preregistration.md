# P3 S3 preregistration: relational-dynamics rate confirmation (F_RDN)

Written and hashed before the declaration. The declaration of this preregistration freezes S3.
- Config: `config/p3_s3_confirmatory.yaml`.
- Code: `src/bi_qmn/p3/runs/p3_s3_confirmatory.py`.
- Runner: `runners/p3/run_p3_s3_confirmatory.py`.

## 1. Claim (the only confirmatory claim of this S3)

**F_RDN.** Under the declared generative distribution, the RDN certification rate exceeds 0.80 in both the medium and the low stratum.

| element | frozen value |
|---|---|
| unit | construction block |
| block success | screened in, ILGC BI `primal_verified`, **and** RDN `certified` |
| abstention | a screened-out block abstains, receives no certificate, and counts as a **failure** |
| denominator | all 160 evaluation blocks of the stratum: refuted, unresolved, BI-unverified and screened-out blocks all stay in |
| test | one-sided exact binomial, H0: p ≤ 0.80, α = 0.05 per stratum (equivalently, the one-sided 95% Clopper–Pearson lower bound > 0.80) |
| pass line | at least **137 of 160** in each stratum (the exact critical count, checked at load) |
| combination | intersection–union: both strata must pass. The claim level is 0.05. The family p-value is the larger stratum p-value |
| analysis | one final analysis: no interim look, no extension, no re-draw |
| planned power | at p₁ = 0.90: 0.971 per stratum; joint lower bound 0.942 (Bonferroni/Fréchet, no independence assumed) |

A single passing stratum is **not** claimed as its own confirmation, since that would need a separate multiplicity rule. Superiority over Model A is not claimed.

## 2. What is frozen here, and where each element came from

| element | value | source |
|---|---|---|
| statistical protocol | `config/p3_statistical_protocol_v2.yaml` at SHA-256 `d4599175…a2c3` | user decisions of 2026-11-20 and 2026-11-21. Frozen by this declaration; any later change refuses the run |
| design | 160 blocks per stratum, pass count 137, planning rate 0.90 | the predeclared D1-v7 design rule (`documents/P3-d1-v7-run.md`), applied as agreed |
| estimator | shared branch-pair estimator, settings of `config/p3_d1_v7.yaml` | D1-v7 (checked at load) |
| support screen | `support_identifiability_screen_v1` (Pauli commutant and disconnected-branch test, ε = 1e-6) | frozen for S3 (user decision 2026-11-21) |
| bodies | the frozen S3 calibration: medium r_diag 0.019870366862317548, r_coh 0.02871110292651041; low r_diag 0.040296296223098556, r_coh 0.05773687224029173 | `outputs/p3_s3_calibration/p3_s3_calibration.json` (SHA-256 2144a7c4…73fb, `calibration_hash` c4f61ad4…0d7f). Cited, never recomputed. Radii checked against the statistical protocol |
| certificates | BI existence with the constraint registry, 80-digit verifier, CLARABEL proposals; RDN `rdn_v1` with the energy-pinched refutation candidates, refined with the optimal classical state when unresolved | the D1-v7 primary-set route (checked at load). No selector, no atlas |

**Known limitation (user decision 2026-11-21).**
- A residual support non-identifiability exists that the frozen screen does not detect: a continuous support-equivalent family, 0.25% of screened-in low records in D1-v7.
- It is reported as a limitation of the coverage claim.
- Such a record is screened in. Its centre can be wrong while BI and RDN still certify. This is a coverage cost inside the body α.
- Characterizing the mechanism is a separate task and not an S3 input.

## 3. Families

- **This S3 confirms F_RDN only**, at α = 0.05.
- **F_frontier** (critical-budget frontier replication) and **F_structural** (matched structural-gain replication) are **not tested** here. Each will be confirmed later, with its own design frozen before its own data.
- The graphical transitions stay recorded in the statistical protocol: after F_RDN passes, each gets 0.025, with Holm. Whether a later preregistration uses that allocation is decided there.
- **Open items of the statistical protocol, closed for this S3.**
  - Frontier and structural α and power: not applicable, because the families are deferred.
  - Exact bootstrap enumeration versus Tango: not applicable, because F_RDN uses the exact binomial.
  - Group-sequential information fractions: not applicable, because there is one final analysis.

## 4. Sealed evaluation bank

| bank | seeds | blocks | strata |
|---|---|---:|---|
| `p3_s3_evaluation_sealed` | construction 53500, observation 53600 | 160 | medium and low. Each construction block is observed once per stratum, as in D1-v7 and the S3 calibration |

- It uses the D1-v7 generative specification (checked).
- It is the first bank in the reserved confirmatory namespace 53500–53999. The declaration checks that no other declared bank uses that namespace.
- It is disjoint from every declared bank in `config/`, checked by construction fingerprint, including:
  - the S3 calibration bank;
  - all D1-v6 and D1-v7 banks;
  - the T1 banks (and T1's withdrawn banks);
  - the P1 sparse atlas A1 banks;
  - the P1 grid.
- Its observation tables are built and screened only by the run, after the declaration.

## 5. Independence from the sparse atlas (P1 A1)

The two preregistrations share frozen inputs, not data or decisions.

| data role | data | used for |
|---|---|---|
| body calibration | `p3_s3_calibration_v1` (frozen) | the bodies of S3, T1 and A1. Cited by hash; nobody recalibrates it |
| atlas threshold fitting | A1 `D_atlas_cal` (with A1 `D_atlas_ref` for the metric) | A1 only |
| atlas evaluation | A1 `D_atlas_eval` (sealed) | A1 only |
| RDN confirmation | `p3_s3_evaluation_sealed` | S3 only |

- **Disjointness.** All four data sets are mutually disjoint by construction fingerprint, checked by both declarations.
- **No outcome dependence.** The A1 outcome cannot change this claim, its rule or its decision. If the sparse atlas fails, F_RDN is still judged as planned.
- **No atlas evidence here.** S3 does not evaluate atlases. The atlas-citable rate Pr(RDN certified | atlas accepted) is an A1 readout.

## 6. Report-only readouts

For each stratum:
- BI terminal counts, RDN status counts and screened-out counts, with the list of failing blocks and their reasons;
- the κ distribution (r_diag is the frozen radius);
- body coverage of the SBP centre, from the exact scores;
- the unobserved-D coefficient error;
- the oracle screen audit.

**Not computed.**
- `difference_vs_model_a` needs a Model A training bank. D1-v6 characterized it (low unobserved-D coefficient error 0.115 for Model A against 0.0107 for SBP).
- `atlas_citable_rate_rdn_given_atlas` belongs to A1.

These two items of the statistical protocol's report-only list are therefore revised here, as the protocol permits ("may revise").

## 7. Decision and status

- **Status `f_rdn_confirmed`** if both strata reach 137/160. Otherwise **`f_rdn_not_confirmed`**, with the per-stratum decisions.
- **Recorded with the status:** per-stratum successes, the exact lower bound, the critical count and the p-value, plus the family p-value (the maximum).
- **One analysis only.** The artifact is immutable, and the run refuses to start if the artifact exists or if any declared file has drifted.

## 8. Execution

- **Declare.** `python runners/p3/run_p3_s3_confirmatory.py declare`. This freezes S3.
- **Run.** `python runners/p3/run_p3_s3_confirmatory.py run --workers 6`.
  - It opens the sealed bank and performs the final analysis.
  - It uses a process pool of 6 workers with 1 BLAS thread each. A worker error stops the run, so no record abstains silently.
- **Expected cost.** About 320 SBP fits and 320 BI and RDN certificates, roughly 40 minutes on 6 workers.
- **Timing of the run.** The run is a separate step from the declaration. It is started only on the user's go-ahead. It does not depend on the A1 result.
