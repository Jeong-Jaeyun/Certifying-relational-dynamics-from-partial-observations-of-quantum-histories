# E1 preregistration: same-budget recovery by one added D context (sealed confirmation)

## Status

**Declared 2026-11-23 on the user's instruction to complete the remaining experiments.** The declaration record is `outputs/m1_e1_confirmatory/m1_e1_confirmatory_declaration.json`.

- With the declaration, this file, `config/m1_e1_confirmatory.yaml` and the import closure of `src/bi_qmn/p3/runs/m1_e1_confirmatory.py` are frozen.
- The run builds and opens the two sealed banks once.
- The user's decisions of 2026-11-22 are fixed below (§2).
- Development evidence: [M1 run](M1-added-measurement-run.md), [evidence package](M1-evidence-package.md), [power and design proposal](E1-power-and-design-proposal.md).
- **E3 (2026-11-23, [run](M1-E3-rule-comparison-run.md)).** The declared carry-forward rule kept `s_Q`: no challenger had a strictly lower 3a tail. The E3 artifact is a dev input.
- **G guard ([run](G-guard-run.md)).** Not adopted under its declared criterion, so the E1 secondary readouts carry no guarded certification. The primary contrast does not use certificates. The evaluation artifact is a dev input.
- **Observed RDN ([run](observed-RDN-run.md)).** A separate certificate of relational dynamics, and not part of E1. Its artifact is a dev input for the record only.
- **Changes since the draft.** Only this status section and three dev-input hashes were added. The design, endpoints, seeds and code are unchanged.

```text
config:      config/m1_e1_confirmatory.yaml
code:        src/bi_qmn/p3/runs/m1_e1_confirmatory.py, runners/p3/run_m1_e1_confirmatory.py
             (reuses, read only: m1_added_measurement.py, m1_added_measurement_posthoc.py, p3_zero_energy_history.py)
tests:       tests/p3/test_m1_e1_confirmatory.py
seeds:       H1 53700 / 53710 / 53720, H2 53740 / 53750 / 53760, bootstrap 53790 (reserved namespace; S3 used 53500 / 53600)
```

## 1. Claim under test

> Under the declared generative law and a fixed extra budget of 2,048 shots, adding **one D context chosen from the support alone** lowers the upper tail (CVaR_0.9) of the held-out diag recovery error by **more than one frozen low diag radius**, compared with spending the same shots on the support contexts.

**Scope.**
- **Populations.** H1 is the mechanism population: screened-in records of the exact 3a class. H2 is the operational population: flag-conditioned screened-in records as they occur, 3a and 3b mixed, label unknown to the user.
- **Information level.** Level i: the estimator is not told the Hamiltonian.
- **Model.** The design is the P1 low design, with pure histories and readout η = 0.
- **What is not claimed.** Nothing about state recovery, other designs, readout or state noise, or the superiority of the s_Q rule over other rules.

## 2. Decisions fixed by the user (2026-11-22)

1. **Primary contrast.** `Δ = CVaR_0.9(C) − CVaR_0.9(A_s)`, diag error on the common held-out set.
2. **Minimum important effect.** `Δ_min` is one frozen low diag radius, `r_diag = 0.040296…`, read from the frozen S3 calibration (SHA-256 checked).
3. **Two co-primary hypotheses.** H1 (enriched 3a) and H2 (natural flag-conditioned), combined by intersection–union.
4. **Sample sizes.** n = 256 per bank.
5. **Banks.** Two independent sealed banks, with separate seeds and no shared construction.
6. **New arm.** C_F is added as a secondary comparator.

## 3. Banks

**Generative law** (both banks). The D1-v7 law: `seeded_construction_bank_p1` with one of four system unitaries drawn uniformly, Dirichlet(8) branch weights with floor 0.15, uniform phases, and η = 0.
- **Mask.** Construction `i` gets a low mask drawn uniformly from the **110 flagged masks** (non-trivial D-mask commutant `R`). The RNG is `sha256("{mask_seed}:{bank_id}:{i}")`.
- **Support sample.** 2,048 shots per support context (6 D + 2 F), with RNG `sha256("{observation_seed}:{bank_id}:low:{i}")`.
- **Screen.** The frozen screen v1 runs on the sampled support. A screened-out construction is never accepted.

**Acceptance**, in index order:
- **H1** (`m1_e1_h1_sealed`) accepts a screened-in construction when its oracle class is 3a: branch condition at the true branch directions below 1e-10. Quota 256, generation cap 4,096.
- **H2** (`m1_e1_h2_sealed`) accepts every screened-in construction. Quota 256, cap 1,024. The class is recorded for description only.
- **Shortfall.** If a quota is not filled by its cap, the shortfall is reported and the analysis runs on the accepted `n`.
- **Ledger.** It counts constructions generated, screened out (by class and reason), screened in, skipped (H1, not 3a) and accepted.

**Independence.**
- The declaration checks that both banks are disjoint by physical fingerprint from each other, from the P1 construction grid and from every declared bank in `config/`. That includes the M1 development bank and the S3 sealed bank.
- No E1 seed was used before.

## 4. Observations and arms

Every sample is drawn before any fit, so no observation depends on a rule.

| arm | data given to the estimator | role |
|---|---|---|
| A0 | support (8 contexts × 2,048) | baseline |
| **C** | support pooled with 256 extra shots per support context (2,304 each) | **primary comparator** (same extra budget, no new context) |
| C_F | support, with all 2,048 extra shots on the observed F context(s) that anticommute with `R`, split equally and pooled (per-context shot counts in the likelihood) | secondary: the strongest reuse of existing contexts |
| **A_s** | support + `Q_s` with 2,048 shots | **primary rule** |
| A_f | support + `Q_f` with 2,048 shots | fixed-order comparator |
| A_o | support + `Q_o` with 2,048 shots | truth oracle (diagnostic) |
| A0_H | support only, with the registered `H_tot` (zero-energy kernel, `zero_energy_history_v1`) | information level ii (diagnostic) |

**Rules and samples.**
- **Candidates.** The unobserved input D contexts whose Pauli anticommutes with `R` (4 or 5). Each gets its own 2,048-shot sample, with RNG `sha256("{observation_seed}:{bank_id}:added:{i}:{context}")`.
- **`Q_s`.** The argmax of the orbit-invariant `s_Q` at the A0 fit's branch vectors.
- **`Q_f`.** The first candidate in sorted context-id order.
- **`Q_o`.** The argmax of the truth `smin_rel` with the candidate added, fixed before any added-shot fit.
- **Ties.** Within relative 1e-12, ties go to the fixed order.
- **C_F sample.** The RNG is `sha256("{observation_seed}:{bank_id}:control_f:{i}")`.
- **C sample.** As in the M1 protocol.
- **Estimator.** Every SBP arm uses the D1-v7 settings (`shared_branch_pair_v1`, 32 starts, deterministic restarts). C_F uses the same likelihood with per-context counts. Arms that pick the same context share one fit.

## 5. Evaluation and statistics

**Error.** The identity-whitened L2 class error `||q̂ − q_exact||` over the outcomes of a context set. **Diag is primary**; coh and the maximum absolute difference are reported.

**Sets.**
- `E = all 20 contexts − support − {Q_s, Q_f}` scores every arm.
- `E_o = E − {Q_o}` scores the oracle block.
- Each arm's own remaining set is reported.

**Statistic.**
- `CVaR_0.9` is the mean of the `ceil(0.1 n)` largest errors (26 of 256), computed per arm.
- **Paired bootstrap.** 10,000 resamples of record indices, with the same index vector for both arms. The RNG is `default_rng(sha256("53790:{bank}")[:16])`. The interval is the 2.5% and 97.5% percentiles of the resampled difference.

**Decision.**

| | rule |
|---|---|
| **H1** passes | the 2.5% bound of `Δ` in the H1 bank exceeds `Δ_min = r_diag` |
| **H2** passes | the 2.5% bound of `Δ` in the H2 bank exceeds `Δ_min` |
| **E1 confirmed** | H1 and H2 both pass (intersection–union; each one-sided at 0.025, so the family level is 0.025) |

**Statements by outcome.**

| outcome | statement |
|---|---|
| `e1_confirmed` | Under the declared law and budget, one support-chosen D context lowers the held-out tail by more than one body radius, both for exact 3a records and for the flag-conditioned population. |
| `mechanism_confirmed_operational_not_shown` | Confirmed for exact 3a; not shown for the operational population. |
| `operational_shown_mechanism_not_shown` | Reported as is. It is unexpected, because the H2 tail is driven by its 3a records. |
| `not_confirmed` | The sealed banks did not confirm a gain above one radius. The intervals and the "above zero" readout are reported as descriptive. |

A bound above 0 but not above `Δ_min` is reported as "a gain was detected but not at the declared margin".

## 6. Secondary readouts (descriptive, no confirmatory claim)

All are computed with the same bootstrap index vectors.

- **On `E`.** `A0 − A_s`, `C_F − A_s`, `A_f − A_s`, `C − C_F`, `A0 − C`, `A0_H − A_s`. Also the primary contrast on the coh class.
- **On `E_o`.** `A0 − A_o`, `A_s − A_o`, `A_f − A_o`, `C − A_s`.
- **Leave one record out.** The range of `Δ` over single-record deletions, and the most influential record.
- **Arm summaries.** CVaR, median, p90 and maximum, per set and class.
- **By class** (H1 is all 3a; H2 has both):
  - per-arm CVaR;
  - A0 and A_s certificates (BI existence with the registry, `rdn_v1`, frozen S3 low bodies) and truth coverage;
  - κ;
  - rank agreement of `s_Q`;
  - picks below half the best `σ_min`;
  - the support extra null.
- **3a records only.** Fiber coverage of the A0 body along the truth's `R` orbit, and the orbit-energy coefficient `B`.
- **Rank stability and A0_H.** `s_Q` agreement with the truth and the A0 orbit distance; the A0_H registered energy and kernel dimension.
- **Unclassified records.** A 3a record whose support extra null is not 1, or a pick that does not restore rank, is listed as unclassified.

**Development facts these readouts check.** On the development bank, the following were seen post hoc and are now declared readouts: A_s raised 3a certification from 7 to 127 of 128, fiber coverage was 123/128, and B ≥ 2/21.

## 7. Failure handling

- **No fit is excluded.** Every SBP and A0_H fit returns a table.
- **Certificates.** An unresolved certificate counts as not certified.
- **Evaluation sets.** A record with an empty evaluation set cannot occur: there are 20 contexts, 8 support contexts and at most 2 picks.
- **C_F.** A record with no observed F context anticommuting with `R` cannot be screened in, because it would be a type-1 mask. If one occurs, the run stops with an error rather than dropping it.
- **Worker integrity.** Workers rebuild each bank and check its fingerprint hash; a mismatch stops the run.

## 8. Integrity and reproducibility

- **Declaration.** It records the config, this document, the dev-input hashes (M1 run, post-hoc, H-known, evidence package, power), the frozen-calibration hash, both banks' fingerprint hashes, the flagged-mask hash and the import-closure snapshot.
- **Drift.** The run refuses any drift and records the post-run drift.
- **One run.** The artifact is written once; a second run is refused.
- **Execution.** A process pool of at most 6 workers, 1 BLAS thread each. A test requires serial and parallel runs to agree on a test-only bank.
- **Labels.** The exact state is used only for the H1 acceptance, the H2 description, `Q_o`, errors, coverage, the registry, A0_H, fiber coverage, the orbit energy and `σ_min`. No such quantity enters A0, C, C_F, A_s or A_f.

## 9. Cost

- H1: about 1,600 SBP fits and 512 certificates.
- H2: about 1,600 fits and 512 certificates.
- About 4–5 hours with 6 workers.

## 10. Deviations

- Any change after declaration requires a new declaration, with the reason and the time the data were seen.
- Analyses added after the result are labelled exploratory.
