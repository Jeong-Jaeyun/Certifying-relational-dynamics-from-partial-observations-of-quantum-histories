# E3 protocol: support-only rules for the single added D context (development)

## Status

**Development protocol, frozen by its declaration.** The user approved it on 2026-11-23 (plan §4 E3), with no reductions.

- **New bank.** The bank is new and is built only by the run.
- **Why a new bank.** The `σ_min` rule was found post hoc on the M1 development bank (record 771), so comparing rules on that bank would be biased.
- **What the result is.** A development finding. A claim that one rule is better than another would need its own sealed comparison.
- **What it decides.** The declared carry-forward rule (§5) decides which rule the E1 preregistration (not yet declared) uses.

```text
config:   config/m1_e3_rules.yaml
code:     src/bi_qmn/p3/runs/m1_e3_rules.py, runners/p3/run_m1_e3_rules.py
tests:    tests/p3/test_m1_e3_rules.py
inputs:   M1 development run and its post-hoc analyses (SHA-256 in the config)
```

## 1. Question

Several support-only rules can choose the one added D context. Which of them keeps the large 3a recovery errors smallest? In particular, does a rule built on the worst-direction separation `σ_min` (Proposition 4 of the draft) do better than the orbit-invariant sensitivity `s_Q`?

## 2. Bank and samples

**Bank** (as in the M1 development protocol). Generation follows the D1-v7 law, in index order:
- **Mask.** Each construction gets a low mask drawn uniformly from the 110 flagged masks.
- **Support.** 2,048 shots per support context.
- **Screen.** Screen v1 runs on the support.
- **Acceptance.** Records are accepted by oracle class until there are **512 type-3a and 128 type-3b** records. The generation cap is 8,192. A shortfall is reported, and the analysis runs on the accepted records.
- **Seeds.** 56201 / 56202 / 56203. They lie outside the reserved namespace, and the bank is checked to be disjoint from every declared bank.

**Samples.**
- **Candidates.** Every candidate (each unobserved input D context that anticommutes with `R`; 4 or 5 per record) gets its own 2,048-shot sample. All samples are drawn before any fit.
- **Fits.** The support alone (A0), plus **the support with each candidate added**. Every rule's choice is therefore fitted, on the same records.

## 3. Rules

| rule | choice | information |
|---|---|---|
| `s_Q` (reference) | largest orbit-invariant sensitivity at the A0 fit | support |
| `σ_min` | largest `σ_min(M_Q)` of the orbit coefficient matrix at the A0 fit | support |
| `σ_min^χ` | largest `σ_min(W M_Q)`, where `W = diag(q̂_Q)^{-1/2}` at the A0 fit: separation in shot-noise units (multinomial Fisher metric) | support |
| local | largest local tangent `2‖b‖` at the fitted point; not orbit-invariant | support |
| fixed | first candidate in sorted context-id order | mask |
| random | a uniformly random candidate, seeded per record (`sha256("56205:{bank_id}:{i}")`) | none |
| oracle | largest truth `smin_rel` with the candidate added | truth (diagnostic) |
| hindsight | the candidate with the smallest error on `E_all` | truth (unattainable reference) |

Ties within relative 1e-12 go to the fixed order.

## 4. Evaluation

**Error.** The identity-whitened L2 class error. **Diag is primary**; coh and the maximum absolute difference are reported.

**Sets.**
- **Pairwise set** (primary). For each declared comparison of rules a and b: `E_ab = all − support − {pick_a, pick_b}`. This is the same construction as the M1 development run.
- **E_all.** All contexts minus the support minus every candidate. It is common to all rules and is used for the all-rule table and for `hindsight`.
- **Own set.** All unobserved contexts minus the rule's own pick. It is used for large-failure counts.

**Statistic.** `CVaR_0.9` per rule (the mean of the 52 largest of 512), with paired bootstrap differences: 10,000 resamples of record indices, seed `sha256("56204:{stratum}")`, interval at 2.5% and 97.5%.

**Declared comparisons.** These are on 3a and the pairwise sets. A positive difference means the second rule has the lower tail.

| | difference |
|---|---|
| C1 | `CVaR(fixed) − CVaR(s_Q)` (replicates P2) |
| C2 | `CVaR(σ_min) − CVaR(s_Q)` |
| C3 | `CVaR(σ_min^χ) − CVaR(s_Q)` |
| C4 | `CVaR(random) − CVaR(s_Q)` |
| C5 | `CVaR(local) − CVaR(s_Q)` |
| C6 | `CVaR(s_Q) − CVaR(oracle)` (gap to the oracle) |
| C7 | `CVaR(σ_min^χ) − CVaR(σ_min)` |

**Reported, not judged.**
- the all-rule table on `E_all` (CVaR, median, p90, maximum; diag and coh);
- large failures (own-set diag error > 0.1) by rule;
- **near-degenerate picks** (`2σ_min` at the truth below the shot-noise scale) by rule, and how many of them failed;
- agreement of each rule's fitted criterion with its value at the truth;
- tie rates;
- the same tables for 3b.

**Report-only: a 3a detector from the support alone.** At the A0 fit, take `max_{P,j} sqrt(<P>_j² + <iPR>_j²)` over the observed F Paulis that anticommute with `R`. Its AUC for 3a against 3b is reported on this enriched bank. The bank is enriched, so this AUC is not a natural-population precision.

## 5. Carry-forward rule for E1 (declared)

- **Default.** `s_Q` stays the E1 primary rule.
- **Replacement.** A challenger (`σ_min` or `σ_min^χ`) replaces it only if `CVaR(s_Q) − CVaR(challenger)` on its pairwise set has a 95% interval entirely above 0; that is, the challenger has a strictly lower 3a tail.
- **Two challengers qualify.** The one with the larger point difference is used.
- **Wording.** A replacement is a design decision for the undeclared E1, not a confirmed superiority. Superiority would need its own sealed comparison.

## 6. Integrity

- **Declaration.** It records the config, this document, the dev-input hashes and an import-closure snapshot. The run refuses on drift.
- **Execution.** A process pool of at most 6 workers, with 1 BLAS thread each. Workers rebuild the bank and check its fingerprint hash. A test requires serial and parallel runs to agree on a test-only bank.
- **Use of the truth.** The exact state is used for the class label, the oracle, hindsight, errors, near-degeneracy and criterion agreement. It never enters a support-only rule.

## 7. Cost

- About 640 × 5.5 ≈ 3,500 SBP fits, plus 640 truth Jacobians.
- About 1.6–1.8 hours on 6 workers.
- No certificates.
