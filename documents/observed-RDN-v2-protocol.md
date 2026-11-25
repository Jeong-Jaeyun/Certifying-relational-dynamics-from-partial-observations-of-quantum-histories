# Observed RDN v2: both-margin conditional test (development protocol)

## Status

**Development protocol, declared before its bank is built.** The declaration freezes this file, `config/observed_rdn_v2.yaml` and the import closure of `src/bi_qmn/p3/runs/observed_rdn_v2.py`.

**Why a v2.** The manuscript's Proposition 1 conditions on both margins of every observed D table. The v1 development run ([run](observed-RDN-run.md)) conditioned only on the system totals and assumed a uniform clock marginal.
- That null holds for every static history in the registered design, because energy-pinched states have a uniform clock marginal in the time basis.
- It fails for a static conditional state seen through a non-uniform clock marginal or an asymmetric clock readout. The table is then `p(τ) p(s)`, which is not τ-flat.
- v2 removes that nuisance parameter. This run measures what that costs in sensitivity and checks the guarantee where v1 breaks.

## 1. The test

For each observed D context `P`, the counts `N(τ, s)` form a K × 2 table with row totals `r_τ` and column totals `c_s`.

- **Null.** Clock reading and system outcome are conditionally independent in every observed D context. Under trusted product measurements this holds whenever the conditional state does not depend on τ, for any clock marginal and any product readout channel.
- **Conditional law.** Given both margins, `N(·, 1)` is multivariate hypergeometric with category sizes `r_τ` and draw size `c_1`, independently across contexts.
- **Statistic.** `G = 2 Σ N log(N n / (r_τ c_s))` over positive cells, summed over observed D contexts. A table with one nonzero row or column has a point-mass conditional law and contributes 0.
- **p-value.** `(1 + #{G_b ≥ G_obs}) / (B + 1)` with `B = 10,000` draws. Ties count in the upper tail.
- **Decision.** Certify at `α = 0.01`; `0.05` is reported.
- **Also reported.** The χ² approximation (df = Σ (rows − 1)(cols − 1) over nonempty margins). The v1 test is computed on the same counts with an independent Monte Carlo stream.

## 2. Fresh development bank

**Bank.** `observed_rdn_v2_dev`, seeds 56601 / 56602 / 56603 and Monte Carlo seed 56604, outside the reserved confirmatory namespace.
- 512 constructions under the D1-v7 generative law.
- Each construction is observed in medium and low with the S3 mask rule.
- Every family has its own multinomial sample. Support counts are drawn per context, from the family's effective tables.
- The declaration checks the bank against every declared bank by physical fingerprint.

**Families** (13 per construction and stratum; 13,312 records).

| family | truth | role |
|---|---|---|
| `tau_flat` | energy-pinched history; uniform clock marginal | null |
| `tau_flat_readout_0.05` | static, symmetric readout η = 0.05 | null |
| `nonuniform_clock_static_0.5` | `σ_C ⊗ ρ̄`: time-diagonal clock with weights `(1 + 0.5 cos(2πτ/K + φ_i))/K`; ρ̄ the static conditional state | null for v2, not for v1 |
| `tau_flat_clock_skew_0.1` | static; with probability 0.1 the time-basis reading is redrawn from `w_ℓ ∝ 1 + cos(2πℓ/K)` (product readout) | null for v2, not for v1 |
| `tau_flat_xor_0.05` | static; with probability 0.05 the system bit is XORed with the parity of the reported reading | outside the product-readout assumption |
| `pure`, `dephased_ν` (ν = 0.5, 0.75, 0.9, 0.95) | relational dynamics, scaled by 1 − ν | sensitivity |
| `readout_0.05`, `pure_clock_skew_0.1` | pure history through product readout | sensitivity |
| `registry_violating` | pure history rotated by a Haar system unitary | sensitivity |

**Existing banks.** The E5 and G-guard supports are regenerated exactly as in the v1 run and joined with the model route's stored outcomes, for v1 and v2 separately.

## 3. Endpoints (development; no confirmatory claim)

- **V1, type I.** v2 certification rate at α = 0.01 on each null family, per stratum and pooled, with Clopper–Pearson intervals. *Consistent* means the lower bound is at most α.
- **V2, sensitivity.** v2 rate per family and stratum.
- **V3, v1 against v2.** Paired decisions on the same counts. The expectation: v1 falsely certifies the two non-uniform-clock nulls; v2 does not.
- **V4, assumption.** v2 rate on `tau_flat_xor_0.05`. It is expected to exceed α. This shows that the physical reading needs product readout.
- **V5, model route.** The joint tables on the E5 and G-guard banks.
- **Dose–response.** v2 and v1 rates by true κ bin on the non-null families.

## 4. Use of the result

- **Sealed preregistration.** The development rates set the families and margins of a sealed preregistration ([sealed preregistration](observed-RDN-v2-sealed-preregistration.md)), written after this run.
- **Manuscript.** The manuscript's development table switches to v2. v1 stays as the comparison that motivates conditioning on the clock margin.

## 5. Integrity

- **Freezing.** The declaration records the config, this document, the dev-input hashes, the bank fingerprint hash, the disjointness check and the import-closure snapshot.
- **Drift.** The run refuses any drift and records the post-run drift.
- **Execution.** A process pool of at most 6 workers, 1 BLAS thread each.
- **Truth labels.** Exact states define the families and the true κ. The test reads only the observed counts.
