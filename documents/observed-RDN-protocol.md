# Observed-RDN protocol: a model-free companion certificate of relational dynamics (development)

## Status

**Development protocol, frozen by its declaration.** The user decided this on 2026-11-23, after the G guard was not adopted ([G-guard run](G-guard-run.md) §5, remedy 1).

- **Scope.** The test is exact by construction. This protocol evaluates its sensitivity and its behaviour on controlled truths, and compares it with the model-based route.
- **Fits.** None are needed.

```text
config:   config/observed_rdn.yaml
code:     src/bi_qmn/p3/runs/observed_rdn.py, runners/p3/run_observed_rdn.py
tests:    tests/p3/test_observed_rdn.py
```

## 1. Why

The model-based RDN certificate is evaluated on the fitted table, which includes the pure-history model's extrapolation to unobserved contexts. As a result:
- Against static histories, it is protected only by the screen (E5).
- Under mild decoherence or readout noise, its calibrated justification fails while its verdict stays correct (E5, G guard).

A test that uses only the observed tables and only the sampling law needs no model, no calibration bank and no coverage.

## 2. The test

**Definition.** For each observed D context `P`:
- the counts `N(τ, s)` over the `K = 8` clock readings `τ` and the system outcomes `s ∈ {0, 1}`;
- the outcome totals `N_s = Σ_τ N(τ, s)`.

**Null hypothesis.** The context's table is τ-flat: `q(τ, s) = m_s` for all `τ`.
- Given `N_s`, the clock readings within each outcome `s` are then multinomial(`N_s`, uniform over `τ`).
- That conditional null has **no nuisance parameter**.

**Statistic.** `G_obs = Σ_P Σ_s Σ_τ 2 N(τ,s) log(K N(τ,s) / N_s)`, summed over the observed D contexts. Their samples are independent.

**p-value.** An exact conditional Monte Carlo p-value: 10,000 null draws per record, `p = (1 + #{G_null ≥ G_obs}) / 10,001`. The χ² approximation with `Σ (K − 1)` degrees of freedom is reported alongside.

**Certification.** **Observed relational dynamics** is certified when `p ≤ α = 0.01` (0.05 is reported).

## 3. What a certification licenses

1. **Every static history is rejected at once.**
   - Every energy-pinched state `E(ρ)` has τ-flat tables in every D context (contract F1).
   - So for any static history, the probability of certifying is at most α. No model, calibration or coverage is involved.
   - A certification therefore states, with error at most α: *the true state is not a static (energy-pinched) history.*
2. **Readout noise keeps the guarantee.**
   - Symmetric clock-label and system-bit readout noise maps τ-flat tables to τ-flat tables.
   - The test stays valid under the registered readout model.
3. **What it does not say.**
   - Nothing about unobserved contexts, the history, or the magnitude of the dynamics beyond the observed modulation.
   - Its sensitivity depends on the observed τ-modulation and the shot count.

## 4. Evaluation

**A. Fresh development bank.** `observed_rdn_v1`, seeds 56401 / 56402 / Haar 56403.
- 512 constructions × 2 strata (S3 mask rule) × 13 families, i.e. 1,024 records per family.
- Families:
  - static;
  - static with readout η = 0.05 (both null families);
  - pure;
  - dephased ν = 0.1, 0.25, 0.5, 0.75, 0.9, 0.95;
  - readout η = 0.02, 0.05, 0.1;
  - registry violating.
- The same construction, mask and shot random numbers are used for every family. The screen decision is recorded on the same support.

**B. Existing development banks** (E5 and the G-guard evaluation).
- Their supports are regenerated deterministically from their seeds. The check is that the observed context lists and the screen decisions match the stored records.
- The results are joined with the stored outcomes of the model-based route.

**Endpoints.**

| | statistic |
|---|---|
| **O1** type I | certification rate on the two null families at α = 0.01, with a Clopper–Pearson 95% upper bound (expected at most 0.01 by construction) |
| **O2** sensitivity | certification rate per family and stratum |
| **O3** dose–response | certification rate by true-κ bin: 0–0.25, 0.25–0.5, 0.5–1, 1–2, 2–4, >4 |
| **O4** versus the model route | on the existing banks, the observed-RDN status of records the model route certified with or without coverage, and of records the screen abstained on |
| **O5** screen agreement | cross-table of screen pass and observed-RDN certification |

## 5. Integrity

- **Declaration.** It records the config, this document, the dev-input hashes and an import-closure snapshot. The run refuses on drift and writes one artifact.
- **Seeds.** Outside the reserved namespace. The fresh bank is checked to be disjoint from every declared bank.
- **Execution.** A process pool of at most 6 workers, with 1 BLAS thread each.
- **Use of the truth.** The exact state defines the families and the true κ, and nothing else. The test uses only the observed counts.
