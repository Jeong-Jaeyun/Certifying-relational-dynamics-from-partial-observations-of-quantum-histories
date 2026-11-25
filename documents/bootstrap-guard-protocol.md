# Bootstrap guard: record-specific goodness-of-fit abstention (development protocol)

## Status

**Development protocol, declared before its bank is built.** The declaration freezes this file, `config/bootstrap_guard.yaml` and the import closure of `src/bi_qmn/p3/runs/bootstrap_guard.py`.

**Background.** The conformal G guard ([run](G-guard-run.md)) was not adopted, for two reasons.
- **False abstention.** Its pure-record false abstention did not transfer: 15/127 = 11.8% against a nominal 5%. The post hoc reading was that the null distribution of the support deviance `G` varies between constructions.
- **Leak.** It let mildly dephased records (ν = 0.1) through. Their bodies did not cover the truth (3 of 18 guarded certified records covered).

This protocol tests the remedy that the first point suggests: calibrate each record against its own fitted model.

## 1. The rule

For a screened-in support:

1. **Fit.** Fit with the frozen estimator (`shared_branch_pair_v1`, 32 starts), giving `θ̂` and the fitted support table in the estimator's output convention.
2. **Deviance.** `G_obs = 2 Σ n p̂ log(p̂ / q̂)` over the support cells.
3. **Bootstrap.** Draw `B = 49` supports from the fitted table (independent multinomial shots per observed context, the record's shot count).
4. **Refit.** Refit each bootstrap support from `θ̂` plus one random start, keep the better optimum and record `G*`.
5. **p-value.** `p = (1 + #{G* ≥ G_obs}) / 50`.
6. **Decision.** Abstain if `p ≤ 0.05`, that is, if at most one `G*` reaches `G_obs`.

**What the rule can and cannot do.**
- **Validity is approximate.** A parametric bootstrap is asymptotically valid, not exact. Its false-abstention rate on pure histories is the first endpoint.
- **Refitting is local.** Warm-starting at `θ̂` can only overstate `G*` relative to a global refit, which makes abstention rarer.
- **Power is not raised.** The rule uses the same statistic as the conformal guard, so its sensitivity to mild misspecification is limited by the same shot noise. The protocol measures how far.

## 2. Bank and families

**Bank.** `bootstrap_guard_v1`, seeds 56701 / 56702 / 56703 and bootstrap seed 56704, outside the reserved namespace.
- 64 constructions under the D1-v7 law, each observed in medium and low with the S3 mask rule.
- Families share each construction's support random numbers.
- The declaration checks disjointness from every declared bank.

| family | truth |
|---|---|
| `pure` | the S3 population (null for the guard) |
| `dephased_ν`, ν = 0.05, 0.1, 0.25 | `(1 − ν)|Ψ⟩⟨Ψ| + ν E(|Ψ⟩⟨Ψ|)` |
| `readout_η`, η = 0.01, 0.02, 0.05 | pure history, symmetric readout error (the estimator assumes η = 0) |

7 families × 2 strata × 64 = 896 records.

## 3. Readouts and endpoints

**Per record.**
- The screen decision.
- The fit and `κ_fit`. Certification is the distance stage of `rdn_v1`, `κ_fit > 1`, the only stage that certifies.
- Joint truth coverage by the frozen S3 bodies.
- `G`, `p_bootstrap` and the bootstrap guard decision.
- The frozen conformal-guard decision (thresholds 144.2 medium, 105.3 low), for a paired comparison.

**Endpoints.** The criteria are those of the conformal guard.

| endpoint | statistic | criterion |
|---|---|---|
| **B1** false abstention | abstentions among screened-in pure records, pooled | consistent if the Clopper–Pearson 95% lower bound ≤ 0.05 |
| **B2** detection | abstentions per misspecified family | reported with intervals |
| **B3** validity | joint truth coverage among guard-passing, certified records, per misspecified family | ≥ 0.90 where at least 20 such records exist |

**Decision.** Adopt the guard only if B1 is consistent **and** every evaluable misspecified family meets B3.
- If it is adopted, the manuscript may state that the model-based truth statement is guarded against the tested misspecifications, at the measured cost.
- If it is not adopted, the model-based truth statement stays conditional on the pure-history model. The manuscript reports the detection curve.

## 4. Integrity

- **Freezing.** The declaration records the config, this document, the dev-input hashes, the bank fingerprint, the disjointness check and the import-closure snapshot.
- **Drift.** The run refuses drift and records the post-run drift.
- **Execution.** At most 6 workers with 1 BLAS thread each. Workers rebuild the constructions and check their hash.
- **Truth labels.** Exact states define the families, the true κ and coverage. The guard reads only the support and its own fit.
