# Observed RDN v2: sealed preregistration (sensitivity and validity of the both-margin test)

## Status

**Sealed confirmatory protocol, declared before its bank is built** (2026-11-23, on the user's instruction to complete the remaining experiments).

- The declaration freezes this file, `config/observed_rdn_v2_sealed.yaml` and the import closure of `src/bi_qmn/p3/runs/observed_rdn_v2.py`. That module was frozen by the development declaration and is unchanged since.
- **Development evidence.** [Development protocol](observed-RDN-v2-protocol.md) and [run](observed-RDN-v2-run.md). The test, threshold, Monte Carlo size, generative law and mask rule are carried over unchanged. The endpoints below were chosen from the development rates (§3).
- **Pre-declaration check.** Before declaring, the S3 failure-analysis code path was smoke-tested on the four S3 failures. No v2 test was computed on any S3 support.

```text
config:   config/observed_rdn_v2_sealed.yaml
code:     src/bi_qmn/p3/runs/observed_rdn_v2.py, runners/p3/run_observed_rdn_v2.py (--stage sealed)
seeds:    bank 53800, observation 53810, haar 53820, Monte Carlo 53830 (reserved namespace; new)
previously allocated in the namespace: 53500, 53600 (S3); 53700, 53710, 53720, 53740, 53750, 53760, 53790 (E1)
```

## 1. Claim under test

> For trusted product measurements with independent shots, the both-margin conditional test (Proposition 1 of the manuscript) at α = 0.01 per record:
> (a) falsely certifies static histories at a rate below 0.02, including static states seen through a non-uniform clock marginal or an asymmetric product clock readout; and
> (b) certifies relational dynamics in the declared families at the stated rates.

**Scope.** The registered P1 design, the D1-v7 generative law and the S3 mask rule, in the medium and low strata. Simulated data only.

## 2. Test, bank and families

**Test.** As declared for development: the G statistic of independence summed over the observed D contexts, its conditional law given both margins of each table (multivariate hypergeometric), and the Monte Carlo rank p-value with B = 10,000. A record certifies when `p ≤ 0.01`.

**Bank.** `observed_rdn_v2_sealed`, 512 constructions under the D1-v7 law.
- Each construction is observed in medium and low with the S3 mask rule. The mask seed key is `p3v2:{bank_id}:{deficit}:{i}`.
- Every family has its own multinomial sample.
- The declaration checks the bank against every declared bank by physical fingerprint.

**Families.** There are 11 per construction and stratum, so 11,264 records in all. Definitions are as in development.

| family | role |
|---|---|
| `tau_flat`, `tau_flat_readout_0.05`, `nonuniform_clock_static_0.5`, `tau_flat_clock_skew_0.1` | null (static conditional state; uniform or non-uniform clock marginal; symmetric or asymmetric product readout) |
| `tau_flat_xor_0.05` | report only: joint readout error, outside the product-readout assumption |
| `pure`, `dephased_0.75`, `dephased_0.9`, `readout_0.05`, `pure_clock_skew_0.1`, `registry_violating` | sensitivity |

`dephased_0.5` and `dephased_0.95` from development are not repeated.

## 3. Endpoints

Every endpoint is an exact one-sided binomial test at α = 0.05 on the v2 certification count at α = 0.01. "Pass line" is the critical count at n = 512, or n = 4,096 for T1.

| endpoint | families | stratum | null hypothesis | pass line | development rate |
|---|---|---|---|---|---|
| **T1** validity | the four null families, pooled | pooled | rate ≥ 0.02 | k ≤ 66 of 4,096 | 46/4,096 (1.12%) |
| **T2** validity, elevated cell | `tau_flat` | low | rate ≥ 0.035 | k ≤ 10 of 512 | 12/512 (2.3%) |
| **S**, near certain (10 tests) | `pure`, `dephased_0.75`, `readout_0.05`, `pure_clock_skew_0.1`, `registry_violating` | medium; low | rate ≤ 0.97 | k ≥ 504 of 512 | 512/512 in every cell |
| **S**, weak dynamics | `dephased_0.9` | medium | rate ≤ 0.80 | k ≥ 425 | 463/512 (0.904 [0.876, 0.928]) |
| **S**, weak dynamics | `dephased_0.9` | low | rate ≤ 0.20 | k ≥ 118 | 170/512 (0.332 [0.291, 0.375]) |

**Choice of margins.**
- **T1.** p0 = 2α bounds the false-certification rate at twice the nominal level. The test is exact, so the expected rate is at most α.
- **T2.** It targets the one development cell whose interval lay above α (12/512, p = 0.0062 against α). The margin 0.035 keeps power at 0.98 if the true rate equals α.
- **Weak dynamics.** The margins sit below the development lower confidence bounds.

**Combination.** The claim passes only if every endpoint passes (intersection–union). Each endpoint is an exact one-sided binomial test at α = 0.05, so the family level is 0.05.

**Independence.** Records that share a construction (two strata, several families) have independent shot samples. Given the constructions, the certification indicators are therefore independent, with heterogeneous rates. For heterogeneous independent indicators with a given mean, the lower tail used by T1 and T2 is no heavier than the binomial one (Hoeffding 1956, for critical counts below the mean minus one). The upper-tail endpoints are evaluated as declared binomial tests on the realized records.

**Planned power.** Exact binomial power for each endpoint:

| endpoint | assumed true rate | power |
|---|---|---|
| T1 | 0.0112 (the development rate) | 0.998 |
| T1 | 0.01 | 0.9999 |
| T2 | 0.01 | 0.984 |
| T2 | 0.0156 | 0.82 |
| near-certain endpoints | 0.995 | 0.9987 each |
| near-certain endpoints | 0.998 | > 0.99999 each |
| `dephased_0.9` medium | 0.876 (the development lower bound) | 0.999 |
| `dephased_0.9` low | 0.291 (the development lower bound) | 0.999 |

- **Joint power.** With the near-certain rates at 0.995 and the type-I rates at α, the joint power is about 0.97 (product approximation). The largest single risk is T2.

**Readings by outcome.**

| outcome | reading |
|---|---|
| claim passes | sealed confirmation (S) of the test's validity, including non-uniform clock marginals and asymmetric product readout, and of its sensitivity in the declared families |
| claim fails | the failing endpoints are reported; validity and sensitivity statements fall back to the development evidence (D) |

## 4. Retrospective S3-support analysis (post hoc)

The same run applies the fixed v2 rule to every evaluation support of the S3 sealed bank. That bank was opened once in 2026 and is regenerated here from its seeds.
- The S3 artifact's SHA-256 is checked before use. Every record must regenerate to its stored block, and the screen decision must reproduce.
- **Reported.** The v2 and v1 certification rates per stratum, and the joint table of S3 success against v2 certification, including the screened-out block.
- **Failure analysis.** For each S3 failure: the D commutant of its mask, the exact class of its construction, a refit with the frozen estimator that must reproduce the stored scores, the truth ratio and, for an exact D commutant, the fiber coverage of the fitted table along the truth's circle.
- **Label.** This is a post hoc analysis of an opened sealed bank. It carries no confirmatory weight.

## 5. Integrity

- **Freezing.** The declaration records the config, this document, the dev-input hashes (including the development artifact), the bank fingerprint hash, the disjointness check and the import-closure snapshot.
- **Drift.** The run refuses any drift and records the post-run drift.
- **One run.** The artifact is written once; a second run is refused.
- **Execution.** A process pool of at most 6 workers, 1 BLAS thread each.
- **Truth labels.** Exact states define the families and κ. The test reads only the observed counts.

## 6. Deviations

Any change after declaration requires a new declaration, with the reason and the time the data were seen.
