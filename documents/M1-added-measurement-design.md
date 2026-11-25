# M1 Q1 verification: risk detection on a fixed support, and recovery by one added D context (protocol)

## Status

**Development protocol. This file is frozen by the declaration** (`outputs/m1_added_measurement/m1_added_measurement_declaration.json`).

- The user approved the design on 2026-11-22, after two review rounds; §2 lists the decisions.
- The bank is built and screened only by the run, after the declaration.
- This is a development bank. Its result is a development finding, not a confirmation.

```text
config:      config/m1_added_measurement.yaml
code:        src/bi_qmn/p3/runs/m1_added_measurement.py, runners/p3/run_m1_added_measurement.py
tests:       tests/p3/test_m1_added_measurement.py
roadmap:     documents/BI-QMN-roadmap.md §4 Q1, §6, §7 item 6
mechanism:   documents/M1-third-mechanism-run.md §2 (types 3a/3b), §7 (added context, s_Q, what the orbit does and does not preserve)
dev inputs:  outputs/m1_third_mechanism/m1_added_context_check.json (SHA256 6dbc104287c314f424430dcea5245b247633fe8103dca74eadfbd3e76e6f0979)
             outputs/m1_third_mechanism/m1_orbit_kappa_check.json  (SHA256 c0f7d8f06adc743c67fa6a12320a99c94ac220d103316d5a1d464151f50718ef)
```

## 1. Question and success

Q1 closes M1 with the question: *which observations leave the history unidentified, and which measurement breaks the ambiguity?* Detection and recovery are separate results.

- **Detection (Q1 ②).** On an exact orbit every observed outcome probability is the same at every orbit point. So more shots of the same contexts add no information that separates the orbit points.
  - A fixed support can give a risk flag: the D-mask commutant, which is threshold-free and needs the mask only.
  - The flag catches 3a and 3b together, so it is not a non-recoverability verdict. Certification performance is reported for 3a and 3b separately.
- **Recovery (Q1 ④).** Add one unobserved D context whose Pauli `Q` anticommutes with `R`. Along the orbit its table moves as `cos 2t <Q> + sin 2t <iQR>`.

**Success (user, 2026-11-22).** Finding a context that breaks the symmetry is not enough. The experiment must show that **one context chosen from the support alone actually reduces the large 3a recovery errors** under the stated shot budget. `smin_rel` motivated the design; the evidence for adoption is the finite-shot recovery result.

## 2. Decisions fixed by the user (2026-11-22)

1. **Primary rule.** Fit the support alone, before the added measurement. At that fit, choose the candidate D context with the largest orbit-invariant sensitivity `s_Q`.
2. **Comparator.** The first candidate in a fixed order, which needs the mask only.
3. **Budget.** Both rules give the added context 2,048 shots.
4. **Bank.** An enriched bank of 128 3a and 128 3b records, generating at most 2,048 flagged constructions.
5. **Primary statistic.** `CVaR_0.9` of the 3a recovery error, computed separately for each arm, and compared by a paired bootstrap over record indices.
6. **Evaluation set.**
   - The common set `E` (the user's "unobserved tables other than the added context") applies to A0, A_s, A_f and C.
   - The oracle diagnostic uses its own set `E_o`. The oracle and every arm it is compared with are scored on that same set.
7. **The oracle candidate is fixed before any added-shot result is seen**, from the truth's `smin_rel`. This avoids picking whichever result came out best.
8. **A failed P1 has a narrow meaning.** It means "this bank did not demonstrate a reduction of the error tail", and nothing about the shot budget follows from it.
9. **C** may improve the estimate of the same support. Its prediction is limited to: it adds no information that separates the points of an exact orbit.
10. **RDN.** The physical object of the RDN decision is invariant along the orbit, but the registered margin `kappa` is not (run report §7). RDN status and the stability of `kappa` are measured separately.

## 3. Development facts behind the design

These are exact-table results at the true branch vectors of the 16 development draws (run report §7), not finite-shot results.

| 3a, screened-in low, 656 combination-draws | fixed order | max `s_Q` |
|---|---:|---:|
| `smin_rel` / best candidate, median | 0.949 | 0.828 |
| `smin_rel` / best candidate, minimum | 0.036 | 0.575 |
| below half of the best candidate | 38 | 0 |
| `smin_rel` median | 0.113 | 0.106 |

- **Worst fixed-order case.** identity, draw 1. The fixed order picks `D,YI` at 0.036 of the best; `s_Q` picks `D,ZZ` at 0.759.
- **The two rules mostly disagree.** They pick the same context in 135 of 656 cases (21%), which is why §6 uses a common set.
- **Exact ties.** In 135 of 656 cases two candidates share the largest `s_Q` at the truth. Both members of every tied pair are at least 0.576 of the best.
- **The pick depends on the state.** Every 3a combination has two different `s_Q` picks across its draws.
- **`s_Q` is orbit-invariant** (drift 1.3e-15). Whether its ranking survives a fit that noise moves *off* the orbit is measured in §7.
- **`kappa` moves along the orbit.** It moves on 576 of the 656 orbits (relative range up to 16%), but stays at 7.9 or more against the frozen low radius.

## 4. Bank

**Constructions.**
- `seeded_construction_bank_p1` with the D1-v7 generative specification draws up to `block_count = 2048` constructions. The bank ID, seed and observation seeds are in the config.
- The generator draws one of four system unitaries uniformly.
- Branch probabilities are Dirichlet(8) with floor 0.15; phases are uniform; readout η = 0.

**Flagged mask.**
- Construction `i` gets a low mask drawn uniformly from the **110 flagged masks**: the low masks whose observed D Paulis have a non-trivial Pauli commutant.
- The draw uses an RNG seeded by `sha256("{mask_seed}:{bank_id}:{i}")`.
- `R` is the unique D-commutant Pauli of the mask.

**Support sample.**
- 2,048 shots per support context: 6 D and 2 F contexts, 16,384 shots in all.
- They are sampled with `sample_shot_batch` from the exact table. The RNG is seeded by `sha256("{observation_seed}:{bank_id}:low:{i}")` with shot start `i * 1e8`, as `build_compact_dev_records` does.

**Acceptance.**
- Constructions are taken in index order.
- Screen v1 runs on the sampled support; a screened-out construction is not accepted.
- A screened-in construction is classified by the oracle. It is **3a** when the branch condition at the true branch directions is `< 1e-10`, and **3b** otherwise.
- It is accepted while its class quota (128 each) is not full. Generation stops when both quotas are full, or after construction 2,047.

**Planned yield.**
- The exhaustive development table gives 41 exact, screened-in combinations among the 440 flagged (unitary, mask) combinations.
- At that rate 3a fills after about 1,374 constructions on average; 3b fills much earlier.
- **If a quota is not filled by the cap, the shortfall is recorded and the analysis runs on the records accepted.** The counts of constructions generated, screened out, and accepted or turned away by class are reported.

## 5. Observations and arms

**Added-context samples.**
- Before any fit, every candidate context of an accepted record gets its own 2,048-shot sample. The candidates are the 4 or 5 unobserved input D contexts whose Pauli anticommutes with `R`.
- The RNG is seeded by `sha256("{observation_seed}:{bank_id}:added:{i}:{context}")`.
- No observation depends on which rule chooses. Two arms that choose the same context see the same sample.

**Control sample.** 256 extra shots per support context, with the RNG seeded by `sha256("{observation_seed}:{bank_id}:control:{i}")`. They are pooled with the support, giving 2,304 shots per context and the same total budget as an added context.

**Estimator.** Every arm uses the D1-v7 SBP estimator settings (`shared_branch_pair_v1`, 32 starts).

| arm | observations | role |
|---|---|---|
| A0 | support | baseline; its fit is where `s_Q` is computed |
| **A_s** | support + `Q_s` | **primary rule** |
| A_f | support + `Q_f` | comparator |
| C | support, 2,304 shots per context | same total budget, no new context |
| A_o | support + `Q_o` | oracle diagnostic |

**Choosing the contexts.**
- **`Q_s`** is the argmax of `s_Q` at the A0 branch vectors (run report §7 formula).
- **`Q_f`** is the first candidate in sorted context-id order.
- **`Q_o`** is the candidate with the largest truth `smin_rel` when it is added. That is the smallest non-gauge singular value of the exact support Jacobian at the true parameters, over the largest. It is fixed from the truth before any added-shot result exists.
- **Ties.** For each rule, values within relative 1e-12 of the maximum count as tied, and ties go to the fixed order.
- **Shared fits.** When two arms choose the same context, they share one fit. The fit is deterministic given its data.

## 6. Evaluation

**Error.** The ILGC-facing class score on a context set `S`: the L2 norm of `q_hat − q_exact` over the outcomes of the contexts in `S` of that class, with identity whitening. **The diag class is primary.** The coh class and the maximum absolute table difference are reported.

**Sets.**
- `E = (all 20 contexts) − support − {Q_s, Q_f}` scores A0, A_s, A_f and C.
- `E_o = E − {Q_o}` scores the oracle diagnostic: A_o, and A0, A_s, A_f and C again, all on `E_o`.
- Each arm's own remaining set (all contexts minus support minus its own added context) is reported too.

**Statistic.**
- For a stratum of `n` records, `CVaR_0.9` of an arm is the mean of its `ceil(0.1 n)` largest errors (13 of 128). It is computed separately for each arm.
- **Paired bootstrap.** Record indices are resampled with replacement (10,000 resamples). The RNG is seeded by `sha256("{bootstrap seed}:{stratum}")`, with the bootstrap seed taken from the config. The same index vector is used for both arms. `CVaR_0.9` of each arm is recomputed on the resample, and the difference is taken. The 95% interval is the 2.5% and 97.5% percentiles.

## 7. Endpoints and reports

**Judged (3a stratum, diag error on `E`).**

| | difference | outcome |
|---|---|---|
| **P1** | `CVaR(A0) − CVaR(A_s)` | **pass** when the interval lies above 0; otherwise "this bank did not demonstrate a reduction of the 3a error tail by the primary rule" |
| **P2** | `CVaR(A_f) − CVaR(A_s)` | interval above 0: `s_Q` has the lower tail. Interval below 0: the fixed order has the lower tail and replaces `s_Q`. Interval contains 0: not distinguished, and `s_Q` is kept on the worst-case rationale of §3 |

**Reported with the same bootstrap.** None of these change P1 or P2.
- `CVaR(A0) − CVaR(A_f)`: whether the comparator reduces the tail.
- `CVaR(C) − CVaR(A_s)`: the same total budget spent on the same contexts versus a new context.
- `CVaR(A0) − CVaR(C)`.
- The oracle block on `E_o`: `A0 − A_o`, `A_s − A_o`, `A_f − A_o`.
- Medians, p90 and maxima of every arm.
- All of the above for the 3b stratum.

**Rank stability** (the user's open question).
- How often `Q_s` at the fit equals the truth's argmax of `s_Q`. A pick inside a truth tie counts as agreement.
- The Spearman correlation of the candidates' `s_Q` at the fit and at the truth.
- The relative error of the fitted `s_Q`.
- For 3a, all three split by the A0 fit's distance from the truth's orbit: the minimum over a grid of `t` of the trace distance between the fitted history and `exp(itR)` applied to the truth.

**RDN and certification of A0 (Q1 ②).** Reported for 3a and 3b separately. The physical object of the RDN decision is invariant along an orbit, but `kappa` is not, so both are measured.
- **Certificates.** BI existence with the constraint registry and the 80-digit verifier, and `rdn_v1`. This is the S3 route, using the frozen S3 low radii read-only from `outputs/p3_s3_calibration/p3_s3_calibration.json` (SHA-256 checked).
- **Coverage.** Whether the truth's class score on all 20 contexts is within the frozen radius, for each class and jointly.
- **`kappa` from the tau-flat distance.** Computed for the exact table and for the fitted table of every arm (distance test only, no refutation step). Reported: the spread of `kappa` across arms and against the truth, and the fraction above 1.

**Structural checks.**
- The extra null of the support Jacobian at the truth (expected 1 for every accepted record).
- The extra null with each fitted arm's context added (expected 0).
- Any record that departs from these is reported as unclassified.

## 8. Cost

- **Fits.** A0, C and A_f for every record, plus A_o and A_s where they choose a new context: at most 1,280 fits.
- **Certificates.** 256 A0 certificates.
- **Time.** The S3 run took about 33 worker-seconds per fit and certificate, so this is about 45–60 minutes with 6 workers.

## 9. Integrity

- **Scope.** Development only, with no sealed bank. Screen v1, S3, A1, T1, D1-v7 and M1-map artifacts and their import closures are read, never changed.
- **Bank isolation.**
  - The construction fingerprints are disjoint from every declared bank.
  - The seeds lie outside the reserved confirmatory range 53500–53999.
- **Declaration.** It records the config, this document, the dev-input and frozen-calibration hashes, and an import-closure snapshot. The run refuses on any drift.
- **Execution.**
  - A process pool of at most 6 workers, with 1 BLAS thread each.
  - Workers rebuild the bank's constructions and check its fingerprint hash.
  - A test requires serial and parallel runs to agree.
- **Labels.** The candidates and the rules came from the development draws of the M1 map. This bank is new, but the result is a development finding.

## 10. What each outcome means for M1

| outcome | M1 statement |
|---|---|
| P1 passes, P2 keeps `s_Q` | one context chosen from the support alone reduces the large 3a errors; the theory document states the recovery condition with `s_Q` as the rule |
| P1 passes, P2 favours the fixed order | recovery is demonstrated; the rule becomes the fixed order, with the reason recorded |
| P1 not shown, `A0 − A_f` above 0 | the fixed order reduced the tail on this bank but the primary rule did not; the rule must be revisited before M1 closes |
| P1 not shown, A_f not shown, oracle `A0 − A_o` above 0 | a single added context can reduce the tail, but neither support-only rule demonstrated it on this bank |
| no arm shows a reduction | this bank did not demonstrate a tail reduction for any single added context. The theory document states recovery as structural (first order) only. A claim about the shot budget would need separate evidence |
| rank stability poor away from the orbit | `s_Q` at the fit is not a reliable proxy; report how it depends on the distance to the orbit |
| quota not filled | report the shortfall; the judged statistics run on the accepted records, with their `n` stated |
