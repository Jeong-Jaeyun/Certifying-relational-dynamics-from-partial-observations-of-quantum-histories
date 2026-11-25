# E4: Hamiltonian information, tolerance and misspecification (development protocol)

## Status

**Development protocol, declared before generation.** The declaration freezes this file, `config/e4_hamiltonian_information.yaml` and the import closure of `src/bi_qmn/p3/runs/e2_e4_extensions.py`.

**Question.** In development, the exact zero-energy constraint of the registered `H_tot` (information level ii) removed the type-3a ambiguity: tail 0.0126 against 0.454 at level i (post hoc). Proposition 5 explains why. The registered energy along the circle is `B sin²t`, with `B ≥ λ₊ = 2/21` on the development records, so the registered tolerance `θ_con = 1e-6` leaves an arc of half-width ≤ 3.2e-3.

E4 asks how much of that gain survives in four situations:
- a looser tolerance;
- a registry whose eigenbasis is slightly wrong;
- a registry whose spectrum is slightly wrong;
- the comparison with, and the combination with, one added measurement.

## 1. Bank (seeds 56901 / 56902 / 56903, outside the reserved namespace)

- **Law.** The registered D1-v7 law. Each construction draws a flagged low mask uniformly.
- **Acceptance.** Screened-in records of exact class 3a, in index order; quota 128, cap 2,048. The procedure is the M1 run's (`build_construction`).
- **Disjointness.** The declaration checks the bank against every declared bank.

## 2. Arms (same support sample in every arm)

All level-ii-type fits use the active-mode coordinates `y` (mode `j`, system index `a`). There, every table is `y†A_o y / y†y` with the estimator's effect matrices. The registered energy is `y†C y / y†y` with `C = ⊕_j (H_S + ω_{m_j})² / s_H²`.

| arm | information | fit |
|---|---|---|
| `A0` | level i (spectrum known, basis free) | `shared_branch_pair_v1`, 32 starts |
| `H_exact` | registered `H`, exact zero energy (level ii) | ML over unit vectors in `ker C` (8 random starts) |
| `H_soft_θ`, θ ∈ {1e-6, 1e-4, 1e-3, 1e-2, 1e-1} | registered `H`, certified tolerance `⟨H²⟩/s_H² ≤ θ` | ML over active-mode pure states subject to `y†Cy/y†y ≤ θ` (SLSQP from the H_exact and A0 optima plus 2 random starts; the best feasible optimum) |
| `H_rot_δ`, δ ∈ {0.01, 0.03, 0.1} | registry with eigenbasis `exp(iδG) U`, `‖G‖_op = 1` (wrong `H`) | exact kernel of the wrong registry |
| `H_eig_0.01` | registry with active eigenvalues shifted by (+ε, −ε, +ε) | exact kernel; expected **empty**, so no exact zero-energy history exists |
| `H_eig_0.01_soft` | the same wrong registry at tolerance 1e-3 | soft fit |
| `A_s` | level i plus the max-`s_Q` context (2,048 shots) | `shared_branch_pair_v1` |
| `H_exact_added` | level ii plus the same added context | exact kernel |

**Per record and arm.**
- **Feasibility.** Whether a fit exists, and the kernel dimension.
- **Error.** The diag error on `E = all − support − {Q_s}`.
- **Coverage.** Joint truth coverage by the frozen S3 low bodies.
- **Orbit angle.** The angle of the fit's nearest point on the truth's `R` circle.
- **Energy.** The fit's energy under the true registry.
- **Likelihood.** The NLL excess over A0 on the support. This shows whether a wrong `H` is visible in the data.

**Per record.** `B`, the arc half-widths for each θ, and the truth's energy under every registry (whether the truth lies outside a wrong registry's constraint set, energy > 1e-6).

**Certificates.** For the first 64 records, BI existence and `rdn_v1` with each arm's own registry: `A0` and `H_exact` with the true registry, and `H_rot_δ` with the rotated registry.

## 3. Readouts (development; descriptive with paired-bootstrap intervals against H_exact)

- **H1, information levels.** CVaR, coverage and orbit angle for A0, H_exact, A_s and H_exact_added.
- **H2, tolerance.** The same for each θ, with the arc half-width. Expectation: for 3a records, coverage falls once the arc exceeds the shot-noise resolution.
- **H3, wrong eigenbasis.** Two cases are distinguished:
  - the truth lies outside the wrong registry's constraint set;
  - the truth's coverage fails even though the fit is feasible.
  The readouts are the NLL excess (whether the support exposes the error) and the BI status.
- **H4, wrong spectrum.** The kernel dimension and the soft arm's CVaR and coverage.

## 4. Use of the result

- **Manuscript.** The Hamiltonian section states the tolerance and misspecification ranges over which level ii removes the ambiguity in this design, as development evidence.
- **Unchanged claim.** The current claim, "exact registered `H`, exact zero energy", stays the stated assumption.

## 5. Integrity

As in the other declared development runs: frozen declaration, drift refusal, at most 6 workers with 1 BLAS thread each, and exact states used only for generation, labels, errors, coverage and geometry.
