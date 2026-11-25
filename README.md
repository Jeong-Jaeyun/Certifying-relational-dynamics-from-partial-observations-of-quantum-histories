# Certifying relational dynamics from partial observations of quantum histories

Code and simulated data for the manuscript by JaeYun Jeong, Ki-Woong Park, and Hwa-Young Jeong. The repository contains the frozen results behind the manuscript, the experiment configurations and runners, and the scripts used to regenerate its figures.

## Read the implementation

[**Numerical code**](src/bi_qmn/) lives in the `bi_qmn` Python package. [**Experiment commands**](runners/p3/) load configurations and call the corresponding Python modules. Start with the implementation for the part of the paper you want to inspect.

| Paper topic | Implementation | Experiment entry point |
| --- | --- | --- |
| Observed relational dynamics | [Fixed-margin test](src/bi_qmn/p3/runs/observed_rdn_v2.py), [uniform-clock test](src/bi_qmn/p3/runs/observed_rdn.py) | [Conditional RDN study](runners/p3/run_observed_rdn_v2.py) |
| Histories, observations, and estimation | [History generator](src/bi_qmn/p1/model/p1_generator.py), [measurement model](src/bi_qmn/p1/model/p1_measurement.py), [history fit](src/bi_qmn/p3/models/p3_shared_branch_pair.py) | [S3 calibration](runners/p3/run_p3_s3_calibration.py) |
| Model-based BI and RDN certification | [Feasibility certificate](src/bi_qmn/ilgc/certificate.py), [verification](src/bi_qmn/ilgc/verifier.py), [RDN adjudication](src/bi_qmn/ilgc/relational_dynamics.py) | [Sealed S3 study](runners/p3/run_p3_s3_confirmatory.py) |
| Ambiguity and identifiability | [Support Jacobian](src/bi_qmn/p3/runs/m1_support_identifiability_map.py), [commutant and exact circles](src/bi_qmn/p3/runs/m1_d_commutant_check.py), [independent Born rule](src/bi_qmn/p3/independent_born.py) | [Orthogonal-history example](runners/p3/run_m1_evidence_package.py) |
| Adding a measurement | [Context selection and comparison](src/bi_qmn/p3/runs/m1_added_measurement.py), [sealed comparison](src/bi_qmn/p3/runs/m1_e1_confirmatory.py) | [E1 study](runners/p3/run_m1_e1_confirmatory.py) |
| Hamiltonian information | [Zero-energy estimator](src/bi_qmn/p3/models/p3_zero_energy_history.py), [E2 and E4 analysis](src/bi_qmn/p3/runs/e2_e4_extensions.py) | [E2 and E4 studies](runners/p3/run_e2_e4.py) |
| Second clock design | [Design, separation and two-point bounds](src/bi_qmn/p3/runs/design2_core.py) | [Pilot](runners/p3/run_design2_pilot.py), [sealed study](runners/p3/run_design2_confirm.py) |

[Shared numerical utilities](src/bi_qmn/core/), [statistical procedures](src/bi_qmn/stats/), and [post hoc manuscript analyses](manuscripts/paper1/analysis/) support these implementations. The package paths match those recorded in the frozen experiment declarations.

## Get the code and data

Install Python 3.11 or newer and [Git LFS](https://git-lfs.com/), then run:

~~~sh
git lfs install
git clone https://github.com/Jeong-Jaeyun/Certifying-relational-dynamics-from-partial-observations-of-quantum-histories.git
cd Certifying-relational-dynamics-from-partial-observations-of-quantum-histories
git lfs pull
python -m pip install -e . matplotlib
~~~

The JSON files under [outputs/](outputs/) are stored in Git LFS. The verification and plotting scripts need the downloaded JSON data.

## Call the conditional test from Python

After installation, pass a mapping from measurement context to a count table. Rows index clock readings and the two columns count the system outcomes.

~~~python
import numpy as np
from bi_qmn.p3.runs.observed_rdn_v2 import both_margin_test

counts = {"D,ZI": np.array([[18, 2], [16, 4], [4, 16], [2, 18]])}
result = both_margin_test(counts, B=10_000, rng=np.random.default_rng(42))
print(result["p_monte_carlo"])
~~~

## Verify and regenerate the figures

From the repository root, check the released artifacts and frozen inputs against [SHA256SUMS](SHA256SUMS). In a shell with GNU coreutils, run:

~~~sh
sha256sum -c SHA256SUMS
~~~

A Python equivalent works on Windows:

~~~sh
python -c "import hashlib; from pathlib import Path; lines = (s.split('  ', 1) for s in Path('SHA256SUMS').read_text().splitlines()); bad = [p for h, p in lines if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]; raise SystemExit('Checksum mismatch: ' + ', '.join(bad) if bad else 0)"
~~~

Regenerate the figures from the frozen data:

~~~sh
python manuscripts/paper1/figures/make_figures.py
python manuscripts/paper1/figures/make_schematic.py
~~~

These commands write figures next to the scripts. The main plotting script verifies the SHA-256 digest of each data input and independently recomputes the Born probabilities for the orthogonal-history example.

## Find the results

| Result | Frozen data |
| --- | --- |
| Conditional test of relational dynamics, including the S3 retrospective | [Conditional test](outputs/observed_rdn_v2) and [uniform-clock comparison](outputs/observed_rdn) |
| Model-based BI and RDN certification | [S3 calibration](outputs/p3_s3_calibration) and [sealed S3 run](outputs/p3_s3_confirmatory) |
| Identifiability classification and explicit orthogonal histories | [Support maps and checks](outputs/m1_third_mechanism) and [reproduction kit](outputs/m1_evidence_package/m1_reproduction_kit_3a.json) |
| Same-budget additional-measurement comparisons | [Development run](outputs/m1_added_measurement), [choice rules](outputs/m1_e3_rules), and [sealed E1 run](outputs/m1_e1_confirmatory) |
| Scope extension and Hamiltonian information | [E2](outputs/e2_scope_extension), [E4](outputs/e4_hamiltonian_information), and [support-only post hoc analyses](outputs/m1_added_measurement_posthoc) |
| Misspecification controls and goodness-of-fit guards | [Controlled truths](outputs/e5_rdn_specificity), [support deviance](outputs/g_guard), [bootstrap guard](outputs/bootstrap_guard), and [post hoc analyses](outputs/paper1_revision_posthoc) |
| Second design | [Structure, pilot, and bounds excluding measured contexts](outputs/design2) and [sealed confirmation](outputs/design2_confirmatory) |

[config/](config/) holds the run settings. [documents/](documents/) contains the protocol files cited by the frozen declarations and the second-design pilot protocol.

The released declarations and run records preserve the original checksums and provenance. The sealed runners protect existing results from being overwritten. The S3 calibration declaration records an earlier version of the statistical protocol; the checked-out version is the one frozen for the subsequent sealed S3 analysis. Figure regeneration uses the released calibration artifact.
