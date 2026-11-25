# Certifying relational dynamics from partial observations of quantum histories

Code and simulated data for the manuscript by JaeYun Jeong, Ki-Woong Park, and Hwa-Young Jeong. The repository contains the frozen results behind the manuscript, the experiment configurations and runners, and the scripts used to regenerate its figures.

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

The original experiment entry points are in [runners/p3/](runners/p3/), their numerical implementations in [src/bi_qmn/](src/bi_qmn/), and standalone manuscript analyses in [manuscripts/paper1/analysis/](manuscripts/paper1/analysis/). [config/](config/) holds the run settings. [documents/](documents/) contains the protocol files cited by the frozen declarations and the second-design pilot protocol.

The released declarations and run records preserve the original checksums and provenance. The sealed runners protect existing results from being overwritten. The S3 calibration declaration records an earlier version of the statistical protocol; the checked-out version is the one frozen for the subsequent sealed S3 analysis. Figure regeneration uses the released calibration artifact.
