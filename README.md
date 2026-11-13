# At-risk transformation for U.S. recession prediction

Replication code for Rahul Billakanti and Minchul Shin, "At-Risk Transformation
for U.S. Recession Prediction," conditionally accepted by the *International
Journal of Forecasting*.

Prepared 2026-11-13. Package version: 0.4.0.dev0.
Contact: Minchul Shin, [minchul.shin@phil.frb.org](mailto:minchul.shin@phil.frb.org).

This public edition contains the main-paper and appendix code, locked Python
dependencies, input checksums, and reference tables and figures. It produces
22 numbered tables and four numbered figures. Figure 2 has three panels,
so the figure output comprises six PDFs. Empirical inputs and saved numerical
panels are excluded. See [data rights and access](docs/data_rights.md) for the
reason and the distinction between public access and redistribution permission.

## Obtain the exact data

Use CPython 3.13 on macOS with Apple Silicon, the tested platform. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and Poppler.
On macOS, `brew install poppler` installs the figure-comparison tool.
If XGBoost reports a missing `libomp` library, run `brew install libomp`.

From this repository's root:

```bash
cd program_v04
uv sync --locked
cd ..
uv pip install --python program_v04/.venv/bin/python xlrd==2.0.2
program_v04/.venv/bin/python scripts/download_inputs.py
program_v04/.venv/bin/python scripts/verify_inputs.py
```

The downloader obtains the 308 inputs used by the paper. It selects exact
historical FRED-MD archive members, reconstructs the archived yield CSV from
the New York Fed workbook, and checks the USRECM target. It stops if a required
snapshot has changed. The extra `xlrd` dependency reads the historical Excel
workbook; it is not part of model estimation.

[Data downloads](docs/data_downloads.md) gives the sources, date ranges,
manual placement instructions, and response to changed provider files.
The [308-file placement list](docs/input_locations.csv) gives each source URL,
archive member, exact destination, and checksum. The download and input checks
do not fit models.
Do not replace an unavailable vintage with `current.csv` or update a checksum
to accept different data.

## Run the replication

From the repository root:

```bash
python3 scripts/run_replication.py
```

This command installs the locked environment, checks the inputs, estimates
every model, generates the artifacts, validates the results, and compares
all 50 output files with the manuscript references. It writes `run_report.json`.
The model environment does not require the download-only Excel parser.

Allow about 2.5 to 4 hours, 8 GiB of RAM, and 2 GiB of free disk space.
The October independent full run took 2 hours 26 minutes on an Apple-Silicon
Mac. No GPU or cluster is needed. Linux and Windows have not been verified.

## Files and outputs

| Location | Contents |
| --- | --- |
| `program_v04/main/` | Main-paper computation and artifact commands |
| `program_v04/appendix/` | Appendix computation, simulation, and artifact commands |
| `program_v04/replication/` | Shared forecasting, data, and validation code |
| `program_v04/data/manifest.csv` | Exact input paths, checksums, shapes, and dates |
| `program_v04/uv.lock` | Locked Python dependencies |
| `program_v04/results/` | Numerical panels created by computation |
| `program_v04/output/` | Tables and figures created by the replication |
| `reference/output/` | Manuscript tables and figures used for comparison |
| `docs/public_sources.json` | Download URLs and exact archive-member mapping |
| `scripts/` | Acquisition, execution, integrity, and comparison commands |

[Artifact commands](program_v04/docs/ARTIFACTS.md) maps each table and figure
to its generator and required saved results. The
[package README](program_v04/README.md) describes separate main and appendix
runs. [Intermediate data](docs/intermediate_data.md) lists the saved bundles.

After a successful full run, regenerate artifacts without fitting models:

```bash
cd program_v04
.venv/bin/python -m program_v04.run_all --input-mode public --artifacts-only
cd ..
program_v04/.venv/bin/python scripts/compare_reference.py
```

## Verification and reporting problems

[Verification scope](docs/verification.md) records the tested code and results.
From the root, `python3 scripts/check_repository.py` checks the distributed
files against `REPOSITORY_MANIFEST.json` without estimating models.

If a command fails, retain its error message, command, operating system,
Python version, and affected input filename or checksum. Contact the author
above. Do not edit a checksum or result fingerprint to bypass a failed check.

## Citation and licensing

Use [CITATION.cff](CITATION.cff) for the software citation and cite the paper
title and authors above. The final publication DOI has not yet been assigned
in this package.

The authors' code is distributed under the [MIT license](LICENSE).

Data-provider terms apply separately to empirical inputs. The included
literature-survey CSVs and reference artifacts are separate from the raw
empirical snapshots omitted from this public edition.
