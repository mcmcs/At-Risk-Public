# Replication package

This package reproduces the empirical and simulation results for “At-Risk
Transformation for U.S. Recession Prediction.” The main profile creates
Figures 1–3 and Tables 1–7. The appendix profile adds Figure 4 and Tables 8–22.

This public-facing edition intentionally excludes the empirical snapshots. Obtain the inputs described in `data/README.md` and place them at the exact `public_path` locations in `data/manifest.csv` before running computation. The USRECM target can be downloaded and verified with `python -m program_v04.download_target`. Other inputs still require the snapshots described in the manifest.

## Run the complete replication

The package requires CPython 3.13 and
[`uv`](https://docs.astral.sh/uv/). From the directory containing this
`program_v04` folder, run:

```bash
cd program_v04
uv sync --locked
.venv/bin/python -m program_v04.run_all --input-mode public
```

The final command performs the heavy computation, stores reusable numerical
results under `results/`, renders every table and figure under `output/` in
manuscript order, and validates the completed package.

To run the two profiles separately:

```bash
.venv/bin/python -m program_v04.main.run_main --input-mode public
.venv/bin/python -m program_v04.appendix.run_appendix --input-mode public
```

The appendix command reuses valid shared main results. The full workflow takes
roughly four hours on the authors’ verified Apple-Silicon environment; allow at
least 8 GiB RAM and 1 GiB free disk space in addition to the package itself.

## Compute once; render individual artifacts

The heavy-computation commands are:

```bash
.venv/bin/python -m program_v04.main.compute_main --input-mode public
.venv/bin/python -m program_v04.appendix.compute_appendix --input-mode public
```

After computation, each artifact can be rendered independently without model
estimation. For example:

```bash
.venv/bin/python -m program_v04.main.artifacts.generate_figure02
.venv/bin/python -m program_v04.appendix.artifacts.generate_table15
```

Run any artifact module with `--help` to see its manuscript label, required
stored results, outputs, and repair command. The complete inventory is in
`docs/ARTIFACTS.md` and `docs/artifact_manifest.csv`.

To rerender all artifacts from existing validated results:

```bash
.venv/bin/python -m program_v04.run_all --input-mode public --artifacts-only
```

To validate a completed package without recomputation:

```bash
.venv/bin/python -m program_v04.validate_all
```

## Data and reproducibility safeguards

`data/manifest.csv` is the machine-readable input contract. It records the
required path, vintage, SHA-256 checksum, size, shape, and date range for every
input. USRECM can be downloaded with `.venv/bin/python -m program_v04.download_target`.
As of September 10, 2026, all 780 paper-sample months matched FRED exactly.
Every download must match the same monthly dates and values. The target's
manifest byte hash and size identify the archived pickle; public verification
allows equivalent serialization. See `data/README.md` for the command and
the other inputs' acquisition and placement instructions.

The program refuses missing or mismatched inputs, partial result bundles,
stale source/configuration fingerprints, and incomplete artifacts. It never
falls back to developer archives or downloads data silently. `results/`
contains reusable machine-readable panels; `output/` contains the final CSV,
LaTeX, and PDF artifacts.

The tested environment is macOS on Apple Silicon with Python 3.13.12 and the
locked dependencies. Linux remains unverified. Numerical file bytes can differ
across environments. Cross-run comparison allows absolute differences up to
`5e-12` in floating-point results, with zero relative tolerance. Schemas,
identifiers, dates, counts, outcomes, and fixed numerical controls must match
exactly. Confidence-interval exclusions of zero and p-value significance
levels must agree.

Generated table text must match exactly after removing only its leading
run-fingerprint comment. Figure pages must have identical pixels when both
PDFs are rendered with the same Poppler installation at 144 dpi. PDF metadata
bytes may differ. These comparison rules supplement each run's exact input
and fingerprint checks; they do not replace them.
