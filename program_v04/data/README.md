# Input data

This public-facing edition intentionally excludes the empirical snapshots. Obtain the inputs described in `data/README.md` and place them at the exact `public_path` locations in `data/manifest.csv` before running computation. The USRECM target can be downloaded and verified with `python -m program_v04.download_target`. Other inputs still require the snapshots described in the manifest.

`manifest.csv` records each input's exact `public_path`, source, vintage,
SHA-256 checksum, byte size, expected shape, and date bounds. The target's
byte hash identifies the archived pickle. Public USRECM verification permits
equivalent pickle serialization only when all monthly dates and values match.
All other inputs retain their byte checks.

Date bounds refer to raw source dates, before the loader normalizes dates to
month starts. The yield snapshot ends on February 28, 2025. Its original
acquisition date remains unknown. On 16 October 2026, a reconstruction from
the official New York Fed workbook matched that snapshot byte for byte.

## Required snapshots

The full profile requires 308 files:

- One final FRED-MD panel from the May 2025 release, with observations through
  April 2025.
- 305 monthly FRED-MD vintages, August 1999 through December 2024.
- One yield-spread snapshot, January 1959 through February 2025.
- One final USRECM chronology, January 1960 through December 2024.

The [St. Louis Fed FRED-MD page](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)
provides the historical archives. Use August 1999 through December 2014 from
the earlier archive and January 2015 through December 2024 from the later
archive. The final panel is `FRED-MD-2025m05.csv` in the later archive.
All 306 required CSVs matched the archived checksums in the 16 October review.
Keep their contents unchanged and place each at its manifest path.
Do not substitute `current.csv`.

The [New York Fed workbook](https://www.newyorkfed.org/medialibrary/media/research/capital_markets/allmonth.xls)
contains the yield series. The repository edition includes a helper that
reconstructs the archived CSV's date labels, precision, and line endings and
requires its original checksum. Its `docs/data_downloads.md` and
`docs/public_sources.json` provide the download instructions and exact archive
member mapping. These files are at the repository root, one level above
`program_v04/`.

## Recession target

From an installed `program_v04/` directory:

```sh
.venv/bin/python -m program_v04.download_target
```

The internal edition already includes the target and can skip this step.
The command downloads [USRECM from FRED](https://fred.stlouisfed.org/series/USRECM),
keeps January 1960 through December 2024, and writes
`data/raw/target/y_target.pkl`. It verifies every monthly date and value before
writing and refuses to overwrite an existing file. Use USRECM, which includes
recession peak months, rather than USREC.

The 16 October 2026 download matched all 780 months exactly, including 104
recession months and 40 during 1990 through 2024. Every download must match the
fixed monthly-value hash in `replication/data.py`. A changed or missing sample
month stops the command. It reports its actual retrieval time and hashes.
The original historical download date remains unknown.

## Retained input and preprocessing

The full manifest also retains an ADS snapshot through 12 July 2025 from the
[Federal Reserve Bank of Philadelphia](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ads).
The paper's current computation does not read it. The private bundle preserves
it; the public acquisition helper does not download a replacement.

The loader applies the FRED-MD transformation codes and fixed variable
exclusions. Imputation, scaling, PCA, and at-risk thresholds use the training
information specified in the code. The real-time exercise repeats preparation
within each vintage. Do not transform the original CSVs in advance.
The target uses the paper's final NBER chronology, not a vintage-specific target.

The main-paper profile requires only the three inputs in its pruned manifest.
The included documentary CSVs support Tables 21 and 22 in the full profile.
Data-provider terms remain separate from the code license.
