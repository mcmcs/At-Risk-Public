# Data downloads and frozen snapshots

The paper uses 308 input files. The public edition contains none of those
files; the downloader installs them locally. The manifest also documents an unused
ADS file, which is not needed and is not downloaded. Data rights and access
conditions are described in [data_rights.md](data_rights.md).
The separately retained confidential journal edition includes the inputs and
does not need downloads. The final submission packet uses the public edition.

## Exact sources

| Input | Required sample or vintage | Official source |
| --- | --- | --- |
| Final FRED-MD panel | May 2025 release, observations through April 2025 | [St. Louis Fed FRED-MD archives](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) |
| 305 real-time FRED-MD panels | August 1999 through December 2024 | Same historical archives |
| Yield-spread CSV | January 1959 through February 2025 | [New York Fed workbook](https://www.newyorkfed.org/medialibrary/media/research/capital_markets/allmonth.xls) |
| Recession target | USRECM, January 1960 through December 2024 | [FRED USRECM](https://fred.stlouisfed.org/series/USRECM) |

On 16 October 2026, all 306 required FRED-MD CSVs matched the frozen manifest
byte for byte. The final snapshot is archive member `FRED-MD-2025m05.csv`.
The real-time sample uses 185 files from the earlier archive and 120 from the
later archive. `public_sources.json` records the download URLs and exact
member-to-input mapping, including the different historical filename styles.

The New York Fed workbook reconstructs the archived yield CSV exactly.
The supplied helper retains its historical date labels, numeric precision,
line endings, and sample bounds. The original author's download date remains
unknown; the verified reconstruction does not establish that historical date.

USRECM includes recession peak months. Do not substitute USREC. The target
must contain all 780 required monthly dates and binary values, including 104
recession months overall and 40 during 1990 through 2024. Its check permits
different pickle serialization only when those dates and values match exactly.

## Download and check

Follow the installation and download commands in the root README.
The downloader keeps provider archives under `downloads/` and writes exact
inputs to the `public_path` entries in `program_v04/data/manifest.csv`.
It writes a successful acquisition record to `download_report.json`.

Existing input files are verified before any download. A mismatching existing
file stops the command. The downloader does not replace it silently.
Each FRED-MD member and the reconstructed yield CSV must match the frozen
SHA-256 and size. The loader also checks the declared shapes and date bounds.
Archive checksums in `public_sources.json` identify the October verification
downloads. Provider archives may gain extra members, so the required member
checksums determine whether the paper's inputs still match.

All commands below are run from the replication folder containing `scripts/`
and `program_v04/`. In the final submission packet, first run:

```bash
cd submission_package_final/replication
```

The downloaded inputs belong at these locations, relative to that folder:

| Input | Destination | Count |
| --- | --- | ---: |
| May 2025 FRED-MD CSV | `program_v04/data/raw/fred_md/latest/fredmd.csv` | 1 |
| Monthly vintages | `program_v04/data/raw/fred_md/vintages/YYYY-MM.csv` | 305 |
| Reconstructed yield spread | `program_v04/data/raw/yield/yieldspread.csv` | 1 |
| Verified recession target | `program_v04/data/raw/target/y_target.pkl` | 1 |

For example, archive member `FRED-MD_2024m03.csv` must be copied to
`program_v04/data/raw/fred_md/vintages/2024-03.csv`. The complete
[input_locations.csv](input_locations.csv) joins the source URL, archive
member, destination, date bounds, and checksum for all 308 inputs. Its paths
are relative to this replication folder; the manifest's `public_path` entries
are relative to `program_v04/`.

The recommended manual-download route is to save the two official ZIPs as
`downloads/1999_2014.zip` and `downloads/2015_2025.zip`, and the workbook as
`downloads/allmonth.xls`, then run `scripts/download_inputs.py` with the
package Python. It uses the cached archives, puts every CSV in the correct
location, converts the workbook, and downloads/verifies USRECM. No need to
open or re-save any CSV in spreadsheet software.

For entirely manual placement:

1. Download the two archives and the workbook linked in `public_sources.json`.
2. Use its member mapping to copy each required CSV, without changing its
   contents, to the corresponding manifest `public_path`.
3. Reconstruct the yield file with the supplied helper:

   ```bash
   program_v04/.venv/bin/python scripts/_reconstruct_yield.py --workbook downloads/allmonth.xls --manifest program_v04/data/manifest.csv --output program_v04/data/raw/yield/yieldspread.csv
   ```

4. From `program_v04/`, run `.venv/bin/python -m program_v04.download_target`.
5. Check inputs without fitting any models:

   ```bash
   program_v04/.venv/bin/python scripts/verify_inputs.py
   ```

To reuse previously downloaded archives without network access, first place
the verified target at its manifest path. Then run the downloader with
`--offline --cache-dir /path/to/downloads`. The cache must contain
`1999_2014.zip`, `2015_2025.zip`, and `allmonth.xls`.

## If a provider changes a file

A changed checksum is a data-version difference. Keep the failed file and
error report separate from the frozen package. Do not substitute `current.csv`,
expand the sample, or change the manifest to make replication pass.

Contact [Minchul Shin](mailto:visiblehand@gmail.com) with the input ID and
observed checksum. The authors retain archived inputs for research records;
any access arrangement remains subject to provider permissions. No private
data bundle is included with the data-free final submission packet.
