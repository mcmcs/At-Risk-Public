# Verification scope

## Current release

The corrected release completed a fresh full run on 18 October 2026. All 22 CSV tables, 22 LaTeX tables, and six figure PDFs passed the reference comparison. Regenerating artifacts left all 51 numerical result and metadata files unchanged.

`staging_checks.json` records the production-code SHA-256, completed stages,
run times, and exit codes. `REPOSITORY_MANIFEST.json` checks every distributed
file. These records distinguish the current code from earlier reviewed copies.

The current release fixes Figure 4's dependence on the plotting style left
by earlier figures. Its grid is now the same in a full run and an isolated
render. The fix changes no model, input, forecast, or score calculation.

## Independent October review

The independent review on 16 October 2026 ran 191 unit tests successfully and
obtained all 308 required inputs from public providers. Every required
FRED-MD file and the reconstructed yield CSV matched its archived byte hash.
The USRECM target matched all required dates and values.

A fresh locked environment on macOS with Apple Silicon and CPython 3.13.13
completed the full computation in 2 hours 25 minutes 45 seconds. It generated
35 numerical panels in 16 result bundles. All 22 CSV tables matched the
manuscript references within absolute tolerance `5e-12`, with a largest
difference of about `2e-12`. All 22 LaTeX tables matched after removal of their
leading run-fingerprint comments. Five of six PDFs matched. The remaining
difference was Figure 4's grid, corrected in the current release.

That review also regenerated all artifacts without changing any of the 51
saved numerical and metadata files. Five numerical panels with retained
historical author counterparts matched those counterparts exactly. The
remaining 30 panels had no such historical counterpart for that comparison.

## Comparison rules

After a complete run, execute this command from the repository root:

```bash
program_v04/.venv/bin/python scripts/compare_reference.py
```

It first runs the package validator, then compares all 50 registered outputs
with `reference/output/`. Floating-point comparisons use absolute tolerance
`5e-12` and zero relative tolerance. Identifiers, dates, counts, outcomes,
selected controls, significance categories, and interval exclusions of zero
must match exactly. LaTeX table text must match after removing its leading
run-fingerprint comment.

Figure comparison uses Poppler's `pdftoppm`, separate from the Python
environment. Both PDFs are rendered at 144 dpi with the same installation,
and their pixels must match. PDF metadata differences do not change a figure.

The reference artifacts preserve the paper's values. Their leading fingerprint
comments identify the runs that produced them. The included
`program_v04/docs/manuscript.tex` contains only table/figure labels in their
original order for numbering checks. Its filename is retained for validator
compatibility. The manuscript text and submission letters are not included.
