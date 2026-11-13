"""Validate a complete run, then compare its artifacts with the paper."""

from __future__ import annotations

import json
from pathlib import Path

from _comparison import compare_frames, compare_pdf, compare_tex, read_comparison_csv
from program_v04.replication.artifacts import artifact_registry
from program_v04.replication.config import PACKAGE_ROOT
from program_v04.validate_all import main as validate_all


def main() -> None:
    validate_all()
    reference = Path(__file__).resolve().parents[1] / "reference"
    counts = {"csv_tables": 0, "tex_tables": 0, "figure_pdfs": 0}
    maximum = 0.0
    for spec in artifact_registry():
        for relative in spec.output_paths:
            expected, observed = reference / relative, PACKAGE_ROOT / relative
            if expected.suffix == ".csv":
                difference = compare_frames(
                    read_comparison_csv(expected), read_comparison_csv(observed),
                    name=relative,
                )
                maximum = max(maximum, difference)
                counts["csv_tables"] += 1
            elif expected.suffix == ".tex":
                compare_tex(expected, observed)
                counts["tex_tables"] += 1
            elif expected.suffix == ".pdf":
                compare_pdf(expected, observed)
                counts["figure_pdfs"] += 1
            else:
                raise ValueError(f"Unexpected artifact: {relative}")
    print(json.dumps({**counts, "maximum_absolute_difference": maximum}, indent=2))


if __name__ == "__main__":
    main()
