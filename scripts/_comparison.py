"""Compare two runs' registered artifacts and their source panels.

This cross-run check supplements each package's own integrity validator.
It does not accept stale fingerprints or establish that a run is complete.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops

from program_v04.replication.artifacts import artifact_registry


# The August 8 audit observed maximum CSV drift of 1.997e-12. This absolute
# bound leaves a small margin; relative tolerance must not grow with scale.
NUMERICAL_ATOL = 5e-12
RASTER_DPI = 144
EXACT_FLOAT_COLUMNS = frozenset(
    {"actual", "recession", "z_yield", "z_sahm", "confidence_level",
     "decision_threshold", "prevalence", "no_skill_pr_auc"}
)
_RUN_COMMENT = re.compile(
    rb"\A% dependency run fingerprints: [0-9a-f]{64}(?:, [0-9a-f]{64})*\r?\n"
)


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, *, name: str) -> float:
    """Keep structure and controls exact; bound differences in float results."""
    try:
        pd.testing.assert_index_equal(left.columns, right.columns, exact=True)
        pd.testing.assert_index_equal(left.index, right.index, exact=True)
        pd.testing.assert_series_equal(left.dtypes, right.dtypes)
        if not left.columns.is_unique:
            raise AssertionError("Duplicate columns")
        maximum = 0.0
        for column in left:
            a, b = left[column], right[column]
            if not pd.api.types.is_float_dtype(a.dtype) or column in EXACT_FLOAT_COLUMNS:
                pd.testing.assert_series_equal(a, b, check_exact=True)
                continue
            pd.testing.assert_series_equal(a.isna(), b.isna())
            av, bv = a.to_numpy(), b.to_numpy()
            np.testing.assert_allclose(av, bv, rtol=0.0, atol=NUMERICAL_ATOL, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(av[finite] - bv[finite]))))
            # Small numeric differences cannot reverse the reported interval
            # exclusions or conventional p-value significance levels.
            if column in {"ci_lower", "ci_upper"}:
                np.testing.assert_array_equal(np.sign(av), np.sign(bv))
            if str(column).startswith("pval"):
                for cutoff in (0.01, 0.05, 0.1):
                    np.testing.assert_array_equal(av < cutoff, bv < cutoff)
        return maximum
    except AssertionError as exc:
        raise ValueError(f"Reproduction mismatch in {name}: {exc}") from exc


def compare_tex(left: Path, right: Path) -> None:
    """Ignore only the generated leading run-fingerprint comment."""
    a = _RUN_COMMENT.sub(b"", left.read_bytes(), count=1)
    b = _RUN_COMMENT.sub(b"", right.read_bytes(), count=1)
    if a != b:
        raise ValueError(f"Displayed table differs: {right}")


def compare_pdf(left: Path, right: Path) -> None:
    """Require equal page counts and pixels using one Poppler installation."""
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("Figure comparison requires pdftoppm from Poppler")
    with tempfile.TemporaryDirectory(prefix="at_risk_rasters_") as directory:
        root = Path(directory)
        pages = []
        for tag, source in (("left", left), ("right", right)):
            subprocess.run(
                [executable, "-png", "-r", str(RASTER_DPI), str(source), str(root / tag)],
                check=True, capture_output=True, text=True,
            )
            pages.append(sorted(root.glob(f"{tag}-*.png")))
        if not pages[0] or len(pages[0]) != len(pages[1]):
            raise ValueError(f"Figure page count differs: {right}")
        for a, b in zip(*pages, strict=True):
            with Image.open(a) as first, Image.open(b) as second:
                first_rgb, second_rgb = first.convert("RGB"), second.convert("RGB")
                if first_rgb.size != second_rgb.size or ImageChops.difference(first_rgb, second_rgb).getbbox():
                    raise ValueError(f"Figure raster differs: {right}, page {b.stem}")


def compare_run_metadata(left: Path, right: Path) -> None:
    """Require the same recorded production code and locked configuration."""
    a, b = json.loads(left.read_text()), json.loads(right.read_text())
    for field in ("source_sha256", "config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(a.get(field, ""))) or a[field] != b.get(field):
            raise ValueError(f"Run {field} differs or is missing: {right}")


def read_comparison_csv(path: Path) -> pd.DataFrame:
    """Parse numeric precision faithfully and retain literal text identifiers."""
    frame = pd.read_csv(path, float_precision="round_trip")
    text = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in frame:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = text[column]
    return frame


def compare_runs(reference: Path, candidate: Path, *, profile: str) -> dict[str, object]:
    """Compare existing outputs without refitting, rerendering, or editing them."""
    if profile not in {"main", "full"}:
        raise ValueError(f"Unknown comparison profile: {profile}")
    specs = artifact_registry("main" if profile == "main" else None)
    if len(specs) != (10 if profile == "main" else 26):
        raise ValueError("Comparison requires the complete selected artifact registry")
    output_paths = sorted({path for spec in specs for path in spec.output_paths})
    panel_paths = sorted({path for spec in specs for path in spec.result_dependencies if path.endswith(".parquet")})
    metadata_paths = sorted({str(Path(path).parent / "metadata.json") for path in panel_paths})
    for root in (reference, candidate):
        for relative in (*output_paths, *panel_paths, *metadata_paths, "data/manifest.csv", "requirements.lock"):
            if not (root / relative).is_file():
                raise FileNotFoundError(f"Missing comparison input: {root / relative}")
    if (reference / "requirements.lock").read_bytes() != (candidate / "requirements.lock").read_bytes():
        raise ValueError("Comparison requires matching locked package versions")
    manifests = [pd.read_csv(root / "data/manifest.csv", dtype=str, keep_default_na=False) for root in (reference, candidate)]
    if profile == "main":
        manifests = [frame[frame["profile"].isin(["main", "shared"])].reset_index(drop=True) for frame in manifests]
    # All metadata is exact, including input hashes, date coverage, and sizes.
    compare_frames(*manifests, name="input manifests")
    for relative in metadata_paths:
        compare_run_metadata(reference / relative, candidate / relative)
    differences = {}
    counts = {"csv_tables": 0, "tex_tables": 0, "figure_pdfs": 0, "source_panels": 0}
    for relative in (*panel_paths, *output_paths):
        left, right = reference / relative, candidate / relative
        if left.suffix == ".parquet":
            differences[relative] = compare_frames(pd.read_parquet(left), pd.read_parquet(right), name=relative)
            counts["source_panels"] += 1
        elif left.suffix == ".csv":
            differences[relative] = compare_frames(
                read_comparison_csv(left), read_comparison_csv(right), name=relative,
            )
            counts["csv_tables"] += 1
        elif left.suffix == ".tex":
            compare_tex(left, right)
            counts["tex_tables"] += 1
        elif left.suffix == ".pdf":
            compare_pdf(left, right)
            counts["figure_pdfs"] += 1
        else:
            raise ValueError(f"Unsupported registered output: {relative}")
    return {
        "profile": profile, **counts, "absolute_tolerance": NUMERICAL_ATOL,
        "relative_tolerance": 0.0, "raster_dpi": RASTER_DPI,
        "maximum_absolute_difference": max(differences.values(), default=0.0),
        "numeric_differences": differences,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference program_v04 directory")
    parser.add_argument("candidate", type=Path, help="Candidate program_v04 directory")
    parser.add_argument("--profile", choices=("main", "full"), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(compare_runs(args.reference, args.candidate, profile=args.profile), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
