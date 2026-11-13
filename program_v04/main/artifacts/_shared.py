"""Small effect-boundary helpers for main artifact entry points."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from program_v04.replication.artifact_io import (
    load_parquet_dependency,
    write_artifact_metadata,
)
from program_v04.replication.artifacts import ArtifactSpec
from program_v04.replication.config import PACKAGE_ROOT
from program_v04.replication.paper import render_generic_latex
from program_v04.replication.provenance import atomic_write_text


def load_dependencies(
    spec: ArtifactSpec,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    panels: dict[str, pd.DataFrame] = {}
    metadata: list[dict[str, object]] = []
    for relative in spec.result_dependencies:
        if not relative.endswith(".parquet"):
            continue
        panel, record = load_parquet_dependency(relative)
        panels[relative] = panel
        metadata.append(record)
    return panels, metadata


def write_table(
    spec: ArtifactSpec,
    frame: pd.DataFrame,
    dependency_metadata: list[dict[str, object]],
) -> None:
    outputs = {Path(relative).suffix: PACKAGE_ROOT / relative for relative in spec.output_paths}
    if set(outputs) != {".csv", ".tex"}:
        raise ValueError(f"{spec.artifact_id}: table must declare one CSV and one TeX output")
    run_ids = sorted({str(item["run_fingerprint"]) for item in dependency_metadata})
    provenance = f"dependency run fingerprints: {', '.join(run_ids)}"
    atomic_write_text(outputs[".csv"], frame.to_csv(index=False))
    atomic_write_text(
        outputs[".tex"],
        render_generic_latex(frame, provenance_comment=provenance),
    )
    write_artifact_metadata(spec, dependency_metadata=dependency_metadata)
