"""Small effect-boundary helpers for appendix artifact entry points."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from program_v04.replication.artifact_io import (
    load_parquet_dependency,
    write_artifact_metadata,
)
from program_v04.replication.artifacts import ArtifactSpec
from program_v04.replication.config import PACKAGE_ROOT, assert_canonical_generation_allowed
from program_v04.replication.paper import render_generic_latex
from program_v04.replication.provenance import atomic_write_text, validate_run_metadata


def load_dependencies(
    spec: ArtifactSpec,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    values: dict[str, object] = {}
    metadata: list[dict[str, object]] = []
    metadata_by_parent: dict[Path, dict[str, object]] = {}
    for relative in spec.result_dependencies:
        path = PACKAGE_ROOT / relative
        if relative.endswith(".parquet"):
            panel, record = load_parquet_dependency(relative)
            values[relative] = panel
            metadata_by_parent[path.parent] = record
            if record not in metadata:
                metadata.append(record)
        elif relative.endswith("metadata.json"):
            assert_canonical_generation_allowed()
            record = metadata_by_parent.get(path.parent)
            if record is None:
                if not path.is_file():
                    raise FileNotFoundError(f"Missing result metadata dependency: {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
                validate_run_metadata(record)
                metadata_by_parent[path.parent] = record
                metadata.append(record)
            values[relative] = record
    return values, metadata


def write_table(
    spec: ArtifactSpec,
    frame: pd.DataFrame,
    dependency_metadata: list[dict[str, object]],
    *,
    tex: str | None = None,
) -> None:
    assert_canonical_generation_allowed()
    outputs = {Path(relative).suffix: PACKAGE_ROOT / relative for relative in spec.output_paths}
    if set(outputs) != {".csv", ".tex"}:
        raise ValueError(f"{spec.artifact_id}: table must declare one CSV and one TeX output")
    run_ids = sorted({str(item["run_fingerprint"]) for item in dependency_metadata})
    provenance = f"dependency run fingerprints: {', '.join(run_ids)}"
    atomic_write_text(outputs[".csv"], frame.to_csv(index=False))
    atomic_write_text(
        outputs[".tex"],
        tex if tex is not None else render_generic_latex(frame, provenance_comment=provenance),
    )
    write_artifact_metadata(spec, dependency_metadata=dependency_metadata)
