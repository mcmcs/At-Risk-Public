"""Fingerprint-gated result reads and artifact-side provenance records."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .artifacts import ArtifactSpec
from .config import PACKAGE_ROOT, assert_canonical_generation_allowed
from .provenance import (
    atomic_write_json,
    fingerprint_file,
    locked_config_fingerprint,
    production_source_fingerprint,
    validate_result_dependency,
)


def load_parquet_dependency(relative_path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load one panel only after its family completion marker verifies it."""
    assert_canonical_generation_allowed()
    path = PACKAGE_ROOT / relative_path
    metadata_path = path.parent / "metadata.json"
    if not path.is_file():
        profile = "appendix" if "/appendix/" in f"/{relative_path}" else "main"
        raise FileNotFoundError(
            f"Missing result dependency {path}. Run: "
            f"python -m program_v04.{profile}.compute_{profile}"
        )
    metadata = validate_result_dependency(path, metadata_path)
    return pd.read_parquet(path), metadata


def artifact_metadata_path(spec: ArtifactSpec) -> Path:
    return PACKAGE_ROOT / "output" / spec.profile / "metadata" / f"{spec.artifact_id}.json"


def write_artifact_metadata(
    spec: ArtifactSpec,
    *,
    dependency_metadata: Iterable[dict[str, object]],
) -> Path:
    """Write an artifact completion marker after every declared output exists."""
    assert_canonical_generation_allowed()
    outputs = tuple(PACKAGE_ROOT / relative for relative in spec.output_paths)
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot complete {spec.artifact_id}; missing outputs={missing}")
    runs = sorted(
        {
            str(metadata["run_fingerprint"])
            for metadata in dependency_metadata
            if "run_fingerprint" in metadata
        }
    )
    payload = {
        "schema_version": 1,
        "complete": True,
        "artifact_id": spec.artifact_id,
        "draft_label": spec.draft_label,
        "generator_module": spec.generator_module,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": locked_config_fingerprint(),
        "source_sha256": production_source_fingerprint()["sha256"],
        "dependency_run_fingerprints": runs,
        "dependencies": [
            asdict(fingerprint_file(PACKAGE_ROOT / relative, relative_to=PACKAGE_ROOT))
            for relative in spec.result_dependencies
            if (PACKAGE_ROOT / relative).is_file()
        ],
        "outputs": [asdict(fingerprint_file(path, relative_to=PACKAGE_ROOT)) for path in outputs],
    }
    path = artifact_metadata_path(spec)
    atomic_write_json(path, payload)
    return path


def validate_artifact_outputs(spec: ArtifactSpec) -> dict[str, object]:
    """Validate one artifact against its live dependencies and outputs."""
    assert_canonical_generation_allowed()
    path = artifact_metadata_path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact metadata is missing: {path}")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("complete") is not True
        or payload.get("artifact_id") != spec.artifact_id
        or payload.get("draft_label") != spec.draft_label
        or payload.get("generator_module") != spec.generator_module
    ):
        raise ValueError(f"Invalid artifact completion marker: {path}")
    if payload.get("config_sha256") != locked_config_fingerprint():
        raise ValueError(f"Artifact was produced by a different locked configuration: {path}")
    if payload.get("source_sha256") != production_source_fingerprint()["sha256"]:
        raise ValueError(f"Artifact was produced by different production source: {path}")

    expected_dependencies = {
        item["path"]: item for item in payload.get("dependencies", [])
    }
    if set(expected_dependencies) != set(spec.result_dependencies):
        raise ValueError(
            f"Artifact dependency membership mismatch for {spec.artifact_id}: "
            f"declared={sorted(expected_dependencies)}, "
            f"required={sorted(spec.result_dependencies)}"
        )
    current_runs: set[str] = set()
    for relative in spec.result_dependencies:
        dependency = PACKAGE_ROOT / relative
        actual = asdict(fingerprint_file(dependency, relative_to=PACKAGE_ROOT))
        if expected_dependencies.get(relative) != actual:
            raise ValueError(f"Artifact dependency fingerprint mismatch: {dependency}")
        if dependency.suffix == ".parquet":
            metadata = validate_result_dependency(
                dependency,
                dependency.parent / "metadata.json",
            )
            current_runs.add(str(metadata["run_fingerprint"]))
    if payload.get("dependency_run_fingerprints") != sorted(current_runs):
        raise ValueError(
            f"Artifact dependency run fingerprints are stale: {spec.artifact_id}"
        )

    expected = {item["path"]: item for item in payload.get("outputs", [])}
    if set(expected) != set(spec.output_paths):
        raise ValueError(
            f"Artifact output membership mismatch for {spec.artifact_id}: "
            f"declared={sorted(expected)}, required={sorted(spec.output_paths)}"
        )
    for relative in spec.output_paths:
        output = PACKAGE_ROOT / relative
        actual = asdict(fingerprint_file(output, relative_to=PACKAGE_ROOT))
        if expected.get(relative) != actual:
            raise ValueError(f"Artifact output fingerprint mismatch: {output}")
    return payload
