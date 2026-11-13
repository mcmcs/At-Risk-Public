"""Deterministic fingerprints, run metadata, and atomic result writes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .config import (
    GLOBAL_SEED,
    HORIZONS,
    LAGS,
    METHOD_DECISIONS,
    PACKAGE_ROOT,
    RESULT_FAMILIES,
    assert_canonical_generation_allowed,
)
from .data import sha256_file


PROVENANCE_SCHEMA_VERSION = 1
LOCKED_DISTRIBUTIONS = (
    "contourpy",
    "cycler",
    "fonttools",
    "joblib",
    "kiwisolver",
    "matplotlib",
    "numpy",
    "packaging",
    "pandas",
    "patsy",
    "pillow",
    "pip",
    "pyarrow",
    "pyparsing",
    "python-dateutil",
    "scikit-learn",
    "scipy",
    "seaborn",
    "setuptools",
    "six",
    "statsmodels",
    "threadpoolctl",
    "tqdm",
    "wheel",
    "xgboost",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value with stable key/order semantics."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    sha256: str
    size_bytes: int


def fingerprint_file(path: Path, *, relative_to: Path | None = None) -> FileFingerprint:
    resolved = path.resolve()
    display = str(resolved.relative_to(relative_to.resolve())) if relative_to else str(resolved)
    return FileFingerprint(display, sha256_file(resolved), resolved.stat().st_size)


def fingerprint_files(paths: Iterable[Path], *, relative_to: Path | None = None) -> tuple[FileFingerprint, ...]:
    return tuple(
        fingerprint_file(path, relative_to=relative_to)
        for path in sorted({Path(path).resolve() for path in paths})
    )


def production_source_fingerprint() -> dict[str, Any]:
    paths = (
        *PACKAGE_ROOT.glob("*.py"),
        *(PACKAGE_ROOT / "replication").glob("*.py"),
        *(PACKAGE_ROOT / "main").rglob("*.py"),
        *(PACKAGE_ROOT / "appendix").rglob("*.py"),
    )
    files = fingerprint_files(paths, relative_to=PACKAGE_ROOT.parent)
    payload = [asdict(item) for item in files]
    return {"sha256": stable_sha256(payload), "files": payload}


def locked_config_payload() -> dict[str, Any]:
    return {
        "global_seed": GLOBAL_SEED,
        "horizons": HORIZONS,
        "lags": LAGS,
        "method_decisions": [asdict(decision) for decision in METHOD_DECISIONS],
        "result_families": [asdict(spec) for spec in RESULT_FAMILIES],
    }


def locked_config_fingerprint() -> str:
    return stable_sha256(locked_config_payload())


def environment_record() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for name in LOCKED_DISTRIBUTIONS:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "missing"
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dependencies": dependencies,
    }


@dataclass(frozen=True)
class RunEvent:
    category: str
    code: str
    count: int
    detail: str

    def validate(self) -> None:
        if self.count < 0:
            raise ValueError(f"Run event count cannot be negative: {self}")
        if not self.category or not self.code or not self.detail:
            raise ValueError(f"Run event fields must be nonempty: {self}")


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    for field in ("forecast_date", "origin_date", "target_date"):
        if field in frame.columns and not frame.empty:
            dates = pd.to_datetime(frame[field], errors="coerce").dropna()
            summary[f"{field}_min"] = str(dates.min().date()) if not dates.empty else None
            summary[f"{field}_max"] = str(dates.max().date()) if not dates.empty else None
    if "actual" in frame.columns:
        actual = pd.to_numeric(frame["actual"], errors="coerce")
        summary["positive_actual_rows"] = int((actual == 1).sum())
        summary["negative_actual_rows"] = int((actual == 0).sum())
    for field in ("exercise", "data_variant", "forecast_family", "horizon", "predictor_key", "model_name"):
        if field in frame.columns:
            values = frame[field].drop_duplicates().sort_values().tolist()
            summary[f"{field}_values"] = values
    return summary


def build_run_metadata(
    *,
    profile: str,
    command: Sequence[str],
    command_options: Mapping[str, Any],
    input_fingerprints: Sequence[FileFingerprint],
    requested_families: Sequence[str],
    completed_families: Sequence[str],
    requested_horizons: Sequence[int],
    completed_horizons: Sequence[int],
    seeds: Mapping[str, int],
    start_time: datetime,
    end_time: datetime,
    result_files: Sequence[FileFingerprint],
    result_summaries: Mapping[str, Mapping[str, Any]],
    events: Sequence[RunEvent] = (),
) -> dict[str, Any]:
    if start_time.tzinfo is None or end_time.tzinfo is None:
        raise ValueError("Run timestamps must be timezone-aware")
    if end_time < start_time:
        raise ValueError("Run end time precedes start time")
    for event in events:
        event.validate()
    source = production_source_fingerprint()
    metadata = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "complete": True,
        "profile": profile,
        "command": list(command),
        "command_options": dict(command_options),
        "config_sha256": locked_config_fingerprint(),
        "source_sha256": source["sha256"],
        "source_files": source["files"],
        "input_files": [asdict(item) for item in input_fingerprints],
        "environment": environment_record(),
        "seeds": dict(seeds),
        "requested_families": list(requested_families),
        "completed_families": list(completed_families),
        "requested_horizons": [int(value) for value in requested_horizons],
        "completed_horizons": [int(value) for value in completed_horizons],
        "started_at_utc": start_time.astimezone(UTC).isoformat(),
        "ended_at_utc": end_time.astimezone(UTC).isoformat(),
        "runtime_seconds": (end_time - start_time).total_seconds(),
        "result_files": [asdict(item) for item in result_files],
        "result_summaries": {key: dict(value) for key, value in result_summaries.items()},
        "events": [asdict(event) for event in events],
    }
    metadata["run_fingerprint"] = stable_sha256(metadata)
    return metadata


def validate_run_metadata(metadata: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "complete",
        "profile",
        "command",
        "command_options",
        "config_sha256",
        "source_sha256",
        "source_files",
        "input_files",
        "environment",
        "seeds",
        "requested_families",
        "completed_families",
        "requested_horizons",
        "completed_horizons",
        "started_at_utc",
        "ended_at_utc",
        "runtime_seconds",
        "result_files",
        "result_summaries",
        "events",
        "run_fingerprint",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Run metadata is missing fields: {missing}")
    if metadata["schema_version"] != PROVENANCE_SCHEMA_VERSION or metadata["complete"] is not True:
        raise ValueError("Run metadata is not a complete supported provenance record")
    unsigned = dict(metadata)
    fingerprint = unsigned.pop("run_fingerprint")
    if stable_sha256(unsigned) != fingerprint:
        raise ValueError("Run metadata fingerprint mismatch")
    if list(metadata["requested_families"]) != list(metadata["completed_families"]):
        raise ValueError("Partial family run cannot be marked complete")
    if list(metadata["requested_horizons"]) != list(metadata["completed_horizons"]):
        raise ValueError("Partial horizon run cannot be marked complete")


@contextmanager
def atomic_output_path(target: Path) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically replace target on success."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(target: Path, payload: bytes) -> None:
    with atomic_output_path(target) as temporary:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def atomic_write_text(target: Path, text: str) -> None:
    atomic_write_bytes(target, text.encode("utf-8"))


def atomic_write_json(target: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    atomic_write_text(target, serialized)


def atomic_write_parquet(target: Path, frame: pd.DataFrame) -> None:
    with atomic_output_path(target) as temporary:
        frame.to_parquet(temporary, index=False)


def write_result_with_metadata(
    *,
    result_path: Path,
    metadata_path: Path,
    frame: pd.DataFrame,
    validate_frame: Callable[[pd.DataFrame], None],
    metadata_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a validated panel first and a complete-marker metadata file last.

    An interruption can leave a panel without metadata, but consumers reject
    that state; it cannot masquerade as a valid completed result bundle.
    """
    assert_canonical_generation_allowed()
    validate_frame(frame)
    atomic_write_parquet(result_path, frame)
    result_fingerprint = fingerprint_file(result_path, relative_to=PACKAGE_ROOT)
    arguments = dict(metadata_arguments)
    arguments["result_files"] = [result_fingerprint]
    arguments["result_summaries"] = {str(result_path.name): summarize_frame(frame)}
    metadata = build_run_metadata(**arguments)
    validate_run_metadata(metadata)
    atomic_write_json(metadata_path, metadata)
    return metadata


def write_result_bundle_with_metadata(
    *,
    result_frames: Mapping[Path, pd.DataFrame],
    metadata_path: Path,
    validators: Mapping[Path, Callable[[pd.DataFrame], None]],
    metadata_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a complete multi-panel family bundle and its marker last.

    Every frame is validated before the first write.  The metadata file is the
    sole completion marker, so an interrupted sequence of atomic panel writes
    is always rejected by consumers as an incomplete bundle.
    """
    assert_canonical_generation_allowed()
    if not result_frames:
        raise ValueError("A result bundle must contain at least one panel")
    if set(result_frames) != set(validators):
        raise ValueError("Result bundle frames and validators have different paths")
    ordered_paths = tuple(sorted((Path(path) for path in result_frames), key=str))
    if metadata_path in ordered_paths:
        raise ValueError("Result metadata path cannot also be a panel path")
    for path in ordered_paths:
        validators[path](result_frames[path])
    for path in ordered_paths:
        atomic_write_parquet(path, result_frames[path])
    fingerprints = [
        fingerprint_file(path, relative_to=PACKAGE_ROOT) for path in ordered_paths
    ]
    arguments = dict(metadata_arguments)
    arguments["result_files"] = fingerprints
    arguments["result_summaries"] = {
        path.name: summarize_frame(result_frames[path]) for path in ordered_paths
    }
    metadata = build_run_metadata(**arguments)
    validate_run_metadata(metadata)
    atomic_write_json(metadata_path, metadata)
    return metadata


def validate_result_bundle(
    result_paths: Sequence[Path],
    metadata_path: Path,
) -> dict[str, Any]:
    """Validate every declared file in one completed result bundle."""
    assert_canonical_generation_allowed()
    paths = tuple(Path(path) for path in result_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Result dependency paths must be nonempty and unique")
    if not metadata_path.is_file() or any(not path.is_file() for path in paths):
        missing = [str(path) for path in (*paths, metadata_path) if not path.is_file()]
        raise FileNotFoundError(f"Incomplete result bundle; missing={missing}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_run_metadata(metadata)
    expected = {item["path"]: item for item in metadata["result_files"]}
    requested_relative = {
        str(path.resolve().relative_to(PACKAGE_ROOT.resolve())) for path in paths
    }
    if requested_relative != set(expected):
        raise ValueError(
            "Result bundle membership differs from metadata: "
            f"requested={sorted(requested_relative)}, declared={sorted(expected)}"
        )
    for path in paths:
        relative = str(path.resolve().relative_to(PACKAGE_ROOT.resolve()))
        actual = fingerprint_file(path, relative_to=PACKAGE_ROOT)
        if asdict(actual) != expected[relative]:
            raise ValueError(f"Result fingerprint mismatch for {path}")
    if metadata["config_sha256"] != locked_config_fingerprint():
        raise ValueError("Result metadata was produced by a different locked configuration")
    if metadata["source_sha256"] != production_source_fingerprint()["sha256"]:
        raise ValueError("Result metadata was produced by different production source")
    return metadata


def validate_result_dependency(result_path: Path, metadata_path: Path) -> dict[str, Any]:
    assert_canonical_generation_allowed()
    if not result_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Incomplete result dependency: require both {result_path} and {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_run_metadata(metadata)
    expected = {item["path"]: item for item in metadata["result_files"]}
    relative = str(result_path.resolve().relative_to(PACKAGE_ROOT.resolve()))
    if relative not in expected:
        raise ValueError(f"Metadata does not declare result file {relative}")
    actual = fingerprint_file(result_path, relative_to=PACKAGE_ROOT)
    if asdict(actual) != expected[relative]:
        raise ValueError(f"Result fingerprint mismatch for {result_path}")
    if metadata["config_sha256"] != locked_config_fingerprint():
        raise ValueError("Result metadata was produced by a different locked configuration")
    if metadata["source_sha256"] != production_source_fingerprint()["sha256"]:
        raise ValueError("Result metadata was produced by different production source")
    return metadata


def monotonic_timer() -> tuple[datetime, float]:
    """Return a timezone-aware wall-clock start and monotonic timer."""
    return datetime.now(UTC), time.monotonic()
