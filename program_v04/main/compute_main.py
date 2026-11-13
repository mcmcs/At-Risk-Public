"""Compute every expensive main-profile result from declared raw inputs."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from program_v04.replication.config import (
    FAMILY_REGISTRY,
    GLOBAL_SEED,
    HORIZONS,
    assert_canonical_generation_allowed,
)
from program_v04.replication.data import (
    validate_forecast_panel,
    validate_importance_panel,
    validate_metric_panel,
)
from program_v04.replication.preparation import prepare_revised_from_declared_inputs
from program_v04.replication.protocols import canonical_main_protocol
from program_v04.replication.provenance import (
    fingerprint_file,
    write_result_bundle_with_metadata,
)
from program_v04.replication.revised import RevisedFamilyRun, run_revised_family
from program_v04.replication.section2 import (
    load_section2_raw,
    resolve_section2_inputs,
    write_section2_results,
    build_section2_data,
)


MAIN_FAMILIES = ("main", "extended", "probit_spread", "parameter_free")


def _write_family(
    run: RevisedFamilyRun,
    *,
    input_mode: str,
    command: tuple[str, ...],
    input_paths: dict[str, Path],
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, object]:
    result_directory = FAMILY_REGISTRY[("revised", run.family)].result_path
    frames = {
        result_directory / "forecasts.parquet": run.forecasts,
        result_directory / "metrics.parquet": run.metrics,
    }
    validators = {
        result_directory / "forecasts.parquet": validate_forecast_panel,
        result_directory / "metrics.parquet": validate_metric_panel,
    }
    if run.importances is not None:
        frames[result_directory / "importances.parquet"] = run.importances
        validators[result_directory / "importances.parquet"] = validate_importance_panel
    return write_result_bundle_with_metadata(
        result_frames=frames,
        metadata_path=result_directory / "metadata.json",
        validators=validators,
        metadata_arguments={
            "profile": "main",
            "command": command,
            "command_options": {
                "input_mode": input_mode,
                "family": run.family,
                "method": run.method_metadata,
            },
            "input_fingerprints": [
                fingerprint_file(path, relative_to=Path(__file__).resolve().parents[2])
                for path in input_paths.values()
            ],
            "requested_families": [run.family],
            "completed_families": [run.family],
            "requested_horizons": list(HORIZONS),
            "completed_horizons": list(HORIZONS),
            "seeds": {"global": GLOBAL_SEED},
            "start_time": started_at,
            "end_time": ended_at,
            "events": run.events,
        },
    )


def compute_main(*, input_mode: Literal["public", "development"]) -> None:
    """Run Section 2 plus the four complete main-owned/shared families."""
    assert_canonical_generation_allowed()
    protocol = canonical_main_protocol()
    command = tuple(sys.argv)

    section_paths = resolve_section2_inputs(mode=input_mode)
    section_started = datetime.now(UTC)
    section_data = build_section2_data(*load_section2_raw(section_paths))
    section_ended = datetime.now(UTC)
    write_section2_results(
        section_data,
        input_paths=section_paths,
        command=command,
        input_mode=input_mode,
        started_at=section_started,
        ended_at=section_ended,
    )
    print("Computed main Section 2 result bundle", flush=True)

    revised = prepare_revised_from_declared_inputs(
        code7_rule=protocol.code7_rule,
        mode=input_mode,
    )
    for family in MAIN_FAMILIES:
        started_at = datetime.now(UTC)
        started_timer = time.monotonic()
        run = run_revised_family(
            revised,
            family,
            protocol,
            horizons=HORIZONS,
        )
        ended_at = datetime.now(UTC)
        _write_family(
            run,
            input_mode=input_mode,
            command=command,
            input_paths=dict(revised.input_paths),
            started_at=started_at,
            ended_at=ended_at,
        )
        print(
            f"Computed revised/{family}: {len(run.forecasts)} forecasts "
            f"in {time.monotonic() - started_timer:.1f}s",
            flush=True,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-mode",
        choices=("public", "development"),
        default="public",
        help=(
            "Resolve checksummed files at public package paths (default) or "
            "the immutable repository development snapshots."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    compute_main(input_mode=args.input_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
