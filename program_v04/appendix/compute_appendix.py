"""Compute every expensive appendix-profile result from declared inputs."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from program_v04.replication.appendix_paper import validate_derived_table
from program_v04.appendix.compute_decision_wealth import compute_decision_wealth
from program_v04.replication.config import (
    FAMILY_REGISTRY,
    GLOBAL_SEED,
    HORIZONS,
    PACKAGE_ROOT,
    assert_canonical_generation_allowed,
)
from program_v04.replication.data import (
    validate_forecast_panel,
    validate_importance_panel,
    validate_metric_panel,
)
from program_v04.replication.preparation import (
    RevisedData,
    prepare_revised_from_declared_inputs,
)
from program_v04.replication.protocols import (
    MainMethodProtocol,
    canonical_main_protocol,
    canonical_realtime_protocol,
)
from program_v04.replication.provenance import (
    RunEvent,
    fingerprint_file,
    validate_result_bundle,
    write_result_bundle_with_metadata,
)
from program_v04.replication.realtime import (
    RealtimeFamilyRun,
    derive_realtime_main,
    resolve_vintage_inputs,
    run_realtime_family,
)
from program_v04.replication.revised import RevisedFamilyRun, run_revised_family
from program_v04.replication.robustness import (
    ASYMMETRIC_LOSS_COST_RATIOS,
    SAMPLING_UNCERTAINTY_ATTEMPT_CAP,
    SAMPLING_UNCERTAINTY_BLOCK_LENGTHS,
    SAMPLING_UNCERTAINTY_CONFIDENCE_LEVEL,
    SAMPLING_UNCERTAINTY_SEED,
    SAMPLING_UNCERTAINTY_VALID_REPLICATIONS,
    build_asymmetric_loss_table,
    build_cumulative_brier_data,
    build_pre2020_robustness_table,
    build_sampling_uncertainty_table,
)
from program_v04.replication.simulation import (
    ReplicationSelectionPolicy,
    SimulationBatch,
    build_diagnostics_table,
    build_main_table,
    canonical_simulation_protocol,
    frozen_step0_config,
    run_simulation_batch,
)


APPENDIX_REVISED_FAMILIES = (
    "threshold",
    "moving_average",
    "nolag",
    "parsimonious",
    "alternative_subsets",
    "expanding_threshold",
)
SHARED_FAMILIES = ("main", "extended")


def _family_paths(family: str) -> tuple[Path, ...]:
    directory = FAMILY_REGISTRY[("revised", family)].result_path
    names = ["forecasts.parquet", "metrics.parquet"]
    if family == "main":
        names.append("importances.parquet")
    return tuple(directory / name for name in names)


def _write_revised_family(
    run: RevisedFamilyRun,
    *,
    input_mode: str,
    command: tuple[str, ...],
    input_paths: dict[str, Path],
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, object]:
    directory = FAMILY_REGISTRY[("revised", run.family)].result_path
    frames = {
        directory / "forecasts.parquet": run.forecasts,
        directory / "metrics.parquet": run.metrics,
    }
    validators = {
        directory / "forecasts.parquet": validate_forecast_panel,
        directory / "metrics.parquet": validate_metric_panel,
    }
    if run.importances is not None:
        frames[directory / "importances.parquet"] = run.importances
        validators[directory / "importances.parquet"] = validate_importance_panel
    return write_result_bundle_with_metadata(
        result_frames=frames,
        metadata_path=directory / "metadata.json",
        validators=validators,
        metadata_arguments={
            "profile": "appendix",
            "command": command,
            "command_options": {
                "input_mode": input_mode,
                "family": run.family,
                "method": run.method_metadata,
            },
            "input_fingerprints": [
                fingerprint_file(path, relative_to=PACKAGE_ROOT.parent)
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


def _load_verified_revised_family(
    family: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    paths = _family_paths(family)
    metadata = validate_result_bundle(paths, paths[0].parent / "metadata.json")
    frames = {path.stem: pd.read_parquet(path) for path in paths}
    validate_forecast_panel(frames["forecasts"], surface=f"reused {family} forecasts")
    validate_metric_panel(frames["metrics"], surface=f"reused {family} metrics")
    if "importances" in frames:
        validate_importance_panel(
            frames["importances"], surface=f"reused {family} importances"
        )
    return frames, metadata


def _shared_families(
    revised: RevisedData,
    protocol: MainMethodProtocol,
    *,
    input_mode: str,
    command: tuple[str, ...],
) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, object]],
    list[dict[str, object]],
]:
    frames_by_family: dict[str, dict[str, pd.DataFrame]] = {}
    metadata_by_family: dict[str, dict[str, object]] = {}
    reuse_records: list[dict[str, object]] = []
    for family in SHARED_FAMILIES:
        try:
            frames, metadata = _load_verified_revised_family(family)
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"Shared revised/{family} is absent or stale; recomputing "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            started_at = datetime.now(UTC)
            run = run_revised_family(revised, family, protocol, horizons=HORIZONS)
            ended_at = datetime.now(UTC)
            metadata = _write_revised_family(
                run,
                input_mode=input_mode,
                command=command,
                input_paths=dict(revised.input_paths),
                started_at=started_at,
                ended_at=ended_at,
            )
            frames = {
                "forecasts": run.forecasts,
                "metrics": run.metrics,
                **({"importances": run.importances} if run.importances is not None else {}),
            }
            reused = False
        else:
            print(
                f"Reusing fingerprint-verified revised/{family} "
                f"run {metadata['run_fingerprint']}",
                flush=True,
            )
            reused = True
        frames_by_family[family] = frames
        metadata_by_family[family] = metadata
        reuse_records.append(
            {
                "family": family,
                "reused": reused,
                "run_fingerprint": metadata["run_fingerprint"],
            }
        )
    return frames_by_family, metadata_by_family, reuse_records


def _write_realtime_family(
    run: RealtimeFamilyRun,
    *,
    input_mode: str,
    command: tuple[str, ...],
    input_paths: tuple[Path, ...],
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, object]:
    directory = FAMILY_REGISTRY[("realtime", run.family)].result_path
    frames = {
        directory / "forecasts.parquet": run.forecasts,
        directory / "metrics.parquet": run.metrics,
    }
    return write_result_bundle_with_metadata(
        result_frames=frames,
        metadata_path=directory / "metadata.json",
        validators={
            directory / "forecasts.parquet": validate_forecast_panel,
            directory / "metrics.parquet": validate_metric_panel,
        },
        metadata_arguments={
            "profile": "appendix",
            "command": command,
            "command_options": {
                "input_mode": input_mode,
                "family": run.family,
                "method": run.method_metadata,
            },
            "input_fingerprints": [
                fingerprint_file(path, relative_to=PACKAGE_ROOT.parent)
                for path in input_paths
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


def _write_derived(
    *,
    main_forecasts: pd.DataFrame,
    extended_forecasts: pd.DataFrame,
    reuse_records: list[dict[str, object]],
    command: tuple[str, ...],
) -> dict[str, object]:
    started_at = datetime.now(UTC)
    bootstrap = build_sampling_uncertainty_table(
        main_forecasts,
        extended_forecasts,
        exercise="revised",
        data_variant="revised_final",
    )
    pre2020 = build_pre2020_robustness_table(
        main_forecasts,
        extended_forecasts,
        exercise="revised",
        data_variant="revised_final",
    )
    cumulative = build_cumulative_brier_data(
        main_forecasts,
        extended_forecasts,
        exercise="revised",
        data_variant="revised_final",
    )
    asymmetric = build_asymmetric_loss_table(
        main_forecasts,
        extended_forecasts,
        exercise="revised",
        data_variant="revised_final",
    )
    ended_at = datetime.now(UTC)
    directory = PACKAGE_ROOT / "results" / "appendix" / "derived"
    frames = {
        directory / "bootstrap.parquet": bootstrap,
        directory / "pre2020.parquet": pre2020,
        directory / "cumulative_brier.parquet": cumulative,
        directory / "asymmetric_loss.parquet": asymmetric,
    }
    validators = {
        path: (
            lambda frame, _surface=path.stem: validate_derived_table(
                frame, surface=f"appendix derived {_surface}"
            )
        )
        for path in frames
    }
    source_files = [
        path
        for family in SHARED_FAMILIES
        for path in _family_paths(family)
        if path.name in {"forecasts.parquet", "metrics.parquet"}
    ]
    return write_result_bundle_with_metadata(
        result_frames=frames,
        metadata_path=directory / "metadata.json",
        validators=validators,
        metadata_arguments={
            "profile": "appendix",
            "command": command,
            "command_options": {
                "family": "derived",
                "method": {
                    "sampling_uncertainty": {
                        "confidence_level": SAMPLING_UNCERTAINTY_CONFIDENCE_LEVEL,
                        "seed": SAMPLING_UNCERTAINTY_SEED,
                        "valid_replications": SAMPLING_UNCERTAINTY_VALID_REPLICATIONS,
                        "attempt_cap": SAMPLING_UNCERTAINTY_ATTEMPT_CAP,
                        "block_lengths": list(SAMPLING_UNCERTAINTY_BLOCK_LENGTHS),
                    },
                    "asymmetric_loss_cost_ratios": list(
                        ASYMMETRIC_LOSS_COST_RATIOS
                    ),
                    "shared_reuse": reuse_records,
                },
            },
            "input_fingerprints": [
                fingerprint_file(path, relative_to=PACKAGE_ROOT)
                for path in source_files
            ],
            "requested_families": ["bootstrap", "pre2020", "cumulative_brier", "asymmetric_loss"],
            "completed_families": ["bootstrap", "pre2020", "cumulative_brier", "asymmetric_loss"],
            "requested_horizons": list(HORIZONS),
            "completed_horizons": list(HORIZONS),
            "seeds": {"bootstrap": SAMPLING_UNCERTAINTY_SEED},
            "start_time": started_at,
            "end_time": ended_at,
            "events": (),
        },
    )


def _write_simulation(command: tuple[str, ...]) -> dict[str, object]:
    protocol = canonical_simulation_protocol()
    config = frozen_step0_config()
    requested = config.b_default
    candidate_cap = (
        requested
        if protocol.selection_policy is ReplicationSelectionPolicy.FIXED_CANDIDATE_SET
        else requested * 10
    )
    started_at = datetime.now(UTC)
    batch = run_simulation_batch(
        requested_replications=requested,
        candidate_cap=candidate_cap,
        config=config,
        protocol=protocol,
        pilot=False,
    )
    ended_at = datetime.now(UTC)
    metrics = batch.metrics_panel()
    diagnostics = batch.diagnostics_panel()
    build_main_table(metrics, config=config)
    build_diagnostics_table(diagnostics, config=config)
    directory = PACKAGE_ROOT / "results" / "appendix" / "simulation"
    frames = {
        directory / "metrics.parquet": metrics,
        directory / "diagnostics.parquet": diagnostics,
    }
    warning_counts = Counter(
        (warning.category, warning.message)
        for outcome in batch.outcomes
        for warning in outcome.fit_warnings
    )
    events = [
        RunEvent(
            "simulation_warning",
            category,
            count,
            message,
        )
        for (category, message), count in sorted(warning_counts.items())
    ]
    if batch.rejected:
        events.append(
            RunEvent(
                "simulation_selection",
                "rejected_candidate",
                len(batch.rejected),
                "Candidate seeds rejected under the selected explicit class-degeneracy policy",
            )
        )

    def validate_metrics(frame: pd.DataFrame) -> None:
        build_main_table(frame, config=config)

    def validate_diagnostics(frame: pd.DataFrame) -> None:
        build_diagnostics_table(frame, config=config)

    return write_result_bundle_with_metadata(
        result_frames=frames,
        metadata_path=directory / "metadata.json",
        validators={
            directory / "metrics.parquet": validate_metrics,
            directory / "diagnostics.parquet": validate_diagnostics,
        },
        metadata_arguments={
            "profile": "appendix",
            "command": command,
            "command_options": {
                "family": "simulation",
                "method": {
                    "batch": batch.metadata(),
                    "config": {
                        "fingerprint": config.fingerprint(
                            c_value=protocol.estimator.c_value
                        ),
                        "horizons": list(config.horizons),
                        "replications": requested,
                    },
                },
            },
            "input_fingerprints": [],
            "requested_families": ["simulation"],
            "completed_families": ["simulation"],
            "requested_horizons": list(config.horizons),
            "completed_horizons": list(config.horizons),
            "seeds": {"simulation_base": config.base_seed},
            "start_time": started_at,
            "end_time": ended_at,
            "events": events,
        },
    )


def _compute_realtime(
    revised: RevisedData,
    *,
    input_mode: Literal["public", "development"],
    command: tuple[str, ...],
) -> None:
    """Compute the complete paired family and derive the displayed subset."""
    vintage_paths = resolve_vintage_inputs(mode=input_mode)
    realtime_started = datetime.now(UTC)
    realtime_extended = run_realtime_family(
        revised,
        "realtime_extended",
        canonical_realtime_protocol(),
        vintage_paths=vintage_paths,
        horizons=HORIZONS,
    )
    realtime_ended = datetime.now(UTC)
    realtime_inputs = tuple(revised.input_paths.values()) + tuple(vintage_paths.values())
    _write_realtime_family(
        realtime_extended,
        input_mode=input_mode,
        command=command,
        input_paths=realtime_inputs,
        started_at=realtime_started,
        ended_at=realtime_ended,
    )
    realtime_main = derive_realtime_main(realtime_extended)
    _write_realtime_family(
        realtime_main,
        input_mode=input_mode,
        command=command,
        input_paths=realtime_inputs,
        started_at=realtime_started,
        ended_at=realtime_ended,
    )
    print(
        f"Computed realtime_extended and derived realtime_main: "
        f"{len(realtime_extended.forecasts)} + {len(realtime_main.forecasts)} rows",
        flush=True,
    )



def compute_realtime(*, input_mode: Literal["public", "development"]) -> None:
    """Recompute only the real-time exercise from declared inputs."""
    assert_canonical_generation_allowed()
    protocol = canonical_realtime_protocol()
    revised = prepare_revised_from_declared_inputs(
        code7_rule=protocol.main.code7_rule, mode=input_mode,
    )
    _compute_realtime(revised, input_mode=input_mode, command=tuple(sys.argv))


def compute_appendix(*, input_mode: Literal["public", "development"]) -> None:
    """Compute shared prerequisites and every appendix-owned result family."""
    assert_canonical_generation_allowed()
    main_protocol = canonical_main_protocol()
    command = tuple(sys.argv)
    revised = prepare_revised_from_declared_inputs(
        code7_rule=main_protocol.code7_rule,
        mode=input_mode,
    )

    shared, _, reuse_records = _shared_families(
        revised,
        main_protocol,
        input_mode=input_mode,
        command=command,
    )
    for family in APPENDIX_REVISED_FAMILIES:
        started_at = datetime.now(UTC)
        timer = time.monotonic()
        run = run_revised_family(revised, family, main_protocol, horizons=HORIZONS)
        ended_at = datetime.now(UTC)
        _write_revised_family(
            run,
            input_mode=input_mode,
            command=command,
            input_paths=dict(revised.input_paths),
            started_at=started_at,
            ended_at=ended_at,
        )
        print(
            f"Computed revised/{family}: {len(run.forecasts)} forecasts "
            f"in {time.monotonic() - timer:.1f}s",
            flush=True,
        )

    _compute_realtime(revised, input_mode=input_mode, command=command)

    _write_derived(
        main_forecasts=shared["main"]["forecasts"],
        extended_forecasts=shared["extended"]["forecasts"],
        reuse_records=reuse_records,
        command=command,
    )
    print("Computed four appendix derived-analysis panels", flush=True)
    compute_decision_wealth(command)
    print("Computed h=3 decision-wealth panels", flush=True)
    _write_simulation(command)
    print("Computed appendix simulation panels", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--simulation-only",
        action="store_true",
        help="Recompute only the 500 accepted simulation replications for Table 20.",
    )
    scope.add_argument(
        "--realtime-only",
        action="store_true",
        help="Recompute only the paired revised/vintage families for Table 15.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("public", "development"),
        default="public",
        help=(
            "Resolve checksummed public inputs (default) or immutable repository "
            "development snapshots."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.realtime_only:
        compute_realtime(input_mode=arguments.input_mode)
    elif arguments.simulation_only:
        assert_canonical_generation_allowed()
        _write_simulation(tuple(sys.argv))
        print("Computed appendix simulation panels", flush=True)
    else:
        compute_appendix(input_mode=arguments.input_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
