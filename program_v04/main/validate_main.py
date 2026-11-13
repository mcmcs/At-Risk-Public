"""Validate all completed main-profile result and artifact contracts."""

from __future__ import annotations

import argparse
import importlib.util

import pandas as pd

from program_v04.replication.artifact_io import validate_artifact_outputs
from program_v04.replication.artifacts import artifact_registry, validate_artifact_registry
from program_v04.replication.config import FAMILY_REGISTRY, assert_canonical_generation_allowed
from program_v04.replication.data import (
    validate_forecast_panel,
    validate_importance_panel,
    validate_metric_panel,
)
from program_v04.replication.provenance import validate_result_bundle
from program_v04.replication.revised import RevisedFamilyRun
from program_v04.replication.parity import validate_featured_cross_family_parity
from program_v04.replication.section2 import load_section2_results


MAIN_FAMILIES = ("main", "extended", "probit_spread", "parameter_free")


def _load_family(family: str) -> RevisedFamilyRun:
    directory = FAMILY_REGISTRY[("revised", family)].result_path
    paths = [directory / "forecasts.parquet", directory / "metrics.parquet"]
    importance_path = directory / "importances.parquet"
    if family == "main":
        paths.append(importance_path)
    metadata = validate_result_bundle(paths, directory / "metadata.json")
    options = metadata.get("command_options")
    method = options.get("method") if isinstance(options, dict) else None
    if not isinstance(method, dict):
        raise ValueError(f"revised/{family} metadata has no method payload")
    run = RevisedFamilyRun(
        family=family,
        forecasts=pd.read_parquet(paths[0]),
        metrics=pd.read_parquet(paths[1]),
        importances=(pd.read_parquet(importance_path) if family == "main" else None),
        method_metadata=dict(method),
        events=(),
    )
    run.validate()
    return run


def validate_main() -> dict[str, int]:
    assert_canonical_generation_allowed()
    inventory = validate_artifact_registry(profile="main")
    main_specs = artifact_registry("main")
    for spec in main_specs:
        if importlib.util.find_spec(spec.generator_module) is None:
            raise ImportError(f"Missing main artifact generator: {spec.generator_module}")
    load_section2_results()
    runs = {family: _load_family(family) for family in MAIN_FAMILIES}
    validate_featured_cross_family_parity(
        runs["main"].forecasts,
        runs["extended"].forecasts,
        exercise="revised",
        data_variant="revised_final",
    )
    for spec in main_specs:
        validate_artifact_outputs(spec)
    return {
        "registered_artifacts": inventory["artifacts"],
        "main_artifacts": len(main_specs),
        "main_families": len(runs),
        "forecast_rows": sum(len(run.forecasts) for run in runs.values()),
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    summary = validate_main()
    print(
        "Main validation OK: "
        f"{summary['main_artifacts']} artifacts, {summary['main_families']} families, "
        f"{summary['forecast_rows']} forecast rows",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
