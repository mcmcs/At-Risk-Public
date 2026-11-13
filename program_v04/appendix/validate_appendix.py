"""Validate all completed appendix-profile result and artifact contracts."""

from __future__ import annotations

import argparse
import importlib.util
import pandas as pd

from program_v04.appendix.compute_appendix import APPENDIX_REVISED_FAMILIES
from program_v04.replication.appendix_paper import (
    load_documentary_rows,
    validate_derived_table,
)
from program_v04.replication.artifact_io import validate_artifact_outputs
from program_v04.replication.artifacts import artifact_registry, validate_artifact_registry
from program_v04.replication.config import (
    FAMILY_REGISTRY,
    PACKAGE_ROOT,
    assert_canonical_generation_allowed,
)
from program_v04.replication.provenance import validate_result_bundle
from program_v04.replication.decision_wealth import validate_decision_wealth
from program_v04.replication.realtime import RealtimeFamilyRun
from program_v04.replication.revised import RevisedFamilyRun
from program_v04.replication.robustness import build_expanding_threshold_comparison
from program_v04.replication.parity import validate_featured_cross_family_parity
from program_v04.replication.simulation import (
    build_diagnostics_table,
    build_main_table,
    frozen_step0_config,
)


def _load_revised(family: str) -> RevisedFamilyRun:
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


def _load_realtime(family: str) -> RealtimeFamilyRun:
    directory = FAMILY_REGISTRY[("realtime", family)].result_path
    paths = [directory / "forecasts.parquet", directory / "metrics.parquet"]
    metadata = validate_result_bundle(paths, directory / "metadata.json")
    options = metadata.get("command_options")
    method = options.get("method") if isinstance(options, dict) else None
    if not isinstance(method, dict):
        raise ValueError(f"realtime/{family} metadata has no method payload")
    run = RealtimeFamilyRun(
        family=family,
        forecasts=pd.read_parquet(paths[0]),
        metrics=pd.read_parquet(paths[1]),
        method_metadata=method,
        events=(),
    )
    run.validate()
    return run


def validate_appendix() -> dict[str, int]:
    assert_canonical_generation_allowed()
    inventory = validate_artifact_registry(profile="appendix")
    specs = artifact_registry("appendix")
    for spec in specs:
        if importlib.util.find_spec(spec.generator_module) is None:
            raise ImportError(f"Missing appendix artifact generator: {spec.generator_module}")

    revised = {
        family: _load_revised(family)
        for family in ("main", "extended", *APPENDIX_REVISED_FAMILIES)
    }
    validate_featured_cross_family_parity(
        revised["main"].forecasts,
        revised["extended"].forecasts,
        exercise="revised",
        data_variant="revised_final",
    )
    build_expanding_threshold_comparison(
        revised["main"].metrics,
        revised["expanding_threshold"].metrics,
        revised["main"].method_metadata,
        revised["expanding_threshold"].method_metadata,
        exercise="revised",
        data_variant="revised_final",
    )
    realtime = {
        family: _load_realtime(family)
        for family in ("realtime_main", "realtime_extended")
    }

    derived_directory = PACKAGE_ROOT / "results" / "appendix" / "derived"
    derived_paths = [
        derived_directory / name
        for name in (
            "bootstrap.parquet",
            "pre2020.parquet",
            "cumulative_brier.parquet",
            "asymmetric_loss.parquet",
        )
    ]
    validate_result_bundle(derived_paths, derived_directory / "metadata.json")
    for path in derived_paths:
        validate_derived_table(pd.read_parquet(path), surface=f"stored {path.stem}")

    wealth_directory = PACKAGE_ROOT / "results" / "appendix" / "decision_wealth"
    wealth_paths = [wealth_directory / "summary.parquet", wealth_directory / "wealth_paths.parquet"]
    validate_result_bundle(wealth_paths, wealth_directory / "metadata.json")
    validate_decision_wealth(pd.read_parquet(wealth_paths[0]), pd.read_parquet(wealth_paths[1]))

    simulation_directory = PACKAGE_ROOT / "results" / "appendix" / "simulation"
    simulation_paths = [
        simulation_directory / "metrics.parquet",
        simulation_directory / "diagnostics.parquet",
    ]
    validate_result_bundle(simulation_paths, simulation_directory / "metadata.json")
    config = frozen_step0_config()
    build_main_table(pd.read_parquet(simulation_paths[0]), config=config)
    build_diagnostics_table(pd.read_parquet(simulation_paths[1]), config=config)

    for survey, filename in (
        ("ijf", "literature_survey_ijf.csv"),
        ("nonijf", "literature_survey_nonijf.csv"),
    ):
        load_documentary_rows(
            PACKAGE_ROOT / "data" / "documentary" / filename,
            survey=survey,
        )
    for spec in specs:
        validate_artifact_outputs(spec)
    return {
        "registered_artifacts": inventory["artifacts"],
        "appendix_artifacts": len(specs),
        "revised_families": len(revised),
        "realtime_families": len(realtime),
        "forecast_rows": sum(len(run.forecasts) for run in revised.values())
        + sum(len(run.forecasts) for run in realtime.values()),
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    summary = validate_appendix()
    print(
        "Appendix validation OK: "
        f"{summary['appendix_artifacts']} artifacts, "
        f"{summary['revised_families']} revised families, "
        f"{summary['realtime_families']} real-time families, "
        f"{summary['forecast_rows']} forecast rows",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
