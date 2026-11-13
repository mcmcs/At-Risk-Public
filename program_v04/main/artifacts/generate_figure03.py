"""Generate main-paper Figure 3 forecast-disagreement plot."""

from __future__ import annotations

from program_v04.main.artifacts._shared import load_dependencies
from program_v04.replication.artifact_io import write_artifact_metadata
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.config import PACKAGE_ROOT
from program_v04.replication.paper import render_main_figure03


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("fig_disagreement_h3", argv)
    panels, metadata = load_dependencies(spec)
    forecasts = panels["results/shared/revised_main/forecasts.parquet"]
    target = (
        forecasts[["forecast_date", "actual"]]
        .drop_duplicates()
        .sort_values("forecast_date")
        .set_index("forecast_date")["actual"]
    )
    if len(spec.output_paths) != 1:
        raise ValueError("Figure 3 must declare exactly one output")
    render_main_figure03(forecasts, target, PACKAGE_ROOT / spec.output_paths[0])
    write_artifact_metadata(spec, dependency_metadata=metadata)
    print("Generated main-paper Figure 3", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
