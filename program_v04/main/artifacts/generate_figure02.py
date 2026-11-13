"""Generate every panel of main-paper Figure 2 from stored forecasts."""

from __future__ import annotations

from program_v04.main.artifacts._shared import load_dependencies
from program_v04.replication.artifact_io import write_artifact_metadata
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.config import PACKAGE_ROOT
from program_v04.replication.paper import render_main_figure02


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("fig_main_probability_plots", argv)
    panels, metadata = load_dependencies(spec)
    forecasts = panels["results/shared/revised_main/forecasts.parquet"]
    target = (
        forecasts[["forecast_date", "actual"]]
        .drop_duplicates()
        .sort_values("forecast_date")
        .set_index("forecast_date")["actual"]
    )
    outputs = tuple(PACKAGE_ROOT / relative for relative in spec.output_paths)
    if len(outputs) != 3:
        raise ValueError("Figure 2 must declare exactly three panel outputs")
    render_main_figure02(forecasts, target, outputs)  # type: ignore[arg-type]
    write_artifact_metadata(spec, dependency_metadata=metadata)
    print("Generated main-paper Figure 2 (three panels)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
