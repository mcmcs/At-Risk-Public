"""Generate main-paper Table 2 from completed forecast-family panels."""

from __future__ import annotations

from program_v04.main.artifacts._shared import load_dependencies, write_table
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.paper import build_main_table02


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_zt_vs_xt", argv)
    panels, metadata = load_dependencies(spec)
    table = build_main_table02(
        panels["results/shared/revised_main/metrics.parquet"],
        panels["results/shared/revised_extended/metrics.parquet"],
        panels["results/main/revised_probit_spread/metrics.parquet"],
        main_forecasts=panels["results/shared/revised_main/forecasts.parquet"],
        extended_forecasts=panels["results/shared/revised_extended/forecasts.parquet"],
    )
    write_table(spec, table, metadata)
    print("Generated main-paper Table 2", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
