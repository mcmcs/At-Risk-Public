"""Generate main-paper Table 3 from completed aggregation-family panels."""

from __future__ import annotations

from program_v04.main.artifacts._shared import load_dependencies, write_table
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.paper import build_main_table03


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_aggregation_comparison", argv)
    panels, metadata = load_dependencies(spec)
    table = build_main_table03(
        panels["results/shared/revised_main/metrics.parquet"],
        panels["results/main/revised_parameter_free/metrics.parquet"],
    )
    write_table(spec, table, metadata)
    print("Generated main-paper Table 3", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
