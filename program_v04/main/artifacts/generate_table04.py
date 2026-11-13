"""Generate main-paper Table 4 forecast-encompassing results."""

from __future__ import annotations

from program_v04.main.artifacts._shared import load_dependencies, write_table
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.paper import build_main_table04
from program_v04.replication.protocols import canonical_main_protocol


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_encompassing_tests", argv)
    panels, metadata = load_dependencies(spec)
    table = build_main_table04(
        panels["results/shared/revised_main/forecasts.parquet"],
        extended_forecasts=panels["results/shared/revised_extended/forecasts.parquet"],
        protocol=canonical_main_protocol(),
    )
    write_table(spec, table, metadata)
    print("Generated main-paper Table 4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
