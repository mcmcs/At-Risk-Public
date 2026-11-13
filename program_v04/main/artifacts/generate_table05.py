"""Generate main-paper Table 5 Brier-score decomposition."""

from __future__ import annotations

from program_v04.main.artifacts._shared import load_dependencies, write_table
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.paper import build_main_table05


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_sse_decomposition_h3", argv)
    panels, metadata = load_dependencies(spec)
    table = build_main_table05(panels["results/shared/revised_main/metrics.parquet"])
    write_table(spec, table, metadata)
    print("Generated main-paper Table 5", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
