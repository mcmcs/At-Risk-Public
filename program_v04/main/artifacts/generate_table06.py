"""Generate main-paper Table 6 feature-importance ranking."""

from __future__ import annotations

from program_v04.main.artifacts._shared import load_dependencies, write_table
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.paper import build_main_table06


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_feature_importance", argv)
    panels, metadata = load_dependencies(spec)
    table = build_main_table06(panels["results/shared/revised_main/importances.parquet"])
    write_table(spec, table, metadata)
    print("Generated main-paper Table 6", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
