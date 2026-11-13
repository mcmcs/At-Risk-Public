"""Generate appendix Table 10 (moving-average robustness)."""

from __future__ import annotations

from program_v04.appendix.artifacts._shared import load_dependencies, write_table
from program_v04.replication.appendix_paper import build_appendix_table10
from program_v04.replication.artifacts import artifact_cli_spec


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_robustness_ma", argv)
    values, metadata = load_dependencies(spec)
    write_table(
        spec,
        build_appendix_table10(
            values[spec.result_dependencies[0]], values[spec.result_dependencies[1]]
        ),
        metadata,
    )
    print("Generated appendix Table 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
