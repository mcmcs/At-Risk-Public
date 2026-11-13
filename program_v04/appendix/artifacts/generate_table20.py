"""Generate appendix Table 20 (skewed-mixture Monte Carlo)."""

from __future__ import annotations

from program_v04.appendix.artifacts._shared import load_dependencies, write_table
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.simulation import build_main_table, frozen_step0_config


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_mc_tailrisk", argv)
    values, metadata = load_dependencies(spec)
    frame = build_main_table(
        values[spec.result_dependencies[0]], config=frozen_step0_config()
    )
    write_table(spec, frame, metadata)
    print("Generated appendix Table 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
