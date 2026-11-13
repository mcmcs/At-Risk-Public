"""Generate appendix Figure 4 (h=3 cumulative Brier-score advantages)."""

from __future__ import annotations

from program_v04.appendix.artifacts._shared import load_dependencies
from program_v04.replication.appendix_paper import render_cumulative_brier_figure
from program_v04.replication.artifact_io import write_artifact_metadata
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.config import PACKAGE_ROOT, assert_canonical_generation_allowed


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("fig_cumulative_brier", argv)
    assert_canonical_generation_allowed()
    values, metadata = load_dependencies(spec)
    if len(spec.output_paths) != 1:
        raise ValueError("Figure 4 must declare exactly one output")
    output = PACKAGE_ROOT / spec.output_paths[0]
    render_cumulative_brier_figure(values[spec.result_dependencies[0]], output)
    write_artifact_metadata(spec, dependency_metadata=metadata)
    print(f"Generated appendix Figure 4: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
