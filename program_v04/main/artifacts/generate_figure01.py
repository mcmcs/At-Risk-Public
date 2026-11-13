"""Generate main-paper Figure 1 from the completed Section 2 result panels."""

from __future__ import annotations

import json

from program_v04.replication.artifact_io import write_artifact_metadata
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.config import PACKAGE_ROOT
from program_v04.replication.section2 import (
    SECTION2_METADATA_PATH,
    load_section2_results,
    render_figure,
)


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("fig_single_binary_examples", argv)
    if len(spec.output_paths) != 1:
        raise ValueError("Figure 1 must declare exactly one output PDF")
    output = PACKAGE_ROOT / spec.output_paths[0]
    render_figure(load_section2_results(), output)
    metadata = json.loads(SECTION2_METADATA_PATH.read_text(encoding="utf-8"))
    write_artifact_metadata(spec, dependency_metadata=[metadata])
    print(f"Generated Figure 1: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
