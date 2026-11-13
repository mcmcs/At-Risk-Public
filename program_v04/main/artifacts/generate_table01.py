"""Generate main-paper Table 1 from the completed Section 2 summary panel."""

from __future__ import annotations

import json

from program_v04.replication.artifact_io import write_artifact_metadata
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.config import PACKAGE_ROOT
from program_v04.replication.provenance import atomic_write_text
from program_v04.replication.section2 import (
    SECTION2_METADATA_PATH,
    load_section2_results,
    render_summary_tex,
)


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_single_binary_summary", argv)
    outputs = {path.rsplit(".", 1)[-1]: PACKAGE_ROOT / path for path in spec.output_paths}
    if set(outputs) != {"csv", "tex"}:
        raise ValueError("Table 1 must declare one CSV and one TeX output")
    summary = load_section2_results().summary
    atomic_write_text(outputs["csv"], summary.to_csv(index=False))
    atomic_write_text(outputs["tex"], render_summary_tex(summary))
    metadata = json.loads(SECTION2_METADATA_PATH.read_text(encoding="utf-8"))
    write_artifact_metadata(spec, dependency_metadata=[metadata])
    print(f"Generated Table 1: {outputs['csv']} and {outputs['tex']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
