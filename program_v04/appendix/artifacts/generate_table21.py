"""Generate appendix Table 21 from versioned IJF survey rows."""

from __future__ import annotations

from program_v04.appendix.artifacts._shared import write_table
from program_v04.replication.appendix_paper import load_documentary_rows, render_documentary_tex
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.config import PACKAGE_ROOT


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_lit_survey", argv)
    frame = load_documentary_rows(
        PACKAGE_ROOT / spec.result_dependencies[0], survey="ijf"
    )
    write_table(spec, frame, [], tex=render_documentary_tex(frame, survey="ijf"))
    print("Generated appendix Table 21")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
