"""Generate appendix Table 18 (X/Z Logit wealth and annualized CE returns)."""

from __future__ import annotations

from program_v04.appendix.artifacts._shared import load_dependencies, write_table
from program_v04.replication.decision_wealth import render_decision_wealth, validate_decision_wealth
from program_v04.replication.artifacts import artifact_cli_spec


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_asymmetric_loss", argv)
    values, metadata = load_dependencies(spec)
    frame = values[spec.result_dependencies[0]]
    paths = values[spec.result_dependencies[1]]
    validate_decision_wealth(frame, paths)
    write_table(spec, frame, metadata, tex=render_decision_wealth(frame))
    print("Generated appendix Table 18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
