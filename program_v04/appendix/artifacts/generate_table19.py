"""Generate appendix Table 19 (expanding threshold calibration)."""

from __future__ import annotations

from program_v04.appendix.artifacts._shared import load_dependencies, write_table
from program_v04.replication.artifacts import artifact_cli_spec
from program_v04.replication.robustness import build_expanding_threshold_comparison


def _method(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise ValueError("Table 19 metadata dependency is malformed")
    options = metadata.get("command_options")
    if not isinstance(options, dict) or not isinstance(options.get("method"), dict):
        raise ValueError("Table 19 metadata has no method payload")
    return dict(options["method"])


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_expanding_threshold", argv)
    values, metadata = load_dependencies(spec)
    main_metrics, main_metadata, expanding_metrics, expanding_metadata = (
        values[relative] for relative in spec.result_dependencies
    )
    frame = build_expanding_threshold_comparison(
        main_metrics,
        expanding_metrics,
        _method(main_metadata),
        _method(expanding_metadata),
        exercise="revised",
        data_variant="revised_final",
    )
    write_table(spec, frame, metadata)
    print("Generated appendix Table 19")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
