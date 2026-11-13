"""Validate every completed result and artifact in the full replication profile."""

from __future__ import annotations

import argparse

from program_v04.appendix.validate_appendix import validate_appendix
from program_v04.main.validate_main import validate_main
from program_v04.replication.artifacts import validate_artifact_registry


def validate_all() -> dict[str, int]:
    """Validate the closed inventory and both profiles without recomputing results."""
    inventory = validate_artifact_registry()
    main_summary = validate_main()
    appendix_summary = validate_appendix()
    if main_summary["registered_artifacts"] != inventory["main"]:
        raise ValueError("Main validator/closed-inventory artifact count mismatch")
    if appendix_summary["registered_artifacts"] != inventory["appendix"]:
        raise ValueError("Appendix validator/closed-inventory artifact count mismatch")
    return {
        "artifacts": inventory["artifacts"],
        "main_artifacts": main_summary["main_artifacts"],
        "appendix_artifacts": appendix_summary["appendix_artifacts"],
        "forecast_rows": (
            main_summary["forecast_rows"] + appendix_summary["forecast_rows"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    summary = validate_all()
    print(
        "Full-package validation OK: "
        f"{summary['artifacts']} artifacts "
        f"({summary['main_artifacts']} main, "
        f"{summary['appendix_artifacts']} appendix), "
        f"{summary['forecast_rows']} profile forecast-row reads",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
