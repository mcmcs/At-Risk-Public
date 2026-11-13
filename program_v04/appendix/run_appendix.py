"""Run appendix computation, all 16 artifacts, and appendix validation."""

from __future__ import annotations

import argparse
import importlib
from typing import Literal

from program_v04.appendix.compute_appendix import compute_appendix
from program_v04.appendix.validate_appendix import validate_appendix
from program_v04.replication.artifacts import artifact_registry
from program_v04.replication.config import assert_canonical_generation_allowed


def run_appendix(
    *,
    input_mode: Literal["public", "development"],
    artifacts_only: bool,
) -> None:
    assert_canonical_generation_allowed()
    if not artifacts_only:
        compute_appendix(input_mode=input_mode)
    for spec in artifact_registry("appendix"):
        module = importlib.import_module(spec.generator_module)
        status = module.main([])
        if status not in (None, 0):
            raise RuntimeError(f"{spec.generator_module} returned status {status}")
    validate_appendix()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("public", "development"), default="public")
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Reuse only fingerprint-valid completed result bundles; do not estimate models.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    run_appendix(
        input_mode=arguments.input_mode,
        artifacts_only=arguments.artifacts_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
