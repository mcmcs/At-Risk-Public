"""Run the main profile, then the appendix profile, from declared inputs."""

from __future__ import annotations

import argparse
from typing import Literal

from program_v04.appendix.run_appendix import run_appendix
from program_v04.main.run_main import run_main
from program_v04.replication.config import assert_canonical_generation_allowed


def run_all(
    *,
    input_mode: Literal["public", "development"],
    artifacts_only: bool,
) -> None:
    assert_canonical_generation_allowed()
    run_main(input_mode=input_mode, artifacts_only=artifacts_only)
    run_appendix(input_mode=input_mode, artifacts_only=artifacts_only)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("public", "development"), default="public")
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Regenerate artifacts and validate existing fingerprint-valid results only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    run_all(input_mode=arguments.input_mode, artifacts_only=arguments.artifacts_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
