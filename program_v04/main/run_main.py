"""Run main computation, all ten artifacts in order, and main validation."""

from __future__ import annotations

import argparse
import importlib
from typing import Literal

from program_v04.main.compute_main import compute_main
from program_v04.main.validate_main import validate_main
from program_v04.replication.artifacts import artifact_registry
from program_v04.replication.config import assert_canonical_generation_allowed


def run_main(
    *,
    input_mode: Literal["public", "development"],
    artifacts_only: bool,
) -> None:
    assert_canonical_generation_allowed()
    if not artifacts_only:
        compute_main(input_mode=input_mode)
    for spec in artifact_registry("main"):
        module = importlib.import_module(spec.generator_module)
        status = module.main([])
        if status not in (None, 0):
            raise RuntimeError(f"{spec.generator_module} returned status {status}")
    validate_main()


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
    args = parse_args(argv)
    run_main(input_mode=args.input_mode, artifacts_only=args.artifacts_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
