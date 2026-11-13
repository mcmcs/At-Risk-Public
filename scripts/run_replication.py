"""Install the locked environment, reproduce artifacts, and compare them."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from check_repository import check


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "program_v04"


def hashes(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Re-estimate every model from the frozen inputs.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "REPOSITORY_MANIFEST.json").read_text())
    private = manifest.get("edition") == "journal-private"
    saved = private and not args.full
    environment = dict(os.environ)
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    environment["MPLCONFIGDIR"] = str(PACKAGE / "output/.matplotlib")
    environment["PYTHONUNBUFFERED"] = "1"
    report = {"status": "running", "mode": "saved" if saved else "full",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "commands": []}
    (ROOT / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")

    def execute(command: list[str], cwd: Path = PACKAGE) -> None:
        report["commands"].append(command)
        subprocess.run(command, cwd=cwd, env=environment, check=True)

    try:
        report["distributed_files_verified"] = check(ROOT)
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("Install uv as described in README.md, then repeat this command.")
        if shutil.which("pdftoppm") is None:
            raise RuntimeError("Figure comparison needs Poppler. On macOS, install it with brew install poppler.")
        execute([uv, "sync", "--locked"])
        python = str(PACKAGE / ".venv/bin/python")
        execute([python, "-c", "from program_v04.replication.data import load_input_manifest, verify_inputs; "
                 "records=tuple(r for r in load_input_manifest() if r.role!='ads_daily'); "
                 "print(f'Checked {len(verify_inputs(records, mode=\"public\"))} required inputs.')"])
        if saved:
            frozen = ROOT / "frozen/results"
            expected = hashes(frozen)
            if not expected:
                raise RuntimeError("The journal bundle has no frozen numerical results")
            destination = PACKAGE / "results"
            if destination.exists():
                if hashes(destination) != expected:
                    raise RuntimeError("Working results differ from the frozen copy. Use a fresh extraction for the saved-results run, or --full to re-estimate.")
            else:
                shutil.copytree(frozen, destination)
            before = hashes(destination)
        command = [python, "-m", "program_v04.run_all", "--input-mode", "public"]
        if saved:
            command.append("--artifacts-only")
        execute(command)
        if saved and hashes(PACKAGE / "results") != before:
            raise RuntimeError("Artifact regeneration altered the saved numerical results")
        execute([python, str(ROOT / "scripts/compare_reference.py")], ROOT)
        report["status"] = "passed"
        report["reference_files_compared"] = 50
        report["frozen_files_verified_after_run"] = check(ROOT)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        (ROOT / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
