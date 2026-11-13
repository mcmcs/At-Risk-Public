"""Check the distributed repository files without estimating models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def check(root: Path) -> int:
    manifest = json.loads((root / "REPOSITORY_MANIFEST.json").read_text())
    failures = []
    seen = set()
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError(f"Invalid manifest path: {relative}")
        seen.add(str(relative))
        path = root / relative
        if path.is_symlink() or not path.is_file():
            failures.append(f"Missing or linked file: {relative}")
        elif path.stat().st_size != record["size_bytes"]:
            failures.append(f"Size differs: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(f"Checksum differs: {relative}")
    if failures:
        raise ValueError("\n".join(failures))
    return len(seen)


if __name__ == "__main__":
    count = check(Path(__file__).resolve().parents[1])
    print(f"Repository integrity: {count} distributed files match the manifest.")
