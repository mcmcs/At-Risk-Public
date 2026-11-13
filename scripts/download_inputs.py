"""Obtain the paper's exact public inputs and reject changed snapshots."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

from program_v04.download_target import download_target
from program_v04.replication.config import PACKAGE_ROOT
from program_v04.replication.data import load_input_manifest, verify_input, verify_inputs


ROOT = Path(__file__).resolve().parents[1]


def fetch(url: str, target: Path, *, offline: bool) -> Path:
    if target.is_file():
        return target
    if offline:
        raise FileNotFoundError(f"Missing cached download: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=target.name + ".", suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            shutil.copyfileobj(response, output)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def install_bytes(record, content: bytes) -> None:
    if len(content) != record.size_bytes or hashlib.sha256(content).hexdigest() != record.sha256:
        raise ValueError(f"{record.input_id}: provider file differs from the frozen manifest; do not substitute it")
    target = PACKAGE_ROOT / record.public_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(content)


def acquire(cache: Path, *, offline: bool) -> dict:
    sources = json.loads((ROOT / "docs/public_sources.json").read_text())
    records = tuple(r for r in load_input_manifest() if r.role != "ads_daily")
    missing = []
    for record in records:
        if (PACKAGE_ROOT / record.public_path).exists():
            verify_input(record, mode="public")
        else:
            missing.append(record)
    downloads = []
    with ExitStack() as stack:
        archives = {}
        for record in missing:
            target = PACKAGE_ROOT / record.public_path
            if record.input_id in sources["fred_md_members"]:
                member = sources["fred_md_members"][record.input_id]
                key = member["archive"]
                if key not in archives:
                    source = sources["archives"][key]
                    archive = fetch(source["url"], cache / source["filename"], offline=offline)
                    archives[key] = stack.enter_context(ZipFile(archive))
                    downloads.append({"url": source["url"], "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
                install_bytes(record, archives[key].read(member["member"]))
            elif record.input_id == "yield_spread":
                from _reconstruct_yield import reconstruct

                source = sources["yield_workbook"]
                workbook = fetch(source["url"], cache / source["filename"], offline=offline)
                reconstruct(workbook, PACKAGE_ROOT / "data/manifest.csv", target)
                downloads.append({"url": source["url"], "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest()})
            elif record.input_id == "usrecm_target_final":
                if offline:
                    raise FileNotFoundError("Offline mode requires the verified target already at data/raw/target/y_target.pkl")
                downloads.append(download_target(target))
            else:
                raise ValueError(f"No acquisition rule for {record.input_id}")
    verified = verify_inputs(records, mode="public")
    return {"status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "verified_inputs": len(verified), "downloads": downloads,
            "verification": "Exact manifest bytes except USRECM, which requires exact monthly dates and values."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "downloads")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    report = {"status": "running"}
    (ROOT / "download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        report = acquire(args.cache_dir.resolve(), offline=args.offline)
    except BaseException as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        (ROOT / "download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
