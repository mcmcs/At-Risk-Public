"""Download and verify the paper's USRECM recession indicator from FRED.

As of September 10, 2026 (America/New_York), FRED's current USRECM values
matched all 780 paper-sample months exactly. Every download repeats that check.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .replication.config import PACKAGE_ROOT
from .replication.data import (
    TARGET_COMPARISON_DATE,
    TARGET_END,
    TARGET_INPUT_ID,
    TARGET_START,
    manifest_by_id,
    sha256_file,
    verify_target_values,
)


FRED_TARGET_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USRECM"


def target_from_fred_csv(content: bytes) -> pd.DataFrame:
    """Select the fixed sample and reject missing, duplicate, or changed months."""
    raw = pd.read_csv(io.BytesIO(content))
    if list(raw.columns) not in (["observation_date", "USRECM"], ["DATE", "USRECM"]):
        raise ValueError("Expected the FRED monthly USRECM CSV, not USREC or another series")
    dates = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0], format="%Y-%m-%d", errors="raise"))
    if dates.isna().any() or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("FRED dates must be valid, sorted, and unique")
    selected = raw.loc[(dates >= TARGET_START) & (dates <= TARGET_END), "USRECM"]
    selected_dates = dates[(dates >= TARGET_START) & (dates <= TARGET_END)]
    recession = pd.Series(
        pd.to_numeric(selected, errors="raise").to_numpy(), index=selected_dates
    )
    verify_target_values(recession)
    # Preserve the forecasting loader's float values and month-start index.
    return pd.DataFrame(
        recession.to_numpy(dtype="float64"),
        index=pd.date_range(TARGET_START, TARGET_END, freq="MS").as_unit("ns"),
        columns=pd.Index(["USRECM"], dtype="object"),
    )


def download_target(output: Path) -> dict[str, object]:
    """Download explicitly, validate before writing, and refuse to overwrite."""
    if output.exists():
        raise FileExistsError(f"Target already exists: {output}")
    with urlopen(FRED_TARGET_URL, timeout=45) as response:
        content = response.read()
    retrieved_at = datetime.now(UTC).isoformat()
    target = target_from_fred_csv(content)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects an input created while the download ran.
    with output.open("xb") as handle:
        try:
            target.to_pickle(handle, protocol=5)
        except BaseException:
            output.unlink()
            raise
    return {
        "source_url": FRED_TARGET_URL,
        "retrieved_at_utc": retrieved_at,
        "initial_comparison_date_America_New_York": TARGET_COMPARISON_DATE,
        "sample_start": TARGET_START,
        "sample_end": TARGET_END,
        "months": len(target),
        "recession_months": int(target["USRECM"].sum()),
        "monthly_values_sha256": verify_target_values(target["USRECM"]),
        "download_sha256": hashlib.sha256(content).hexdigest(),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        help="New output path; defaults to the target path in data/manifest.csv",
    )
    args = parser.parse_args(argv)
    output = args.output or PACKAGE_ROOT / manifest_by_id()[TARGET_INPUT_ID].public_path
    print(json.dumps(download_target(output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
