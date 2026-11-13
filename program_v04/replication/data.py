"""Declared-input resolution, manifest verification, and result schemas."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import PACKAGE_ROOT
from .validation import (
    require_binary,
    require_boolean,
    require_columns,
    require_deterministic_order,
    require_finite_numeric,
    require_forecast_calendar,
    require_month_start,
    require_nonempty,
    require_nonempty_strings,
    require_nonnegative_integer,
    require_non_null,
    require_probability,
    require_unique_keys,
)


REPO_ROOT = PACKAGE_ROOT.parent
MANIFEST_PATH = PACKAGE_ROOT / "data" / "manifest.csv"

TARGET_INPUT_ID = "usrecm_target_final"
TARGET_START = "1960-01-01"
TARGET_END = "2024-12-01"
# As of 2026-09-10 (America/New_York), the FRED USRECM download matched
# all 780 frozen monthly values, including 104 recession months (40 in 1990-2024).
# This is the comparison date, not the unknown original acquisition date.
TARGET_COMPARISON_DATE = "2026-09-10"
TARGET_VALUES_SHA256 = "6994f06c6bba9ef26a5aa29265babc34665d40db092ed93f0cec531d8566f484"

FORECAST_KEYS = (
    "exercise",
    "data_variant",
    "forecast_family",
    "horizon",
    "forecast_date",
    "origin_date",
    "target_date",
    "predictor_key",
    "model_name",
)
FORECAST_COLUMNS = (*FORECAST_KEYS, "probability", "actual")

METRIC_KEYS = (
    "exercise",
    "data_variant",
    "forecast_family",
    "horizon",
    "predictor_key",
    "model_name",
)
METRIC_COLUMNS = (
    *METRIC_KEYS,
    "num_forecasts",
    "num_positive_actuals",
    "num_negative_actuals",
    "pr_auc",
    "roc_auc",
    "brier",
    "brier_recession",
    "brier_expansion",
    "reliability",
    "resolution",
    "uncertainty",
)

IMPORTANCE_KEYS = (
    "exercise",
    "data_variant",
    "forecast_family",
    "horizon",
    "forecast_date",
    "predictor_key",
    "model_name",
    "feature_name",
)
IMPORTANCE_COLUMNS = (
    *IMPORTANCE_KEYS,
    "base_feature_name",
    "display_name",
    "sector",
    "importance",
    "importance_type",
    "is_degenerate_model",
    "degenerate_reason",
)


@dataclass(frozen=True)
class InputRecord:
    input_id: str
    profile: str
    role: str
    public_path: str
    development_source_path: str
    source_name: str
    source_url: str
    acquisition_date: str
    vintage: str
    license_status: str
    redistribution_allowed: bool
    sha256: str
    size_bytes: int
    expected_rows: int
    expected_columns: int
    date_column: str
    date_start: str
    date_end: str
    notes: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "InputRecord":
        redistribution = row["redistribution_allowed"].lower()
        if redistribution not in {"true", "false"}:
            raise ValueError(
                f"{row['input_id']}: redistribution_allowed must be true or false, "
                f"found {row['redistribution_allowed']!r}"
            )
        for field in ("public_path", "development_source_path"):
            candidate = Path(row[field])
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"{row['input_id']}: unsafe {field} {row[field]!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"{row['input_id']}: invalid SHA-256 {row['sha256']!r}")
        return cls(
            input_id=row["input_id"],
            profile=row["profile"],
            role=row["role"],
            public_path=row["public_path"],
            development_source_path=row["development_source_path"],
            source_name=row["source_name"],
            source_url=row["source_url"],
            acquisition_date=row["acquisition_date"],
            vintage=row["vintage"],
            license_status=row["license_status"],
            redistribution_allowed=redistribution == "true",
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
            expected_rows=int(row["expected_rows"]),
            expected_columns=int(row["expected_columns"]),
            date_column=row["date_column"],
            date_start=row["date_start"],
            date_end=row["date_end"],
            notes=row["notes"],
        )


def load_input_manifest(path: Path = MANIFEST_PATH) -> tuple[InputRecord, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        records = tuple(InputRecord.from_row(row) for row in csv.DictReader(handle))
    if not records:
        raise ValueError(f"Input manifest is empty: {path}")
    identifiers = [record.input_id for record in records]
    public_paths = [record.public_path for record in records]
    development_paths = [record.development_source_path for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Input manifest contains duplicate input_id values")
    if len(set(public_paths)) != len(public_paths):
        raise ValueError("Input manifest contains duplicate public_path values")
    if len(set(development_paths)) != len(development_paths):
        raise ValueError("Input manifest contains duplicate development_source_path values")
    return records


def manifest_by_id(path: Path = MANIFEST_PATH) -> dict[str, InputRecord]:
    return {record.input_id: record for record in load_input_manifest(path)}


def resolve_input_path(
    record: InputRecord,
    *,
    mode: Literal["public", "development"] = "public",
) -> Path:
    """Resolve one declared input without a hidden archive fallback."""
    if mode == "public":
        path = PACKAGE_ROOT / record.public_path
        repair = f"Place the declared input at {path}"
    elif mode == "development":
        path = REPO_ROOT / record.development_source_path
        repair = f"Restore the immutable development snapshot at {path}"
    else:
        raise ValueError(f"Unknown input resolution mode: {mode}")
    if not path.is_file():
        expected = f"expected SHA-256 {record.sha256}"
        if mode == "public" and record.input_id == TARGET_INPUT_ID:
            repair += "; run python -m program_v04.download_target"
            expected = f"expected monthly-value SHA-256 {TARGET_VALUES_SHA256}"
        raise FileNotFoundError(
            f"Missing declared input {record.input_id}: {path}. {repair}; "
            f"{expected}. No fallback path is permitted."
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input(
    record: InputRecord,
    *,
    mode: Literal["public", "development"] = "public",
    verify_shape: bool = True,
) -> Path:
    """Verify input identity, allowing equivalent public USRECM serialization.

    Public USRECM always requires exact monthly dates and values. Its manifest
    byte hash and size describe the archived pickle. All other inputs, and
    development-mode USRECM, retain the original byte checks.
    """
    path = resolve_input_path(record, mode=mode)
    if mode == "public" and record.input_id == TARGET_INPUT_ID:
        verify_target_values(load_target_snapshot(path))
    else:
        actual_size = path.stat().st_size
        if actual_size != record.size_bytes:
            raise ValueError(
                f"{record.input_id}: size mismatch at {path}; "
                f"expected {record.size_bytes}, found {actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != record.sha256:
            raise ValueError(
                f"{record.input_id}: checksum mismatch at {path}; "
                f"expected {record.sha256}, found {actual_hash}"
            )
    if verify_shape:
        rows, columns, date_column, date_start, date_end = input_shape_and_dates(
            path, record.role
        )
        if (int(rows), int(columns)) != (record.expected_rows, record.expected_columns):
            raise ValueError(
                f"{record.input_id}: shape mismatch at {path}; expected "
                f"({record.expected_rows}, {record.expected_columns}), found ({rows}, {columns})"
            )
        if date_column != record.date_column:
            raise ValueError(
                f"{record.input_id}: date column mismatch at {path}; "
                f"expected {record.date_column!r}, found {date_column!r}"
            )
        if (date_start, date_end) != (record.date_start, record.date_end):
            raise ValueError(
                f"{record.input_id}: date coverage mismatch at {path}; "
                f"expected {record.date_start} through {record.date_end}, "
                f"found {date_start} through {date_end}"
            )
    return path


def verify_inputs(
    records: tuple[InputRecord, ...] | None = None,
    *,
    mode: Literal["public", "development"] = "public",
    verify_shape: bool = True,
) -> dict[str, Path]:
    records = load_input_manifest() if records is None else records
    return {
        record.input_id: verify_input(record, mode=mode, verify_shape=verify_shape)
        for record in records
    }


_DASHED_YIELD_DATE_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")


def parse_yield_date(value: object) -> pd.Timestamp:
    """Parse the term-spread snapshot's deterministic 2017 format switch."""
    text = str(value).strip()
    if _DASHED_YIELD_DATE_RE.fullmatch(text):
        day, month, year_text = text.split("-")
        two_digit_year = int(year_text)
        year = 1900 + two_digit_year if two_digit_year >= 59 else 2000 + two_digit_year
        return pd.Timestamp(f"{year:04d}-{month}-{int(day):02d}")
    return pd.Timestamp(text)


def input_shape_and_dates(path: Path, role: str) -> tuple[int, int, str, str, str]:
    """Inspect raw dates without the forecasting loaders' month-start normalization.

    Only the initial FRED-MD transformation-code row is excluded from date
    parsing. Invalid or missing observation dates must not shorten coverage.
    """
    if path.suffix == ".pkl":
        value = pd.read_pickle(path)
        rows = len(value)
        columns = value.shape[1] if isinstance(value, pd.DataFrame) else 1
        date_column = "index"
        dates = pd.DatetimeIndex(pd.to_datetime(value.index, errors="raise"))
    else:
        frame = pd.read_csv(path, low_memory=False)
        rows, columns = frame.shape
        date_column = str(frame.columns[0])
        values = frame.iloc[:, 0].astype(str).str.strip()
        if role == "yield_spread":
            dates = pd.DatetimeIndex(values.map(parse_yield_date))
        elif role == "ads_daily":
            dates = pd.DatetimeIndex(
                pd.to_datetime(values, format="%Y:%m:%d", errors="raise")
            )
        elif role in {"fred_md_final", "fred_md_vintage"}:
            if not values.empty and values.iloc[0] == "Transform:":
                values = values.iloc[1:]
            dates = pd.DatetimeIndex(
                pd.to_datetime(values, format="%m/%d/%Y", errors="raise")
            )
        else:
            raise ValueError(f"Unknown dated input role {role!r}: {path}")
    if dates.empty or dates.hasnans:
        raise ValueError(f"Input contains missing observation dates: {path}")
    return (
        int(rows), int(columns), date_column,
        str(dates.min().date()), str(dates.max().date()),
    )


def load_fred_md_snapshot(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load one FRED-MD snapshot into raw monthly values and integer tcodes."""
    frame = pd.read_csv(path, low_memory=False)
    if frame.shape[0] < 2 or frame.shape[1] < 2 or frame.columns[0] != "sasdate":
        raise ValueError(f"Malformed FRED-MD snapshot: {path}")
    tcodes = pd.to_numeric(frame.iloc[0, 1:], errors="coerce")
    if tcodes.isna().any() or not (tcodes % 1 == 0).all():
        raise ValueError(f"FRED-MD transformation-code row is incomplete: {path}")
    tcodes = tcodes.astype(int)
    data = frame.iloc[1:].copy().rename(columns={"sasdate": "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError(f"FRED-MD snapshot contains invalid dates: {path}")
    data = data.set_index("date")
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError(f"FRED-MD dates must be sorted and unique: {path}")
    expected = pd.date_range(data.index.min(), data.index.max(), freq="MS")
    if not data.index.equals(expected):
        raise ValueError(f"FRED-MD dates must form a gapless monthly index: {path}")
    tcodes.index = data.columns
    return data, tcodes


def load_yield_spread_snapshot(path: Path) -> pd.Series:
    """Load the monthly ten-year-minus-three-month term-spread snapshot."""
    frame = pd.read_csv(path, usecols=["Date", "Spread"])
    frame["Date"] = frame["Date"].map(parse_yield_date)
    frame["Spread"] = pd.to_numeric(frame["Spread"], errors="coerce")
    if frame[["Date", "Spread"]].isna().any().any():
        raise ValueError(f"Yield-spread snapshot contains invalid dates or values: {path}")
    if frame["Date"].duplicated().any():
        raise ValueError(f"Yield-spread snapshot contains duplicate source dates: {path}")
    result = (
        frame.set_index("Date")["Spread"]
        .sort_index()
        .resample("MS")
        .first()
        .rename("yield_spread")
    )
    expected = pd.date_range(result.index.min(), result.index.max(), freq="MS")
    if not result.index.equals(expected) or result.isna().any():
        raise ValueError(f"Yield-spread snapshot does not form a complete monthly series: {path}")
    return result


def load_target_snapshot(path: Path) -> pd.Series:
    """Load the frozen one-column final-chronology USRECM target snapshot."""
    target = pd.read_pickle(path)
    if not isinstance(target, pd.DataFrame) or list(target.columns) != ["USRECM"]:
        raise ValueError("Target input must be a one-column DataFrame named USRECM")
    recession = pd.to_numeric(target["USRECM"], errors="coerce").rename("recession")
    recession.index = pd.DatetimeIndex(pd.to_datetime(recession.index, errors="coerce"))
    if recession.index.isna().any() or recession.isna().any():
        raise ValueError("Target snapshot contains invalid dates or values")
    if not recession.index.is_unique or not recession.index.is_monotonic_increasing:
        raise ValueError("Target snapshot dates must be sorted and unique")
    expected = pd.date_range(recession.index.min(), recession.index.max(), freq="MS")
    if not recession.index.equals(expected):
        raise ValueError("Target snapshot must have a gapless monthly index")
    observed = set(recession.unique())
    if not observed <= {0, 1}:
        raise ValueError(f"Target snapshot must be binary; found {sorted(observed)}")
    return recession


def verify_target_values(recession: pd.Series) -> str:
    """Require the paper's exact target using a serialization-independent hash.

    Canonical bytes are ASCII CSV: ``date,USRECM\n`` followed by sorted
    ``YYYY-MM-DD,0\n`` or ``YYYY-MM-DD,1\n`` rows, including the final newline.
    No tolerance, date filling, or recession-count-only comparison is used.
    """
    expected_dates = pd.date_range(TARGET_START, TARGET_END, freq="MS")
    if not recession.index.equals(expected_dates):
        raise ValueError("USRECM must contain exactly the monthly dates 1960-01 through 2024-12")
    if recession.isna().any() or not set(recession.unique()) <= {0, 1}:
        raise ValueError("USRECM values must be nonmissing and binary")
    payload = "date,USRECM\n" + "".join(
        f"{date:%Y-%m-%d},{int(value)}\n" for date, value in recession.items()
    )
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    if digest != TARGET_VALUES_SHA256:
        raise ValueError(
            "USRECM monthly values differ from the paper's verified series; "
            f"expected {TARGET_VALUES_SHA256}, found {digest}. Do not replace the reference hash."
        )
    return digest


def validate_forecast_panel(frame: pd.DataFrame, *, surface: str = "forecast_panel") -> None:
    require_nonempty(frame, surface=surface)
    require_columns(frame, FORECAST_COLUMNS, surface=surface)
    require_non_null(frame, FORECAST_COLUMNS, surface=surface)
    require_unique_keys(frame, FORECAST_KEYS, surface=surface)
    require_deterministic_order(frame, FORECAST_KEYS, surface=surface)
    require_forecast_calendar(frame, surface=surface)
    require_probability(frame["probability"], surface=surface)
    require_binary(frame["actual"], surface=surface, field="actual")


def validate_metric_panel(frame: pd.DataFrame, *, surface: str = "metric_panel") -> None:
    require_nonempty(frame, surface=surface)
    require_columns(frame, METRIC_COLUMNS, surface=surface)
    require_non_null(frame, METRIC_COLUMNS, surface=surface)
    require_unique_keys(frame, METRIC_KEYS, surface=surface)
    require_deterministic_order(frame, METRIC_KEYS, surface=surface)
    counts = {
        field: require_nonnegative_integer(frame[field], surface=surface, field=field)
        for field in ("num_forecasts", "num_positive_actuals", "num_negative_actuals")
    }
    if not (
        counts["num_positive_actuals"] + counts["num_negative_actuals"]
        == counts["num_forecasts"]
    ).all():
        raise ValueError(
            f"{surface}: num_positive_actuals + num_negative_actuals must equal num_forecasts"
        )
    if (counts["num_forecasts"] == 0).any():
        raise ValueError(f"{surface}: num_forecasts must be positive")
    for field in (
        "pr_auc",
        "roc_auc",
        "brier",
        "brier_recession",
        "brier_expansion",
        "reliability",
        "resolution",
        "uncertainty",
    ):
        require_probability(frame[field], surface=surface, field=field)


def validate_importance_panel(frame: pd.DataFrame, *, surface: str = "importance_panel") -> None:
    require_nonempty(frame, surface=surface)
    require_columns(frame, IMPORTANCE_COLUMNS, surface=surface)
    required = tuple(column for column in IMPORTANCE_COLUMNS if column != "degenerate_reason")
    require_non_null(frame, required, surface=surface)
    require_unique_keys(frame, IMPORTANCE_KEYS, surface=surface)
    require_deterministic_order(frame, IMPORTANCE_KEYS, surface=surface)
    require_month_start(
        frame["forecast_date"], surface=surface, field="forecast_date"
    )
    require_finite_numeric(frame["importance"], surface=surface, field="importance")
    for field in ("base_feature_name", "display_name", "sector", "importance_type"):
        require_nonempty_strings(frame[field], surface=surface, field=field)
    degenerate = require_boolean(
        frame["is_degenerate_model"], surface=surface, field="is_degenerate_model"
    )
    reasons = frame["degenerate_reason"]
    invalid_reason = degenerate & reasons.map(
        lambda value: not isinstance(value, str) or not value.strip()
    )
    if invalid_reason.any():
        raise ValueError(f"{surface}: every degenerate model must have a nonempty reason")
