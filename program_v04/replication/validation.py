"""Reusable fail-fast invariants for v04 data and result surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def require_columns(frame: pd.DataFrame, required: Iterable[str], *, surface: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{surface}: missing required columns {missing}")


def require_nonempty(frame: pd.DataFrame, *, surface: str) -> None:
    if frame.empty:
        raise ValueError(f"{surface}: frame is empty")


def require_non_null(frame: pd.DataFrame, columns: Sequence[str], *, surface: str) -> None:
    require_columns(frame, columns, surface=surface)
    bad = {column: int(frame[column].isna().sum()) for column in columns if frame[column].isna().any()}
    if bad:
        raise ValueError(f"{surface}: null values in required fields {bad}")


def require_unique_keys(frame: pd.DataFrame, keys: Sequence[str], *, surface: str) -> None:
    require_columns(frame, keys, surface=surface)
    duplicated = frame.duplicated(subset=list(keys), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(keys)].head(5).to_dict("records")
        raise ValueError(f"{surface}: duplicate primary keys; examples={examples}")


def require_deterministic_order(frame: pd.DataFrame, keys: Sequence[str], *, surface: str) -> None:
    require_columns(frame, keys, surface=surface)
    expected = frame.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    observed = frame.reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError(f"{surface}: rows are not in deterministic key order {list(keys)}")


def require_month_start(values: pd.Series, *, surface: str, field: str) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{surface}: {field} contains invalid or missing dates")
    if not (dates.dt.day == 1).all():
        examples = dates[dates.dt.day != 1].head(5).astype(str).tolist()
        raise ValueError(f"{surface}: {field} is not month-start; examples={examples}")
    return dates


def require_gapless_monthly_index(index: pd.Index, *, surface: str) -> None:
    dates = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if dates.isna().any():
        raise ValueError(f"{surface}: index contains invalid dates")
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError(f"{surface}: monthly index must be sorted and unique")
    if len(dates) > 1:
        expected = pd.date_range(dates.min(), dates.max(), freq="MS")
        if not dates.equals(expected):
            missing = expected.difference(dates).astype(str).tolist()[:5]
            raise ValueError(f"{surface}: monthly index has gaps; examples={missing}")


def require_forecast_calendar(frame: pd.DataFrame, *, surface: str) -> None:
    """Enforce target/origin/horizon semantics on every forecast row."""
    fields = ["horizon", "forecast_date", "origin_date", "target_date"]
    require_non_null(frame, fields, surface=surface)
    forecast = require_month_start(frame["forecast_date"], surface=surface, field="forecast_date")
    origin = require_month_start(frame["origin_date"], surface=surface, field="origin_date")
    target = require_month_start(frame["target_date"], surface=surface, field="target_date")
    if not forecast.equals(target):
        raise ValueError(f"{surface}: forecast_date must equal the evaluated target_date")
    month_gap = (target.dt.year - origin.dt.year) * 12 + (target.dt.month - origin.dt.month)
    horizons = pd.to_numeric(frame["horizon"], errors="coerce")
    mismatch = month_gap != horizons
    if mismatch.any():
        examples = frame.loc[mismatch, fields].head(5).to_dict("records")
        raise ValueError(f"{surface}: target-origin gap differs from horizon; examples={examples}")


def require_probability(values: pd.Series, *, surface: str, field: str = "probability") -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{surface}: {field} must be finite and in [0, 1]")


def require_binary(values: pd.Series, *, surface: str, field: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = set(numeric.dropna().unique())
    if numeric.isna().any() or not observed <= {0, 1}:
        raise ValueError(f"{surface}: {field} must contain only non-null 0/1 values; found={sorted(observed)}")


def require_finite_numeric(values: pd.Series, *, surface: str, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{surface}: {field} must contain only finite numeric values")
    return numeric


def require_nonnegative_integer(values: pd.Series, *, surface: str, field: str) -> pd.Series:
    numeric = require_finite_numeric(values, surface=surface, field=field)
    if (numeric < 0).any() or (numeric % 1 != 0).any():
        raise ValueError(f"{surface}: {field} must contain nonnegative integers")
    return numeric


def require_boolean(values: pd.Series, *, surface: str, field: str) -> pd.Series:
    if values.isna().any() or not all(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"{surface}: {field} must contain only non-null booleans")
    return values.astype(bool)


def require_nonempty_strings(values: pd.Series, *, surface: str, field: str) -> None:
    valid = values.map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not valid.all():
        raise ValueError(f"{surface}: {field} must contain only nonempty strings")
