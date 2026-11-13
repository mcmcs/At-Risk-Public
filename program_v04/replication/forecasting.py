"""Pure recursive-calendar and information-set contracts for forecast runners."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from .config import REALTIME_OOS_START, REVISED_OOS_START


class RealtimeCalendarPolicy(StrEnum):
    """Candidate treatment of a vintage panel that ends before its target."""

    LEGACY_SHIFT_TRUNCATED_PANEL = "legacy_shift_truncated_panel"
    LATEST_AVAILABLE_OBSERVATION = "latest_available_observation"


class PredictionRowPolicy(StrEnum):
    EXACT_TARGET = "exact_target"
    LEGACY_LAST_AVAILABLE = "legacy_last_available"


class RealtimeCalibrationPolicy(StrEnum):
    """Threshold and penalty information-set sharing rule (RF-006)."""

    LEGACY_REVISED_SHARED = "legacy_revised_shared"
    VARIANT_SPECIFIC = "variant_specific"


class PredictorCompositionPolicy(StrEnum):
    """Predictor-membership rule for revised/vintage comparison (RF-020)."""

    LEGACY_VARIANT_SPECIFIC = "legacy_variant_specific"
    COMMON_LOCKED = "common_locked"


def _month_distance(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return int((later.to_period("M") - earlier.to_period("M")).n)


def _require_month_start(value: pd.Timestamp, *, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp != timestamp.to_period("M").to_timestamp():
        raise ValueError(f"{field} must be a month-start date")
    return timestamp


def _validate_monthly_index(index: pd.Index, *, surface: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{surface}: index must be a DatetimeIndex")
    if index.empty or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{surface}: index must be nonempty, sorted, and unique")
    normalized = index.to_period("M").to_timestamp()
    if not normalized.equals(index):
        raise ValueError(f"{surface}: index must contain month-start dates")
    expected = pd.date_range(index.min(), index.max(), freq="MS")
    if not expected.equals(index):
        raise ValueError(f"{surface}: monthly index must be gapless")
    return index


@dataclass(frozen=True)
class ForecastPoint:
    target_date: pd.Timestamp
    origin_date: pd.Timestamp
    horizon: int
    vintage_key: str | None = None

    def validate(self) -> None:
        target = _require_month_start(self.target_date, field="target_date")
        origin = _require_month_start(self.origin_date, field="origin_date")
        if self.horizon < 1:
            raise ValueError("Forecast horizon must be positive")
        if _month_distance(target, origin) != self.horizon:
            raise ValueError("Forecast target/origin gap does not equal the horizon")
        if self.vintage_key is not None and self.vintage_key != origin.strftime("%Y-%m"):
            raise ValueError("Real-time vintage key must equal the forecast origin month")


def revised_forecast_calendar(
    target_index: pd.Index,
    *,
    horizon: int,
    oos_start: str = REVISED_OOS_START,
) -> tuple[ForecastPoint, ...]:
    """Return the locked revised target/origin calendar."""
    index = _validate_monthly_index(target_index, surface="revised calendar")
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    points = tuple(
        ForecastPoint(
            target_date=target,
            origin_date=target - pd.DateOffset(months=horizon),
            horizon=horizon,
        )
        for target in index[index >= pd.Timestamp(oos_start)]
    )
    if not points:
        raise ValueError("Revised forecast calendar is empty")
    for point in points:
        point.validate()
    return points


def realtime_forecast_calendar(
    target_index: pd.Index,
    *,
    horizon: int,
    first_origin: str = REALTIME_OOS_START,
) -> tuple[ForecastPoint, ...]:
    """Return targets whose horizon-specific origin has a declared vintage."""
    index = _validate_monthly_index(target_index, surface="real-time calendar")
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    first_target = pd.Timestamp(first_origin) + pd.DateOffset(months=horizon)
    points = tuple(
        ForecastPoint(
            target_date=target,
            origin_date=target - pd.DateOffset(months=horizon),
            horizon=horizon,
            vintage_key=(target - pd.DateOffset(months=horizon)).strftime("%Y-%m"),
        )
        for target in index[index >= first_target]
    )
    if not points:
        raise ValueError("Real-time forecast calendar is empty")
    for point in points:
        point.validate()
    return points


@dataclass(frozen=True)
class AlignedForecastPanel:
    frame: pd.DataFrame
    policy: RealtimeCalendarPolicy
    shift_months: int
    latest_available_date: pd.Timestamp
    prediction_row_date: pd.Timestamp
    contemporaneous_source_date: pd.Timestamp

    def validate(self, point: ForecastPoint) -> None:
        point.validate()
        _validate_monthly_index(self.frame.index, surface="aligned forecast panel")
        if self.shift_months < 1:
            raise ValueError("Aligned forecast shift must be positive")
        if self.prediction_row_date > point.target_date:
            raise ValueError("Prediction row cannot follow the target")
        expected_source = self.prediction_row_date - pd.DateOffset(months=self.shift_months)
        if self.contemporaneous_source_date != expected_source:
            raise ValueError("Aligned panel source-date provenance is inconsistent")


def align_panel_for_forecast(
    base_panel: pd.DataFrame,
    point: ForecastPoint,
    *,
    calendar_policy: RealtimeCalendarPolicy,
) -> AlignedForecastPanel:
    """Align one final or vintage panel and retain exact source-date provenance."""
    point.validate()
    _validate_monthly_index(base_panel.index, surface="forecast base panel")
    if base_panel.columns.has_duplicates or base_panel.empty:
        raise ValueError("Forecast base panel must be nonempty with unique columns")
    if not isinstance(calendar_policy, RealtimeCalendarPolicy):
        raise ValueError(f"Unsupported real-time calendar policy: {calendar_policy}")

    latest_available = pd.Timestamp(base_panel.index.max())
    if point.vintage_key is not None and latest_available > point.origin_date:
        raise ValueError("Vintage panel contains predictor observations after the forecast origin")
    if calendar_policy is RealtimeCalendarPolicy.LEGACY_SHIFT_TRUNCATED_PANEL:
        shift_months = point.horizon
        shifted = base_panel.shift(shift_months).loc[: point.target_date]
    elif latest_available >= point.target_date:
        shift_months = point.horizon
        shifted = base_panel.shift(shift_months).loc[: point.target_date]
    else:
        shift_months = _month_distance(point.target_date, latest_available)
        if shift_months < point.horizon:
            raise ValueError("Vintage latest observation is inconsistent with the horizon")
        extended_index = pd.date_range(
            base_panel.index.min(), point.target_date, freq="MS"
        )
        shifted = base_panel.reindex(extended_index).shift(shift_months).loc[: point.target_date]

    training_slice = shifted.loc[: point.origin_date]
    valid_columns = training_slice.columns[~training_slice.isna().all()]
    shifted = shifted.loc[:, list(valid_columns)]
    if shifted.empty or shifted.shape[1] == 0:
        raise ValueError("Aligned forecast panel has no training-supported predictors")
    prediction_row_date = (
        point.target_date if point.target_date in shifted.index else pd.Timestamp(shifted.index.max())
    )
    aligned = AlignedForecastPanel(
        frame=shifted,
        policy=calendar_policy,
        shift_months=shift_months,
        latest_available_date=latest_available,
        prediction_row_date=prediction_row_date,
        contemporaneous_source_date=(
            prediction_row_date - pd.DateOffset(months=shift_months)
        ),
    )
    aligned.validate(point)
    return aligned


@dataclass(frozen=True)
class PredictionRow:
    frame: pd.DataFrame
    row_date: pd.Timestamp


def select_prediction_row(
    predictor_data: pd.DataFrame,
    target_date: pd.Timestamp,
    *,
    row_policy: PredictionRowPolicy,
) -> PredictionRow:
    """Select one prediction row without an implicit vintage fallback."""
    _validate_monthly_index(predictor_data.index, surface="prediction-row input")
    target = _require_month_start(target_date, field="prediction target")
    if not isinstance(row_policy, PredictionRowPolicy):
        raise ValueError(f"Unsupported prediction-row policy: {row_policy}")
    if target in predictor_data.index:
        row_date = target
    elif row_policy is PredictionRowPolicy.LEGACY_LAST_AVAILABLE:
        available = predictor_data.loc[:target]
        if available.empty:
            raise ValueError("No prediction row exists on or before the target")
        row_date = pd.Timestamp(available.index.max())
    else:
        raise KeyError(f"Exact prediction row is missing for target {target.date()}")
    row = predictor_data.loc[[row_date]].copy()
    if row.isna().any().any():
        raise ValueError(f"Prediction row {row_date.date()} contains missing values")
    return PredictionRow(frame=row, row_date=row_date)


@dataclass(frozen=True)
class PredictorCompositionLock:
    columns: tuple[str, ...]
    source_panel_columns: tuple[tuple[str, tuple[str, ...]], ...]

    def validate(self) -> None:
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Composition lock columns must be nonempty and unique")
        if len(self.source_panel_columns) < 2:
            raise ValueError("Composition lock requires at least two source panels")
        names = [name for name, _columns in self.source_panel_columns]
        if len(set(names)) != len(names):
            raise ValueError("Composition lock source names must be unique")
        for name, columns in self.source_panel_columns:
            if not columns or len(set(columns)) != len(columns):
                raise ValueError(
                    f"Composition lock source {name} columns must be nonempty and unique"
                )
        common = set(self.source_panel_columns[0][1])
        for _name, columns in self.source_panel_columns[1:]:
            common &= set(columns)
        ordered_common = tuple(
            column for column in self.source_panel_columns[0][1] if column in common
        )
        if self.columns != ordered_common:
            raise ValueError(
                "Composition lock is not the exact ordered source-panel intersection"
            )


def lock_common_predictors(
    panels: dict[str, pd.DataFrame],
    *,
    registry_order: tuple[str, ...],
) -> PredictorCompositionLock:
    """Return one evaluation-wide predictor intersection in locked registry order."""
    if len(panels) < 2:
        raise ValueError("Common predictor lock requires at least two panels")
    if not registry_order or len(set(registry_order)) != len(registry_order):
        raise ValueError("Predictor registry order must be nonempty and unique")
    common = set(registry_order)
    for name, panel in panels.items():
        if panel.empty or panel.columns.has_duplicates:
            raise ValueError(f"{name}: composition panel must be nonempty and unique")
        common &= set(panel.columns)
    locked = tuple(variable for variable in registry_order if variable in common)
    if not locked:
        raise ValueError("No predictor is common across the declared panels")
    composition_lock = PredictorCompositionLock(
        columns=locked,
        source_panel_columns=tuple(
            (name, tuple(column for column in registry_order if column in panel.columns))
            for name, panel in sorted(panels.items())
        ),
    )
    composition_lock.validate()
    return composition_lock


def apply_realtime_composition(
    revised_panel: pd.DataFrame,
    vintage_panel: pd.DataFrame,
    *,
    composition_policy: PredictorCompositionPolicy,
    composition_lock: PredictorCompositionLock | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply an explicit RF-020 membership rule to the paired variants."""
    if not isinstance(composition_policy, PredictorCompositionPolicy):
        raise ValueError(f"Unsupported composition policy: {composition_policy}")
    if composition_policy is PredictorCompositionPolicy.LEGACY_VARIANT_SPECIFIC:
        if composition_lock is not None:
            raise ValueError("Legacy composition cannot accept a common composition lock")
        return {
            "revised_common_window": revised_panel.copy(),
            "realtime_vintage": vintage_panel.copy(),
        }
    if composition_lock is None:
        raise ValueError("Common composition requires an evaluation-wide lock")
    composition_lock.validate()
    locked_columns = composition_lock.columns
    missing_revised = [column for column in locked_columns if column not in revised_panel]
    missing_vintage = [column for column in locked_columns if column not in vintage_panel]
    if missing_revised or missing_vintage:
        raise KeyError(
            "Locked common predictors are missing: "
            f"revised={missing_revised}, vintage={missing_vintage}"
        )
    return {
        "revised_common_window": revised_panel.loc[:, list(locked_columns)].copy(),
        "realtime_vintage": vintage_panel.loc[:, list(locked_columns)].copy(),
    }


def calibration_panels_by_variant(
    revised_panel: pd.DataFrame,
    vintage_panel: pd.DataFrame,
    *,
    calibration_policy: RealtimeCalibrationPolicy,
    calibration_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Choose threshold-calibration inputs without an implicit RF-006 rule."""
    if not isinstance(calibration_policy, RealtimeCalibrationPolicy):
        raise ValueError(f"Unsupported calibration policy: {calibration_policy}")
    end = _require_month_start(calibration_end, field="calibration_end")
    revised_calibration = revised_panel.loc[:end]
    vintage_calibration = vintage_panel.loc[:end]
    if revised_calibration.empty or vintage_calibration.empty:
        raise ValueError("Real-time calibration panel is empty through the declared end")
    if calibration_policy is RealtimeCalibrationPolicy.LEGACY_REVISED_SHARED:
        return {
            "revised_common_window": revised_calibration,
            "realtime_vintage": revised_calibration,
        }
    return {
        "revised_common_window": revised_calibration,
        "realtime_vintage": vintage_calibration,
    }


def tuning_scope(
    data_variant: str,
    *,
    calibration_policy: RealtimeCalibrationPolicy,
) -> str:
    """Return the frozen-C namespace implied by the RF-006 candidate."""
    allowed = {"revised_common_window", "realtime_vintage"}
    if data_variant not in allowed:
        raise ValueError(f"Unknown real-time data variant: {data_variant}")
    if not isinstance(calibration_policy, RealtimeCalibrationPolicy):
        raise ValueError(f"Unsupported calibration policy: {calibration_policy}")
    if calibration_policy is RealtimeCalibrationPolicy.LEGACY_REVISED_SHARED:
        return "revised_shared"
    return data_variant


@dataclass(frozen=True)
class RecursiveDesign:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_predict: pd.DataFrame
    prediction_row_date: pd.Timestamp


def build_recursive_design(
    predictor_data: pd.DataFrame,
    target: pd.Series,
    point: ForecastPoint,
    *,
    prediction_row_policy: PredictionRowPolicy,
    minimum_training_rows: int,
) -> RecursiveDesign:
    """Align one expanding-window design and fail on incomplete required rows."""
    point.validate()
    if minimum_training_rows < 1:
        raise ValueError("Minimum training rows must be positive")
    if not target.index.is_unique or not target.index.is_monotonic_increasing:
        raise ValueError("Recursive target index must be sorted and unique")
    _validate_monthly_index(target.index, surface="recursive target")
    if point.target_date not in target.index:
        raise KeyError(f"Recursive target is missing {point.target_date.date()}")
    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_target.isna().any() or not set(numeric_target.unique()) <= {0, 1}:
        raise ValueError("Recursive target must be strictly binary")
    prediction = select_prediction_row(
        predictor_data,
        point.target_date,
        row_policy=prediction_row_policy,
    )
    X_train_candidate = predictor_data.loc[: point.origin_date]
    common = target.loc[: point.origin_date].index.intersection(
        X_train_candidate.index, sort=False
    )
    X_train = X_train_candidate.loc[common].copy()
    y_train = target.loc[common].copy()
    if len(X_train) < minimum_training_rows:
        raise ValueError(
            f"Recursive design has {len(X_train)} rows; requires {minimum_training_rows}"
        )
    if X_train.isna().any().any() or prediction.frame.isna().any().any():
        raise ValueError("Recursive design contains missing predictors")
    if list(X_train.columns) != list(prediction.frame.columns):
        raise ValueError("Recursive training and prediction columns differ")
    values = np.concatenate(
        [X_train.to_numpy(dtype=float).ravel(), prediction.frame.to_numpy(dtype=float).ravel()]
    )
    if not np.isfinite(values).all():
        raise ValueError("Recursive design predictors must be finite")
    return RecursiveDesign(
        X_train=X_train,
        y_train=y_train,
        X_predict=prediction.frame,
        prediction_row_date=prediction.row_date,
    )
