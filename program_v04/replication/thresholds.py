"""Audited threshold estimands for at-risk state construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from .transforms import RollingBoundary, rolling_signal
from .validation import require_binary


RANK_TIE_METHOD = "average"


class RecessionSamplePolicy(StrEnum):
    """Missing-rank rule for variable and sector threshold calibration (RF-022)."""

    LEGACY_COMPLETE_CASE_ROWS = "legacy_complete_case_rows"
    PER_VARIABLE_AVAILABLE_ROWS = "per_variable_available_rows"


@dataclass(frozen=True)
class ThresholdCalibration:
    """Threshold values plus the exact calibration information-set provenance."""

    global_by_window: dict[int, float]
    sector: dict[str, float]
    variable: dict[str, float]
    recession_sample_policy: RecessionSamplePolicy
    calibration_columns: tuple[str, ...]
    predictor_end: str
    target_end: str

    def validate(self) -> None:
        if not self.global_by_window:
            raise ValueError("Threshold calibration has no global quantile")
        if not self.calibration_columns or len(set(self.calibration_columns)) != len(
            self.calibration_columns
        ):
            raise ValueError("Threshold calibration columns must be nonempty and unique")
        if not isinstance(self.recession_sample_policy, RecessionSamplePolicy):
            raise ValueError("Threshold calibration has an invalid recession-sample policy")
        for field, value in (("predictor_end", self.predictor_end), ("target_end", self.target_end)):
            try:
                pd.Timestamp(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Threshold calibration {field} is not a valid date") from exc
        for namespace, values in (
            ("global", self.global_by_window),
            ("sector", self.sector),
            ("variable", self.variable),
        ):
            for key, value in values.items():
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{namespace} threshold {key!r} is outside [0, 1]")


def _target_series(target: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("Threshold calibration target frame must have exactly one column")
        target = target.iloc[:, 0]
    if not target.index.is_unique:
        raise ValueError("Threshold calibration target index must be unique")
    require_binary(target, surface="threshold_calibration", field="target")
    return pd.to_numeric(target)


def unify_cycle_direction(
    predictors: pd.DataFrame,
    *,
    counter_cyclical: set[str] | frozenset[str],
) -> pd.DataFrame:
    """Negate counter-cyclical signals so low unified values mean at risk."""
    unified = predictors.copy()
    for variable in counter_cyclical:
        if variable in unified.columns:
            unified[variable] = -unified[variable]
    return unified


def _recession_index(predictors: pd.DataFrame, target: pd.Series) -> pd.Index:
    common = predictors.index.intersection(target.index, sort=False)
    recession = common[target.loc[common].to_numpy() == 1]
    if len(recession) == 0:
        raise ValueError("Threshold calibration sample contains no recession observations")
    return recession


def empirical_unified_quantile(
    predictors: pd.DataFrame,
    target: pd.Series | pd.DataFrame,
    *,
    counter_cyclical: set[str] | frozenset[str],
    momentum_window: int = 1,
    rolling_boundary: RollingBoundary | None = None,
    rank_method: str = RANK_TIE_METHOD,
) -> float:
    """Median across variables of median recession-period empirical ranks."""
    if predictors.empty:
        raise ValueError("Threshold calibration predictors are empty")
    if momentum_window > 1 and rolling_boundary is None:
        raise ValueError("Moving-average threshold calibration requires RollingBoundary (RF-011)")
    boundary = rolling_boundary or RollingBoundary.COMPLETE
    target_series = _target_series(target)
    recession = _recession_index(predictors, target_series)
    unified = unify_cycle_direction(predictors, counter_cyclical=counter_cyclical)
    medians: list[float] = []
    for variable in unified.columns:
        signal = rolling_signal(unified[variable], momentum_window, boundary=boundary)
        ranks = signal.rank(pct=True, method=rank_method)
        observed = ranks.loc[ranks.index.intersection(recession, sort=False)].dropna()
        if not observed.empty:
            medians.append(float(observed.median()))
    if not medians:
        raise ValueError("No variable has a defined recession-period rank")
    return float(np.median(np.asarray(medians, dtype=float)))


def empirical_variable_quantiles(
    predictors: pd.DataFrame,
    target: pd.Series | pd.DataFrame,
    *,
    counter_cyclical: set[str] | frozenset[str],
    recession_sample_policy: RecessionSamplePolicy,
    rank_method: str = RANK_TIE_METHOD,
) -> dict[str, float]:
    """Variable-specific median recession ranks under an explicit missingness rule."""
    target_series = _target_series(target)
    recession = _recession_index(predictors, target_series)
    ranks = unify_cycle_direction(
        predictors, counter_cyclical=counter_cyclical
    ).rank(pct=True, method=rank_method)
    recession_ranks = ranks.loc[ranks.index.intersection(recession, sort=False)]
    if recession_sample_policy is RecessionSamplePolicy.LEGACY_COMPLETE_CASE_ROWS:
        recession_ranks = recession_ranks.dropna()
    elif recession_sample_policy is not RecessionSamplePolicy.PER_VARIABLE_AVAILABLE_ROWS:
        raise ValueError(f"Unsupported recession-sample policy: {recession_sample_policy}")
    medians = recession_ranks.median(skipna=True)
    values = {name: float(value) for name, value in medians.items() if pd.notna(value)}
    if not values:
        raise ValueError("No variable-specific recession quantile is defined")
    return values


def empirical_sector_quantiles(
    predictors: pd.DataFrame,
    target: pd.Series | pd.DataFrame,
    *,
    counter_cyclical: set[str] | frozenset[str],
    variable_groups: dict[str, tuple[str, ...]],
    recession_sample_policy: RecessionSamplePolicy,
    rank_method: str = RANK_TIE_METHOD,
) -> dict[str, float]:
    """Median across variable-specific recession ranks within each sector."""
    variable = empirical_variable_quantiles(
        predictors,
        target,
        counter_cyclical=counter_cyclical,
        recession_sample_policy=recession_sample_policy,
        rank_method=rank_method,
    )
    sectors: dict[str, float] = {}
    for sector, members in variable_groups.items():
        present = [variable[name] for name in members if name in variable]
        if present:
            sectors[sector] = float(np.median(np.asarray(present, dtype=float)))
    if not sectors:
        raise ValueError("No sector-specific recession quantile is defined")
    return sectors


def calibrate_thresholds(
    predictors: pd.DataFrame,
    target: pd.Series | pd.DataFrame,
    *,
    windows: tuple[int, ...],
    rolling_boundary: RollingBoundary | None,
    counter_cyclical: set[str] | frozenset[str],
    variable_groups: dict[str, tuple[str, ...]],
    recession_sample_policy: RecessionSamplePolicy,
    rank_method: str = RANK_TIE_METHOD,
) -> ThresholdCalibration:
    """Calibrate every requested threshold estimand on one training slice."""
    if not windows or any(window < 1 for window in windows) or len(set(windows)) != len(windows):
        raise ValueError("Threshold windows must be unique positive integers")
    target_series = _target_series(target)
    global_by_window = {
        window: empirical_unified_quantile(
            predictors,
            target,
            counter_cyclical=counter_cyclical,
            momentum_window=window,
            rolling_boundary=rolling_boundary if window > 1 else None,
            rank_method=rank_method,
        )
        for window in windows
    }
    calibration = ThresholdCalibration(
        global_by_window=global_by_window,
        sector=empirical_sector_quantiles(
            predictors,
            target,
            counter_cyclical=counter_cyclical,
            variable_groups=variable_groups,
            recession_sample_policy=recession_sample_policy,
            rank_method=rank_method,
        ),
        variable=empirical_variable_quantiles(
            predictors,
            target,
            counter_cyclical=counter_cyclical,
            recession_sample_policy=recession_sample_policy,
            rank_method=rank_method,
        ),
        recession_sample_policy=recession_sample_policy,
        calibration_columns=tuple(predictors.columns),
        predictor_end=pd.Timestamp(predictors.index.max()).isoformat(),
        target_end=pd.Timestamp(target_series.index.max()).isoformat(),
    )
    calibration.validate()
    return calibration
