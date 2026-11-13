"""Audited feature-importance extraction and paper-facing rollups."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from .data import validate_importance_panel
from .predictors import infer_sector, strip_lag_suffix, to_display_name


class UnsupportedImportancePolicy(StrEnum):
    RAISE = "raise"
    RECORD_ZERO_VECTOR = "record_zero_vector"


class DegenerateImportancePolicy(StrEnum):
    RAISE = "raise"
    EXCLUDE_RECORDED = "exclude_recorded"


class EpisodeEndpointPolicy(StrEnum):
    """Candidate endpoint semantics for the RF-010 pre-recession windows."""

    LEGACY_DECLARED = "legacy_declared"
    END_BEFORE_PEAK = "end_before_peak"


class IncompleteEpisodePolicy(StrEnum):
    """Candidate treatment of the six-month observable 1990 window (RF-010)."""

    REQUIRE_COMPLETE = "require_complete"
    ALLOW_AVAILABLE_WITH_COUNT = "allow_available_with_count"


@dataclass(frozen=True)
class FeatureImportance:
    values: pd.Series
    importance_type: str
    is_degenerate_model: bool
    degenerate_reason: str

    def validate(self, feature_names: tuple[str, ...]) -> None:
        if tuple(self.values.index) != feature_names:
            raise ValueError("Importance values do not match the declared feature order")
        values = self.values.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("Importance values must be finite and nonnegative")
        if not self.importance_type:
            raise ValueError("Importance type must be nonempty")
        if self.is_degenerate_model and not self.degenerate_reason:
            raise ValueError("Degenerate importance requires a reason")
        if not self.is_degenerate_model and self.degenerate_reason:
            raise ValueError("Nondegenerate importance cannot carry a degeneracy reason")


def _validate_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Feature names must be nonempty and unique")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Feature names must be nonempty strings")
    return names


def _xgboost_feature_map(
    raw_keys: Iterable[str],
    feature_names: tuple[str, ...],
) -> dict[str, str]:
    mapped: dict[str, str] = {}
    unknown: list[str] = []
    for raw_key in raw_keys:
        if raw_key in feature_names:
            mapped[raw_key] = raw_key
        elif raw_key.startswith("f") and raw_key[1:].isdigit():
            position = int(raw_key[1:])
            if 0 <= position < len(feature_names):
                mapped[raw_key] = feature_names[position]
            else:
                unknown.append(raw_key)
        else:
            unknown.append(raw_key)
    if unknown:
        raise ValueError(f"XGBoost returned unmapped feature keys: {sorted(unknown)}")
    return mapped


def extract_feature_importance(
    model_name: str,
    fitted_model: object,
    feature_names: Iterable[str],
    *,
    unsupported_policy: UnsupportedImportancePolicy,
) -> FeatureImportance:
    """Return a feature-aligned vector with explicit unsupported handling."""
    names = _validate_feature_names(feature_names)
    if not isinstance(unsupported_policy, UnsupportedImportancePolicy):
        raise ValueError(f"Unsupported importance policy: {unsupported_policy}")
    if model_name in {"Logit_L2", "Logit_L1", "ElasticNet"}:
        coefficients = np.asarray(getattr(fitted_model, "coef_", None), dtype=float)
        if coefficients.shape != (1, len(names)):
            raise ValueError(f"{model_name}: coefficient shape does not match features")
        values = np.abs(coefficients[0])
        is_degenerate = float(values.sum()) <= 0.0
        result = FeatureImportance(
            values=pd.Series(values, index=names, dtype=float),
            importance_type="abs_coef",
            is_degenerate_model=is_degenerate,
            degenerate_reason=(
                "logistic fit returned no nonzero coefficients" if is_degenerate else ""
            ),
        )
    elif model_name == "RandomForest":
        values = np.asarray(getattr(fitted_model, "feature_importances_", None), dtype=float)
        if values.shape != (len(names),):
            raise ValueError("RandomForest importance shape does not match features")
        result = FeatureImportance(
            values=pd.Series(values, index=names, dtype=float),
            importance_type="gini",
            is_degenerate_model=float(values.sum()) <= 0.0,
            degenerate_reason=(
                "forest returned no positive Gini importance"
                if float(values.sum()) <= 0.0
                else ""
            ),
        )
    elif model_name == "XGBoost":
        booster = fitted_model.get_booster()
        score_map = booster.get_score(importance_type="gain")
        translated = _xgboost_feature_map(score_map, names)
        values = pd.Series(0.0, index=names, dtype=float)
        for raw_key, value in score_map.items():
            values.loc[translated[raw_key]] = float(value)
        is_degenerate = float(values.sum()) <= 0.0
        result = FeatureImportance(
            values=values,
            importance_type="gain",
            is_degenerate_model=is_degenerate,
            degenerate_reason=(
                "booster returned no positive gain scores" if is_degenerate else ""
            ),
        )
    elif unsupported_policy is UnsupportedImportancePolicy.RAISE:
        raise NotImplementedError(f"{model_name}: importance extraction is unsupported")
    else:
        result = FeatureImportance(
            values=pd.Series(0.0, index=names, dtype=float),
            importance_type="unsupported",
            is_degenerate_model=True,
            degenerate_reason=f"{model_name}: importance extraction is unsupported",
        )
    result.validate(names)
    return result


def build_importance_rows(
    *,
    exercise: str,
    data_variant: str,
    forecast_family: str,
    horizon: int,
    forecast_date: pd.Timestamp,
    predictor_key: str,
    model_name: str,
    result: FeatureImportance,
) -> list[dict[str, object]]:
    """Emit one long-form row for every declared feature of one fitted model."""
    names = _validate_feature_names(result.values.index)
    result.validate(names)
    rows: list[dict[str, object]] = []
    for feature_name, value in result.values.items():
        base_name = strip_lag_suffix(feature_name)
        rows.append(
            {
                "exercise": exercise,
                "data_variant": data_variant,
                "forecast_family": forecast_family,
                "horizon": horizon,
                "forecast_date": pd.Timestamp(forecast_date),
                "predictor_key": predictor_key,
                "model_name": model_name,
                "feature_name": feature_name,
                "base_feature_name": base_name,
                "display_name": to_display_name(base_name),
                "sector": infer_sector(base_name),
                "importance": float(value),
                "importance_type": result.importance_type,
                "is_degenerate_model": result.is_degenerate_model,
                "degenerate_reason": result.degenerate_reason,
            }
        )
    return rows


def _importance_subset(
    panel: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
    forecast_family: str,
    horizon: int,
    predictor_key: str,
    model_name: str,
    degeneracy_policy: DegenerateImportancePolicy,
) -> pd.DataFrame:
    validate_importance_panel(panel, surface="importance summary source")
    if not isinstance(exercise, str) or not exercise:
        raise ValueError("Importance summary exercise must be a nonempty string")
    if not isinstance(degeneracy_policy, DegenerateImportancePolicy):
        raise ValueError(f"Unsupported degeneracy policy: {degeneracy_policy}")
    subset = panel[
        (panel["exercise"] == exercise)
        & (panel["data_variant"] == data_variant)
        & (panel["forecast_family"] == forecast_family)
        & (panel["horizon"] == horizon)
        & (panel["predictor_key"] == predictor_key)
        & (panel["model_name"] == model_name)
    ].copy()
    if subset.empty:
        raise ValueError("Importance summary source selection is empty")
    degenerate_dates = subset.loc[
        subset["is_degenerate_model"], "forecast_date"
    ].drop_duplicates()
    if not degenerate_dates.empty:
        if degeneracy_policy is DegenerateImportancePolicy.RAISE:
            raise RuntimeError(
                f"Importance summary contains {len(degenerate_dates)} degenerate fits"
            )
        subset = subset[~subset["is_degenerate_model"]].copy()
    if subset.empty:
        raise RuntimeError("No nondegenerate importance dates remain")
    return subset


def summarize_top_features(
    feature_importances: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
    forecast_family: str,
    horizon: int,
    predictor_key: str,
    model_name: str,
    top_n: int,
    degeneracy_policy: DegenerateImportancePolicy,
) -> pd.DataFrame:
    """Aggregate lags and average deterministic within-date importance shares."""
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    subset = _importance_subset(
        feature_importances,
        exercise=exercise,
        data_variant=data_variant,
        forecast_family=forecast_family,
        horizon=horizon,
        predictor_key=predictor_key,
        model_name=model_name,
        degeneracy_policy=degeneracy_policy,
    )
    per_date = subset.groupby(
        ["forecast_date", "base_feature_name", "display_name"],
        as_index=False,
        sort=False,
    )["importance"].sum()
    totals = per_date.groupby("forecast_date", as_index=False, sort=False)[
        "importance"
    ].sum().rename(columns={"importance": "total_importance"})
    per_date = per_date.merge(totals, on="forecast_date", validate="many_to_one")
    if (per_date["total_importance"] <= 0.0).any():
        raise RuntimeError("Importance summary contains a zero-total date")
    per_date["importance_share_pct"] = (
        100.0 * per_date["importance"] / per_date["total_importance"]
    )
    summary = per_date.groupby(
        ["base_feature_name", "display_name"], as_index=False, sort=False
    )["importance_share_pct"].mean()
    return (
        summary.sort_values(
            ["importance_share_pct", "base_feature_name"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(top_n)
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class EpisodeDefinition:
    label: str
    peak_date: pd.Timestamp
    legacy_start: pd.Timestamp
    legacy_end: pd.Timestamp


@dataclass(frozen=True)
class EpisodeWindow:
    label: str
    peak_date: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp
    endpoint_policy: EpisodeEndpointPolicy

    def validate(self) -> None:
        dates = [pd.Timestamp(value) for value in (self.peak_date, self.start, self.end)]
        if any(value != value.to_period("M").to_timestamp() for value in dates):
            raise ValueError(f"{self.label}: episode dates must be month-start")
        if self.start > self.end:
            raise ValueError(f"{self.label}: episode window starts after it ends")
        months = int((self.end.to_period("M") - self.start.to_period("M")).n) + 1
        if months != 12:
            raise ValueError(f"{self.label}: episode window must declare twelve months")
        if not isinstance(self.endpoint_policy, EpisodeEndpointPolicy):
            raise ValueError(f"{self.label}: invalid episode endpoint policy")


PRE_RECESSION_EPISODES = (
    EpisodeDefinition(
        "1990",
        pd.Timestamp("1990-07-01"),
        pd.Timestamp("1989-07-01"),
        pd.Timestamp("1990-06-01"),
    ),
    EpisodeDefinition(
        "2001",
        pd.Timestamp("2001-03-01"),
        pd.Timestamp("2000-04-01"),
        pd.Timestamp("2001-03-01"),
    ),
    EpisodeDefinition(
        "2008",
        pd.Timestamp("2007-12-01"),
        pd.Timestamp("2007-01-01"),
        pd.Timestamp("2007-12-01"),
    ),
    EpisodeDefinition(
        "2020",
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2019-03-01"),
        pd.Timestamp("2020-02-01"),
    ),
)


def build_episode_windows(
    *,
    endpoint_policy: EpisodeEndpointPolicy,
    episodes: tuple[EpisodeDefinition, ...] = PRE_RECESSION_EPISODES,
) -> tuple[EpisodeWindow, ...]:
    """Build explicit legacy or peak-exclusive twelve-month RF-010 windows."""
    if not isinstance(endpoint_policy, EpisodeEndpointPolicy):
        raise ValueError(f"Unsupported episode endpoint policy: {endpoint_policy}")
    windows: list[EpisodeWindow] = []
    for episode in episodes:
        if endpoint_policy is EpisodeEndpointPolicy.LEGACY_DECLARED:
            start, end = episode.legacy_start, episode.legacy_end
        else:
            end = episode.peak_date - pd.DateOffset(months=1)
            start = end - pd.DateOffset(months=11)
        window = EpisodeWindow(
            label=episode.label,
            peak_date=episode.peak_date,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            endpoint_policy=endpoint_policy,
        )
        window.validate()
        windows.append(window)
    return tuple(windows)


def summarize_sector_contributions(
    feature_importances: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
    forecast_family: str,
    horizon: int,
    predictor_key: str,
    model_name: str,
    windows: tuple[EpisodeWindow, ...],
    degeneracy_policy: DegenerateImportancePolicy,
    incomplete_policy: IncompleteEpisodePolicy,
) -> pd.DataFrame:
    """Compute window-total sector shares with explicit RF-010 completeness."""
    if not windows:
        raise ValueError("Sector contributions require episode windows")
    if not isinstance(incomplete_policy, IncompleteEpisodePolicy):
        raise ValueError(f"Unsupported incomplete episode policy: {incomplete_policy}")
    for window in windows:
        window.validate()
    subset = _importance_subset(
        feature_importances,
        exercise=exercise,
        data_variant=data_variant,
        forecast_family=forecast_family,
        horizon=horizon,
        predictor_key=predictor_key,
        model_name=model_name,
        degeneracy_policy=degeneracy_policy,
    )
    per_date_sector = subset.groupby(
        ["forecast_date", "sector"], as_index=False, sort=False
    )["importance"].sum()
    rows: list[dict[str, object]] = []
    for window in windows:
        expected_dates = pd.date_range(window.start, window.end, freq="MS")
        selected = per_date_sector[
            per_date_sector["forecast_date"].between(window.start, window.end)
        ].copy()
        observed_dates = pd.DatetimeIndex(
            selected["forecast_date"].drop_duplicates().sort_values()
        )
        off_grid_dates = observed_dates.difference(expected_dates)
        if len(off_grid_dates):
            examples = off_grid_dates.astype(str).tolist()[:5]
            raise RuntimeError(
                f"{window.label}: episode contains off-grid forecast dates; "
                f"examples={examples}"
            )
        missing_dates = expected_dates.difference(observed_dates)
        if len(observed_dates) + len(missing_dates) != len(expected_dates):
            raise RuntimeError(f"{window.label}: episode coverage counts are incoherent")
        if len(missing_dates) and incomplete_policy is IncompleteEpisodePolicy.REQUIRE_COMPLETE:
            raise RuntimeError(
                f"{window.label}: episode window is missing {len(missing_dates)} of "
                f"{len(expected_dates)} forecast months"
            )
        if selected.empty:
            raise RuntimeError(f"{window.label}: episode has no observable importance dates")
        total = float(selected["importance"].sum())
        if total <= 0.0:
            raise RuntimeError(f"{window.label}: episode has zero total importance")
        by_sector = selected.groupby("sector", as_index=False, sort=False)[
            "importance"
        ].sum()
        for row in by_sector.itertuples(index=False):
            rows.append(
                {
                    "episode_label": window.label,
                    "peak_date": window.peak_date,
                    "window_start": window.start,
                    "window_end": window.end,
                    "endpoint_policy": window.endpoint_policy.value,
                    "incomplete_policy": incomplete_policy.value,
                    "expected_month_count": len(expected_dates),
                    "observed_month_count": len(observed_dates),
                    "missing_month_count": len(missing_dates),
                    "sector": row.sector,
                    "share_pct": 100.0 * float(row.importance) / total,
                }
            )
    return pd.DataFrame(rows)
