"""Audited shared robustness estimands and featured-model contracts."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import (
    validate_forecast_panel,
    validate_metric_panel,
)
from .parity import (
    FEATURED_DUPLICATED_PAIRS,
    FEATURED_MODEL_SOURCES,
    _require_surface_selector,
    validate_featured_cross_family_parity,
)
from .validation import (
    require_binary,
    require_gapless_monthly_index,
    require_month_start,
)


FEATURED_MODEL_IDS = {
    "proposed": ("Zt_Disagg", "Logit_L2"),
    "xt_logit_l2": ("Xt_Full", "Logit_L2"),
    "xt_xgboost": ("Xt_Full", "XGBoost"),
    "xt_random_forest": ("Xt_Full", "RandomForest"),
}
FEATURED_MODEL_LABELS = {
    "proposed": r"$Z_t$ + Logit-L2",
    "xt_logit_l2": r"$X_t$ + Logit-L2",
    "xt_xgboost": r"$X_t$ + XGBoost",
    "xt_random_forest": r"$X_t$ + Random Forest",
}
FEATURED_BENCHMARK_IDS = (
    "xt_logit_l2",
    "xt_xgboost",
    "xt_random_forest",
)
FEATURED_BENCHMARK_COLORS = {
    "xt_logit_l2": "#2166AC",
    "xt_xgboost": "#1B7837",
    "xt_random_forest": "#B35806",
}

SAMPLING_UNCERTAINTY_SEED = 20260727
SAMPLING_UNCERTAINTY_VALID_REPLICATIONS = 10_000
SAMPLING_UNCERTAINTY_ATTEMPT_CAP = 50_000
SAMPLING_UNCERTAINTY_BLOCK_LENGTHS = (12, 6, 24)
SAMPLING_UNCERTAINTY_PRIMARY_BLOCK_LENGTH = 12
SAMPLING_UNCERTAINTY_CONFIDENCE_LEVEL = 0.90
ASYMMETRIC_LOSS_COST_RATIOS = (1, 2, 5, 10, 20)


def _positive_unique_ints(values: Iterable[int], *, field: str) -> tuple[int, ...]:
    result = tuple(values)
    if (
        not result
        or len(set(result)) != len(result)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in result
        )
    ):
        raise ValueError(f"{field} must contain unique positive integers")
    return result


def build_featured_common_forecasts(
    main_forecasts: pd.DataFrame,
    extended_forecasts: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
    horizon: int,
) -> pd.DataFrame:
    """Return a complete common target-month panel for the featured models."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Featured-model horizon must be a positive integer")
    validate_featured_cross_family_parity(
        main_forecasts,
        extended_forecasts,
        exercise=exercise,
        data_variant=data_variant,
    )
    family_panels = {"main": main_forecasts, "extended": extended_forecasts}
    common: pd.DataFrame | None = None
    expected_dates: pd.DatetimeIndex | None = None
    actual_columns: list[str] = []

    for model_id, pair in FEATURED_MODEL_IDS.items():
        family = FEATURED_MODEL_SOURCES[pair]
        source = family_panels[family]
        selected = source[
            (source["exercise"] == exercise)
            & (source["data_variant"] == data_variant)
            & (source["forecast_family"] == family)
            & (source["horizon"] == horizon)
            & (source["predictor_key"] == pair[0])
            & (source["model_name"] == pair[1])
        ][["target_date", "actual", "probability"]].copy()
        if selected.empty:
            raise ValueError(f"Featured common panel is missing {model_id} at h={horizon}")
        if selected["target_date"].duplicated().any():
            raise ValueError(
                f"Featured common panel has duplicate target dates for {model_id}"
            )
        selected = selected.sort_values("target_date", kind="mergesort").reset_index(
            drop=True
        )
        model_dates = pd.DatetimeIndex(selected["target_date"])
        require_gapless_monthly_index(
            model_dates,
            surface=f"featured {model_id} target dates at h={horizon}",
        )
        if expected_dates is None:
            expected_dates = model_dates
        elif not model_dates.equals(expected_dates):
            raise ValueError(
                f"Featured common panel target dates differ for {model_id} at h={horizon}"
            )

        actual_column = f"actual__{model_id}"
        actual_columns.append(actual_column)
        selected = selected.rename(
            columns={"actual": actual_column, "probability": model_id}
        )
        common = (
            selected
            if common is None
            else common.merge(selected, on="target_date", how="inner", validate="one_to_one")
        )

    if common is None or common.empty:
        raise ValueError(f"Featured common panel is empty at h={horizon}")
    reference_actual = common[actual_columns[0]]
    if any(not common[column].equals(reference_actual) for column in actual_columns[1:]):
        raise ValueError(f"Featured common panel outcomes differ at h={horizon}")
    if set(reference_actual.unique()) != {0, 1}:
        raise ValueError(f"Featured common panel requires both outcome classes at h={horizon}")

    return (
        common[["target_date", actual_columns[0], *FEATURED_MODEL_IDS]]
        .rename(columns={actual_columns[0]: "actual"})
        .sort_values("target_date", kind="mergesort")
        .reset_index(drop=True)
    )


def stationary_bootstrap_counts(
    actual: np.ndarray,
    *,
    expected_block_length: int,
    valid_replications: int,
    attempt_cap: int,
    rng: np.random.Generator,
    batch_size: int = 1_000,
) -> tuple[np.ndarray, dict[str, int]]:
    """Draw class-mixed circular stationary-bootstrap count vectors."""
    for field, value, minimum in (
        ("expected_block_length", expected_block_length, 2),
        ("valid_replications", valid_replications, 1),
        ("attempt_cap", attempt_cap, 1),
        ("batch_size", batch_size, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"Stationary bootstrap {field} must be an integer >= {minimum}")
    if attempt_cap < valid_replications:
        raise ValueError("Stationary bootstrap requires attempt_cap >= valid_replications")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Stationary bootstrap requires an explicit numpy Generator")

    outcome = np.asarray(actual, dtype=float)
    if outcome.ndim != 1 or outcome.size == 0 or not np.isfinite(outcome).all():
        raise ValueError("Stationary bootstrap requires a finite nonempty outcome vector")
    if set(np.unique(outcome)) != {0.0, 1.0}:
        raise ValueError("Stationary bootstrap requires both binary outcome classes")
    sample_size = len(outcome)
    accepted: list[np.ndarray] = []
    attempted = 0
    degenerate = 0
    accepted_count = 0
    restart_probability = 1.0 / expected_block_length

    while accepted_count < valid_replications and attempted < attempt_cap:
        draw_count = min(batch_size, attempt_cap - attempted)
        indices = np.empty((draw_count, sample_size), dtype=np.int32)
        indices[:, 0] = rng.integers(0, sample_size, size=draw_count)
        for position in range(1, sample_size):
            restart = rng.random(draw_count) < restart_probability
            new_start = rng.integers(0, sample_size, size=draw_count)
            indices[:, position] = np.where(
                restart,
                new_start,
                (indices[:, position - 1] + 1) % sample_size,
            )

        positive_counts = outcome[indices].sum(axis=1)
        valid_mask = (positive_counts > 0) & (positive_counts < sample_size)
        needed = valid_replications - accepted_count
        valid_positions = np.flatnonzero(valid_mask)
        if len(valid_positions) >= needed:
            prefix_length = int(valid_positions[needed - 1]) + 1
            indices = indices[:prefix_length]
            valid_mask = valid_mask[:prefix_length]
        else:
            prefix_length = draw_count

        attempted += prefix_length
        degenerate += int((~valid_mask).sum())
        valid_indices = indices[valid_mask]
        if len(valid_indices):
            counts = np.zeros((len(valid_indices), sample_size), dtype=np.uint32)
            row_indices = np.repeat(np.arange(len(valid_indices)), sample_size)
            np.add.at(counts, (row_indices, valid_indices.ravel()), 1)
            accepted.append(counts)
            accepted_count += len(valid_indices)

    if accepted_count < valid_replications:
        raise RuntimeError(
            "Stationary bootstrap reached the attempt cap "
            f"({attempted} attempts, {accepted_count} valid, {degenerate} degenerate)"
        )
    weights = np.concatenate(accepted, axis=0)[:valid_replications]
    if not (weights.sum(axis=1) == sample_size).all():
        raise RuntimeError("Stationary bootstrap returned invalid count totals")
    counts = {
        "attempted_replications": attempted,
        "valid_replications": valid_replications,
        "degenerate_replications": degenerate,
    }
    if attempted != valid_replications + degenerate:
        raise RuntimeError("Stationary bootstrap replication counts are incoherent")
    return weights, counts


def weighted_average_precision(
    actual: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute exact tie-aware weighted average precision for many draws."""
    outcome = np.asarray(actual, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    sample_weights = np.asarray(weights)
    if outcome.ndim != 1 or probability.shape != outcome.shape:
        raise ValueError("Weighted average precision arrays must be aligned vectors")
    if set(np.unique(outcome)) != {0.0, 1.0}:
        raise ValueError("Weighted average precision requires both binary classes")
    if not np.isfinite(probability).all() or not (
        (probability >= 0.0) & (probability <= 1.0)
    ).all():
        raise ValueError("Weighted average precision requires finite probabilities in [0, 1]")
    if (
        sample_weights.ndim != 2
        or sample_weights.shape[0] == 0
        or sample_weights.shape[1] != len(outcome)
        or not np.isfinite(sample_weights.astype(float)).all()
        or (sample_weights < 0).any()
    ):
        raise ValueError("Weighted average precision requires aligned nonnegative weights")
    weighted_positive = sample_weights @ outcome
    weighted_negative = sample_weights @ (1.0 - outcome)
    if (weighted_positive <= 0).any() or (weighted_negative <= 0).any():
        raise ValueError("Weighted average precision requires both classes in every draw")

    order = np.argsort(-probability, kind="mergesort")
    sorted_probabilities = probability[order]
    sorted_actual = outcome[order]
    group_starts = np.r_[
        0,
        np.flatnonzero(sorted_probabilities[1:] != sorted_probabilities[:-1]) + 1,
    ]
    sorted_weights = sample_weights[:, order].astype(float, copy=False)
    group_weights = np.add.reduceat(sorted_weights, group_starts, axis=1)
    group_positive_weights = np.add.reduceat(
        sorted_weights * sorted_actual,
        group_starts,
        axis=1,
    )
    cumulative_weights = np.cumsum(group_weights, axis=1)
    cumulative_positives = np.cumsum(group_positive_weights, axis=1)
    precision = np.divide(
        cumulative_positives,
        cumulative_weights,
        out=np.zeros_like(cumulative_positives),
        where=cumulative_weights > 0,
    )
    total_positives = cumulative_positives[:, -1]
    return np.sum(group_positive_weights * precision, axis=1) / total_positives


def build_sampling_uncertainty_table(
    main_forecasts: pd.DataFrame,
    extended_forecasts: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
    horizons: tuple[int, ...] = (3, 6, 12),
    block_lengths: tuple[int, ...] = SAMPLING_UNCERTAINTY_BLOCK_LENGTHS,
    valid_replications: int = SAMPLING_UNCERTAINTY_VALID_REPLICATIONS,
    attempt_cap: int = SAMPLING_UNCERTAINTY_ATTEMPT_CAP,
    seed: int = SAMPLING_UNCERTAINTY_SEED,
    confidence_level: float = SAMPLING_UNCERTAINTY_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Build paired circular stationary-bootstrap percentile intervals."""
    horizon_values = _positive_unique_ints(horizons, field="Bootstrap horizons")
    block_values = _positive_unique_ints(
        block_lengths,
        field="Bootstrap block lengths",
    )
    if min(block_values) <= 1:
        raise ValueError("Bootstrap block lengths must exceed one")
    if SAMPLING_UNCERTAINTY_PRIMARY_BLOCK_LENGTH not in block_values:
        raise ValueError("Bootstrap block lengths must include the primary 12-month rule")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Bootstrap seed must be an integer")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0 < confidence_level < 1
    ):
        raise ValueError("Bootstrap confidence level must lie strictly between zero and one")
    tail = float((Decimal("1") - Decimal(str(confidence_level))) / 2)
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for horizon in horizon_values:
        common = build_featured_common_forecasts(
            main_forecasts,
            extended_forecasts,
            exercise=exercise,
            data_variant=data_variant,
            horizon=horizon,
        )
        actual = common["actual"].to_numpy(dtype=float)
        probabilities = {
            model_id: common[model_id].to_numpy(dtype=float)
            for model_id in FEATURED_MODEL_IDS
        }
        point_pr_auc = {
            model_id: float(average_precision_score(actual, values))
            for model_id, values in probabilities.items()
        }
        point_brier = {
            model_id: float(np.mean((values - actual) ** 2))
            for model_id, values in probabilities.items()
        }

        for expected_block_length in block_values:
            weights, replication_counts = stationary_bootstrap_counts(
                actual,
                expected_block_length=expected_block_length,
                valid_replications=valid_replications,
                attempt_cap=attempt_cap,
                rng=rng,
            )
            bootstrap_pr_auc = {
                model_id: weighted_average_precision(actual, values, weights)
                for model_id, values in probabilities.items()
            }
            bootstrap_brier = {
                model_id: weights @ ((values - actual) ** 2) / len(actual)
                for model_id, values in probabilities.items()
            }

            for benchmark_id in FEATURED_BENCHMARK_IDS:
                differences = {
                    "pr_auc": (
                        point_pr_auc["proposed"] - point_pr_auc[benchmark_id],
                        bootstrap_pr_auc["proposed"]
                        - bootstrap_pr_auc[benchmark_id],
                        "proposed_minus_benchmark",
                    ),
                    "brier": (
                        point_brier[benchmark_id] - point_brier["proposed"],
                        bootstrap_brier[benchmark_id]
                        - bootstrap_brier["proposed"],
                        "benchmark_minus_proposed",
                    ),
                }
                for metric, (point_estimate, bootstrap_values, orientation) in (
                    differences.items()
                ):
                    ci_lower, ci_upper = np.quantile(
                        bootstrap_values,
                        [tail, 1 - tail],
                    )
                    records.append(
                        {
                            "horizon": horizon,
                            "benchmark_id": benchmark_id,
                            "benchmark_label": FEATURED_MODEL_LABELS[benchmark_id],
                            "metric": metric,
                            "difference_orientation": orientation,
                            "bootstrap_type": "stationary_circular",
                            "interval_method": "percentile",
                            "confidence_level": confidence_level,
                            "point_estimate": point_estimate,
                            "ci_lower": float(ci_lower),
                            "ci_upper": float(ci_upper),
                            "expected_block_length": expected_block_length,
                            "is_primary": (
                                expected_block_length
                                == SAMPLING_UNCERTAINTY_PRIMARY_BLOCK_LENGTH
                            ),
                            "seed": seed,
                            "target_month_count": len(actual),
                            "recession_month_count": int(actual.sum()),
                            **replication_counts,
                        }
                    )
    columns = [
        "horizon",
        "benchmark_id",
        "benchmark_label",
        "metric",
        "difference_orientation",
        "bootstrap_type",
        "interval_method",
        "confidence_level",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "expected_block_length",
        "is_primary",
        "seed",
        "target_month_count",
        "recession_month_count",
        "attempted_replications",
        "valid_replications",
        "degenerate_replications",
    ]
    return pd.DataFrame(records, columns=columns)


def build_pre2020_robustness_table(
    main_forecasts: pd.DataFrame,
    extended_forecasts: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
    horizons: tuple[int, ...] = (3, 6, 12),
    pre2020_boundary: str = "2020-01-01",
) -> pd.DataFrame:
    """Compare full and target-date-restricted pre-2020 metrics without refit."""
    horizon_values = _positive_unique_ints(horizons, field="Pre-2020 horizons")
    boundary_series = pd.Series([pre2020_boundary])
    boundary = require_month_start(
        boundary_series,
        surface="pre-2020 robustness",
        field="boundary",
    ).iloc[0]
    records: list[dict[str, object]] = []
    boundary_rule = f"target_date_before_{boundary:%Y_%m_%d}"

    for horizon in horizon_values:
        common = build_featured_common_forecasts(
            main_forecasts,
            extended_forecasts,
            exercise=exercise,
            data_variant=data_variant,
            horizon=horizon,
        )
        windows = (
            ("full", "Full evaluation", "all_common_target_dates", common),
            (
                "pre_2020",
                "Pre-2020 evaluation",
                boundary_rule,
                common[common["target_date"] < boundary].reset_index(drop=True),
            ),
        )
        for window_id, window_label, window_rule, window in windows:
            if window.empty or set(window["actual"].unique()) != {0, 1}:
                raise ValueError(
                    f"Pre-2020 window {window_id} at h={horizon} requires both classes"
                )
            actual = window["actual"].to_numpy(dtype=float)
            target_month_count = len(window)
            recession_month_count = int(actual.sum())
            no_skill_pr_auc = recession_month_count / target_month_count
            for model_id, pair in FEATURED_MODEL_IDS.items():
                probabilities = window[model_id].to_numpy(dtype=float)
                records.append(
                    {
                        "analysis_type": "evaluation_window_restriction",
                        "window_id": window_id,
                        "window_label": window_label,
                        "window_rule": window_rule,
                        "window_start": window["target_date"].min(),
                        "window_end": window["target_date"].max(),
                        "horizon": horizon,
                        "model_id": model_id,
                        "model_label": FEATURED_MODEL_LABELS[model_id],
                        "source_family": FEATURED_MODEL_SOURCES[pair],
                        "predictor_key": pair[0],
                        "model_name": pair[1],
                        "pr_auc": float(average_precision_score(actual, probabilities)),
                        "brier": float(np.mean((probabilities - actual) ** 2)),
                        "target_month_count": target_month_count,
                        "recession_month_count": recession_month_count,
                        "no_skill_pr_auc": no_skill_pr_auc,
                    }
                )
    columns = [
        "analysis_type",
        "window_id",
        "window_label",
        "window_rule",
        "window_start",
        "window_end",
        "horizon",
        "model_id",
        "model_label",
        "source_family",
        "predictor_key",
        "model_name",
        "pr_auc",
        "brier",
        "target_month_count",
        "recession_month_count",
        "no_skill_pr_auc",
    ]
    return pd.DataFrame(records, columns=columns)


def build_cumulative_brier_data(
    main_forecasts: pd.DataFrame,
    extended_forecasts: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
    horizons: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Build period and cumulative benchmark-minus-proposed squared errors."""
    panels: list[pd.DataFrame] = []
    for horizon in _positive_unique_ints(horizons, field="Cumulative horizons"):
        common = build_featured_common_forecasts(
            main_forecasts,
            extended_forecasts,
            exercise=exercise,
            data_variant=data_variant,
            horizon=horizon,
        )
        actual = common["actual"].to_numpy(dtype=float)
        proposed_probability = common["proposed"].to_numpy(dtype=float)
        proposed_squared_error = (proposed_probability - actual) ** 2
        for benchmark_id in FEATURED_BENCHMARK_IDS:
            benchmark_probability = common[benchmark_id].to_numpy(dtype=float)
            benchmark_squared_error = (benchmark_probability - actual) ** 2
            period_advantage = benchmark_squared_error - proposed_squared_error
            panels.append(
                pd.DataFrame(
                    {
                        "horizon": horizon,
                        "benchmark_id": benchmark_id,
                        "benchmark_label": FEATURED_MODEL_LABELS[benchmark_id],
                        "benchmark_color": FEATURED_BENCHMARK_COLORS[benchmark_id],
                        "target_date": common["target_date"],
                        "actual": actual,
                        "proposed_probability": proposed_probability,
                        "benchmark_probability": benchmark_probability,
                        "proposed_squared_error": proposed_squared_error,
                        "benchmark_squared_error": benchmark_squared_error,
                        "period_brier_advantage": period_advantage,
                        "cumulative_brier_advantage": np.cumsum(period_advantage),
                        "difference_orientation": (
                            "benchmark_minus_proposed_squared_error"
                        ),
                    }
                )
            )
    columns = [
        "horizon",
        "benchmark_id",
        "benchmark_label",
        "benchmark_color",
        "target_date",
        "actual",
        "proposed_probability",
        "benchmark_probability",
        "proposed_squared_error",
        "benchmark_squared_error",
        "period_brier_advantage",
        "cumulative_brier_advantage",
        "difference_orientation",
    ]
    return pd.concat(panels, ignore_index=True)[columns]


def recession_month_spans(
    target_dates: pd.Series,
    actual: pd.Series,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Return contiguous recession windows with an exclusive end month."""
    if not isinstance(target_dates, pd.Series) or not isinstance(actual, pd.Series):
        raise TypeError("Recession spans require pandas Series")
    if len(target_dates) != len(actual) or len(actual) == 0:
        raise ValueError("Recession spans require equal nonempty vectors")
    frame = pd.DataFrame(
        {
            "target_date": require_month_start(
                target_dates,
                surface="recession spans",
                field="target_date",
            ),
            "actual": actual.to_numpy(),
        }
    ).sort_values("target_date", kind="mergesort")
    if frame["target_date"].duplicated().any():
        raise ValueError("Recession spans require unique target dates")
    require_gapless_monthly_index(
        pd.DatetimeIndex(frame["target_date"]),
        surface="recession spans",
    )
    require_binary(frame["actual"], surface="recession spans", field="actual")

    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    recession_start: pd.Timestamp | None = None
    for target_date, outcome in frame.itertuples(index=False):
        if outcome == 1 and recession_start is None:
            recession_start = target_date
        elif outcome == 0 and recession_start is not None:
            spans.append((recession_start, target_date))
            recession_start = None
    if recession_start is not None:
        spans.append(
            (
                recession_start,
                frame["target_date"].iloc[-1] + pd.DateOffset(months=1),
            )
        )
    return tuple(spans)


def build_asymmetric_loss_table(
    main_forecasts: pd.DataFrame,
    extended_forecasts: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
    horizons: tuple[int, ...] = (3, 6, 12),
    false_negative_costs: tuple[int, ...] = ASYMMETRIC_LOSS_COST_RATIOS,
) -> pd.DataFrame:
    """Evaluate fixed loss-implied rules without evaluation-sample tuning."""
    horizon_values = _positive_unique_ints(horizons, field="Asymmetric horizons")
    costs = _positive_unique_ints(
        false_negative_costs,
        field="False-negative costs",
    )
    records: list[dict[str, object]] = []
    for horizon in horizon_values:
        common = build_featured_common_forecasts(
            main_forecasts,
            extended_forecasts,
            exercise=exercise,
            data_variant=data_variant,
            horizon=horizon,
        )
        actual = common["actual"].to_numpy(dtype=float)
        sample_size = len(actual)
        recession_month_count = int(actual.sum())
        prevalence = recession_month_count / sample_size

        for false_negative_cost in costs:
            decision_threshold = 1 / (1 + false_negative_cost)
            always_expansion_loss = false_negative_cost * prevalence
            always_recession_loss = 1 - prevalence
            best_constant_loss = min(always_expansion_loss, always_recession_loss)
            if always_expansion_loss < always_recession_loss:
                best_constant_action = "always_expansion"
            elif always_recession_loss < always_expansion_loss:
                best_constant_action = "always_recession"
            else:
                best_constant_action = "tie"

            for model_id, pair in FEATURED_MODEL_IDS.items():
                probabilities = common[model_id].to_numpy(dtype=float)
                recession_action = probabilities >= decision_threshold
                false_positives = int(np.sum(recession_action & (actual == 0)))
                false_negatives = int(np.sum(~recession_action & (actual == 1)))
                model_loss = (
                    false_positives + false_negative_cost * false_negatives
                ) / sample_size
                records.append(
                    {
                        "horizon": horizon,
                        "model_id": model_id,
                        "model_label": FEATURED_MODEL_LABELS[model_id],
                        "source_family": FEATURED_MODEL_SOURCES[pair],
                        "predictor_key": pair[0],
                        "model_name": pair[1],
                        "false_positive_cost": 1,
                        "false_negative_cost": false_negative_cost,
                        "decision_threshold": decision_threshold,
                        "threshold_rule": "recession_if_probability_greater_or_equal",
                        "threshold_selection": "loss_implied_not_evaluation_tuned",
                        "target_month_count": sample_size,
                        "recession_month_count": recession_month_count,
                        "prevalence": prevalence,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                        "model_loss": model_loss,
                        "always_expansion_loss": always_expansion_loss,
                        "always_recession_loss": always_recession_loss,
                        "best_constant_action": best_constant_action,
                        "best_constant_loss": best_constant_loss,
                        "normalized_decision_value": (
                            1 - model_loss / best_constant_loss
                        ),
                        "assessment_scope": (
                            "fixed_rule_comparison_not_structural_welfare"
                        ),
                        "calibration_limitation": (
                            "loss_threshold_is_bayes_optimal_only_for_"
                            "calibrated_probabilities"
                        ),
                    }
                )
    columns = [
        "horizon",
        "model_id",
        "model_label",
        "source_family",
        "predictor_key",
        "model_name",
        "false_positive_cost",
        "false_negative_cost",
        "decision_threshold",
        "threshold_rule",
        "threshold_selection",
        "target_month_count",
        "recession_month_count",
        "prevalence",
        "false_positives",
        "false_negatives",
        "model_loss",
        "always_expansion_loss",
        "always_recession_loss",
        "best_constant_action",
        "best_constant_loss",
        "normalized_decision_value",
        "assessment_scope",
        "calibration_limitation",
    ]
    return pd.DataFrame(records, columns=columns)


def _metric_row(
    panel: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
    forecast_family: str,
    horizon: int,
    predictor_key: str,
) -> pd.Series:
    selected = panel[
        (panel["exercise"] == exercise)
        & (panel["data_variant"] == data_variant)
        & (panel["forecast_family"] == forecast_family)
        & (panel["horizon"] == horizon)
        & (panel["predictor_key"] == predictor_key)
        & (panel["model_name"] == "Logit_L2")
    ]
    if len(selected) != 1:
        raise ValueError(
            "Threshold comparison needs one metric row for "
            f"{forecast_family}/{predictor_key}/h={horizon}"
        )
    return selected.iloc[0]


def _quantile_from_metadata(
    payload: object,
    *,
    horizon: int,
    surface: str,
) -> float:
    if not isinstance(payload, dict):
        raise ValueError(f"{surface}: global_quantiles must be a mapping")
    horizon_payload = payload.get(str(horizon))
    if not isinstance(horizon_payload, dict) or "1" not in horizon_payload:
        raise ValueError(f"{surface}: missing h={horizon}, window=1 quantile")
    value = float(horizon_payload["1"])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{surface}: initial quantile must be finite and in [0, 1]")
    return value


def _oos_start_from_metadata(
    metadata: dict[str, object],
    *,
    surface: str,
) -> pd.Timestamp:
    value = metadata.get("oos_start")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{surface} metadata has no declared OOS date")
    return require_month_start(
        pd.Series([value]),
        surface=f"{surface} metadata",
        field="oos_start",
    ).iloc[0]


def build_expanding_threshold_comparison(
    main_metrics: pd.DataFrame,
    expanding_metrics: pd.DataFrame,
    main_metadata: dict[str, object],
    expanding_metadata: dict[str, object],
    *,
    exercise: str,
    data_variant: str,
    horizons: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Compare frozen and expanding global-threshold specifications."""
    _require_surface_selector(exercise, data_variant)
    validate_metric_panel(main_metrics, surface="main threshold metrics")
    validate_metric_panel(expanding_metrics, surface="expanding threshold metrics")
    if main_metadata.get("threshold_mode") != "global":
        raise ValueError("Main threshold metadata must declare frozen global mode")
    if expanding_metadata.get("threshold_mode") != "expanding_global":
        raise ValueError("Expanding threshold metadata must declare expanding_global mode")
    if (
        expanding_metadata.get("threshold_recalibration_rule")
        != "recompute_global_quantile_at_each_forecast_origin"
        or expanding_metadata.get("threshold_recalibration_inputs")
        != "unshifted_X_transformed_and_y_target_through_train_end"
    ):
        raise ValueError("Expanding threshold metadata has an invalid recalibration rule")
    quantile_paths = expanding_metadata.get("global_quantile_path")
    if not isinstance(quantile_paths, dict):
        raise ValueError("Expanding threshold metadata has no quantile-path mapping")
    main_oos_start = _oos_start_from_metadata(
        main_metadata,
        surface="Main threshold",
    )
    expanding_oos_start = _oos_start_from_metadata(
        expanding_metadata,
        surface="Expanding threshold",
    )
    if main_oos_start != expanding_oos_start:
        raise ValueError("Main and expanding threshold OOS declarations differ")

    records: list[dict[str, object]] = []
    for horizon in _positive_unique_ints(horizons, field="Expanding horizons"):
        frozen_quantile = _quantile_from_metadata(
            main_metadata.get("global_quantiles"),
            horizon=horizon,
            surface="main threshold metadata",
        )
        expanding_initial_quantile = _quantile_from_metadata(
            expanding_metadata.get("global_quantiles"),
            horizon=horizon,
            surface="expanding threshold metadata",
        )
        path = quantile_paths.get(str(horizon))
        if not isinstance(path, list) or not path:
            raise ValueError(f"Expanding threshold has no quantile path at h={horizon}")
        path_frame = pd.DataFrame(path)
        required_columns = {
            "forecast_date",
            "train_end",
            "predictor_key",
            "window_size",
            "quantile",
        }
        if not required_columns.issubset(path_frame.columns):
            raise ValueError(f"Expanding quantile path is incomplete at h={horizon}")
        forecast_dates = require_month_start(
            path_frame["forecast_date"],
            surface=f"expanding quantile path h={horizon}",
            field="forecast_date",
        )
        train_end = require_month_start(
            path_frame["train_end"],
            surface=f"expanding quantile path h={horizon}",
            field="train_end",
        )
        if (
            not forecast_dates.is_monotonic_increasing
            or forecast_dates.duplicated().any()
        ):
            raise ValueError(
                "Expanding quantile path dates are not ordered and unique "
                f"at h={horizon}"
            )
        require_gapless_monthly_index(
            pd.DatetimeIndex(forecast_dates),
            surface=f"expanding quantile path h={horizon}",
        )
        if (
            not path_frame["predictor_key"].eq("Zt_Disagg_ExpandingQ").all()
            or not path_frame["window_size"].eq(1).all()
        ):
            raise ValueError(f"Expanding quantile path keys are invalid at h={horizon}")
        expected_train_end = forecast_dates - pd.DateOffset(months=horizon)
        if not train_end.reset_index(drop=True).equals(
            expected_train_end.reset_index(drop=True)
        ):
            raise ValueError(f"Expanding quantile path timing is invalid at h={horizon}")
        quantiles = pd.to_numeric(path_frame["quantile"], errors="coerce")
        if (
            quantiles.isna().any()
            or not np.isfinite(quantiles).all()
            or not quantiles.between(0.0, 1.0, inclusive="both").all()
        ):
            raise ValueError(
                "Expanding path quantiles must be finite and in [0, 1] "
                f"at h={horizon}"
            )

        main_metric = _metric_row(
            main_metrics,
            exercise=exercise,
            data_variant=data_variant,
            forecast_family="main",
            horizon=horizon,
            predictor_key="Zt_Disagg",
        )
        expanding_metric = _metric_row(
            expanding_metrics,
            exercise=exercise,
            data_variant=data_variant,
            forecast_family="expanding_threshold",
            horizon=horizon,
            predictor_key="Zt_Disagg_ExpandingQ",
        )
        expected_count = int(main_metric["num_forecasts"])
        if (
            int(expanding_metric["num_forecasts"]) != expected_count
            or int(expanding_metric["num_positive_actuals"])
            != int(main_metric["num_positive_actuals"])
            or len(path_frame) != expected_count
        ):
            raise ValueError(f"Expanding quantile path and metric counts differ at h={horizon}")
        if forecast_dates.iloc[0] != expanding_oos_start:
            raise ValueError(
                "Expanding quantile path does not start at declared OOS date "
                f"at h={horizon}"
            )
        if (
            abs(expanding_initial_quantile - frozen_quantile) > 1e-15
            or abs(float(quantiles.iloc[0]) - frozen_quantile) > 1e-15
        ):
            raise ValueError(
                f"First expanding quantile differs from the frozen quantile at h={horizon}"
            )

        specifications = (
            (
                "frozen_initial",
                "Initial quantile frozen",
                main_metric,
                "main",
                "Zt_Disagg",
                frozen_quantile,
                frozen_quantile,
                frozen_quantile,
                frozen_quantile,
                1,
                "initial_pre_oos_global_quantile_frozen",
            ),
            (
                "expanding",
                "Expanding quantile",
                expanding_metric,
                "expanding_threshold",
                "Zt_Disagg_ExpandingQ",
                float(quantiles.iloc[0]),
                float(quantiles.iloc[-1]),
                float(quantiles.min()),
                float(quantiles.max()),
                len(path_frame),
                str(expanding_metadata["threshold_recalibration_rule"]),
            ),
        )
        for (
            threshold_scheme,
            threshold_label,
            metric,
            source_family,
            predictor_key,
            initial_quantile,
            final_quantile,
            minimum_quantile,
            maximum_quantile,
            quantile_update_count,
            recalibration_rule,
        ) in specifications:
            records.append(
                {
                    "horizon": horizon,
                    "threshold_scheme": threshold_scheme,
                    "threshold_label": threshold_label,
                    "source_family": source_family,
                    "predictor_key": predictor_key,
                    "model_name": "Logit_L2",
                    "num_forecasts": int(metric["num_forecasts"]),
                    "num_positive_actuals": int(metric["num_positive_actuals"]),
                    "pr_auc": float(metric["pr_auc"]),
                    "brier": float(metric["brier"]),
                    "initial_global_quantile": initial_quantile,
                    "final_global_quantile": final_quantile,
                    "minimum_global_quantile": minimum_quantile,
                    "maximum_global_quantile": maximum_quantile,
                    "quantile_update_count": quantile_update_count,
                    "recalibration_rule": recalibration_rule,
                }
            )
    columns = [
        "horizon",
        "threshold_scheme",
        "threshold_label",
        "source_family",
        "predictor_key",
        "model_name",
        "num_forecasts",
        "num_positive_actuals",
        "pr_auc",
        "brier",
        "initial_global_quantile",
        "final_global_quantile",
        "minimum_global_quantile",
        "maximum_global_quantile",
        "quantile_update_count",
        "recalibration_rule",
    ]
    return pd.DataFrame(records, columns=columns)
