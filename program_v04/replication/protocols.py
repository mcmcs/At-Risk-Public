"""Explicit main-profile method protocol and author-resolution binding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .config import METHOD_DECISIONS, assert_canonical_generation_allowed
from .forecasting import (
    PredictionRowPolicy,
    PredictorCompositionPolicy,
    RealtimeCalendarPolicy,
    RealtimeCalibrationPolicy,
)
from .importances import EpisodeEndpointPolicy, IncompleteEpisodePolicy
from .metrics import BrierReferenceRule, EncompassingCovariance
from .models import CVPreprocessingPolicy, ClassImbalanceRule
from .thresholds import RecessionSamplePolicy
from .transforms import Code7Rule, RollingBoundary, TailInequality


class RankTiePolicy(StrEnum):
    AVERAGE_PERCENT_RANK = "average_percent_rank"
    RIGHT_CONTINUOUS_ECDF = "right_continuous_ecdf"

    @property
    def pandas_method(self) -> str:
        return {
            RankTiePolicy.AVERAGE_PERCENT_RANK: "average",
            RankTiePolicy.RIGHT_CONTINUOUS_ECDF: "max",
        }[self]


class LogisticObjectivePolicy(StrEnum):
    BALANCED_WITH_INTERCEPT = "balanced_with_intercept"
    UNWEIGHTED_WITH_INTERCEPT = "unweighted_with_intercept"

    @property
    def imbalance_rule(self) -> ClassImbalanceRule:
        return {
            LogisticObjectivePolicy.BALANCED_WITH_INTERCEPT: ClassImbalanceRule.BALANCED,
            LogisticObjectivePolicy.UNWEIGHTED_WITH_INTERCEPT: ClassImbalanceRule.UNWEIGHTED,
        }[self]


class TuningInitializationPolicy(StrEnum):
    STRICT_PER_KEY = "strict_per_key"


class TuningFailurePolicy(StrEnum):
    LEGACY_C_ONE_FALLBACK = "legacy_c_one_fallback"
    STRICT_RAISE = "strict_raise"


@dataclass(frozen=True)
class MainMethodProtocol:
    """Every disputed choice reachable from the main profile, with no defaults."""

    code7_rule: Code7Rule
    rolling_boundary: RollingBoundary
    tail_inequality: TailInequality
    rank_tie_policy: RankTiePolicy
    recession_sample_policy: RecessionSamplePolicy
    cv_preprocessing_policy: CVPreprocessingPolicy
    logistic_objective_policy: LogisticObjectivePolicy
    encompassing_covariance: EncompassingCovariance
    brier_reference_rule: BrierReferenceRule
    episode_endpoint_policy: EpisodeEndpointPolicy
    incomplete_episode_policy: IncompleteEpisodePolicy
    tuning_initialization_policy: TuningInitializationPolicy
    tuning_failure_policy: TuningFailurePolicy

    def validate(self) -> None:
        enum_fields = (
            ("code7_rule", self.code7_rule, Code7Rule),
            ("rolling_boundary", self.rolling_boundary, RollingBoundary),
            ("tail_inequality", self.tail_inequality, TailInequality),
            ("rank_tie_policy", self.rank_tie_policy, RankTiePolicy),
            ("recession_sample_policy", self.recession_sample_policy, RecessionSamplePolicy),
            ("cv_preprocessing_policy", self.cv_preprocessing_policy, CVPreprocessingPolicy),
            ("logistic_objective_policy", self.logistic_objective_policy, LogisticObjectivePolicy),
            ("encompassing_covariance", self.encompassing_covariance, EncompassingCovariance),
            ("brier_reference_rule", self.brier_reference_rule, BrierReferenceRule),
            ("episode_endpoint_policy", self.episode_endpoint_policy, EpisodeEndpointPolicy),
            ("incomplete_episode_policy", self.incomplete_episode_policy, IncompleteEpisodePolicy),
            ("tuning_initialization_policy", self.tuning_initialization_policy, TuningInitializationPolicy),
            ("tuning_failure_policy", self.tuning_failure_policy, TuningFailurePolicy),
        )
        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise ValueError(f"Main protocol {name} has invalid value {value!r}")

    @property
    def imbalance_rule(self) -> ClassImbalanceRule:
        return self.logistic_objective_policy.imbalance_rule

    @property
    def fit_intercept(self) -> bool:
        return True

    @property
    def rank_method(self) -> str:
        return self.rank_tie_policy.pandas_method

    def as_metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "code7_rule": self.code7_rule.value,
            "rolling_boundary": self.rolling_boundary.value,
            "tail_inequality": self.tail_inequality.value,
            "rank_tie_policy": self.rank_tie_policy.value,
            "rank_method": self.rank_method,
            "recession_sample_policy": self.recession_sample_policy.value,
            "cv_preprocessing_policy": self.cv_preprocessing_policy.value,
            "logistic_objective_policy": self.logistic_objective_policy.value,
            "imbalance_rule": self.imbalance_rule.value,
            "fit_intercept": self.fit_intercept,
            "encompassing_covariance": self.encompassing_covariance.value,
            "brier_reference_rule": self.brier_reference_rule.value,
            "episode_endpoint_policy": self.episode_endpoint_policy.value,
            "incomplete_episode_policy": self.incomplete_episode_policy.value,
            "tuning_initialization_policy": self.tuning_initialization_policy.value,
            "tuning_failure_policy": self.tuning_failure_policy.value,
        }


REQUIRED_MAIN_RESOLUTIONS = {
    "RF-004",
    "RF-007",
    "RF-009",
    "RF-010",
    "RF-011",
    "RF-012",
    "RF-013",
    "RF-016",
    "RF-017",
    "RF-021",
    "RF-022",
    "RF-025",
}


def main_protocol_from_resolutions(resolutions: Mapping[str, str]) -> MainMethodProtocol:
    """Bind exact ledger resolution values to an executable main protocol."""
    missing = sorted(REQUIRED_MAIN_RESOLUTIONS - set(resolutions))
    extras = sorted(set(resolutions) - REQUIRED_MAIN_RESOLUTIONS)
    if missing or extras:
        raise ValueError(f"Main protocol resolution keys mismatch: missing={missing}, extras={extras}")
    try:
        endpoint_text, incomplete_text = resolutions["RF-010"].split("+", 1)
        protocol = MainMethodProtocol(
            code7_rule=Code7Rule(resolutions["RF-004"]),
            rolling_boundary=RollingBoundary(resolutions["RF-011"]),
            tail_inequality=TailInequality(resolutions["RF-009"]),
            rank_tie_policy=RankTiePolicy(resolutions["RF-012"]),
            recession_sample_policy=RecessionSamplePolicy(resolutions["RF-022"]),
            cv_preprocessing_policy=CVPreprocessingPolicy(resolutions["RF-013"]),
            logistic_objective_policy=LogisticObjectivePolicy(resolutions["RF-017"]),
            encompassing_covariance=EncompassingCovariance(resolutions["RF-007"]),
            brier_reference_rule=BrierReferenceRule(resolutions["RF-016"]),
            episode_endpoint_policy=EpisodeEndpointPolicy(endpoint_text),
            incomplete_episode_policy=IncompleteEpisodePolicy(incomplete_text),
            tuning_initialization_policy=TuningInitializationPolicy(resolutions["RF-021"]),
            tuning_failure_policy=TuningFailurePolicy(resolutions["RF-025"]),
        )
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid main author-resolution value: {exc}") from exc
    protocol.validate()
    return protocol


def canonical_main_protocol() -> MainMethodProtocol:
    """Return the author-selected protocol only after the global gate closes."""
    assert_canonical_generation_allowed()
    selected = {
        decision.finding_id: decision.selected_resolution
        for decision in METHOD_DECISIONS
        if decision.finding_id in REQUIRED_MAIN_RESOLUTIONS
    }
    if any(value is None for value in selected.values()):
        raise RuntimeError("Resolved main decisions must record exact executable resolution values")
    return main_protocol_from_resolutions(
        {finding_id: str(value) for finding_id, value in selected.items()}
    )


def frozen_legacy_main_protocol() -> MainMethodProtocol:
    """Explicit characterization of the frozen main-surface conventions."""
    protocol = MainMethodProtocol(
        code7_rule=Code7Rule.LEGACY_TWELVE_MONTH_PERCENT,
        rolling_boundary=RollingBoundary.LEGACY_PARTIAL,
        tail_inequality=TailInequality.STRICT,
        rank_tie_policy=RankTiePolicy.AVERAGE_PERCENT_RANK,
        recession_sample_policy=RecessionSamplePolicy.LEGACY_COMPLETE_CASE_ROWS,
        cv_preprocessing_policy=CVPreprocessingPolicy.LEGACY_PRECOMPUTED,
        logistic_objective_policy=LogisticObjectivePolicy.BALANCED_WITH_INTERCEPT,
        encompassing_covariance=EncompassingCovariance.LEGACY_IID,
        brier_reference_rule=BrierReferenceRule.LEGACY_HALF,
        episode_endpoint_policy=EpisodeEndpointPolicy.LEGACY_DECLARED,
        incomplete_episode_policy=IncompleteEpisodePolicy.ALLOW_AVAILABLE_WITH_COUNT,
        tuning_initialization_policy=TuningInitializationPolicy.STRICT_PER_KEY,
        tuning_failure_policy=TuningFailurePolicy.LEGACY_C_ONE_FALLBACK,
    )
    protocol.validate()
    return protocol


def recommended_candidate_main_protocol() -> MainMethodProtocol:
    """Decision-neutral recommended candidate; never used implicitly."""
    protocol = MainMethodProtocol(
        code7_rule=Code7Rule.PUBLISHED_DIFFERENCE_OF_PERCENT_CHANGE,
        rolling_boundary=RollingBoundary.COMPLETE,
        tail_inequality=TailInequality.STRICT,
        rank_tie_policy=RankTiePolicy.AVERAGE_PERCENT_RANK,
        recession_sample_policy=RecessionSamplePolicy.PER_VARIABLE_AVAILABLE_ROWS,
        cv_preprocessing_policy=CVPreprocessingPolicy.FOLD_PURE,
        logistic_objective_policy=LogisticObjectivePolicy.BALANCED_WITH_INTERCEPT,
        encompassing_covariance=EncompassingCovariance.HAC_HORIZON,
        brier_reference_rule=BrierReferenceRule.SAMPLE_PREVALENCE,
        episode_endpoint_policy=EpisodeEndpointPolicy.END_BEFORE_PEAK,
        incomplete_episode_policy=IncompleteEpisodePolicy.REQUIRE_COMPLETE,
        tuning_initialization_policy=TuningInitializationPolicy.STRICT_PER_KEY,
        tuning_failure_policy=TuningFailurePolicy.STRICT_RAISE,
    )
    protocol.validate()
    return protocol


@dataclass(frozen=True)
class RealtimeMethodProtocol:
    """Every disputed choice reachable from the real-time appendix profile."""

    main: MainMethodProtocol
    calendar_policy: RealtimeCalendarPolicy
    prediction_row_policy: PredictionRowPolicy
    calibration_policy: RealtimeCalibrationPolicy
    composition_policy: PredictorCompositionPolicy

    def validate(self) -> None:
        self.main.validate()
        enum_fields = (
            ("calendar_policy", self.calendar_policy, RealtimeCalendarPolicy),
            ("prediction_row_policy", self.prediction_row_policy, PredictionRowPolicy),
            ("calibration_policy", self.calibration_policy, RealtimeCalibrationPolicy),
            ("composition_policy", self.composition_policy, PredictorCompositionPolicy),
        )
        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise ValueError(f"Real-time protocol {name} has invalid value {value!r}")
        coherent_calendar_rows = {
            RealtimeCalendarPolicy.LEGACY_SHIFT_TRUNCATED_PANEL:
                PredictionRowPolicy.LEGACY_LAST_AVAILABLE,
            RealtimeCalendarPolicy.LATEST_AVAILABLE_OBSERVATION:
                PredictionRowPolicy.EXACT_TARGET,
        }
        expected = coherent_calendar_rows[self.calendar_policy]
        if self.prediction_row_policy is not expected:
            raise ValueError(
                "Real-time calendar and prediction-row policies are incoherent: "
                f"{self.calendar_policy.value} requires {expected.value}"
            )

    def as_metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "main": self.main.as_metadata(),
            "calendar_policy": self.calendar_policy.value,
            "prediction_row_policy": self.prediction_row_policy.value,
            "calibration_policy": self.calibration_policy.value,
            "composition_policy": self.composition_policy.value,
        }


REQUIRED_REALTIME_RESOLUTIONS = {"RF-005", "RF-006", "RF-020"}


def realtime_protocol_from_resolutions(
    main: MainMethodProtocol,
    resolutions: Mapping[str, str],
) -> RealtimeMethodProtocol:
    """Bind exact ledger values to an executable real-time protocol."""
    missing = sorted(REQUIRED_REALTIME_RESOLUTIONS - set(resolutions))
    extras = sorted(set(resolutions) - REQUIRED_REALTIME_RESOLUTIONS)
    if missing or extras:
        raise ValueError(
            f"Real-time protocol resolution keys mismatch: missing={missing}, extras={extras}"
        )
    try:
        calendar_text, row_text = resolutions["RF-005"].split("+", 1)
        protocol = RealtimeMethodProtocol(
            main=main,
            calendar_policy=RealtimeCalendarPolicy(calendar_text),
            prediction_row_policy=PredictionRowPolicy(row_text),
            calibration_policy=RealtimeCalibrationPolicy(resolutions["RF-006"]),
            composition_policy=PredictorCompositionPolicy(resolutions["RF-020"]),
        )
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid real-time author-resolution value: {exc}") from exc
    protocol.validate()
    return protocol


def canonical_realtime_protocol() -> RealtimeMethodProtocol:
    """Return the author-selected real-time protocol after the global gate closes."""
    assert_canonical_generation_allowed()
    selected = {
        decision.finding_id: decision.selected_resolution
        for decision in METHOD_DECISIONS
        if decision.finding_id in REQUIRED_REALTIME_RESOLUTIONS
    }
    if any(value is None for value in selected.values()):
        raise RuntimeError(
            "Resolved real-time decisions must record exact executable resolution values"
        )
    return realtime_protocol_from_resolutions(
        canonical_main_protocol(),
        {finding_id: str(value) for finding_id, value in selected.items()},
    )


def frozen_legacy_realtime_protocol() -> RealtimeMethodProtocol:
    """Explicit characterization of the frozen v03 real-time conventions."""
    protocol = RealtimeMethodProtocol(
        main=frozen_legacy_main_protocol(),
        calendar_policy=RealtimeCalendarPolicy.LEGACY_SHIFT_TRUNCATED_PANEL,
        prediction_row_policy=PredictionRowPolicy.LEGACY_LAST_AVAILABLE,
        calibration_policy=RealtimeCalibrationPolicy.LEGACY_REVISED_SHARED,
        composition_policy=PredictorCompositionPolicy.LEGACY_VARIANT_SPECIFIC,
    )
    protocol.validate()
    return protocol


def recommended_candidate_realtime_protocol() -> RealtimeMethodProtocol:
    """Decision-neutral recommended real-time candidate; never used implicitly."""
    protocol = RealtimeMethodProtocol(
        main=recommended_candidate_main_protocol(),
        calendar_policy=RealtimeCalendarPolicy.LATEST_AVAILABLE_OBSERVATION,
        prediction_row_policy=PredictionRowPolicy.EXACT_TARGET,
        calibration_policy=RealtimeCalibrationPolicy.VARIANT_SPECIFIC,
        composition_policy=PredictorCompositionPolicy.COMMON_LOCKED,
    )
    protocol.validate()
    return protocol
