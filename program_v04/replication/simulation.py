"""Audited, decision-neutral core for the appendix skewed-mixture simulation.

The frozen simulation is characterized exactly, but none of its disputed
choices is a public default.  A caller must supply a :class:`SimulationProtocol`
that names the seed-selection, rank, clipping, cutoff, standardization,
estimator, and warning policies.  This keeps RF-008, RF-018, and RF-019 behind
the author-decision gate while making every candidate executable.

This module is deliberately independent of the frozen ``simulation/`` package
so it can be installed as part of the public full-profile wheel.  It performs
no filesystem writes; M4 owns persistence and provenance orchestration.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from .models import ClassImbalanceRule
from .transforms import TailInequality


class ReplicationSelectionPolicy(StrEnum):
    """How candidate seeds contribute to the reported Monte Carlo sample."""

    CONDITIONAL_ACCEPTED_SEED_WALK = "conditional_accepted_seed_walk"
    FIXED_CANDIDATE_SET = "fixed_candidate_set"


class SimulationRankPolicy(StrEnum):
    """Empirical-rank convention used to calibrate the global threshold."""

    RIGHT_CONTINUOUS_ECDF = "right_continuous_ecdf"
    AVERAGE_TIED_PERCENT_RANK = "average_tied_percent_rank"


class SimulationClipPolicy(StrEnum):
    """Treatment of the raw cross-variable threshold quantile."""

    LEGACY_WINSORIZE_005_040 = "legacy_winsorize_005_040"
    UNCLIPPED = "unclipped"


class PredictorStandardizationPolicy(StrEnum):
    """Which simulated predictor stacks receive training-only scaling."""

    BOTH_STACKS_TRAIN_ONLY = "both_stacks_train_only"
    CONTINUOUS_ONLY_TRAIN_ONLY = "continuous_only_train_only"
    NONE = "none"


class SimulationWarningPolicy(StrEnum):
    """How estimator warnings are surfaced and retained."""

    LEGACY_SUPPRESS_ALL = "legacy_suppress_all"
    LEGACY_RECORD_ALL = "legacy_record_all"
    RECORD_CONVERGENCE_RAISE_UNEXPECTED = "record_convergence_raise_unexpected"
    RAISE_ALL = "raise_all"


@dataclass(frozen=True)
class SimulationConfig:
    """Complete DGP, sample, threshold-bound, and seed configuration."""

    b_default: int = 500
    b_pilot: int = 100
    t_burn: int = 200
    t_train: int = 360
    t_test: int = 420
    lags: tuple[int, ...] = (0, 3, 6, 12)
    horizons: tuple[int, ...] = (3, 6, 12)
    n_variables: int = 50
    p11: float = 0.95
    pi_stress: float = 0.087
    p0_range: tuple[float, float] = (0.05, 0.15)
    p1_range: tuple[float, float] = (0.55, 0.70)
    rho_range: tuple[float, float] = (0.20, 0.50)
    gamma_range: tuple[float, float] = (0.05, 0.20)
    sigma_range: tuple[float, float] = (0.80, 1.20)
    event_magnitude_range: tuple[float, float] = (1.00, 1.40)
    rho_factor: float = 0.85
    t_degrees_freedom: float = 5.0
    idiosyncratic_noise_scale: float = 0.30
    tau_bounds: tuple[float, float] = (0.05, 0.40)
    per_variable_rank_quantile: float = 0.50
    ladder_step: int = 0
    base_seed: int = 20260425

    @property
    def lag_max(self) -> int:
        return max(self.lags)

    @property
    def horizon_max(self) -> int:
        return max(self.horizons)

    @property
    def t_use(self) -> int:
        return self.t_train + self.t_test + self.lag_max + self.horizon_max

    @property
    def t_full(self) -> int:
        return self.t_burn + self.t_use

    @property
    def train_origin_lo(self) -> int:
        return self.lag_max + 1

    @property
    def train_origin_hi(self) -> int:
        return self.lag_max + self.t_train

    @property
    def test_origin_lo(self) -> int:
        return self.train_origin_hi + 1

    @property
    def test_origin_hi(self) -> int:
        return self.train_origin_hi + self.t_test

    @property
    def d_train_hi(self) -> int:
        return self.train_origin_hi

    @property
    def p01(self) -> float:
        return self.pi_stress * (1.0 - self.p11) / (1.0 - self.pi_stress)

    @property
    def t_innovation_scale(self) -> float:
        return float(
            np.sqrt(
                self.t_degrees_freedom / (self.t_degrees_freedom - 2.0)
            )
        )

    def validate(self) -> None:
        if self.b_default < 1 or self.b_pilot < 1:
            raise ValueError("Simulation replication counts must be positive")
        if self.t_burn < 0 or self.t_train < 1 or self.t_test < 1:
            raise ValueError("Simulation sample lengths are invalid")
        if not self.lags or self.lags[0] != 0 or tuple(sorted(set(self.lags))) != self.lags:
            raise ValueError("Simulation lags must be sorted, unique, and begin at zero")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("Simulation horizons must be positive and nonempty")
        if self.test_origin_lo - self.horizon_max < self.train_origin_lo:
            raise ValueError("Simulation has no training targets available at the first test origin")
        if self.n_variables < 1:
            raise ValueError("Simulation variable count must be positive")
        if self.test_origin_hi + self.horizon_max != self.t_use:
            raise ValueError("Simulation target rows do not end at T_USE")
        if self.train_origin_lo - 1 < self.lag_max:
            raise ValueError("Simulation design lacks its required lag prefix")
        if not 0.0 < self.pi_stress < 1.0 or not 0.0 < self.p11 < 1.0:
            raise ValueError("Stress-chain probabilities must lie strictly in (0, 1)")
        if not 0.0 < self.p01 < 1.0:
            raise ValueError("Implied p01 must lie strictly in (0, 1)")
        for name, bounds in (
            ("p0", self.p0_range),
            ("p1", self.p1_range),
            ("rho", self.rho_range),
            ("gamma", self.gamma_range),
            ("sigma", self.sigma_range),
            ("event magnitude", self.event_magnitude_range),
            ("tau", self.tau_bounds),
        ):
            if len(bounds) != 2 or not all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid {name} range: {bounds}")
        if not 0.0 <= self.p0_range[0] <= self.p0_range[1] < 1.0:
            raise ValueError("p0 range must lie in [0, 1)")
        if not 0.0 <= self.p1_range[0] <= self.p1_range[1] < 1.0:
            raise ValueError("p1 range must lie in [0, 1)")
        if self.p1_range[0] < self.p0_range[1]:
            raise ValueError("State-one event probabilities must exceed state-zero probabilities")
        if not 0.0 <= self.tau_bounds[0] <= self.tau_bounds[1] <= 1.0:
            raise ValueError("Tau bounds must lie in [0, 1]")
        if not 0.0 <= self.per_variable_rank_quantile <= 1.0:
            raise ValueError("Per-variable rank quantile must lie in [0, 1]")
        if not 0.0 <= self.rho_factor < 1.0:
            raise ValueError("Factor persistence must lie in [0, 1)")
        if self.t_degrees_freedom <= 2.0:
            raise ValueError("Student-t degrees of freedom must exceed two")
        if self.idiosyncratic_noise_scale < 0.0:
            raise ValueError("Idiosyncratic noise scale cannot be negative")
        if self.ladder_step not in range(5):
            raise ValueError("Simulation ladder step must be one of 0, 1, 2, 3, 4")

    def fingerprint(self, *, c_value: float) -> str:
        """Return the frozen metadata layout with the executed estimator C."""
        self.validate()
        if not np.isfinite(c_value) or c_value <= 0.0:
            raise ValueError("Fingerprint logistic C must be finite and positive")
        return "|".join(
            [
                f"B={self.b_default}",
                f"T_USE={self.t_use}",
                f"N={self.n_variables}",
                f"LAGS={self.lags}",
                f"H={self.horizons}",
                f"P11={self.p11}",
                f"PI_STRESS={self.pi_stress}",
                f"P0=[{self.p0_range[0]},{self.p0_range[1]}]",
                f"P1=[{self.p1_range[0]},{self.p1_range[1]}]",
                f"RHO=[{self.rho_range[0]},{self.rho_range[1]}]",
                f"GAMMA=[{self.gamma_range[0]},{self.gamma_range[1]}]",
                f"SIGMA=[{self.sigma_range[0]},{self.sigma_range[1]}]",
                (
                    f"A=[{self.event_magnitude_range[0]},"
                    f"{self.event_magnitude_range[1]}]"
                ),
                f"RHO_G={self.rho_factor}",
                f"TDOF={self.t_degrees_freedom}",
                f"S_EPS={self.idiosyncratic_noise_scale}",
                f"TAU=[{self.tau_bounds[0]},{self.tau_bounds[1]}]",
                f"Q_PER_VAR={self.per_variable_rank_quantile}",
                f"C={float(c_value)}",
                f"LADDER_STEP={self.ladder_step}",
                f"BASE_SEED={self.base_seed}",
            ]
        )

    def for_ladder_step(self, step: int) -> SimulationConfig:
        """Return the plan's explicit ladder configuration without environment state."""
        if step not in range(5):
            raise ValueError("Simulation ladder step must be one of 0, 1, 2, 3, 4")
        return replace(
            self,
            p1_range=(0.60, 0.75) if step >= 1 else (0.55, 0.70),
            idiosyncratic_noise_scale=0.20 if step >= 2 else 0.30,
            gamma_range=(0.00, 0.10) if step >= 3 else (0.05, 0.20),
            event_magnitude_range=(1.20, 1.60) if step >= 4 else (1.00, 1.40),
            ladder_step=step,
        )


@dataclass(frozen=True)
class SimulationEstimatorSpec:
    """Fully explicit estimator and preprocessing contract."""

    c_value: float
    imbalance_rule: ClassImbalanceRule
    fit_intercept: bool
    max_iter: int
    random_state: int
    standardization_policy: PredictorStandardizationPolicy

    def validate(self) -> None:
        if not np.isfinite(self.c_value) or self.c_value <= 0.0:
            raise ValueError("Simulation logistic C must be finite and positive")
        if not isinstance(self.imbalance_rule, ClassImbalanceRule):
            raise ValueError("Simulation class-imbalance rule is invalid")
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("Simulation intercept rule must be Boolean")
        if self.max_iter < 1:
            raise ValueError("Simulation maximum iteration count must be positive")
        if not isinstance(self.random_state, int):
            raise ValueError("Simulation estimator random state must be an integer")
        if not isinstance(self.standardization_policy, PredictorStandardizationPolicy):
            raise ValueError("Simulation standardization policy is invalid")

    def as_metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "c_value": float(self.c_value),
            "imbalance_rule": self.imbalance_rule.value,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "standardization_policy": self.standardization_policy.value,
        }


@dataclass(frozen=True)
class SimulationProtocol:
    """Every disputed simulation method choice; intentionally has no defaults."""

    selection_policy: ReplicationSelectionPolicy
    rank_policy: SimulationRankPolicy
    clip_policy: SimulationClipPolicy
    tail_inequality: TailInequality
    estimator: SimulationEstimatorSpec
    warning_policy: SimulationWarningPolicy

    def validate(self) -> None:
        if not isinstance(self.selection_policy, ReplicationSelectionPolicy):
            raise ValueError("Simulation seed-selection policy is invalid")
        if not isinstance(self.rank_policy, SimulationRankPolicy):
            raise ValueError("Simulation rank policy is invalid")
        if not isinstance(self.clip_policy, SimulationClipPolicy):
            raise ValueError("Simulation clipping policy is invalid")
        if not isinstance(self.tail_inequality, TailInequality):
            raise ValueError("Simulation tail inequality is invalid")
        if not isinstance(self.warning_policy, SimulationWarningPolicy):
            raise ValueError("Simulation warning policy is invalid")
        self.estimator.validate()

    def as_metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "selection_policy": self.selection_policy.value,
            "rank_policy": self.rank_policy.value,
            "clip_policy": self.clip_policy.value,
            "tail_inequality": self.tail_inequality.value,
            "warning_policy": self.warning_policy.value,
            "estimator": self.estimator.as_metadata(),
        }


def _stable_mapping_hash(value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_step0_config() -> SimulationConfig:
    """Construct the frozen ladder-step-zero DGP configuration."""
    config = SimulationConfig()
    config.validate()
    return config


def frozen_legacy_protocol() -> SimulationProtocol:
    """Name, without endorsing, every choice in the frozen appendix run."""
    protocol = SimulationProtocol(
        selection_policy=ReplicationSelectionPolicy.CONDITIONAL_ACCEPTED_SEED_WALK,
        rank_policy=SimulationRankPolicy.RIGHT_CONTINUOUS_ECDF,
        clip_policy=SimulationClipPolicy.LEGACY_WINSORIZE_005_040,
        tail_inequality=TailInequality.WEAK,
        estimator=SimulationEstimatorSpec(
            c_value=1.0,
            imbalance_rule=ClassImbalanceRule.UNWEIGHTED,
            fit_intercept=True,
            max_iter=2000,
            random_state=42,
            standardization_policy=PredictorStandardizationPolicy.BOTH_STACKS_TRAIN_ONLY,
        ),
        warning_policy=SimulationWarningPolicy.LEGACY_SUPPRESS_ALL,
    )
    protocol.validate()
    return protocol


REQUIRED_SIMULATION_RESOLUTIONS = {"RF-008", "RF-018", "RF-019"}


def simulation_protocol_from_resolutions(
    resolutions: Mapping[str, str],
) -> SimulationProtocol:
    """Bind exact author-resolution values to the simulation implementation."""
    missing = sorted(REQUIRED_SIMULATION_RESOLUTIONS - set(resolutions))
    extras = sorted(set(resolutions) - REQUIRED_SIMULATION_RESOLUTIONS)
    if missing or extras:
        raise ValueError(
            f"Simulation protocol resolution keys mismatch: missing={missing}, extras={extras}"
        )
    try:
        (
            rank_text,
            clip_text,
            tail_text,
            imbalance_text,
            standardization_text,
        ) = resolutions["RF-019"].split("+", 4)
        protocol = SimulationProtocol(
            selection_policy=ReplicationSelectionPolicy(resolutions["RF-008"]),
            rank_policy=SimulationRankPolicy(rank_text),
            clip_policy=SimulationClipPolicy(clip_text),
            tail_inequality=TailInequality(tail_text),
            estimator=SimulationEstimatorSpec(
                c_value=1.0,
                imbalance_rule=ClassImbalanceRule(imbalance_text),
                fit_intercept=True,
                max_iter=2000,
                random_state=42,
                standardization_policy=PredictorStandardizationPolicy(
                    standardization_text
                ),
            ),
            warning_policy=SimulationWarningPolicy(resolutions["RF-018"]),
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid simulation author-resolution value: {exc}") from exc
    protocol.validate()
    return protocol


def canonical_simulation_protocol() -> SimulationProtocol:
    """Return the selected simulation protocol after the global gate closes."""
    from .config import METHOD_DECISIONS, assert_canonical_generation_allowed

    assert_canonical_generation_allowed()
    selected = {
        decision.finding_id: decision.selected_resolution
        for decision in METHOD_DECISIONS
        if decision.finding_id in REQUIRED_SIMULATION_RESOLUTIONS
    }
    if any(value is None for value in selected.values()):
        raise RuntimeError(
            "Resolved simulation decisions must record exact executable resolution values"
        )
    return simulation_protocol_from_resolutions(
        {finding_id: str(value) for finding_id, value in selected.items()}
    )


def recommended_candidate_protocol() -> SimulationProtocol:
    """Decision-neutral recommended simulation candidate; never used implicitly."""
    protocol = SimulationProtocol(
        selection_policy=ReplicationSelectionPolicy.FIXED_CANDIDATE_SET,
        rank_policy=SimulationRankPolicy.AVERAGE_TIED_PERCENT_RANK,
        clip_policy=SimulationClipPolicy.UNCLIPPED,
        tail_inequality=TailInequality.STRICT,
        estimator=SimulationEstimatorSpec(
            c_value=1.0,
            imbalance_rule=ClassImbalanceRule.BALANCED,
            fit_intercept=True,
            max_iter=2000,
            random_state=42,
            standardization_policy=(
                PredictorStandardizationPolicy.CONTINUOUS_ONLY_TRAIN_ONLY
            ),
        ),
        warning_policy=(
            SimulationWarningPolicy.RECORD_CONVERGENCE_RAISE_UNEXPECTED
        ),
    )
    protocol.validate()
    return protocol


@dataclass(frozen=True)
class SimulatedReplication:
    seed: int
    stress: np.ndarray
    target: np.ndarray
    event: np.ndarray
    continuous: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    rho: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    event_magnitude: np.ndarray


@dataclass(frozen=True)
class GlobalThreshold:
    tau_raw: float
    tau: float
    tau_clipped: bool
    cutoffs: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class DesignMatrices:
    train: np.ndarray
    test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class FitWarningRecord:
    horizon: int
    model: str
    category: str
    message: str


@dataclass(frozen=True)
class ReplicationOutcome:
    seed: int
    accepted: bool
    rejection_reason: str | None
    candidate_stress_frequency: float
    metrics: pd.DataFrame | None
    diagnostics: pd.DataFrame | None
    fit_warnings: tuple[FitWarningRecord, ...]


@dataclass(frozen=True)
class SimulationBatch:
    """In-memory seed schedule and outcomes; rejected candidates are never hidden."""

    protocol: SimulationProtocol
    config: SimulationConfig
    requested_replications: int
    candidate_cap: int
    pilot: bool
    outcomes: tuple[ReplicationOutcome, ...]

    @property
    def accepted(self) -> tuple[ReplicationOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.accepted)

    @property
    def rejected(self) -> tuple[ReplicationOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if not outcome.accepted)

    def metrics_panel(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for replication, outcome in enumerate(self.accepted):
            if outcome.metrics is None:
                continue
            frame = outcome.metrics.copy()
            frame.insert(0, "replication", replication)
            frame.insert(1, "seed", int(outcome.seed))
            frames.append(frame)
        if not frames:
            return pd.DataFrame(
                columns=["replication", "seed", "horizon", "model", "pr_auc", "brier"]
            )
        return pd.concat(frames, ignore_index=True)

    def diagnostics_panel(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for replication, outcome in enumerate(self.accepted):
            if outcome.diagnostics is None:
                continue
            frame = outcome.diagnostics.copy()
            frame.insert(0, "replication", replication)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def metadata(self) -> dict[str, object]:
        warnings_flat = [record for outcome in self.outcomes for record in outcome.fit_warnings]
        candidate_frequencies = np.array(
            [outcome.candidate_stress_frequency for outcome in self.outcomes],
            dtype=float,
        )
        accepted_frequencies = np.array(
            [outcome.candidate_stress_frequency for outcome in self.accepted],
            dtype=float,
        )
        rejected_frequencies = np.array(
            [outcome.candidate_stress_frequency for outcome in self.rejected],
            dtype=float,
        )
        protocol_metadata = self.protocol.as_metadata()
        calendar = simulation_calendar(self.config)
        configuration_fingerprint = self.config.fingerprint(
            c_value=self.protocol.estimator.c_value
        )
        protocol_fingerprint = _stable_mapping_hash(protocol_metadata)
        batch_fingerprint = _stable_mapping_hash(
            {
                "configuration_fingerprint": configuration_fingerprint,
                "calendar": calendar,
                "protocol": protocol_metadata,
                "requested_replications": self.requested_replications,
                "candidate_cap": self.candidate_cap,
                "pilot": self.pilot,
            }
        )
        return {
            "requested_replications": self.requested_replications,
            "candidate_cap": self.candidate_cap,
            "pilot": self.pilot,
            "candidate_count": len(self.outcomes),
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "rejection_rate": len(self.rejected) / max(1, len(self.outcomes)),
            "accepted_seeds": [outcome.seed for outcome in self.accepted],
            "rejected_seeds": [
                {"seed": outcome.seed, "reason": outcome.rejection_reason}
                for outcome in self.rejected
            ],
            "warning_count": len(warnings_flat),
            "warning_categories": sorted({record.category for record in warnings_flat}),
            "candidate_mean_stress_frequency": float(candidate_frequencies.mean()),
            "accepted_mean_stress_frequency": (
                float(accepted_frequencies.mean())
                if accepted_frequencies.size
                else None
            ),
            "rejected_mean_stress_frequency": (
                float(rejected_frequencies.mean())
                if rejected_frequencies.size
                else None
            ),
            "protocol": protocol_metadata,
            "calendar": calendar,
            "configuration_fingerprint": configuration_fingerprint,
            "protocol_fingerprint": protocol_fingerprint,
            "fingerprint": batch_fingerprint,
        }


class SimulationAttemptCapExceeded(RuntimeError):
    """Conditional seed walk exhausted its declared cap before its target."""

    def __init__(self, partial_batch: SimulationBatch):
        self.partial_batch = partial_batch
        super().__init__(
            "Simulation candidate cap exhausted before the accepted-replication "
            f"target: candidates={len(partial_batch.outcomes)}, "
            f"accepted={len(partial_batch.accepted)}, "
            f"requested={partial_batch.requested_replications}"
        )


def _origin_rows(lo_1idx: int, hi_1idx: int) -> np.ndarray:
    return np.arange(lo_1idx - 1, hi_1idx, dtype=int)


def training_origin_rows(config: SimulationConfig, horizon: int) -> np.ndarray:
    """Return zero-based origins whose targets are known at the first test origin.

    A label dated at the forecast origin is available. The fixed fit therefore
    retains candidate pairs with origin + horizon <= first test origin.
    """
    config.validate()
    if horizon not in config.horizons:
        raise ValueError(f"Unsupported simulation horizon: {horizon}")
    last_origin = min(config.train_origin_hi, config.test_origin_lo - horizon)
    return _origin_rows(config.train_origin_lo, last_origin)


def simulation_calendar(config: SimulationConfig) -> dict[str, object]:
    """Describe the estimation and threshold samples in one-based rows."""
    return {
        "rule": "training_target_at_or_before_first_test_origin",
        "row_numbering": "one_based_after_burn_in",
        "threshold_reference": [1, config.d_train_hi],
        "test_origins": [config.test_origin_lo, config.test_origin_hi],
        "training_by_horizon": {
            str(horizon): {
                "origin_first": int(rows[0] + 1),
                "origin_last": int(rows[-1] + 1),
                "target_last": int(rows[-1] + 1 + horizon),
                "pairs": int(rows.size),
            }
            for horizon in config.horizons
            for rows in [training_origin_rows(config, horizon)]
        },
    }


def _simulate_markov_chain(
    rng: np.random.Generator,
    length: int,
    config: SimulationConfig,
) -> np.ndarray:
    stress = np.empty(length, dtype=np.int8)
    stress[0] = 1 if rng.uniform() < config.pi_stress else 0
    uniforms = rng.uniform(size=length - 1)
    for time in range(1, length):
        probability = config.p11 if stress[time - 1] == 1 else config.p01
        stress[time] = 1 if uniforms[time - 1] < probability else 0
    return stress


def _simulate_factor(
    rng: np.random.Generator,
    length: int,
    config: SimulationConfig,
) -> np.ndarray:
    factor = np.empty(length, dtype=np.float64)
    factor[0] = rng.standard_normal()
    innovations = rng.standard_normal(length - 1)
    scale = float(np.sqrt(1.0 - config.rho_factor**2))
    for time in range(1, length):
        factor[time] = (
            config.rho_factor * factor[time - 1] + scale * innovations[time - 1]
        )
    return factor


def _simulate_mixture_indicator(
    rng: np.random.Generator,
    stress: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
) -> np.ndarray:
    probabilities = np.where(stress[:, None] == 1, p1[None, :], p0[None, :])
    return (rng.uniform(size=probabilities.shape) < probabilities).astype(np.int8)


def mixture_innovation_conditional_mean(
    probability: np.ndarray | float,
    event_magnitude: np.ndarray | float,
) -> np.ndarray:
    """Return the analytic conditional mean of the two mixture branches."""
    probability_array = np.asarray(probability, dtype=float)
    magnitude_array = np.asarray(event_magnitude, dtype=float)
    if (
        np.any(probability_array < 0.0)
        or np.any(probability_array >= 1.0)
        or np.any(magnitude_array < 0.0)
    ):
        raise ValueError("Mixture probabilities/magnitudes are outside their valid ranges")
    return (
        probability_array * -magnitude_array
        + (1.0 - probability_array)
        * probability_array
        * magnitude_array
        / (1.0 - probability_array)
    )


def _build_mixture_innovation(
    event: np.ndarray,
    noise: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    event_magnitude: np.ndarray,
    stress: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    probabilities = np.where(stress[:, None] == 1, p1[None, :], p0[None, :])
    magnitude = event_magnitude[None, :]
    event_float = event.astype(np.float64)
    return (
        -magnitude * event_float
        + probabilities * magnitude / (1.0 - probabilities) * (1.0 - event_float)
        + config.idiosyncratic_noise_scale * noise
    )


def _simulate_continuous_panel(
    rho: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    factor: np.ndarray,
    innovation: np.ndarray,
) -> np.ndarray:
    drift = gamma[None, :] * factor[:, None] + sigma[None, :] * innovation
    panel = np.zeros((len(factor), len(rho)), dtype=np.float64)
    for time in range(1, len(factor)):
        panel[time] = rho * panel[time - 1] + drift[time]
    return panel


def simulate_replication(seed: int, *, config: SimulationConfig) -> SimulatedReplication:
    """Generate one panel using the frozen, documented RNG draw order."""
    config.validate()
    if not isinstance(seed, int):
        raise ValueError("Simulation seed must be an integer")
    rng = np.random.default_rng(seed)
    stress_full = _simulate_markov_chain(rng, config.t_full, config)
    p0 = rng.uniform(*config.p0_range, size=config.n_variables)
    p1 = rng.uniform(*config.p1_range, size=config.n_variables)
    rho = rng.uniform(*config.rho_range, size=config.n_variables)
    gamma = rng.uniform(*config.gamma_range, size=config.n_variables)
    sigma = rng.uniform(*config.sigma_range, size=config.n_variables)
    event_magnitude = rng.uniform(
        *config.event_magnitude_range, size=config.n_variables
    )
    factor_full = _simulate_factor(rng, config.t_full, config)
    event_full = _simulate_mixture_indicator(rng, stress_full, p0, p1)
    noise_full = (
        rng.standard_t(
            df=config.t_degrees_freedom,
            size=(config.t_full, config.n_variables),
        )
        / config.t_innovation_scale
    )
    innovation_full = _build_mixture_innovation(
        event_full,
        noise_full,
        p0,
        p1,
        event_magnitude,
        stress_full,
        config,
    )
    continuous_full = _simulate_continuous_panel(
        rho, gamma, sigma, factor_full, innovation_full
    )
    usable = slice(config.t_burn, config.t_full)
    stress = stress_full[usable].astype(np.int8)
    return SimulatedReplication(
        seed=seed,
        stress=stress,
        target=stress.copy(),
        event=event_full[usable].astype(np.int8),
        continuous=continuous_full[usable],
        p0=p0,
        p1=p1,
        rho=rho,
        gamma=gamma,
        sigma=sigma,
        event_magnitude=event_magnitude,
    )


def _rank_values(
    sample: np.ndarray,
    values: np.ndarray,
    *,
    policy: SimulationRankPolicy,
) -> np.ndarray:
    if policy is SimulationRankPolicy.RIGHT_CONTINUOUS_ECDF:
        return np.searchsorted(np.sort(sample), values, side="right") / float(len(sample))
    if policy is SimulationRankPolicy.AVERAGE_TIED_PERCENT_RANK:
        sorted_sample = np.sort(sample)
        left = np.searchsorted(sorted_sample, values, side="left")
        right = np.searchsorted(sorted_sample, values, side="right")
        return ((left + 1.0 + right) / 2.0) / float(len(sample))
    raise ValueError(f"Unsupported simulation rank policy: {policy}")


def build_global_threshold(
    continuous: np.ndarray,
    target: np.ndarray,
    *,
    config: SimulationConfig,
    rank_policy: SimulationRankPolicy,
    clip_policy: SimulationClipPolicy,
    tail_inequality: TailInequality,
) -> GlobalThreshold:
    """Build the single cross-variable threshold under explicit RF-019 choices."""
    config.validate()
    if continuous.shape != (config.t_use, config.n_variables):
        raise ValueError("Continuous simulation panel has the wrong shape")
    if target.shape != (config.t_use,) or not np.isin(target, [0, 1]).all():
        raise ValueError("Simulation target must be a binary T_USE vector")
    if not isinstance(rank_policy, SimulationRankPolicy):
        raise ValueError("Simulation rank policy is invalid")
    if not isinstance(clip_policy, SimulationClipPolicy):
        raise ValueError("Simulation clipping policy is invalid")
    if not isinstance(tail_inequality, TailInequality):
        raise ValueError("Simulation tail inequality is invalid")

    training_rows = _origin_rows(1, config.d_train_hi)
    recession_rows = training_rows[target[training_rows] == 1]
    if recession_rows.size == 0:
        raise ValueError("Empty R: no contemporaneous training recessions")
    reference = continuous[training_rows]
    recession_values = continuous[recession_rows]
    ranks = np.empty_like(recession_values, dtype=float)
    for variable in range(config.n_variables):
        ranks[:, variable] = _rank_values(
            reference[:, variable],
            recession_values[:, variable],
            policy=rank_policy,
        )
    per_variable = np.quantile(
        ranks,
        config.per_variable_rank_quantile,
        axis=0,
        method="linear",
    )
    tau_raw = float(np.median(per_variable))
    if clip_policy is SimulationClipPolicy.LEGACY_WINSORIZE_005_040:
        tau = max(config.tau_bounds[0], min(config.tau_bounds[1], tau_raw))
    else:
        tau = tau_raw
    clipped = bool(tau != tau_raw)
    cutoffs = np.quantile(reference, tau, axis=0, method="linear")
    if tail_inequality is TailInequality.WEAK:
        state = continuous <= cutoffs[None, :]
    else:
        state = continuous < cutoffs[None, :]
    return GlobalThreshold(
        tau_raw=tau_raw,
        tau=float(tau),
        tau_clipped=clipped,
        cutoffs=cutoffs,
        state=state.astype(np.int8),
    )


def build_design(
    panel: np.ndarray,
    target: np.ndarray,
    horizon: int,
    *,
    config: SimulationConfig,
) -> DesignMatrices:
    """Build lag stacks using training labels known at the first test origin."""
    config.validate()
    if horizon not in config.horizons:
        raise ValueError(f"Unsupported simulation horizon: {horizon}")
    if panel.shape != (config.t_use, config.n_variables):
        raise ValueError("Simulation design panel has the wrong shape")
    if target.shape != (config.t_use,):
        raise ValueError("Simulation design target has the wrong shape")
    train_rows = training_origin_rows(config, horizon)
    test_rows = _origin_rows(config.test_origin_lo, config.test_origin_hi)

    def lag_stack(rows: np.ndarray) -> np.ndarray:
        return np.concatenate([panel[rows - lag] for lag in config.lags], axis=1)

    return DesignMatrices(
        train=lag_stack(train_rows),
        test=lag_stack(test_rows),
        y_train=target[train_rows + horizon],
        y_test=target[test_rows + horizon],
    )


def standardize_train_test(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen training-only population-standard-deviation scaling."""
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Simulation train/test matrices must be aligned and two-dimensional")
    if train.shape[0] == 0 or test.shape[0] == 0 or not np.isfinite(train).all():
        raise ValueError("Simulation standardization requires nonempty finite training data")
    mean = train.mean(axis=0)
    standard_deviation = train.std(axis=0, ddof=0)
    safe = np.where(standard_deviation < 1e-12, 1.0, standard_deviation)
    return (train - mean) / safe, (test - mean) / safe


def _target_problem(target: np.ndarray, config: SimulationConfig) -> str | None:
    test_rows = _origin_rows(config.test_origin_lo, config.test_origin_hi)
    for horizon in config.horizons:
        train_rows = training_origin_rows(config, horizon)
        train_target = target[train_rows + horizon]
        test_target = target[test_rows + horizon]
        if train_target.sum() in (0, train_target.size):
            return f"single_class_train_h{horizon}"
        if test_target.sum() in (0, test_target.size):
            return f"single_class_test_h{horizon}"
    return None


def _make_simulation_logistic(spec: SimulationEstimatorSpec) -> LogisticRegression:
    spec.validate()
    return LogisticRegression(
        solver="lbfgs",
        C=float(spec.c_value),
        l1_ratio=0,
        fit_intercept=spec.fit_intercept,
        max_iter=spec.max_iter,
        random_state=spec.random_state,
        class_weight=(
            "balanced"
            if spec.imbalance_rule is ClassImbalanceRule.BALANCED
            else None
        ),
    )


def _fit_predict(
    train: np.ndarray,
    target: np.ndarray,
    test: np.ndarray,
    *,
    estimator: SimulationEstimatorSpec,
    warning_policy: SimulationWarningPolicy,
) -> tuple[np.ndarray, tuple[tuple[str, str], ...]]:
    model = _make_simulation_logistic(estimator)
    if warning_policy is SimulationWarningPolicy.LEGACY_SUPPRESS_ALL:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train, target)
            probability = model.predict_proba(test)[:, 1]
        recorded: tuple[tuple[str, str], ...] = ()
    elif warning_policy is SimulationWarningPolicy.RAISE_ALL:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.fit(train, target)
            probability = model.predict_proba(test)[:, 1]
        recorded = ()
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train, target)
            probability = model.predict_proba(test)[:, 1]
        recorded = tuple(
            (warning.category.__name__, str(warning.message)) for warning in caught
        )
        if warning_policy is SimulationWarningPolicy.RECORD_CONVERGENCE_RAISE_UNEXPECTED:
            unexpected = [
                warning
                for warning in caught
                if not issubclass(warning.category, ConvergenceWarning)
            ]
            if unexpected:
                first = unexpected[0]
                raise RuntimeError(
                    "Unexpected simulation estimator warning: "
                    f"{first.category.__name__}: {first.message}"
                )
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Simulation estimator returned invalid probabilities")
    return probability, recorded


def _stress_frequencies(
    stress: np.ndarray,
) -> tuple[float, float, float]:
    previous = stress[:-1]
    current = stress[1:]
    n11 = int(((previous == 1) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n00 = int(((previous == 0) & (current == 0)).sum())
    p11 = float("nan") if n11 + n10 == 0 else n11 / (n11 + n10)
    p01 = float("nan") if n01 + n00 == 0 else n01 / (n01 + n00)
    return float(stress.mean()), float(p11), float(p01)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _build_diagnostics(
    replication: SimulatedReplication,
    threshold: GlobalThreshold,
    config: SimulationConfig,
) -> pd.DataFrame:
    test_rows = _origin_rows(config.test_origin_lo, config.test_origin_hi)
    row: dict[str, object] = {"seed": int(replication.seed)}
    for horizon in config.horizons:
        train_rows = training_origin_rows(config, horizon)
        row[f"recession_freq_train_h{horizon}"] = float(
            replication.target[train_rows + horizon].mean()
        )
        row[f"recession_freq_test_h{horizon}"] = float(
            replication.target[test_rows + horizon].mean()
        )
    row["recession_freq_train_contemp"] = float(
        replication.target[_origin_rows(1, config.train_origin_hi)].mean()
    )
    stress_frequency, empirical_p11, empirical_p01 = _stress_frequencies(
        replication.stress
    )
    row.update(
        {
            "stress_freq": stress_frequency,
            "emp_p11": empirical_p11,
            "emp_p01": empirical_p01,
            "theoretical_pi_stress": float(config.pi_stress),
            "theoretical_p11": float(config.p11),
            "theoretical_p01": float(config.p01),
            "tau_raw": float(threshold.tau_raw),
            "tau": float(threshold.tau),
            "tau_clipped": int(threshold.tau_clipped),
            "pr_B": float(replication.event.mean()),
            "pr_z": float(threshold.state.mean()),
            "pr_z_eq_B": float((threshold.state == replication.event).mean()),
        }
    )
    event_positive = int(replication.event.sum())
    state_positive = int(threshold.state.sum())
    overlap = int((threshold.state & replication.event).sum())
    row["pr_z_given_B"] = (
        overlap / event_positive if event_positive > 0 else float("nan")
    )
    row["pr_B_given_z"] = (
        overlap / state_positive if state_positive > 0 else float("nan")
    )
    stress_float = replication.stress.astype(float)
    row["corr_xbar_S"] = _correlation(
        replication.continuous.mean(axis=1), stress_float
    )
    row["corr_zbar_S"] = _correlation(threshold.state.mean(axis=1), stress_float)
    row["corr_Bbar_S"] = _correlation(replication.event.mean(axis=1), stress_float)
    return pd.DataFrame([row])


def run_one_replication(
    seed: int,
    *,
    config: SimulationConfig,
    protocol: SimulationProtocol,
    pilot: bool,
) -> ReplicationOutcome:
    """Generate and evaluate one candidate; all method choices are required."""
    config.validate()
    protocol.validate()
    replication = simulate_replication(seed, config=config)
    candidate_stress_frequency = float(replication.stress.mean())
    target_problem = _target_problem(replication.target, config)
    if target_problem is not None:
        return ReplicationOutcome(
            seed, False, target_problem, candidate_stress_frequency, None, None, ()
        )
    try:
        threshold = build_global_threshold(
            replication.continuous,
            replication.target,
            config=config,
            rank_policy=protocol.rank_policy,
            clip_policy=protocol.clip_policy,
            tail_inequality=protocol.tail_inequality,
        )
    except ValueError as exc:
        if "Empty R" not in str(exc):
            raise
        return ReplicationOutcome(
            seed, False, "empty_R", candidate_stress_frequency, None, None, ()
        )
    diagnostics = _build_diagnostics(replication, threshold, config)
    if pilot:
        return ReplicationOutcome(
            seed, True, None, candidate_stress_frequency, None, diagnostics, ()
        )

    metric_rows: list[dict[str, object]] = []
    fit_warning_rows: list[FitWarningRecord] = []
    for horizon in config.horizons:
        for model_name, panel, continuous_stack in (
            ("Xt_Logit_L2", replication.continuous, True),
            ("Zt_Logit_L2", threshold.state, False),
        ):
            design = build_design(panel, replication.target, horizon, config=config)
            standardize = (
                protocol.estimator.standardization_policy
                is PredictorStandardizationPolicy.BOTH_STACKS_TRAIN_ONLY
                or (
                    protocol.estimator.standardization_policy
                    is PredictorStandardizationPolicy.CONTINUOUS_ONLY_TRAIN_ONLY
                    and continuous_stack
                )
            )
            if standardize:
                train, test = standardize_train_test(design.train, design.test)
            else:
                train, test = design.train, design.test
            probability, warning_rows = _fit_predict(
                train,
                design.y_train,
                test,
                estimator=protocol.estimator,
                warning_policy=protocol.warning_policy,
            )
            fit_warning_rows.extend(
                FitWarningRecord(horizon, model_name, category, message)
                for category, message in warning_rows
            )
            metric_rows.append(
                {
                    "horizon": int(horizon),
                    "model": model_name,
                    "pr_auc": float(
                        average_precision_score(design.y_test, probability)
                    ),
                    "brier": float(brier_score_loss(design.y_test, probability)),
                }
            )
    return ReplicationOutcome(
        seed,
        True,
        None,
        candidate_stress_frequency,
        pd.DataFrame(metric_rows),
        diagnostics,
        tuple(fit_warning_rows),
    )


def run_simulation_batch(
    *,
    requested_replications: int,
    candidate_cap: int,
    config: SimulationConfig,
    protocol: SimulationProtocol,
    pilot: bool,
) -> SimulationBatch:
    """Run one explicit seed policy without writing or silently dropping candidates."""
    config.validate()
    protocol.validate()
    if not isinstance(requested_replications, int) or isinstance(
        requested_replications, bool
    ) or requested_replications < 1:
        raise ValueError("Requested simulation replication count must be positive")
    if not isinstance(candidate_cap, int) or isinstance(candidate_cap, bool):
        raise ValueError("Simulation candidate cap must be an integer")
    if candidate_cap < requested_replications:
        raise ValueError(
            "Simulation candidate cap must be at least the requested replication count"
        )
    outcomes: list[ReplicationOutcome] = []
    offset = 0
    while True:
        if len(outcomes) >= candidate_cap:
            partial_batch = SimulationBatch(
                protocol=protocol,
                config=config,
                requested_replications=requested_replications,
                candidate_cap=candidate_cap,
                pilot=pilot,
                outcomes=tuple(outcomes),
            )
            raise SimulationAttemptCapExceeded(partial_batch)
        outcome = run_one_replication(
            config.base_seed + offset,
            config=config,
            protocol=protocol,
            pilot=pilot,
        )
        outcomes.append(outcome)
        offset += 1
        if protocol.selection_policy is ReplicationSelectionPolicy.FIXED_CANDIDATE_SET:
            if len(outcomes) == requested_replications:
                break
        elif sum(item.accepted for item in outcomes) == requested_replications:
            break
    return SimulationBatch(
        protocol=protocol,
        config=config,
        requested_replications=requested_replications,
        candidate_cap=candidate_cap,
        pilot=pilot,
        outcomes=tuple(outcomes),
    )


def _standard_error(values: np.ndarray) -> float:
    finite = values[~np.isnan(values)]
    if finite.size <= 1:
        return float("nan")
    return float(finite.std(ddof=1) / np.sqrt(finite.size))


def _format_mean_standard_error(mean: float, standard_error: float) -> str:
    if np.isnan(mean):
        return "n/a"
    if np.isnan(standard_error):
        return f"{mean:.3f}"
    return f"{mean:.3f} ({standard_error:.3f})"


def _validate_complete_metrics(metrics: pd.DataFrame, config: SimulationConfig) -> None:
    required = {"replication", "seed", "horizon", "model", "pr_auc", "brier"}
    if set(metrics.columns) != required or metrics.empty:
        raise ValueError("Simulation metrics panel has an invalid schema or is empty")
    expected_models = {"Xt_Logit_L2", "Zt_Logit_L2"}
    for replication, group in metrics.groupby("replication", sort=False):
        keys = set(zip(group["horizon"], group["model"], strict=True))
        expected = {(horizon, model) for horizon in config.horizons for model in expected_models}
        if keys != expected or len(group) != len(expected):
            raise ValueError(f"Simulation replication {replication} is incomplete")
    if not np.isfinite(metrics[["pr_auc", "brier"]].to_numpy(dtype=float)).all():
        raise ValueError("Simulation metrics contain nonfinite values")


def build_main_table(
    metrics: pd.DataFrame,
    *,
    config: SimulationConfig,
) -> pd.DataFrame:
    """Build the frozen Table 20 estimands from a complete metrics panel."""
    _validate_complete_metrics(metrics, config)
    pr_auc = metrics.pivot(
        index="replication", columns=["horizon", "model"], values="pr_auc"
    )
    brier = metrics.pivot(
        index="replication", columns=["horizon", "model"], values="brier"
    )
    rows: list[dict[str, object]] = []
    for horizon in config.horizons:
        pr_x = pr_auc[(horizon, "Xt_Logit_L2")].to_numpy()
        pr_z = pr_auc[(horizon, "Zt_Logit_L2")].to_numpy()
        brier_x = brier[(horizon, "Xt_Logit_L2")].to_numpy()
        brier_z = brier[(horizon, "Zt_Logit_L2")].to_numpy()
        difference_pr = pr_z - pr_x
        difference_brier = brier_z - brier_x
        rows.append(
            {
                "Horizon": horizon,
                "PR AUC $X_t$": _format_mean_standard_error(
                    float(np.mean(pr_x)), _standard_error(pr_x)
                ),
                "PR AUC $Z_t$": _format_mean_standard_error(
                    float(np.mean(pr_z)), _standard_error(pr_z)
                ),
                r"$\Delta$ PR AUC (Z$-$X)": _format_mean_standard_error(
                    float(np.mean(difference_pr)), _standard_error(difference_pr)
                ),
                "Win rate (PR, Z$>$X)": f"{100 * np.mean(difference_pr > 0):.1f}%",
                "Brier $X_t$": _format_mean_standard_error(
                    float(np.mean(brier_x)), _standard_error(brier_x)
                ),
                "Brier $Z_t$": _format_mean_standard_error(
                    float(np.mean(brier_z)), _standard_error(brier_z)
                ),
                r"$\Delta$ Brier (Z$-$X)": _format_mean_standard_error(
                    float(np.mean(difference_brier)), _standard_error(difference_brier)
                ),
                "Win rate (Brier, Z$<$X)": (
                    f"{100 * np.mean(difference_brier < 0):.1f}%"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics_table(
    diagnostics: pd.DataFrame,
    *,
    config: SimulationConfig,
) -> pd.DataFrame:
    """Build the frozen reader-facing diagnostics summary."""
    required = {
        "tau",
        "tau_clipped",
        "pr_B",
        "pr_z",
        "pr_z_eq_B",
        "pr_z_given_B",
        "pr_B_given_z",
        "corr_zbar_S",
        "corr_Bbar_S",
        "corr_xbar_S",
        "stress_freq",
        "emp_p11",
        "emp_p01",
        *{f"recession_freq_test_h{horizon}" for horizon in config.horizons},
    }
    if diagnostics.empty or not required.issubset(diagnostics.columns):
        raise ValueError("Simulation diagnostics panel is empty or incomplete")
    means = diagnostics.mean(numeric_only=True)
    rows = [
        {
            "Horizon": str(horizon),
            r"$\Pr(R=1)$ test": f"{means[f'recession_freq_test_h{horizon}']:.3f}",
        }
        for horizon in config.horizons
    ]
    rows.append({"Horizon": "all", r"$\Pr(R=1)$ test": "--"})
    table = pd.DataFrame(rows)
    sample_wide = {
        r"$\bar\tau$": f"{means['tau']:.3f}",
        r"$\tau$ clip rate": f"{means['tau_clipped']:.3f}",
        r"$\Pr(B=1)$": f"{means['pr_B']:.3f}",
        r"$\Pr(z=1)$": f"{means['pr_z']:.3f}",
        r"$\Pr(z=B)$": f"{means['pr_z_eq_B']:.3f}",
        r"$\Pr(z|B=1)$": f"{means['pr_z_given_B']:.3f}",
        r"$\Pr(B|z=1)$": f"{means['pr_B_given_z']:.3f}",
        r"$\mathrm{corr}(\bar z, S)$": f"{means['corr_zbar_S']:.3f}",
        r"$\mathrm{corr}(\bar B, S)$": f"{means['corr_Bbar_S']:.3f}",
        r"$\mathrm{corr}(\bar x, S)$": f"{means['corr_xbar_S']:.3f}",
        r"$\Pr(S=1)$": f"{means['stress_freq']:.3f}",
        r"$\hat p_{11}$": f"{means['emp_p11']:.3f}",
        r"$\hat p_{01}$": f"{means['emp_p01']:.4f}",
    }
    for column, value in sample_wide.items():
        table[column] = ""
        table.loc[table["Horizon"] == "all", column] = value
    return table
