"""Locked package configuration and fail-closed author-decision gate.

This module may describe legacy/result-family membership before the author
selects corrected estimands, but no canonical compute entry point may proceed
until :func:`assert_canonical_generation_allowed` succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
RESULTS_ROOT = PACKAGE_ROOT / "results"
OUTPUT_ROOT = PACKAGE_ROOT / "output"
DOCS_ROOT = PACKAGE_ROOT / "docs"

GLOBAL_SEED = 42
HORIZONS = (3, 6, 12)
LAGS = (3, 6, 12)
BASELINE_SMOOTHING_WINDOW = 1
REVISED_OOS_START = "1990-01-01"
REALTIME_OOS_START = "1999-08-01"


class DecisionStatus(StrEnum):
    """Permitted states for an audited author decision."""

    PENDING = "pending"
    RESOLVED = "resolved"
    ACCEPTED_LIMITATION = "accepted_limitation"


@dataclass(frozen=True)
class MethodDecision:
    finding_id: str
    topic: str
    recommended_resolution: str
    status: DecisionStatus = DecisionStatus.PENDING
    selected_resolution: str | None = None

    def validate(self) -> None:
        if self.status is DecisionStatus.PENDING and self.selected_resolution is not None:
            raise ValueError(f"{self.finding_id}: pending decision cannot select a resolution")
        if self.status is not DecisionStatus.PENDING and not self.selected_resolution:
            raise ValueError(f"{self.finding_id}: resolved decision must record its resolution")


METHOD_DECISIONS = (
    MethodDecision(
        "RF-004", "FRED-MD transformation code 7", "Use the published code-7 formula.",
        DecisionStatus.RESOLVED, "published_difference_of_percent_change",
    ),
    MethodDecision(
        "RF-005", "real-time forecast calendar", "Use the latest observation actually present in each vintage.",
        DecisionStatus.RESOLVED, "latest_available_observation+exact_target",
    ),
    MethodDecision(
        "RF-006", "real-time calibration information set", "Calibrate thresholds and penalties within each vintage information set.",
        DecisionStatus.RESOLVED, "variant_specific",
    ),
    MethodDecision(
        "RF-007", "encompassing covariance", "Use a horizon-linked HAC covariance rule.",
        DecisionStatus.ACCEPTED_LIMITATION, "legacy_iid",
    ),
    MethodDecision(
        "RF-008", "simulation acceptance conditioning", "Use a fixed candidate-seed design with explicit degenerate handling.",
        DecisionStatus.ACCEPTED_LIMITATION, "conditional_accepted_seed_walk",
    ),
    MethodDecision(
        "RF-009", "tail inequality", "Preserve strict code and correct the manuscript definition.",
        DecisionStatus.RESOLVED, "strict",
    ),
    MethodDecision(
        "RF-010", "pre-recession windows", "Use twelve months ending one month before each peak.",
        DecisionStatus.RESOLVED, "legacy_declared+allow_available_with_count",
    ),
    MethodDecision(
        "RF-011", "moving-average boundary", "Require a complete smoothing window.",
        DecisionStatus.RESOLVED, "legacy_partial",
    ),
    MethodDecision(
        "RF-012", "rank tie semantics", "Preserve average ranks and define the tie rule.",
        DecisionStatus.RESOLVED, "average_percent_rank",
    ),
    MethodDecision(
        "RF-013", "regularization CV preprocessing", "Refit preprocessing inside every penalty-validation fold.",
        DecisionStatus.RESOLVED, "fold_pure",
    ),
    MethodDecision(
        "RF-014", "data redistribution", "Distribute acquisition instructions unless redistribution is approved.",
        DecisionStatus.ACCEPTED_LIMITATION,
        "internal_bundle+public_acquisition_instructions_no_redistribution",
    ),
    MethodDecision(
        "RF-015", "target provenance", "Record the frozen checksum and disclose unrecoverable acquisition provenance.",
        DecisionStatus.ACCEPTED_LIMITATION,
        "accepted_frozen_checksum_unrecoverable_acquisition_provenance",
    ),
    MethodDecision(
        "RF-016", "Brier no-information reference", "Use prevalence as the constant no-information reference.",
        DecisionStatus.RESOLVED, "sample_prevalence",
    ),
    MethodDecision(
        "RF-017", "penalized-logit objective", "Specify balanced weights and an unpenalized intercept.",
        DecisionStatus.RESOLVED, "balanced_with_intercept",
    ),
    MethodDecision(
        "RF-018", "simulation warning policy", "Use explicit invariants and narrow warning handling.",
        DecisionStatus.ACCEPTED_LIMITATION, "legacy_suppress_all",
    ),
    MethodDecision(
        "RF-019", "simulation procedure parity", "Align simulation rules or disclose every distinct convention.",
        DecisionStatus.RESOLVED,
        "right_continuous_ecdf+legacy_winsorize_005_040+weak+unweighted+both_stacks_train_only",
    ),
    MethodDecision(
        "RF-020", "real-time predictor composition", "Use an explicitly common locked predictor set.",
        DecisionStatus.RESOLVED, "legacy_variant_specific",
    ),
    MethodDecision(
        "RF-021", "regularization initialization", "Track initialization per predictor and horizon and fail fast.",
        DecisionStatus.RESOLVED, "strict_per_key",
    ),
    MethodDecision(
        "RF-022",
        "recession-rank missingness for variable and sector thresholds",
        "Use each variable's available recession ranks and disclose or regenerate the affected robustness results.",
        DecisionStatus.RESOLVED,
        "per_variable_available_rows",
    ),
    MethodDecision(
        "RF-023",
        "simulation machine-readable numerical reproducibility",
        "Require same-environment double-run evidence and document the legacy-panel ulp drift.",
        DecisionStatus.ACCEPTED_LIMITATION,
        "same_environment_double_run_and_documented_legacy_ulp_limit",
    ),
    MethodDecision(
        "RF-025",
        "term-spread regularization fallback after class-degenerate CV",
        "Either redesign the pre-OOS time-series folds so every training fold is estimable or disclose and explicitly retain fixed C=1 for the term-spread logit.",
        DecisionStatus.RESOLVED,
        "strict_raise",
    ),
)


def pending_method_decisions() -> tuple[MethodDecision, ...]:
    """Return unresolved decisions in stable finding order."""
    return tuple(decision for decision in METHOD_DECISIONS if decision.status is DecisionStatus.PENDING)


def assert_canonical_generation_allowed() -> None:
    """Fail closed until every M1 author decision has an explicit disposition."""
    for decision in METHOD_DECISIONS:
        decision.validate()
    pending = pending_method_decisions()
    if pending:
        identifiers = ", ".join(decision.finding_id for decision in pending)
        raise RuntimeError(
            "Canonical program_v04 generation is disabled pending author decisions: "
            f"{identifiers}. Resolve the M1 decision gate and update both "
            "replication/config.py and docs/discrepancy_ledger.csv before computing results."
        )


@dataclass(frozen=True)
class ResultFamilySpec:
    """Locked membership and ownership for one normalized result family."""

    exercise: str
    family: str
    owner: str
    result_subdirectory: str
    predictor_models: tuple[tuple[str, str], ...]
    consumers: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.exercise, self.family

    @property
    def result_path(self) -> Path:
        return RESULTS_ROOT / self.result_subdirectory


def _pairs(*entries: tuple[str, tuple[str, ...]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (predictor, model)
        for predictor, models in entries
        for model in models
    )


MAIN_PAIRS = _pairs(
    ("Zt_Disagg", ("Logit_L2", "XGBoost")),
    ("Zt_Avg", ("Logit_L2", "XGBoost")),
    ("Zt_PCA", ("Logit_L2", "XGBoost")),
    ("Xt_Full", ("Logit_L2", "XGBoost")),
    ("Xt_PCA", ("Logit_L2",)),
)

NOLAG_PAIRS = _pairs(
    ("Zt_Disagg", ("Logit_L2", "XGBoost")),
    ("Zt_Avg", ("Logit_L2", "XGBoost")),
    ("Zt_PCA", ("Logit_L2", "XGBoost")),
    ("Xt_Full", ("Logit_L2", "XGBoost")),
    ("Xt_PCA", ("Logit_L2",)),
)

PARSIMONIOUS_PAIRS = _pairs(
    ("Zt_Parsimonious", ("Logit_L2",)),
    ("Xt_Parsimonious", ("XGBoost",)),
    ("Yield_Univariate", ("XGBoost",)),
)

# RF-025 removes only these non-artifact logit members. Reconciliation tools
# consume this explicit contract so their absence cannot be mistaken for
# unexplained numerical drift.
RF025_REMOVED_PAIRS = (
    ("main", "Yield", "Logit_L2"),
    ("nolag", "Yield", "Logit_L2"),
    ("parsimonious", "Yield_Univariate", "Logit_L2"),
)

EXTENDED_PAIRS = _pairs(
    ("Zt_Disagg", ("Logit_L2", "Logit_L1", "ElasticNet", "XGBoost", "RandomForest", "MLP")),
    ("Xt_Full", ("Logit_L2", "Logit_L1", "ElasticNet", "XGBoost", "RandomForest", "MLP")),
)

REALTIME_MAIN_PAIRS = _pairs(
    ("Zt_Disagg", ("Logit_L2",)),
    ("Zt_PCA", ("Logit_L2", "XGBoost")),
    ("Xt_Full", ("Logit_L2", "XGBoost", "RandomForest")),
    ("Xt_PCA", ("Logit_L2", "XGBoost", "RandomForest")),
)

REALTIME_EXTENDED_PAIRS = _pairs(
    ("Zt_Disagg", ("Logit_L2", "XGBoost")),
    ("Xt_Full", ("Logit_L2", "XGBoost", "RandomForest")),
    ("Zt_Avg", ("Logit_L2", "XGBoost")),
    ("Zt_PCA", ("Logit_L2", "XGBoost")),
    ("Xt_PCA", ("Logit_L2", "XGBoost", "RandomForest")),
)

RESULT_FAMILIES = (
    ResultFamilySpec("revised", "main", "shared", "shared/revised_main", MAIN_PAIRS, ("main", "appendix")),
    ResultFamilySpec("revised", "extended", "shared", "shared/revised_extended", EXTENDED_PAIRS, ("main", "appendix")),
    ResultFamilySpec("revised", "probit_spread", "main", "main/revised_probit_spread", (("Yield_Spread", "Probit"), ("Yield_Spread_Binary", "Probit")), ("main",)),
    ResultFamilySpec("revised", "parameter_free", "main", "main/revised_parameter_free", (("Zt_Share", "Parameter_Free"),), ("main",)),
    ResultFamilySpec("revised", "threshold", "appendix", "appendix/revised_threshold", _pairs(("Zt_Disagg_SecQ", ("Logit_L2",)), ("Zt_Disagg_VarQ", ("Logit_L2",)), ("Zt_PCA_SecQ", ("Logit_L2",)), ("Zt_PCA_VarQ", ("Logit_L2",))), ("appendix",)),
    ResultFamilySpec("revised", "moving_average", "appendix", "appendix/revised_moving_average", _pairs(*tuple((f"{base}_MA{window}", ("Logit_L2",)) for window in (3, 6, 12) for base in ("Zt_Disagg", "Zt_PCA"))), ("appendix",)),
    ResultFamilySpec("revised", "nolag", "appendix", "appendix/revised_nolag", NOLAG_PAIRS, ("appendix",)),
    ResultFamilySpec("revised", "parsimonious", "appendix", "appendix/revised_parsimonious", PARSIMONIOUS_PAIRS, ("appendix",)),
    ResultFamilySpec("revised", "alternative_subsets", "appendix", "appendix/revised_alternative_subsets", _pairs(("Zt_Spreads", ("Logit_L2",)), ("Xt_Spreads", ("XGBoost",)), ("Zt_RealEcon", ("Logit_L2",)), ("Xt_RealEcon", ("XGBoost",))), ("appendix",)),
    ResultFamilySpec("revised", "expanding_threshold", "appendix", "appendix/revised_expanding_threshold", (("Zt_Disagg_ExpandingQ", "Logit_L2"),), ("appendix",)),
    ResultFamilySpec("realtime", "realtime_main", "appendix", "appendix/realtime_main", REALTIME_MAIN_PAIRS, ("appendix",)),
    ResultFamilySpec("realtime", "realtime_extended", "appendix", "appendix/realtime_extended", REALTIME_EXTENDED_PAIRS, ("appendix",)),
)


def family_registry() -> dict[tuple[str, str], ResultFamilySpec]:
    """Return the validated family registry keyed by exercise and family."""
    registry = {spec.key: spec for spec in RESULT_FAMILIES}
    if len(registry) != len(RESULT_FAMILIES):
        raise ValueError("Duplicate exercise/family key in RESULT_FAMILIES")
    for spec in RESULT_FAMILIES:
        if not spec.predictor_models:
            raise ValueError(f"{spec.key}: empty predictor/model membership")
        if len(set(spec.predictor_models)) != len(spec.predictor_models):
            raise ValueError(f"{spec.key}: duplicate predictor/model pair")
        if spec.owner == "shared" and set(spec.consumers) != {"main", "appendix"}:
            raise ValueError(f"{spec.key}: shared family must serve both profiles")
        if Path(spec.result_subdirectory).is_absolute() or ".." in Path(spec.result_subdirectory).parts:
            raise ValueError(f"{spec.key}: unsafe result subdirectory")
    return registry


FAMILY_REGISTRY = family_registry()
