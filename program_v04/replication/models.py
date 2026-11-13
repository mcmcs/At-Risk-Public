"""Audited estimator, time-series validation, and frozen-tuning contracts."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier

from .validation import require_binary

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - depends on the native runtime
    xgb = None
    XGBOOST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    XGBOOST_IMPORT_ERROR = None


GLOBAL_SEED = 42
C_GRID_SIZE = 30
C_VALUES = tuple(float(value) for value in np.logspace(-3, 1, C_GRID_SIZE))
MIN_TUNING_ROWS = 50


class LogisticPenalty(StrEnum):
    L2 = "l2"
    L1 = "l1"
    ELASTICNET = "elasticnet"


class ClassImbalanceRule(StrEnum):
    """Candidate interpretation of the weighted objective in RF-017."""

    UNWEIGHTED = "unweighted"
    BALANCED = "balanced"


class CVPreprocessingPolicy(StrEnum):
    """Whether feature preprocessing is rebuilt inside each C-selection fold."""

    LEGACY_PRECOMPUTED = "legacy_precomputed"
    FOLD_PURE = "fold_pure"


class ModelSet(StrEnum):
    DEFAULT = "default"
    EXTENDED = "extended"


@dataclass(frozen=True)
class LogisticSpec:
    penalty: LogisticPenalty
    c_value: float
    imbalance_rule: ClassImbalanceRule
    fit_intercept: bool
    random_state: int = GLOBAL_SEED

    def validate(self) -> None:
        if not isinstance(self.penalty, LogisticPenalty):
            raise ValueError(f"Unsupported logistic penalty: {self.penalty}")
        if not np.isfinite(self.c_value) or self.c_value <= 0.0:
            raise ValueError("Logistic C must be finite and positive")
        if not isinstance(self.imbalance_rule, ClassImbalanceRule):
            raise ValueError(f"Unsupported class-imbalance rule: {self.imbalance_rule}")
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("fit_intercept must be Boolean")

    @property
    def solver(self) -> str:
        return {
            LogisticPenalty.L2: "lbfgs",
            LogisticPenalty.L1: "liblinear",
            LogisticPenalty.ELASTICNET: "saga",
        }[self.penalty]

    @property
    def l1_ratio(self) -> float:
        return {
            LogisticPenalty.L2: 0.0,
            LogisticPenalty.L1: 1.0,
            LogisticPenalty.ELASTICNET: 0.5,
        }[self.penalty]

    @property
    def max_iter(self) -> int:
        return 2000 if self.penalty is LogisticPenalty.ELASTICNET else 1000


def make_logistic(spec: LogisticSpec) -> LogisticRegression:
    """Construct the exact version-locked logistic estimator from an explicit objective."""
    spec.validate()
    class_weight = (
        "balanced" if spec.imbalance_rule is ClassImbalanceRule.BALANCED else None
    )
    return LogisticRegression(
        solver=spec.solver,
        C=float(spec.c_value),
        l1_ratio=spec.l1_ratio,
        max_iter=spec.max_iter,
        random_state=spec.random_state,
        class_weight=class_weight,
        fit_intercept=spec.fit_intercept,
    )


def get_model_templates(
    *,
    optimal_c_l2: float,
    optimal_c_l1: float,
    optimal_c_elasticnet: float,
    model_set: ModelSet,
    imbalance_rule: ClassImbalanceRule,
    fit_intercept: bool,
) -> dict[str, object]:
    """Return unfitted templates while keeping RF-017 choices explicit."""
    if not isinstance(model_set, ModelSet):
        raise ValueError(f"Unsupported model set: {model_set}")
    templates: dict[str, object] = {
        "Logit_L2": make_logistic(
            LogisticSpec(
                LogisticPenalty.L2,
                optimal_c_l2,
                imbalance_rule,
                fit_intercept,
            )
        )
    }
    if xgb is None:
        raise RuntimeError(f"Locked XGBoost runtime is unavailable: {XGBOOST_IMPORT_ERROR}")
    templates["XGBoost"] = xgb.XGBClassifier(
        random_state=GLOBAL_SEED,
        eval_metric="logloss",
    )
    if model_set is ModelSet.EXTENDED:
        templates.update(
            {
                "RandomForest": RandomForestClassifier(
                    random_state=GLOBAL_SEED,
                    n_estimators=100,
                    class_weight=(
                        "balanced"
                        if imbalance_rule is ClassImbalanceRule.BALANCED
                        else None
                    ),
                ),
                "Logit_L1": make_logistic(
                    LogisticSpec(
                        LogisticPenalty.L1,
                        optimal_c_l1,
                        imbalance_rule,
                        fit_intercept,
                    )
                ),
                "ElasticNet": make_logistic(
                    LogisticSpec(
                        LogisticPenalty.ELASTICNET,
                        optimal_c_elasticnet,
                        imbalance_rule,
                        fit_intercept,
                    )
                ),
                "MLP": MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    random_state=GLOBAL_SEED,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            }
        )
    return templates


def _require_finite_frame(frame: pd.DataFrame, *, surface: str) -> None:
    if frame.empty:
        raise ValueError(f"{surface}: feature frame is empty")
    if frame.columns.has_duplicates or not frame.index.is_unique:
        raise ValueError(f"{surface}: feature columns and index must be unique")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{surface}: features must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{surface}: features must be finite")


def fit_predict_single_model(
    model_name: str,
    model_template: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_predict: pd.DataFrame,
    *,
    imbalance_rule: ClassImbalanceRule,
) -> tuple[float, object]:
    """Fit one required model and return a validated one-row probability."""
    if not isinstance(imbalance_rule, ClassImbalanceRule):
        raise ValueError(f"Unsupported class-imbalance rule: {imbalance_rule}")
    _require_finite_frame(X_train, surface=f"{model_name} training")
    _require_finite_frame(X_predict, surface=f"{model_name} prediction")
    if len(X_predict) != 1:
        raise ValueError(f"{model_name}: prediction frame must contain exactly one row")
    if list(X_train.columns) != list(X_predict.columns):
        raise ValueError(f"{model_name}: training and prediction columns differ")
    if not X_train.index.equals(y_train.index):
        raise ValueError(f"{model_name}: training features and target index differ")
    require_binary(y_train, surface=model_name, field="training target")
    if pd.Series(y_train).nunique() != 2:
        raise ValueError(f"{model_name}: training target must contain both classes")

    model = clone(model_template)
    if model_name == "XGBoost":
        positives = int((y_train == 1).sum())
        negatives = int((y_train == 0).sum())
        scale = negatives / positives if imbalance_rule is ClassImbalanceRule.BALANCED else 1.0
        model.set_params(scale_pos_weight=scale)
    elif hasattr(model, "class_weight"):
        model.set_params(
            class_weight=(
                "balanced" if imbalance_rule is ClassImbalanceRule.BALANCED else None
            )
        )

    model.fit(X_train, y_train)
    probability = float(model.predict_proba(X_predict)[:, 1][0])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{model_name}: estimator returned an invalid probability")
    return probability, model


@dataclass(frozen=True)
class CVFold:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    preprocessing_policy: CVPreprocessingPolicy

    def validate(self) -> None:
        _require_finite_frame(self.X_train, surface="CV training")
        _require_finite_frame(self.X_validation, surface="CV validation")
        if list(self.X_train.columns) != list(self.X_validation.columns):
            raise ValueError("CV training and validation columns differ")
        if not self.X_train.index.equals(self.y_train.index):
            raise ValueError("CV training features and target index differ")
        if not self.X_validation.index.equals(self.y_validation.index):
            raise ValueError("CV validation features and target index differ")
        if not self.X_train.index.is_monotonic_increasing:
            raise ValueError("CV training index must be sorted")
        if not self.X_validation.index.is_monotonic_increasing:
            raise ValueError("CV validation index must be sorted")
        if self.X_train.index.max() >= self.X_validation.index.min():
            raise ValueError("CV validation observations must follow training observations")
        require_binary(self.y_train, surface="CV", field="training target")
        require_binary(self.y_validation, surface="CV", field="validation target")
        if self.y_train.nunique() != 2:
            raise ValueError("CV training target must contain both classes")
        if not isinstance(self.preprocessing_policy, CVPreprocessingPolicy):
            raise ValueError("CV fold has an invalid preprocessing policy")


class FoldPreprocessor(Protocol):
    """Fresh per-fold transformer whose fit surface never receives validation rows."""

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame: ...

    def transform(self, X_validation: pd.DataFrame) -> pd.DataFrame: ...


FoldPreprocessorFactory = Callable[[], FoldPreprocessor]


def build_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    preprocessing_policy: CVPreprocessingPolicy,
    fold_preprocessor_factory: FoldPreprocessorFactory | None = None,
    n_splits: int = 5,
) -> tuple[CVFold, ...]:
    """Build time-ordered folds under an explicit RF-013 preprocessing policy."""
    if not isinstance(preprocessing_policy, CVPreprocessingPolicy):
        raise ValueError(f"Unsupported CV preprocessing policy: {preprocessing_policy}")
    if n_splits < 2:
        raise ValueError("Time-series CV requires at least two splits")
    if not X.index.equals(y.index):
        raise ValueError("C-selection features and target must have identical indexes")
    if not X.index.is_unique or not X.index.is_monotonic_increasing:
        raise ValueError("C-selection index must be sorted and unique")
    require_binary(y, surface="C selection", field="target")
    if preprocessing_policy is CVPreprocessingPolicy.LEGACY_PRECOMPUTED:
        if fold_preprocessor_factory is not None:
            raise ValueError("Legacy precomputed CV cannot accept a fold preprocessor factory")
        _require_finite_frame(X, surface="precomputed CV")
    elif fold_preprocessor_factory is None:
        raise ValueError("Fold-pure CV requires a fold preprocessor factory (RF-013)")

    folds: list[CVFold] = []
    preprocessor_identities: set[int] = set()
    preprocessors: list[FoldPreprocessor] = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_positions, validation_positions in splitter.split(X):
        X_train_raw = X.iloc[train_positions]
        X_validation_raw = X.iloc[validation_positions]
        y_train_raw = y.iloc[train_positions]
        if preprocessing_policy is CVPreprocessingPolicy.FOLD_PURE:
            if fold_preprocessor_factory is None:
                raise RuntimeError("Fold-pure preprocessor factory disappeared")
            preprocessor = fold_preprocessor_factory()
            identity = id(preprocessor)
            if identity in preprocessor_identities:
                raise ValueError("Fold-pure preprocessor factory must return a fresh object")
            preprocessor_identities.add(identity)
            preprocessors.append(preprocessor)
            X_train = preprocessor.fit_transform(
                X_train_raw.copy(),
                y_train_raw.copy(),
            )
            X_validation = preprocessor.transform(X_validation_raw.copy())
            if not X_train.index.isin(X_train_raw.index).all():
                raise ValueError("Fold preprocessor introduced non-training rows")
            if not X_validation.index.isin(X_validation_raw.index).all():
                raise ValueError("Fold preprocessor introduced non-validation rows")
        else:
            X_train = X_train_raw.copy()
            X_validation = X_validation_raw.copy()
        fold = CVFold(
            X_train=X_train,
            y_train=y.loc[X_train.index].copy(),
            X_validation=X_validation,
            y_validation=y.loc[X_validation.index].copy(),
            preprocessing_policy=preprocessing_policy,
        )
        fold.validate()
        folds.append(fold)
    return tuple(folds)


@dataclass(frozen=True)
class CSelectionResult:
    selected_c: float
    selected_average_brier: float | None
    average_brier_by_c: tuple[tuple[float, float], ...]
    rejected_candidates: tuple[tuple[float, str], ...]
    preprocessing_policy: CVPreprocessingPolicy
    n_splits: int

    def as_metadata(self) -> dict[str, object]:
        return {
            "selected_c": self.selected_c,
            "selected_average_brier": self.selected_average_brier,
            "average_brier_by_c": [
                {"c": c_value, "average_brier": score}
                for c_value, score in self.average_brier_by_c
            ],
            "rejected_candidates": [
                {"c": c_value, "reason": reason}
                for c_value, reason in self.rejected_candidates
            ],
            "preprocessing_policy": self.preprocessing_policy.value,
            "n_splits": self.n_splits,
        }


def select_logistic_c(
    folds: tuple[CVFold, ...],
    *,
    penalty: LogisticPenalty,
    c_values: Iterable[float] = C_VALUES,
    imbalance_rule: ClassImbalanceRule,
    fit_intercept: bool,
) -> CSelectionResult:
    """Minimize mean fold Brier score, retaining first-in-grid tie behavior."""
    if not folds:
        raise ValueError("C selection requires at least one validation fold")
    for fold in folds:
        fold.validate()
    policies = {fold.preprocessing_policy for fold in folds}
    if len(policies) != 1:
        raise ValueError("C-selection folds do not share one preprocessing policy")
    preprocessing_policy = next(iter(policies))
    candidates = tuple(float(value) for value in c_values)
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or not all(np.isfinite(value) and value > 0.0 for value in candidates)
    ):
        raise ValueError("C grid must contain unique finite positive values")

    scores: list[tuple[float, float]] = []
    rejected: list[tuple[float, str]] = []
    for c_value in candidates:
        fold_scores: list[float] = []
        try:
            for fold in folds:
                model = make_logistic(
                    LogisticSpec(
                        penalty,
                        c_value,
                        imbalance_rule,
                        fit_intercept,
                    )
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    model.fit(fold.X_train, fold.y_train)
                probability = model.predict_proba(fold.X_validation)[:, 1]
                fold_scores.append(
                    float(
                        np.mean(
                            (
                                fold.y_validation.to_numpy(dtype=float)
                                - probability
                            )
                            ** 2
                        )
                    )
                )
        except (ConvergenceWarning, ValueError) as exc:
            rejected.append((c_value, f"{type(exc).__name__}: {exc}"))
            continue
        scores.append((c_value, float(np.mean(fold_scores))))
    if not scores:
        raise RuntimeError("Every C candidate failed; no silent C=1 fallback is allowed")
    selected_c, selected_score = min(scores, key=lambda item: item[1])
    return CSelectionResult(
        selected_c=selected_c,
        selected_average_brier=selected_score,
        average_brier_by_c=tuple(scores),
        rejected_candidates=tuple(rejected),
        preprocessing_policy=preprocessing_policy,
        n_splits=len(folds),
    )


def optimize_logistic_c(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    penalty: LogisticPenalty,
    preprocessing_policy: CVPreprocessingPolicy,
    imbalance_rule: ClassImbalanceRule,
    fit_intercept: bool,
    fold_preprocessor_factory: FoldPreprocessorFactory | None = None,
    c_values: Iterable[float] = C_VALUES,
    n_splits: int = 5,
) -> CSelectionResult:
    """Build audited folds and select C without an implicit RF-013/RF-017 choice."""
    if len(X) < MIN_TUNING_ROWS:
        raise ValueError(
            f"C selection requires at least {MIN_TUNING_ROWS} rows; "
            "the legacy C=1 small-sample branch must be invoked explicitly"
        )
    folds = build_cv_folds(
        X,
        y,
        preprocessing_policy=preprocessing_policy,
        fold_preprocessor_factory=fold_preprocessor_factory,
        n_splits=n_splits,
    )
    return select_logistic_c(
        folds,
        penalty=penalty,
        c_values=c_values,
        imbalance_rule=imbalance_rule,
        fit_intercept=fit_intercept,
    )


@dataclass(frozen=True, order=True)
class TuningKey:
    data_scope: str
    predictor_key: str
    horizon: int
    penalty: LogisticPenalty

    def validate(self) -> None:
        if not self.data_scope or not self.predictor_key:
            raise ValueError("Tuning scope and predictor key must be nonempty")
        if self.horizon < 1:
            raise ValueError("Tuning horizon must be positive")
        if not isinstance(self.penalty, LogisticPenalty):
            raise ValueError("Tuning penalty is invalid")


@dataclass(frozen=True)
class FrozenTuningRecord:
    selected_c: float
    selected_at: pd.Timestamp
    training_end: pd.Timestamp
    training_rows: int
    training_columns: int
    preprocessing_policy: CVPreprocessingPolicy

    def validate(self) -> None:
        if not np.isfinite(self.selected_c) or self.selected_c <= 0.0:
            raise ValueError("Frozen tuning C must be finite and positive")
        selected_at = pd.Timestamp(self.selected_at)
        training_end = pd.Timestamp(self.training_end)
        if pd.isna(selected_at) or pd.isna(training_end):
            raise ValueError("Frozen tuning dates must be valid")
        if training_end > selected_at:
            raise ValueError("Frozen tuning training end cannot follow selection date")
        if self.training_rows < 1 or self.training_columns < 1:
            raise ValueError("Frozen tuning design dimensions must be positive")
        if not isinstance(self.preprocessing_policy, CVPreprocessingPolicy):
            raise ValueError("Frozen tuning preprocessing policy is invalid")
        for field, timestamp in (("selected_at", selected_at), ("training_end", training_end)):
            if timestamp != timestamp.to_period("M").to_timestamp():
                raise ValueError(f"Frozen tuning {field} must be a month-start date")


class FrozenTuningRegistry:
    """Require per-key initialization on the designated first forecast date (RF-021)."""

    def __init__(self, designated_dates: dict[int, pd.Timestamp]) -> None:
        if not designated_dates:
            raise ValueError("Frozen tuning registry requires designated dates")
        self._designated_dates = {
            int(horizon): pd.Timestamp(date) for horizon, date in designated_dates.items()
        }
        if any(horizon < 1 for horizon in self._designated_dates):
            raise ValueError("Frozen tuning horizons must be positive")
        if any(
            date != date.to_period("M").to_timestamp()
            for date in self._designated_dates.values()
        ):
            raise ValueError("Designated tuning dates must be month-start dates")
        self._records: dict[TuningKey, FrozenTuningRecord] = {}

    def initialize(self, key: TuningKey, record: FrozenTuningRecord) -> None:
        key.validate()
        record.validate()
        if key.horizon not in self._designated_dates:
            raise KeyError(f"No designated tuning date for horizon {key.horizon}")
        expected = self._designated_dates[key.horizon]
        if pd.Timestamp(record.selected_at) != expected:
            raise RuntimeError(
                f"{key}: tuning must initialize on {expected.date()}, "
                f"not {pd.Timestamp(record.selected_at).date()} (RF-021)"
            )
        latest_permitted_training_end = expected - pd.DateOffset(months=key.horizon)
        if pd.Timestamp(record.training_end) > latest_permitted_training_end:
            raise RuntimeError(
                f"{key}: tuning training end must not follow forecast origin "
                f"{latest_permitted_training_end.date()} (RF-021)"
            )
        if key in self._records:
            raise RuntimeError(f"{key}: frozen tuning state is already initialized")
        self._records[key] = record

    def require(self, key: TuningKey, *, forecast_date: pd.Timestamp) -> FrozenTuningRecord:
        key.validate()
        requested_at = pd.Timestamp(forecast_date)
        if pd.isna(requested_at) or requested_at != requested_at.to_period("M").to_timestamp():
            raise ValueError("Frozen tuning forecast date must be a valid month-start date")
        if key not in self._records:
            raise RuntimeError(
                f"{key}: tuning was not initialized on its designated first date (RF-021)"
            )
        record = self._records[key]
        if record.selected_at > requested_at:
            raise RuntimeError(f"{key}: attempted to use tuning selected in the future")
        return record

    def records(self) -> dict[TuningKey, FrozenTuningRecord]:
        return dict(self._records)
