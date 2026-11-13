"""Audited recursive revised-data forecast families.

The numerical engine is policy-explicit and therefore usable for legacy
characterization and synthetic checks.  Canonical entry points remain behind
the author-decision gate in :mod:`program_v04.replication.config`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import FAMILY_REGISTRY, LAGS, REVISED_OOS_START
from .data import (
    FORECAST_COLUMNS,
    FORECAST_KEYS,
    IMPORTANCE_COLUMNS,
    IMPORTANCE_KEYS,
    METRIC_COLUMNS,
    validate_forecast_panel,
    validate_importance_panel,
    validate_metric_panel,
)
from .forecasting import (
    PredictionRowPolicy,
    build_recursive_design,
    revised_forecast_calendar,
)
from .importances import (
    UnsupportedImportancePolicy,
    build_importance_rows,
    extract_feature_importance,
)
from .metrics import build_metric_panel
from .models import (
    C_VALUES,
    CSelectionResult,
    CVPreprocessingPolicy,
    ClassImbalanceRule,
    FoldPreprocessor,
    FrozenTuningRecord,
    FrozenTuningRegistry,
    LogisticPenalty,
    ModelSet,
    TuningKey,
    fit_predict_single_model,
    get_model_templates,
    optimize_logistic_c,
)
from .predictors import (
    COUNTER_CYCLICAL_VARS,
    PREDICTOR_SPECS,
    VARIABLE_GROUPS,
    PredictorSpec,
    PredictorType,
    add_lags,
    build_predictor_data,
    modeling_frame,
)
from .preparation import RevisedData
from .protocols import MainMethodProtocol, TuningFailurePolicy
from .provenance import RunEvent
from .thresholds import ThresholdCalibration, calibrate_thresholds


MINIMUM_RECURSIVE_ROWS = 50
STANDARD_REVISED_FAMILIES = frozenset(
    {
        "main",
        "extended",
        "threshold",
        "moving_average",
        "nolag",
        "parsimonious",
        "alternative_subsets",
        "expanding_threshold",
    }
)


@dataclass(frozen=True)
class RevisedFamilyRun:
    family: str
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    importances: pd.DataFrame | None
    method_metadata: dict[str, object]
    events: tuple[RunEvent, ...]

    def validate(self) -> None:
        validate_forecast_panel(self.forecasts, surface=f"revised/{self.family} forecasts")
        validate_metric_panel(self.metrics, surface=f"revised/{self.family} metrics")
        if self.method_metadata.get("oos_start") != REVISED_OOS_START:
            raise ValueError(
                f"revised/{self.family} method metadata must declare the locked "
                f"target OOS start {REVISED_OOS_START}"
            )
        expected = set(FAMILY_REGISTRY[("revised", self.family)].predictor_models)
        observed = set(
            self.forecasts[["predictor_key", "model_name"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if observed != expected:
            raise ValueError(
                f"revised/{self.family} membership mismatch: "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        counts = self.forecasts.groupby(
            ["horizon", "predictor_key", "model_name"], sort=False
        )["forecast_date"].nunique()
        if counts.nunique() != 1:
            raise ValueError(f"revised/{self.family} has incomplete predictor/model paths")
        if self.importances is not None:
            validate_importance_panel(
                self.importances, surface=f"revised/{self.family} importances"
            )
            importance_pairs = set(
                self.importances[["predictor_key", "model_name"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            if importance_pairs != expected:
                raise ValueError(
                    f"revised/{self.family} importance membership differs from forecasts"
                )
        for event in self.events:
            event.validate()


def _family_pairs(family: str) -> tuple[tuple[str, str], ...]:
    try:
        spec = FAMILY_REGISTRY[("revised", family)]
    except KeyError as exc:
        raise KeyError(f"Unknown revised result family: {family}") from exc
    return spec.predictor_models


def _predictor_models(family: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    grouped: dict[str, list[str]] = {}
    for predictor, model in _family_pairs(family):
        grouped.setdefault(predictor, []).append(model)
    return tuple((predictor, tuple(models)) for predictor, models in grouped.items())


def _family_lags(family: str) -> tuple[int, ...]:
    if family in {"nolag", "parameter_free"}:
        return ()
    return LAGS


def _calibrate_for_spec(
    data: RevisedData,
    spec: PredictorSpec,
    *,
    calibration_end: pd.Timestamp,
    protocol: MainMethodProtocol,
) -> ThresholdCalibration | None:
    if spec.predictor_type in {PredictorType.XT_FULL, PredictorType.XT_PCA, PredictorType.YIELD}:
        return None
    calibration_source = modeling_frame(data.transformed, spec).loc[:calibration_end]
    calibration_target = data.target.loc[:calibration_end]
    return calibrate_thresholds(
        calibration_source,
        calibration_target,
        windows=(spec.window_size,),
        rolling_boundary=(protocol.rolling_boundary if spec.window_size > 1 else None),
        counter_cyclical=COUNTER_CYCLICAL_VARS,
        variable_groups=VARIABLE_GROUPS,
        recession_sample_policy=protocol.recession_sample_policy,
        rank_method=protocol.rank_method,
    )


def _shifted_source(data: RevisedData, spec: PredictorSpec, horizon: int) -> pd.DataFrame:
    base = data.yield_spread if spec.predictor_type is PredictorType.YIELD else data.transformed
    return modeling_frame(base, spec).shift(horizon)


def _build_design_panel(
    source: pd.DataFrame,
    spec: PredictorSpec,
    *,
    train_end: pd.Timestamp,
    full_end: pd.Timestamp,
    thresholds: ThresholdCalibration | None,
    protocol: MainMethodProtocol,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    predictor = build_predictor_data(
        spec,
        source.loc[:full_end],
        train_end=train_end,
        thresholds=thresholds,
        rolling_boundary=(protocol.rolling_boundary if spec.window_size > 1 else None),
        tail_inequality=protocol.tail_inequality,
    )
    return add_lags(predictor, lags).dropna()


class _FoldPurePredictor(FoldPreprocessor):
    """Rebuild every sample-dependent predictor operation inside one CV fold."""

    def __init__(
        self,
        *,
        data: RevisedData,
        shifted_source: pd.DataFrame,
        spec: PredictorSpec,
        horizon: int,
        protocol: MainMethodProtocol,
        lags: tuple[int, ...],
    ) -> None:
        self.data = data
        self.shifted_source = shifted_source
        self.spec = spec
        self.horizon = horizon
        self.protocol = protocol
        self.lags = lags
        self.train_end: pd.Timestamp | None = None
        self.thresholds: ThresholdCalibration | None = None

    def _build(self, full_end: pd.Timestamp) -> pd.DataFrame:
        if self.train_end is None:
            raise RuntimeError("Fold preprocessor was not fitted")
        return _build_design_panel(
            self.shifted_source,
            self.spec,
            train_end=self.train_end,
            full_end=full_end,
            thresholds=self.thresholds,
            protocol=self.protocol,
            lags=self.lags,
        )

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        if X_train.empty or not X_train.index.equals(y_train.index):
            raise ValueError("Fold-pure training inputs are empty or misaligned")
        self.train_end = pd.Timestamp(X_train.index.max())
        self.thresholds = _calibrate_for_spec(
            self.data,
            self.spec,
            calibration_end=self.train_end,
            protocol=self.protocol,
        )
        built = self._build(self.train_end)
        missing = X_train.index.difference(built.index)
        if len(missing):
            raise ValueError(f"Fold preprocessing lost training dates: {missing.tolist()[:5]}")
        return built.loc[X_train.index]

    def transform(self, X_validation: pd.DataFrame) -> pd.DataFrame:
        if X_validation.empty:
            raise ValueError("Fold-pure validation input is empty")
        built = self._build(pd.Timestamp(X_validation.index.max()))
        missing = X_validation.index.difference(built.index)
        if len(missing):
            raise ValueError(f"Fold preprocessing lost validation dates: {missing.tolist()[:5]}")
        return built.loc[X_validation.index]


def _penalties_for_models(models: tuple[str, ...]) -> tuple[LogisticPenalty, ...]:
    mapping = {
        "Logit_L2": LogisticPenalty.L2,
        "Logit_L1": LogisticPenalty.L1,
        "ElasticNet": LogisticPenalty.ELASTICNET,
    }
    return tuple(mapping[model] for model in models if model in mapping)


def _tune_predictor(
    *,
    data: RevisedData,
    family: str,
    spec: PredictorSpec,
    models: tuple[str, ...],
    horizon: int,
    protocol: MainMethodProtocol,
    source: pd.DataFrame,
    thresholds: ThresholdCalibration | None,
    first_target: pd.Timestamp,
    lags: tuple[int, ...],
    registry: FrozenTuningRegistry,
    c_values: Iterable[float],
) -> dict[LogisticPenalty, object]:
    penalties = _penalties_for_models(models)
    if not penalties:
        return {}
    origin = first_target - pd.DateOffset(months=horizon)
    panel = _build_design_panel(
        source,
        spec,
        train_end=origin,
        full_end=first_target,
        thresholds=thresholds,
        protocol=protocol,
        lags=lags,
    )
    point = revised_forecast_calendar(data.target.index, horizon=horizon)[0]
    design = build_recursive_design(
        panel,
        data.target,
        point,
        prediction_row_policy=PredictionRowPolicy.EXACT_TARGET,
        minimum_training_rows=MINIMUM_RECURSIVE_ROWS,
    )
    results: dict[LogisticPenalty, object] = {}
    for penalty in penalties:
        factory = None
        tuning_X = design.X_train
        if protocol.cv_preprocessing_policy is CVPreprocessingPolicy.FOLD_PURE:
            tuning_X = source.reindex(design.X_train.index)

            def factory(
                *,
                _data: RevisedData = data,
                _source: pd.DataFrame = source,
                _spec: PredictorSpec = spec,
                _horizon: int = horizon,
                _protocol: MainMethodProtocol = protocol,
                _lags: tuple[int, ...] = lags,
            ) -> _FoldPurePredictor:
                return _FoldPurePredictor(
                    data=_data,
                    shifted_source=_source,
                    spec=_spec,
                    horizon=_horizon,
                    protocol=_protocol,
                    lags=_lags,
                )

        try:
            result = optimize_logistic_c(
                tuning_X,
                design.y_train,
                penalty=penalty,
                preprocessing_policy=protocol.cv_preprocessing_policy,
                imbalance_rule=protocol.imbalance_rule,
                fit_intercept=protocol.fit_intercept,
                fold_preprocessor_factory=factory,
                c_values=c_values,
            )
        except (ValueError, RuntimeError) as exc:
            if protocol.tuning_failure_policy is not TuningFailurePolicy.LEGACY_C_ONE_FALLBACK:
                raise RuntimeError(
                    f"C selection failed for revised/{family}, {spec.name}, "
                    f"h={horizon}, penalty={penalty.value}"
                ) from exc
            result = CSelectionResult(
                selected_c=1.0,
                selected_average_brier=None,
                average_brier_by_c=(),
                rejected_candidates=((1.0, f"legacy fallback after {type(exc).__name__}: {exc}"),),
                preprocessing_policy=protocol.cv_preprocessing_policy,
                n_splits=5,
            )
        key = TuningKey(f"revised/{family}", spec.name, horizon, penalty)
        registry.initialize(
            key,
            FrozenTuningRecord(
                selected_c=result.selected_c,
                selected_at=first_target,
                training_end=origin,
                training_rows=len(design.X_train),
                training_columns=design.X_train.shape[1],
                preprocessing_policy=protocol.cv_preprocessing_policy,
            ),
        )
        results[penalty] = result
    return results


def _standard_family_method_metadata(
    *,
    family: str,
    protocol: MainMethodProtocol,
    lags: tuple[int, ...],
    tuning: dict[str, object],
    thresholds: dict[str, object],
    global_quantiles: dict[str, dict[str, float]],
    global_quantile_path: dict[str, list[dict[str, object]]],
    forecast_limit: int | None,
) -> dict[str, object]:
    """Build the standard-family method payload consumed by artifact gates."""
    if family not in STANDARD_REVISED_FAMILIES:
        raise ValueError(f"{family} is not a standard revised family")
    return {
        "protocol": protocol.as_metadata(),
        "oos_start": REVISED_OOS_START,
        "lags": list(lags),
        "tuning": tuning,
        "thresholds": thresholds,
        "threshold_mode": (
            "expanding_global" if family == "expanding_threshold" else "global"
        ),
        "global_quantiles": global_quantiles,
        "global_quantile_path": global_quantile_path,
        "threshold_recalibration_rule": (
            "recompute_global_quantile_at_each_forecast_origin"
            if family == "expanding_threshold"
            else "frozen_initial_global_quantile"
        ),
        "threshold_recalibration_inputs": (
            "unshifted_X_transformed_and_y_target_through_train_end"
            if family == "expanding_threshold"
            else "initial_unshifted_X_transformed_and_y_target"
        ),
        "forecast_limit": forecast_limit,
    }


def _standard_family(
    data: RevisedData,
    family: str,
    protocol: MainMethodProtocol,
    *,
    horizons: tuple[int, ...],
    forecast_limit: int | None,
    c_values: Iterable[float],
) -> RevisedFamilyRun:
    if family not in STANDARD_REVISED_FAMILIES:
        raise ValueError(f"{family} is not a standard revised family")
    pairs = _predictor_models(family)
    lags = _family_lags(family)
    calendars = {
        horizon: revised_forecast_calendar(data.target.index, horizon=horizon)
        for horizon in horizons
    }
    designated = {horizon: calendar[0].target_date for horizon, calendar in calendars.items()}
    tuning_registry = FrozenTuningRegistry(designated)
    forecast_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    tuning_metadata: dict[str, object] = {}
    threshold_metadata: dict[str, object] = {}
    global_quantiles: dict[str, dict[str, float]] = {}
    global_quantile_path: dict[str, list[dict[str, object]]] = {}
    event_counts: dict[tuple[str, str, str], int] = {}
    collect_importances = family == "main"

    for horizon in horizons:
        full_calendar = calendars[horizon]
        selected_calendar = (
            full_calendar
            if forecast_limit is None
            else full_calendar[-int(forecast_limit) :]
        )
        first_target = full_calendar[0].target_date
        initial_origin = full_calendar[0].origin_date
        for predictor_key, models in pairs:
            spec = PREDICTOR_SPECS[predictor_key]
            source = _shifted_source(data, spec, horizon)
            thresholds = _calibrate_for_spec(
                data,
                spec,
                calibration_end=initial_origin,
                protocol=protocol,
            )
            if thresholds is not None:
                threshold_metadata[f"h{horizon}/{predictor_key}"] = {
                    "global_by_window": thresholds.global_by_window,
                    "sector": thresholds.sector,
                    "variable": thresholds.variable,
                    "predictor_end": thresholds.predictor_end,
                    "target_end": thresholds.target_end,
                    "recession_sample_policy": thresholds.recession_sample_policy.value,
                }
                if predictor_key in {"Zt_Disagg", "Zt_Disagg_ExpandingQ"}:
                    global_quantiles[str(horizon)] = {
                        str(window): float(value)
                        for window, value in thresholds.global_by_window.items()
                    }
            tuning = _tune_predictor(
                data=data,
                family=family,
                spec=spec,
                models=models,
                horizon=horizon,
                protocol=protocol,
                source=source,
                thresholds=thresholds,
                first_target=first_target,
                lags=lags,
                registry=tuning_registry,
                c_values=c_values,
            )
            tuning_metadata[f"h{horizon}/{predictor_key}"] = {
                penalty.value: result.as_metadata() for penalty, result in tuning.items()
            }
            c_by_penalty = {
                penalty: tuning_registry.require(
                    TuningKey(f"revised/{family}", predictor_key, horizon, penalty),
                    forecast_date=first_target,
                ).selected_c
                for penalty in _penalties_for_models(models)
            }
            model_set = ModelSet.EXTENDED if family == "extended" else ModelSet.DEFAULT
            templates = get_model_templates(
                optimal_c_l2=c_by_penalty.get(LogisticPenalty.L2, 1.0),
                optimal_c_l1=c_by_penalty.get(LogisticPenalty.L1, 1.0),
                optimal_c_elasticnet=c_by_penalty.get(LogisticPenalty.ELASTICNET, 1.0),
                model_set=model_set,
                imbalance_rule=protocol.imbalance_rule,
                fit_intercept=protocol.fit_intercept,
            )
            for point in selected_calendar:
                active_thresholds = thresholds
                if family == "expanding_threshold":
                    active_thresholds = _calibrate_for_spec(
                        data,
                        spec,
                        calibration_end=point.origin_date,
                        protocol=protocol,
                    )
                    if active_thresholds is None:
                        raise RuntimeError("Expanding-threshold calibration disappeared")
                    for window, quantile in active_thresholds.global_by_window.items():
                        global_quantile_path.setdefault(str(horizon), []).append(
                            {
                                "forecast_date": point.target_date,
                                "train_end": point.origin_date,
                                "predictor_key": predictor_key,
                                "window_size": int(window),
                                "quantile": float(quantile),
                            }
                        )
                panel = _build_design_panel(
                    source,
                    spec,
                    train_end=point.origin_date,
                    full_end=point.target_date,
                    thresholds=active_thresholds,
                    protocol=protocol,
                    lags=lags,
                )
                design = build_recursive_design(
                    panel,
                    data.target,
                    point,
                    prediction_row_policy=PredictionRowPolicy.EXACT_TARGET,
                    minimum_training_rows=MINIMUM_RECURSIVE_ROWS,
                )
                for model_name in models:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        probability, fitted = fit_predict_single_model(
                            model_name,
                            templates[model_name],
                            design.X_train,
                            design.y_train,
                            design.X_predict,
                            imbalance_rule=protocol.imbalance_rule,
                        )
                    for warning in caught:
                        key = (
                            "estimator_warning",
                            warning.category.__name__,
                            str(warning.message),
                        )
                        event_counts[key] = event_counts.get(key, 0) + 1
                    forecast_rows.append(
                        {
                            "exercise": "revised",
                            "data_variant": "revised_final",
                            "forecast_family": family,
                            "horizon": horizon,
                            "forecast_date": point.target_date,
                            "origin_date": point.origin_date,
                            "target_date": point.target_date,
                            "predictor_key": predictor_key,
                            "model_name": model_name,
                            "probability": probability,
                            "actual": float(data.target.loc[point.target_date]),
                        }
                    )
                    if collect_importances:
                        importance = extract_feature_importance(
                            model_name,
                            fitted,
                            design.X_train.columns,
                            unsupported_policy=UnsupportedImportancePolicy.RAISE,
                        )
                        importance_rows.extend(
                            build_importance_rows(
                                exercise="revised",
                                data_variant="revised_final",
                                forecast_family=family,
                                horizon=horizon,
                                forecast_date=point.target_date,
                                predictor_key=predictor_key,
                                model_name=model_name,
                                result=importance,
                            )
                        )

    forecasts = (
        pd.DataFrame(forecast_rows, columns=list(FORECAST_COLUMNS))
        .sort_values(list(FORECAST_KEYS), kind="mergesort")
        .reset_index(drop=True)
    )
    metrics = build_metric_panel(forecasts)
    importances = None
    if collect_importances:
        importances = (
            pd.DataFrame(importance_rows, columns=list(IMPORTANCE_COLUMNS))
            .sort_values(list(IMPORTANCE_KEYS), kind="mergesort")
            .reset_index(drop=True)
        )
    events = tuple(
        RunEvent(category, code, count, detail)
        for (category, code, detail), count in sorted(event_counts.items())
    )
    result = RevisedFamilyRun(
        family=family,
        forecasts=forecasts,
        metrics=metrics,
        importances=importances,
        method_metadata=_standard_family_method_metadata(
            family=family,
            protocol=protocol,
            lags=lags,
            tuning=tuning_metadata,
            thresholds=threshold_metadata,
            global_quantiles=global_quantiles,
            global_quantile_path=global_quantile_path,
            forecast_limit=forecast_limit,
        ),
        events=events,
    )
    result.validate()
    return result


def _probit_family(
    data: RevisedData,
    protocol: MainMethodProtocol,
    *,
    horizons: tuple[int, ...],
    forecast_limit: int | None,
) -> RevisedFamilyRun:
    pre_oos = data.yield_spread.index < pd.Timestamp(REVISED_OOS_START)
    spread_train = data.yield_spread.loc[pre_oos, "YIELD_SPREAD"]
    target_train = data.target.loc[pre_oos]
    recession_dates = target_train[target_train == 1].index
    tau = float(
        spread_train.rank(pct=True, method=protocol.rank_method)
        .loc[recession_dates]
        .median()
    )
    cutoff = float(spread_train.quantile(tau))
    if protocol.tail_inequality.value == "strict":
        binary = (data.yield_spread["YIELD_SPREAD"] < cutoff).astype(float)
    else:
        binary = (data.yield_spread["YIELD_SPREAD"] <= cutoff).astype(float)
    sources = (
        ("Yield_Spread", data.yield_spread),
        ("Yield_Spread_Binary", binary.rename("YIELD_SPREAD_BIN").to_frame()),
    )
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        calendar = revised_forecast_calendar(data.target.index, horizon=horizon)
        if forecast_limit is not None:
            calendar = calendar[-int(forecast_limit) :]
        for predictor_key, frame in sources:
            design_panel = add_lags(frame.shift(horizon), LAGS).dropna()
            for point in calendar:
                design = build_recursive_design(
                    design_panel.loc[: point.target_date],
                    data.target,
                    point,
                    prediction_row_policy=PredictionRowPolicy.EXACT_TARGET,
                    minimum_training_rows=MINIMUM_RECURSIVE_ROWS,
                )
                X_train = sm.add_constant(design.X_train, has_constant="add")
                X_predict = sm.add_constant(
                    design.X_predict, has_constant="add"
                ).reindex(columns=X_train.columns, fill_value=0.0)
                fitted = sm.Probit(design.y_train, X_train).fit(disp=0, maxiter=100)
                if not bool(fitted.mle_retvals.get("converged", False)):
                    raise RuntimeError(
                        f"{predictor_key}, h={horizon}, {point.target_date.date()}: "
                        "probit did not converge"
                    )
                probability = float(fitted.predict(X_predict).iloc[0])
                if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise RuntimeError("Probit returned an invalid probability")
                rows.append(
                    {
                        "exercise": "revised",
                        "data_variant": "revised_final",
                        "forecast_family": "probit_spread",
                        "horizon": horizon,
                        "forecast_date": point.target_date,
                        "origin_date": point.origin_date,
                        "target_date": point.target_date,
                        "predictor_key": predictor_key,
                        "model_name": "Probit",
                        "probability": probability,
                        "actual": float(data.target.loc[point.target_date]),
                    }
                )
    forecasts = (
        pd.DataFrame(rows, columns=list(FORECAST_COLUMNS))
        .sort_values(list(FORECAST_KEYS), kind="mergesort")
        .reset_index(drop=True)
    )
    result = RevisedFamilyRun(
        family="probit_spread",
        forecasts=forecasts,
        metrics=build_metric_panel(forecasts),
        importances=None,
        method_metadata={
            "protocol": protocol.as_metadata(),
            "oos_start": REVISED_OOS_START,
            "illustration_tau": tau,
            "illustration_threshold_value": cutoff,
            "lags": list(LAGS),
            "forecast_limit": forecast_limit,
        },
        events=(),
    )
    result.validate()
    return result


def _parameter_free_family(
    data: RevisedData,
    protocol: MainMethodProtocol,
    *,
    horizons: tuple[int, ...],
    forecast_limit: int | None,
) -> RevisedFamilyRun:
    spec = PREDICTOR_SPECS["Zt_Disagg"]
    rows: list[dict[str, object]] = []
    threshold_metadata: dict[str, object] = {}
    for horizon in horizons:
        full_calendar = revised_forecast_calendar(data.target.index, horizon=horizon)
        calendar = full_calendar if forecast_limit is None else full_calendar[-int(forecast_limit) :]
        initial_origin = full_calendar[0].origin_date
        thresholds = _calibrate_for_spec(
            data,
            spec,
            calibration_end=initial_origin,
            protocol=protocol,
        )
        if thresholds is None:
            raise RuntimeError("Parameter-free family requires at-risk thresholds")
        threshold_metadata[str(horizon)] = thresholds.global_by_window
        source = _shifted_source(data, spec, horizon)
        for point in calendar:
            states = _build_design_panel(
                source,
                spec,
                train_end=point.origin_date,
                full_end=point.target_date,
                thresholds=thresholds,
                protocol=protocol,
                lags=(),
            )
            if point.target_date not in states.index:
                raise KeyError(f"Parameter-free prediction row missing {point.target_date.date()}")
            probability = float(states.loc[point.target_date].mean())
            rows.append(
                {
                    "exercise": "revised",
                    "data_variant": "revised_final",
                    "forecast_family": "parameter_free",
                    "horizon": horizon,
                    "forecast_date": point.target_date,
                    "origin_date": point.origin_date,
                    "target_date": point.target_date,
                    "predictor_key": "Zt_Share",
                    "model_name": "Parameter_Free",
                    "probability": probability,
                    "actual": float(data.target.loc[point.target_date]),
                }
            )
    forecasts = (
        pd.DataFrame(rows, columns=list(FORECAST_COLUMNS))
        .sort_values(list(FORECAST_KEYS), kind="mergesort")
        .reset_index(drop=True)
    )
    result = RevisedFamilyRun(
        family="parameter_free",
        forecasts=forecasts,
        metrics=build_metric_panel(forecasts),
        importances=None,
        method_metadata={
            "protocol": protocol.as_metadata(),
            "oos_start": REVISED_OOS_START,
            "thresholds": threshold_metadata,
            "lags": [],
            "forecast_limit": forecast_limit,
        },
        events=(),
    )
    result.validate()
    return result


def run_revised_family(
    data: RevisedData,
    family: str,
    protocol: MainMethodProtocol,
    *,
    horizons: tuple[int, ...],
    forecast_limit: int | None = None,
    c_values: Iterable[float] = C_VALUES,
) -> RevisedFamilyRun:
    """Run one complete locked family under a fully explicit protocol."""
    data.validate()
    protocol.validate()
    c_grid = tuple(float(value) for value in c_values)
    if not c_grid:
        raise ValueError("Revised C grid cannot be empty")
    if not horizons or len(set(horizons)) != len(horizons) or any(value < 1 for value in horizons):
        raise ValueError("Revised horizons must be unique positive integers")
    if forecast_limit is not None and forecast_limit < 1:
        raise ValueError("Forecast limit must be positive")
    _family_pairs(family)
    if family == "probit_spread":
        return _probit_family(
            data, protocol, horizons=horizons, forecast_limit=forecast_limit
        )
    if family == "parameter_free":
        return _parameter_free_family(
            data, protocol, horizons=horizons, forecast_limit=forecast_limit
        )
    return _standard_family(
        data,
        family,
        protocol,
        horizons=horizons,
        forecast_limit=forecast_limit,
        c_values=c_grid,
    )
