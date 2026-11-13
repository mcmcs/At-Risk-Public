"""Audited recursive real-time forecast families.

The engine keeps the four disputed real-time choices explicit: the vintage
calendar, prediction-row selection, calibration sharing, and predictor
composition.  It can therefore characterize the frozen v03 panels without
making those legacy conventions canonical.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from .config import FAMILY_REGISTRY, HORIZONS, LAGS
from .data import (
    FORECAST_COLUMNS,
    FORECAST_KEYS,
    InputRecord,
    load_fred_md_snapshot,
    manifest_by_id,
    validate_forecast_panel,
    validate_metric_panel,
    verify_input,
)
from .forecasting import (
    ForecastPoint,
    PredictorCompositionLock,
    PredictorCompositionPolicy,
    RealtimeCalibrationPolicy,
    align_panel_for_forecast,
    apply_realtime_composition,
    build_recursive_design,
    calibration_panels_by_variant,
    lock_common_predictors,
    realtime_forecast_calendar,
    tuning_scope,
)
from .metrics import build_metric_panel
from .models import (
    C_VALUES,
    CSelectionResult,
    CVPreprocessingPolicy,
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
    ALL_GROUPED_VARIABLES,
    COUNTER_CYCLICAL_VARS,
    PREDICTOR_SPECS,
    VARIABLE_GROUPS,
    VARS_TO_REMOVE,
    PredictorSpec,
    PredictorType,
    add_lags,
    build_predictor_data,
    modeling_frame,
)
from .preparation import RevisedData
from .protocols import RealtimeMethodProtocol, TuningFailurePolicy
from .provenance import RunEvent
from .thresholds import ThresholdCalibration, calibrate_thresholds
from .transforms import (
    generate_at_risk_states,
    generate_pca_factors,
    impute_training_only,
    transform_panel,
)


MINIMUM_RECURSIVE_ROWS = 50
REALTIME_VARIANTS = ("revised_common_window", "realtime_vintage")
REALTIME_FAMILIES = ("realtime_main", "realtime_extended")


@dataclass(frozen=True)
class RealtimeFamilyRun:
    family: str
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    method_metadata: dict[str, object]
    events: tuple[RunEvent, ...]

    def validate(self) -> None:
        if self.family not in REALTIME_FAMILIES:
            raise ValueError(f"Unknown real-time family: {self.family}")
        validate_forecast_panel(
            self.forecasts, surface=f"realtime/{self.family} forecasts"
        )
        validate_metric_panel(self.metrics, surface=f"realtime/{self.family} metrics")
        expected_pairs = set(
            FAMILY_REGISTRY[("realtime", self.family)].predictor_models
        )
        observed_variants = set(self.forecasts["data_variant"])
        if observed_variants != set(REALTIME_VARIANTS):
            raise ValueError(
                f"realtime/{self.family} variants mismatch: {sorted(observed_variants)}"
            )
        for variant, group in self.forecasts.groupby("data_variant", sort=False):
            observed_pairs = set(
                group[["predictor_key", "model_name"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            if observed_pairs != expected_pairs:
                raise ValueError(
                    f"realtime/{self.family}/{variant} membership mismatch: "
                    f"missing={sorted(expected_pairs - observed_pairs)}, "
                    f"extra={sorted(observed_pairs - expected_pairs)}"
                )
        counts = self.forecasts.groupby(
            ["data_variant", "horizon", "predictor_key", "model_name"],
            sort=False,
        )["forecast_date"].nunique()
        by_variant_horizon = counts.groupby(level=[0, 1]).nunique()
        if (by_variant_horizon != 1).any():
            raise ValueError(f"realtime/{self.family} has incomplete model paths")
        by_horizon = counts.groupby(level=1).nunique()
        if (by_horizon != 1).any():
            raise ValueError(
                f"realtime/{self.family} revised/vintage path lengths differ"
            )
        expected_counts = self.method_metadata.get("expected_forecasts_by_horizon")
        if not isinstance(expected_counts, dict):
            raise ValueError(f"realtime/{self.family} has no expected path-count metadata")
        observed_counts = {
            str(int(horizon)): int(group.iloc[0])
            for horizon, group in counts.groupby(level=1)
        }
        if observed_counts != expected_counts:
            raise ValueError(
                f"realtime/{self.family} forecast counts differ from the locked calendar: "
                f"observed={observed_counts}, expected={expected_counts}"
            )
        for event in self.events:
            event.validate()


def vintage_manifest_records(
    records: Mapping[str, InputRecord] | None = None,
) -> tuple[InputRecord, ...]:
    """Return the exact declared 1999-08--2024-12 vintage registry."""
    registry = manifest_by_id() if records is None else dict(records)
    vintages = tuple(
        sorted(
            (record for record in registry.values() if record.role == "fred_md_vintage"),
            key=lambda record: record.vintage,
        )
    )
    if len(vintages) != 305:
        raise ValueError(f"Expected 305 declared FRED-MD vintages, found {len(vintages)}")
    expected = tuple(
        month.strftime("%Y-%m")
        for month in pd.date_range("1999-08-01", "2024-12-01", freq="MS")
    )
    observed = tuple(record.vintage for record in vintages)
    if observed != expected:
        raise ValueError("Declared vintage registry is not the gapless 1999-08--2024-12 calendar")
    if any(record.input_id != f"fred_md_vintage_{record.vintage}" for record in vintages):
        raise ValueError("Vintage input identifiers do not match their declared months")
    return vintages


def resolve_vintage_inputs(
    *,
    mode: Literal["public", "development"] = "public",
    records: Mapping[str, InputRecord] | None = None,
) -> dict[str, Path]:
    """Resolve, checksum, and shape-check all declared historical vintages."""
    return {
        record.vintage: verify_input(record, mode=mode, verify_shape=True)
        for record in vintage_manifest_records(records)
    }


def load_vintage_transformed(
    path: Path,
    *,
    protocol: RealtimeMethodProtocol,
) -> pd.DataFrame:
    """Load one declared vintage with the protocol's explicit code-7 rule."""
    protocol.validate()
    raw, tcodes = load_fred_md_snapshot(path)
    transformed = transform_panel(
        raw,
        tcodes,
        code7_rule=protocol.main.code7_rule,
    )
    retained = [column for column in transformed if column not in VARS_TO_REMOVE]
    if not retained:
        raise ValueError(f"Vintage has no retained predictors: {path}")
    result = transformed.loc[:, retained].copy()
    if result.columns.has_duplicates:
        raise ValueError(f"Vintage has duplicate transformed predictors: {path}")
    return result


def _family_predictor_models(
    family: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if family not in REALTIME_FAMILIES:
        raise ValueError(f"Unknown real-time family: {family}")
    grouped: dict[str, list[str]] = {}
    for predictor, model in FAMILY_REGISTRY[("realtime", family)].predictor_models:
        grouped.setdefault(predictor, []).append(model)
    return tuple((predictor, tuple(models)) for predictor, models in grouped.items())


def _calibrate_spec(
    panel: pd.DataFrame,
    target: pd.Series,
    spec: PredictorSpec,
    *,
    calibration_end: pd.Timestamp,
    protocol: RealtimeMethodProtocol,
) -> ThresholdCalibration | None:
    if spec.predictor_type in {PredictorType.XT_FULL, PredictorType.XT_PCA}:
        return None
    source = modeling_frame(panel, spec).loc[:calibration_end]
    return calibrate_thresholds(
        source,
        target.loc[:calibration_end],
        windows=(spec.window_size,),
        rolling_boundary=(
            protocol.main.rolling_boundary if spec.window_size > 1 else None
        ),
        counter_cyclical=COUNTER_CYCLICAL_VARS,
        variable_groups=VARIABLE_GROUPS,
        recession_sample_policy=protocol.main.recession_sample_policy,
        rank_method=protocol.main.rank_method,
    )


def _application_thresholds(
    calibration: ThresholdCalibration | None,
    columns: tuple[str, ...],
) -> ThresholdCalibration | None:
    """Adapt a frozen scalar threshold to the columns present in one vintage."""
    if calibration is None:
        return None
    if not columns:
        raise ValueError("Threshold application requires at least one predictor")
    if tuple(calibration.calibration_columns) == columns:
        return calibration
    applied = ThresholdCalibration(
        global_by_window=dict(calibration.global_by_window),
        sector={
            sector: value
            for sector, value in calibration.sector.items()
            if any(column in VARIABLE_GROUPS.get(sector, ()) for column in columns)
        },
        variable={
            column: calibration.variable[column]
            for column in columns
            if column in calibration.variable
        },
        recession_sample_policy=calibration.recession_sample_policy,
        calibration_columns=columns,
        predictor_end=calibration.predictor_end,
        target_end=calibration.target_end,
    )
    applied.validate()
    return applied


def _build_predictor_panel(
    aligned: pd.DataFrame,
    spec: PredictorSpec,
    *,
    train_end: pd.Timestamp,
    thresholds: ThresholdCalibration | None,
    protocol: RealtimeMethodProtocol,
) -> pd.DataFrame:
    if (
        protocol.composition_policy
        is PredictorCompositionPolicy.LEGACY_VARIANT_SPECIFIC
        and spec.subset is None
        and spec.predictor_type
        in {PredictorType.ZT_DISAGG, PredictorType.ZT_AVG, PredictorType.ZT_PCA}
    ):
        # Frozen v03 fitted the KNN imputer on every column present in a
        # vintage, then retained only the 122 grouped variables when it formed
        # states.  Some vintages contain replacement/extra series, so selecting
        # the grouped variables before imputation changes the historical panel.
        # Keep this ordering only inside the explicit legacy-composition path.
        train = aligned.loc[:train_end]
        imputed = impute_training_only(train, aligned)
        grouped = [column for column in ALL_GROUPED_VARIABLES if column in imputed]
        if not grouped:
            raise ValueError(f"{spec.name}: vintage has no grouped predictors")
        selected = imputed.loc[:, grouped]
        applied = _application_thresholds(thresholds, tuple(grouped))
        if applied is None:
            raise ValueError(f"{spec.name}: at-risk predictor lost its threshold")
        states = generate_at_risk_states(
            selected,
            applied.global_by_window[spec.window_size],
            counter_cyclical=COUNTER_CYCLICAL_VARS,
            variable_to_sector={
                variable: sector
                for sector, variables in VARIABLE_GROUPS.items()
                for variable in variables
            },
            threshold_type=spec.threshold_type.value,
            sector_thresholds=applied.sector,
            variable_thresholds=applied.variable,
            window_size=spec.window_size,
            rolling_boundary=(
                protocol.main.rolling_boundary if spec.window_size > 1 else None
            ),
            tail_inequality=protocol.main.tail_inequality,
            reference_df=selected.loc[:train_end],
        )
        if spec.predictor_type is PredictorType.ZT_DISAGG:
            predictor = states
        elif spec.predictor_type is PredictorType.ZT_AVG:
            predictor = states.mean(axis=1, skipna=False).to_frame(name="Zt_Avg")
        else:
            predictor, _loadings = generate_pca_factors(
                states.loc[:train_end], states, standardize=False
            )
        return add_lags(predictor, LAGS).dropna()

    selected = modeling_frame(aligned, spec)
    applied = _application_thresholds(thresholds, tuple(selected.columns))
    predictor = build_predictor_data(
        spec,
        selected,
        train_end=train_end,
        thresholds=applied,
        rolling_boundary=(
            protocol.main.rolling_boundary if spec.window_size > 1 else None
        ),
        tail_inequality=protocol.main.tail_inequality,
    )
    return add_lags(predictor, LAGS).dropna()


class _RealtimeFoldPredictor(FoldPreprocessor):
    """Rebuild sample-dependent real-time preprocessing inside one CV fold."""

    def __init__(
        self,
        *,
        aligned: pd.DataFrame,
        calibration_panel: pd.DataFrame,
        target: pd.Series,
        spec: PredictorSpec,
        protocol: RealtimeMethodProtocol,
    ) -> None:
        self.aligned = aligned
        self.calibration_panel = calibration_panel
        self.target = target
        self.spec = spec
        self.protocol = protocol
        self.train_end: pd.Timestamp | None = None
        self.thresholds: ThresholdCalibration | None = None

    def _build(self, full_end: pd.Timestamp) -> pd.DataFrame:
        if self.train_end is None:
            raise RuntimeError("Fold preprocessor was not fitted")
        return _build_predictor_panel(
            self.aligned.loc[:full_end],
            self.spec,
            train_end=self.train_end,
            thresholds=self.thresholds,
            protocol=self.protocol,
        )

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        if X_train.empty or not X_train.index.equals(y_train.index):
            raise ValueError("Fold-pure real-time training inputs are empty or misaligned")
        self.train_end = pd.Timestamp(X_train.index.max())
        self.thresholds = _calibrate_spec(
            self.calibration_panel,
            self.target,
            self.spec,
            calibration_end=self.train_end,
            protocol=self.protocol,
        )
        built = self._build(self.train_end)
        missing = X_train.index.difference(built.index)
        if len(missing):
            raise ValueError(
                f"Fold-pure real-time preprocessing lost training dates: {missing.tolist()[:5]}"
            )
        return built.loc[X_train.index]

    def transform(self, X_validation: pd.DataFrame) -> pd.DataFrame:
        if X_validation.empty:
            raise ValueError("Fold-pure real-time validation input is empty")
        built = self._build(pd.Timestamp(X_validation.index.max()))
        missing = X_validation.index.difference(built.index)
        if len(missing):
            raise ValueError(
                f"Fold-pure real-time preprocessing lost validation dates: {missing.tolist()[:5]}"
            )
        return built.loc[X_validation.index]


def _composition_lock(
    revised: pd.DataFrame,
    vintage_paths: Mapping[str, Path],
    *,
    policy: PredictorCompositionPolicy,
) -> PredictorCompositionLock | None:
    if policy is PredictorCompositionPolicy.LEGACY_VARIANT_SPECIFIC:
        return None
    signatures: dict[str, pd.DataFrame] = {"revised": revised.iloc[[0]]}
    for vintage, path in vintage_paths.items():
        header = pd.read_csv(path, nrows=1).columns[1:]
        retained = [column for column in header if column not in VARS_TO_REMOVE]
        signatures[f"vintage_{vintage}"] = pd.DataFrame(
            np.zeros((1, len(retained))), columns=retained
        )
    return lock_common_predictors(
        signatures,
        registry_order=tuple(ALL_GROUPED_VARIABLES),
    )


def _threshold_metadata(
    calibration: ThresholdCalibration | None,
) -> dict[str, object] | None:
    if calibration is None:
        return None
    return {
        "global_by_window": calibration.global_by_window,
        "sector": calibration.sector,
        "variable": calibration.variable,
        "calibration_columns": list(calibration.calibration_columns),
        "predictor_end": calibration.predictor_end,
        "target_end": calibration.target_end,
        "recession_sample_policy": calibration.recession_sample_policy.value,
    }


def _initialize_tuning(
    *,
    family: str,
    variant: str,
    spec: PredictorSpec,
    models: tuple[str, ...],
    horizon: int,
    point: ForecastPoint,
    design_panel: pd.DataFrame,
    aligned_panel: pd.DataFrame,
    calibration_panel: pd.DataFrame,
    target: pd.Series,
    protocol: RealtimeMethodProtocol,
    registry: FrozenTuningRegistry,
    c_values: Iterable[float],
    event_counts: dict[tuple[str, str, str], int],
) -> CSelectionResult | None:
    if "Logit_L2" not in models:
        return None
    scope = tuning_scope(variant, calibration_policy=protocol.calibration_policy)
    key = TuningKey(f"realtime/{family}/{scope}", spec.name, horizon, LogisticPenalty.L2)
    try:
        existing = registry.require(key, forecast_date=point.target_date)
    except RuntimeError:
        existing = None
    if existing is not None:
        return None
    design = build_recursive_design(
        design_panel,
        target,
        point,
        prediction_row_policy=protocol.prediction_row_policy,
        minimum_training_rows=MINIMUM_RECURSIVE_ROWS,
    )
    factory = None
    tuning_X = design.X_train
    if protocol.main.cv_preprocessing_policy is CVPreprocessingPolicy.FOLD_PURE:
        tuning_X = aligned_panel.reindex(design.X_train.index)

        def factory(
            *,
            _aligned: pd.DataFrame = aligned_panel,
            _calibration: pd.DataFrame = calibration_panel,
            _target: pd.Series = target,
            _spec: PredictorSpec = spec,
            _protocol: RealtimeMethodProtocol = protocol,
        ) -> _RealtimeFoldPredictor:
            return _RealtimeFoldPredictor(
                aligned=_aligned,
                calibration_panel=_calibration,
                target=_target,
                spec=_spec,
                protocol=_protocol,
            )

    try:
        result = optimize_logistic_c(
            tuning_X,
            design.y_train,
            penalty=LogisticPenalty.L2,
            preprocessing_policy=protocol.main.cv_preprocessing_policy,
            imbalance_rule=protocol.main.imbalance_rule,
            fit_intercept=protocol.main.fit_intercept,
            fold_preprocessor_factory=factory,
            c_values=c_values,
        )
    except (ValueError, RuntimeError) as exc:
        if protocol.main.tuning_failure_policy is not TuningFailurePolicy.LEGACY_C_ONE_FALLBACK:
            raise RuntimeError(
                f"C selection failed for realtime/{family}/{variant}, "
                f"{spec.name}, h={horizon}"
            ) from exc
        reason = f"legacy fallback after {type(exc).__name__}: {exc}"
        result = CSelectionResult(
            selected_c=1.0,
            selected_average_brier=None,
            average_brier_by_c=(),
            rejected_candidates=((1.0, reason),),
            preprocessing_policy=protocol.main.cv_preprocessing_policy,
            n_splits=5,
        )
        event_key = ("tuning_fallback", "legacy_c_one", reason)
        event_counts[event_key] = event_counts.get(event_key, 0) + 1
    registry.initialize(
        key,
        FrozenTuningRecord(
            selected_c=result.selected_c,
            selected_at=point.target_date,
            training_end=point.origin_date,
            training_rows=len(design.X_train),
            training_columns=design.X_train.shape[1],
            preprocessing_policy=protocol.main.cv_preprocessing_policy,
        ),
    )
    return result


def run_realtime_family(
    revised: RevisedData,
    family: str,
    protocol: RealtimeMethodProtocol,
    *,
    vintage_paths: Mapping[str, Path],
    horizons: tuple[int, ...] = HORIZONS,
    forecast_limit: int | None = None,
    c_values: Iterable[float] = C_VALUES,
) -> RealtimeFamilyRun:
    """Run one complete paired revised/vintage real-time family."""
    revised.validate()
    protocol.validate()
    if forecast_limit is not None and int(forecast_limit) < 1:
        raise ValueError("forecast_limit must be positive when supplied")
    predictor_models = _family_predictor_models(family)
    if set(vintage_paths) != {
        month.strftime("%Y-%m")
        for month in pd.date_range("1999-08-01", "2024-12-01", freq="MS")
    }:
        raise ValueError("Real-time runner requires all 305 declared vintage paths")
    calendars = {
        horizon: realtime_forecast_calendar(revised.target.index, horizon=horizon)
        for horizon in horizons
    }
    designated = {
        horizon: calendar[0].target_date for horizon, calendar in calendars.items()
    }
    tuning_registry = FrozenTuningRegistry(designated)
    composition_lock = _composition_lock(
        revised.transformed,
        vintage_paths,
        policy=protocol.composition_policy,
    )
    vintage_cache: dict[str, pd.DataFrame] = {}

    def vintage(key: str) -> pd.DataFrame:
        if key not in vintage_cache:
            vintage_cache[key] = load_vintage_transformed(
                vintage_paths[key], protocol=protocol
            )
        return vintage_cache[key]

    forecast_rows: list[dict[str, object]] = []
    tuning_metadata: dict[str, object] = {}
    threshold_metadata: dict[str, object] = {}
    alignment_counts: dict[tuple[str, int, int, int], int] = {}
    event_counts: dict[tuple[str, str, str], int] = {}

    for horizon, full_calendar in calendars.items():
        selected_calendar = (
            full_calendar
            if forecast_limit is None
            else full_calendar[-int(forecast_limit):]
        )
        initial_point = full_calendar[0]
        if initial_point.vintage_key is None:
            raise RuntimeError("Real-time initial point lost its vintage key")
        initial_composed = apply_realtime_composition(
            revised.transformed,
            vintage(initial_point.vintage_key),
            composition_policy=protocol.composition_policy,
            composition_lock=composition_lock,
        )
        calibration_panels = calibration_panels_by_variant(
            initial_composed["revised_common_window"],
            initial_composed["realtime_vintage"],
            calibration_policy=protocol.calibration_policy,
            calibration_end=initial_point.origin_date,
        )
        thresholds_by_key: dict[tuple[str, str], ThresholdCalibration | None] = {}

        for variant in REALTIME_VARIANTS:
            base = initial_composed[variant]
            alignment_point = (
                ForecastPoint(
                    initial_point.target_date,
                    initial_point.origin_date,
                    horizon,
                )
                if variant == "revised_common_window"
                else initial_point
            )
            aligned = align_panel_for_forecast(
                base,
                alignment_point,
                calendar_policy=protocol.calendar_policy,
            )
            for predictor_key, models in predictor_models:
                spec = PREDICTOR_SPECS[predictor_key]
                thresholds = _calibrate_spec(
                    calibration_panels[variant],
                    revised.target,
                    spec,
                    calibration_end=initial_point.origin_date,
                    protocol=protocol,
                )
                thresholds_by_key[(variant, predictor_key)] = thresholds
                threshold_metadata[f"h{horizon}/{variant}/{predictor_key}"] = (
                    _threshold_metadata(thresholds)
                )
                design_panel = _build_predictor_panel(
                    aligned.frame,
                    spec,
                    train_end=initial_point.origin_date,
                    thresholds=thresholds,
                    protocol=protocol,
                )
                tuning = _initialize_tuning(
                    family=family,
                    variant=variant,
                    spec=spec,
                    models=models,
                    horizon=horizon,
                    point=initial_point,
                    design_panel=design_panel,
                    aligned_panel=aligned.frame,
                    calibration_panel=calibration_panels[variant],
                    target=revised.target,
                    protocol=protocol,
                    registry=tuning_registry,
                    c_values=c_values,
                    event_counts=event_counts,
                )
                if tuning is not None:
                    scope = tuning_scope(
                        variant, calibration_policy=protocol.calibration_policy
                    )
                    tuning_metadata[f"h{horizon}/{scope}/{predictor_key}"] = (
                        tuning.as_metadata()
                    )

        for point in selected_calendar:
            if point.vintage_key is None:
                raise RuntimeError("Real-time forecast point lost its vintage key")
            composed = apply_realtime_composition(
                revised.transformed,
                vintage(point.vintage_key),
                composition_policy=protocol.composition_policy,
                composition_lock=composition_lock,
            )
            for variant in REALTIME_VARIANTS:
                alignment_point = (
                    ForecastPoint(point.target_date, point.origin_date, horizon)
                    if variant == "revised_common_window"
                    else point
                )
                aligned = align_panel_for_forecast(
                    composed[variant],
                    alignment_point,
                    calendar_policy=protocol.calendar_policy,
                )
                for predictor_key, models in predictor_models:
                    spec = PREDICTOR_SPECS[predictor_key]
                    panel = _build_predictor_panel(
                        aligned.frame,
                        spec,
                        train_end=point.origin_date,
                        thresholds=thresholds_by_key[(variant, predictor_key)],
                        protocol=protocol,
                    )
                    design = build_recursive_design(
                        panel,
                        revised.target,
                        point,
                        prediction_row_policy=protocol.prediction_row_policy,
                        minimum_training_rows=MINIMUM_RECURSIVE_ROWS,
                    )
                    row_gap = int(
                        (point.target_date.to_period("M") -
                         design.prediction_row_date.to_period("M")).n
                    )
                    alignment_key = (variant, horizon, aligned.shift_months, row_gap)
                    alignment_counts[alignment_key] = alignment_counts.get(alignment_key, 0) + 1
                    scope = tuning_scope(
                        variant, calibration_policy=protocol.calibration_policy
                    )
                    registry_record = tuning_registry.require(
                        TuningKey(
                            f"realtime/{family}/{scope}",
                            predictor_key,
                            horizon,
                            LogisticPenalty.L2,
                        ),
                        forecast_date=point.target_date,
                    )
                    templates = get_model_templates(
                        optimal_c_l2=registry_record.selected_c,
                        optimal_c_l1=1.0,
                        optimal_c_elasticnet=1.0,
                        model_set=ModelSet.EXTENDED,
                        imbalance_rule=protocol.main.imbalance_rule,
                        fit_intercept=protocol.main.fit_intercept,
                    )
                    for model_name in models:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            probability, _fitted = fit_predict_single_model(
                                model_name,
                                templates[model_name],
                                design.X_train,
                                design.y_train,
                                design.X_predict,
                                imbalance_rule=protocol.main.imbalance_rule,
                            )
                        for warning in caught:
                            event_key = (
                                "estimator_warning",
                                warning.category.__name__,
                                str(warning.message),
                            )
                            event_counts[event_key] = event_counts.get(event_key, 0) + 1
                        forecast_rows.append(
                            {
                                "exercise": "realtime",
                                "data_variant": variant,
                                "forecast_family": family,
                                "horizon": horizon,
                                "forecast_date": point.target_date,
                                "origin_date": point.origin_date,
                                "target_date": point.target_date,
                                "predictor_key": predictor_key,
                                "model_name": model_name,
                                "probability": probability,
                                "actual": float(revised.target.loc[point.target_date]),
                            }
                        )

    forecasts = (
        pd.DataFrame(forecast_rows, columns=list(FORECAST_COLUMNS))
        .sort_values(list(FORECAST_KEYS), kind="mergesort")
        .reset_index(drop=True)
    )
    metrics = build_metric_panel(forecasts)
    events = tuple(
        RunEvent(category, code, count, detail)
        for (category, code, detail), count in sorted(event_counts.items())
    )
    result = RealtimeFamilyRun(
        family=family,
        forecasts=forecasts,
        metrics=metrics,
        method_metadata={
            "protocol": protocol.as_metadata(),
            "lags": list(LAGS),
            "vintage_registry": {
                "count": len(vintage_paths),
                "first": min(vintage_paths),
                "last": max(vintage_paths),
            },
            "composition_lock": (
                None if composition_lock is None else list(composition_lock.columns)
            ),
            "tuning": tuning_metadata,
            "thresholds": threshold_metadata,
            "expected_forecasts_by_horizon": {
                str(horizon): len(
                    calendar
                    if forecast_limit is None
                    else calendar[-int(forecast_limit):]
                )
                for horizon, calendar in calendars.items()
            },
            "alignment_counts": [
                {
                    "data_variant": variant,
                    "horizon": horizon,
                    "shift_months": shift_months,
                    "prediction_row_gap_months": row_gap,
                    "predictor_date_applications": count,
                }
                for (variant, horizon, shift_months, row_gap), count
                in sorted(alignment_counts.items())
            ],
            "forecast_limit": forecast_limit,
        },
        events=events,
    )
    result.validate()
    return result


def derive_realtime_main(extended: RealtimeFamilyRun) -> RealtimeFamilyRun:
    """Derive the registered headline subset from one extended computation."""
    extended.validate()
    if extended.family != "realtime_extended":
        raise ValueError("Only realtime_extended can be the source of realtime_main")
    allowed = set(FAMILY_REGISTRY[("realtime", "realtime_main")].predictor_models)
    mask = pd.Series(
        [
            (predictor, model) in allowed
            for predictor, model in extended.forecasts[
                ["predictor_key", "model_name"]
            ].itertuples(index=False, name=None)
        ],
        index=extended.forecasts.index,
    )
    forecasts = extended.forecasts.loc[mask].copy()
    forecasts["forecast_family"] = "realtime_main"
    forecasts = forecasts.sort_values(list(FORECAST_KEYS), kind="mergesort").reset_index(drop=True)
    result = RealtimeFamilyRun(
        family="realtime_main",
        forecasts=forecasts,
        metrics=build_metric_panel(forecasts),
        method_metadata={
            **extended.method_metadata,
            "derived_from_family": "realtime_extended",
            "subset_rule": "registered realtime_main predictor/model membership",
        },
        events=extended.events,
    )
    result.validate()
    return result
