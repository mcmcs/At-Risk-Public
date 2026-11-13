"""Pure, training-only transformations for revised and real-time predictors.

Disputed behaviors are explicit arguments.  In particular, transformation
code 7 and incomplete moving-average windows have no silent canonical choice.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


N_PCA_FACTORS = 8


class Code7Rule(StrEnum):
    """Audited candidate definitions for FRED-MD transformation code 7."""

    LEGACY_TWELVE_MONTH_PERCENT = "legacy_twelve_month_percent"
    PUBLISHED_DIFFERENCE_OF_PERCENT_CHANGE = "published_difference_of_percent_change"


class RollingBoundary(StrEnum):
    """Audited candidate treatments of an incomplete initial rolling window."""

    LEGACY_PARTIAL = "legacy_partial"
    COMPLETE = "complete"


class TailInequality(StrEnum):
    """Audited candidate threshold inequalities."""

    STRICT = "strict"
    WEAK = "weak"


def transform_series(
    series: pd.Series,
    code: int,
    *,
    code7_rule: Code7Rule | None = None,
) -> pd.Series:
    """Apply one documented FRED-MD transformation exactly once.

    Codes 1--6 are undisputed.  Code 7 requires an explicit audited rule so a
    caller cannot silently inherit the legacy mismatch (RF-004).
    """
    if code == 1:
        return series.copy()
    if code == 2:
        return series.diff()
    if code == 3:
        return series.diff().diff()
    if code == 4:
        return np.log(series)
    if code == 5:
        return np.log(series).diff()
    if code == 6:
        return np.log(series).diff().diff()
    if code != 7:
        raise ValueError(f"Unknown FRED-MD transformation code: {code}")
    if code7_rule is None:
        raise ValueError("FRED-MD transformation code 7 requires an explicit Code7Rule (RF-004)")
    if code7_rule is Code7Rule.LEGACY_TWELVE_MONTH_PERCENT:
        return series.pct_change(periods=12, fill_method=None) * 100.0
    if code7_rule is Code7Rule.PUBLISHED_DIFFERENCE_OF_PERCENT_CHANGE:
        one_period_change = series / series.shift(1) - 1.0
        return one_period_change.diff()
    raise ValueError(f"Unsupported code-7 rule: {code7_rule}")


def transform_panel(
    raw: pd.DataFrame,
    transformation_codes: pd.Series,
    *,
    code7_rule: Code7Rule | None = None,
) -> pd.DataFrame:
    """Transform every declared column once, preserving input order and index."""
    if raw.columns.has_duplicates:
        raise ValueError("Raw FRED-MD panel has duplicate column names")
    missing = [column for column in raw.columns if column not in transformation_codes.index]
    extras = [column for column in transformation_codes.index if column not in raw.columns]
    if missing or extras:
        raise ValueError(
            f"Transformation-code registry mismatch: missing={missing}, unexpected={extras}"
        )
    transformed = {
        column: transform_series(
            raw[column],
            int(transformation_codes.loc[column]),
            code7_rule=code7_rule,
        )
        for column in raw.columns
    }
    return pd.DataFrame(transformed, index=raw.index)


def _validate_training_transform_inputs(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> None:
    if train_df.empty:
        raise ValueError("Training slice is empty")
    if full_df.empty:
        raise ValueError("Full transform slice is empty")
    if train_df.columns.has_duplicates or full_df.columns.has_duplicates:
        raise ValueError("Training-only transform received duplicate columns")
    if list(train_df.columns) != list(full_df.columns):
        raise ValueError("Training and full transform slices must have identical ordered columns")
    if not train_df.index.is_unique or not full_df.index.is_unique:
        raise ValueError("Training and full transform indices must be unique")


def _drop_training_all_nan(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_nan = train_df.columns[train_df.isna().all()].tolist()
    return train_df.drop(columns=all_nan), full_df.drop(columns=all_nan)


def impute_training_only(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Fit KNN imputation on the training slice and score the supplied slice."""
    _validate_training_transform_inputs(train_df, full_df)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    train_use, full_use = _drop_training_all_nan(train_df, full_df)
    if train_use.shape[1] == 0:
        raise ValueError("Every predictor is missing throughout the training slice")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputer.fit(train_use)
    values = imputer.transform(full_use)
    return pd.DataFrame(values, index=full_use.index, columns=full_use.columns)


def scale_training_only(train_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Fit standardization on the training slice and score the supplied slice."""
    _validate_training_transform_inputs(train_df, full_df)
    if train_df.isna().any().any() or full_df.isna().any().any():
        raise ValueError("Scaling requires already-imputed finite inputs")
    scaler = StandardScaler()
    scaler.fit(train_df)
    return pd.DataFrame(
        scaler.transform(full_df),
        index=full_df.index,
        columns=full_df.columns,
    )


def orient_pca_components(
    factors: pd.DataFrame,
    loadings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orient each component so its first maximum-absolute loading is positive."""
    if list(factors.columns) != list(loadings.columns):
        raise ValueError("PCA factor and loading component names differ")
    oriented_factors = factors.copy()
    oriented_loadings = loadings.copy()
    for component in oriented_loadings.columns:
        loading = oriented_loadings[component]
        if loading.empty:
            continue
        anchor = loading.abs().idxmax()
        if float(loading.loc[anchor]) < 0.0:
            oriented_loadings[component] = -oriented_loadings[component]
            oriented_factors[component] = -oriented_factors[component]
    return oriented_factors, oriented_loadings


def generate_pca_factors(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame | None = None,
    *,
    n_factors: int = N_PCA_FACTORS,
    standardize: bool = True,
    n_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit deterministic, training-only PCA and score the supplied full slice."""
    full_df = train_df if full_df is None else full_df
    _validate_training_transform_inputs(train_df, full_df)
    if n_factors < 1:
        raise ValueError("n_factors must be positive")
    imputed = impute_training_only(train_df, full_df, n_neighbors=n_neighbors)
    train_imputed = imputed.loc[train_df.index]
    if standardize:
        final = scale_training_only(train_imputed, imputed)
        train_final = final.loc[train_df.index]
    else:
        final = imputed
        train_final = train_imputed

    nonconstant = train_final.var()[lambda values: values >= 1e-10].index.tolist()
    if not nonconstant:
        raise ValueError("PCA training slice has no nonconstant predictor")
    train_final = train_final[nonconstant]
    final = final[nonconstant]
    component_count = min(n_factors, train_final.shape[0], train_final.shape[1])
    if component_count < 1:
        raise ValueError("PCA component count is zero")

    estimator = PCA(n_components=component_count, random_state=random_state)
    estimator.fit(train_final)
    columns = [f"PCA_Factor_{number}" for number in range(1, component_count + 1)]
    factors = pd.DataFrame(estimator.transform(final), index=final.index, columns=columns)
    loadings = pd.DataFrame(estimator.components_.T, index=final.columns, columns=columns)
    return orient_pca_components(factors, loadings)


def rolling_signal(
    series: pd.Series,
    window_size: int,
    *,
    boundary: RollingBoundary,
) -> pd.Series:
    """Return an audited moving average under an explicit boundary rule."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    min_periods = 1 if boundary is RollingBoundary.LEGACY_PARTIAL else window_size
    return series.rolling(window=window_size, min_periods=min_periods).mean()


def generate_at_risk_states(
    full_imputed: pd.DataFrame,
    threshold: float,
    *,
    counter_cyclical: set[str] | frozenset[str],
    variable_to_sector: dict[str, str],
    threshold_type: str = "global",
    sector_thresholds: dict[str, float] | None = None,
    variable_thresholds: dict[str, float] | None = None,
    window_size: int = 1,
    rolling_boundary: RollingBoundary | None = None,
    tail_inequality: TailInequality,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate binary at-risk states using training-reference quantiles only."""
    if full_imputed.empty or reference_df.empty:
        raise ValueError("At-risk state construction requires nonempty full and reference frames")
    if list(full_imputed.columns) != list(reference_df.columns):
        raise ValueError("At-risk full/reference frames must have identical ordered columns")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Global threshold quantile must lie in [0, 1]")
    if threshold_type not in {"global", "sector", "variable"}:
        raise ValueError(f"Unknown threshold type: {threshold_type}")
    if window_size > 1 and rolling_boundary is None:
        raise ValueError("Moving-average states require an explicit RollingBoundary (RF-011)")
    boundary = rolling_boundary or RollingBoundary.COMPLETE

    states: dict[str, pd.Series] = {}
    for variable in full_imputed.columns:
        full_signal = rolling_signal(full_imputed[variable], window_size, boundary=boundary)
        reference_signal = rolling_signal(reference_df[variable], window_size, boundary=boundary)
        quantile = threshold
        if threshold_type == "variable":
            if variable_thresholds is None or variable not in variable_thresholds:
                raise KeyError(f"Missing variable threshold for {variable}")
            quantile = variable_thresholds[variable]
        elif threshold_type == "sector":
            if variable not in variable_to_sector:
                raise KeyError(f"Missing sector mapping for {variable}")
            sector = variable_to_sector[variable]
            if sector_thresholds is None or sector not in sector_thresholds:
                raise KeyError(f"Missing sector threshold for {sector}")
            quantile = sector_thresholds[sector]
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Threshold quantile for {variable} lies outside [0, 1]")

        is_counter = variable in counter_cyclical
        cutoff = reference_signal.quantile(1.0 - quantile if is_counter else quantile)
        if pd.isna(cutoff):
            raise ValueError(f"Threshold calibration produced NaN for {variable}")
        if tail_inequality is TailInequality.STRICT:
            indicator = full_signal.gt(cutoff) if is_counter else full_signal.lt(cutoff)
        elif tail_inequality is TailInequality.WEAK:
            indicator = full_signal.ge(cutoff) if is_counter else full_signal.le(cutoff)
        else:
            raise ValueError(f"Unsupported tail inequality: {tail_inequality}")
        state = indicator.astype(float)
        state[full_signal.isna()] = np.nan
        states[variable] = state
    return pd.DataFrame(states, index=full_imputed.index)
