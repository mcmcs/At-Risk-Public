"""Audited forecast metrics, references, and encompassing inference."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning as StatsmodelsConvergenceWarning,
)
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

from .data import METRIC_COLUMNS, METRIC_KEYS, validate_forecast_panel
from .validation import require_binary, require_probability


LEGACY_LOGIT_CLIP = 1e-6
LEGACY_MURPHY_BINS = 10


class MetricName(StrEnum):
    PR_AUC = "pr_auc"
    ROC_AUC = "roc_auc"
    BRIER = "brier"


class BrierReferenceRule(StrEnum):
    """Candidate interpretation of the no-information Brier reference (RF-016)."""

    LEGACY_HALF = "legacy_constant_half"
    SAMPLE_PREVALENCE = "sample_prevalence"


class EncompassingCovariance(StrEnum):
    """Candidate covariance rule for the forecast-encompassing probit (RF-007)."""

    LEGACY_IID = "legacy_iid"
    HAC_HORIZON = "hac_horizon"


def _validated_metric_arrays(
    actuals: pd.Series,
    probabilities: pd.Series,
    *,
    surface: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(actuals, pd.Series) or not isinstance(probabilities, pd.Series):
        raise TypeError(f"{surface}: actuals and probabilities must be pandas Series")
    if not actuals.index.equals(probabilities.index):
        raise ValueError(f"{surface}: actual and probability indexes differ")
    if actuals.empty or not actuals.index.is_unique:
        raise ValueError(f"{surface}: evaluation index must be nonempty and unique")
    require_binary(actuals, surface=surface, field="actual")
    require_probability(probabilities, surface=surface)
    if actuals.nunique() != 2:
        raise ValueError(f"{surface}: evaluation outcome must contain both classes")
    return (
        actuals.to_numpy(dtype=float),
        probabilities.to_numpy(dtype=float),
    )


def compute_metric(
    actuals: pd.Series,
    probabilities: pd.Series,
    *,
    metric: MetricName,
) -> float:
    """Compute one complete-sample metric with no silent row dropping."""
    if not isinstance(metric, MetricName):
        raise ValueError(f"Unsupported metric: {metric}")
    y_true, y_prob = _validated_metric_arrays(
        actuals,
        probabilities,
        surface=f"{metric.value} metric",
    )
    if metric is MetricName.PR_AUC:
        return float(average_precision_score(y_true, y_prob))
    if metric is MetricName.ROC_AUC:
        return float(roc_auc_score(y_true, y_prob))
    return float(brier_score_loss(y_true, y_prob))


def brier_score_decomposition(
    actuals: pd.Series,
    probabilities: pd.Series,
    *,
    n_bins: int = LEGACY_MURPHY_BINS,
) -> dict[str, float]:
    """Characterize v03's equal-width grouped Murphy components."""
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("Murphy decomposition requires at least two bins")
    y_true, y_prob = _validated_metric_arrays(
        actuals,
        probabilities,
        surface="Brier decomposition",
    )
    base_rate = float(y_true.mean())
    uncertainty = base_rate * (1.0 - base_rate)
    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    bin_index = np.digitize(y_prob, bins)
    reliability = 0.0
    resolution = 0.0
    total = len(y_true)
    for bucket in range(1, n_bins + 1):
        mask = bin_index == bucket
        if not np.any(mask):
            continue
        weight = float(mask.sum()) / total
        actual_mean = float(y_true[mask].mean())
        probability_mean = float(y_prob[mask].mean())
        reliability += weight * (actual_mean - probability_mean) ** 2
        resolution += weight * (actual_mean - base_rate) ** 2
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def pr_auc_no_skill_reference(actuals: pd.Series) -> float:
    """Return the positive-class prevalence used by the paper's PR reference."""
    require_binary(actuals, surface="PR AUC reference", field="actual")
    if actuals.empty:
        raise ValueError("PR AUC reference requires observations")
    return float(actuals.mean())


@dataclass(frozen=True)
class BrierReference:
    rule: BrierReferenceRule
    constant_probability: float
    brier: float
    observations: int
    positive_observations: int


def brier_no_information_reference(
    actuals: pd.Series,
    *,
    rule: BrierReferenceRule,
) -> BrierReference:
    """Evaluate an explicit RF-016 constant-probability reference."""
    if not isinstance(rule, BrierReferenceRule):
        raise ValueError(f"Unsupported Brier reference rule: {rule}")
    require_binary(actuals, surface="Brier reference", field="actual")
    if actuals.empty:
        raise ValueError("Brier reference requires observations")
    probability = 0.5 if rule is BrierReferenceRule.LEGACY_HALF else float(actuals.mean())
    values = actuals.to_numpy(dtype=float)
    return BrierReference(
        rule=rule,
        constant_probability=probability,
        brier=float(np.mean((values - probability) ** 2)),
        observations=len(values),
        positive_observations=int(values.sum()),
    )


def build_metric_panel(
    forecast_panel: pd.DataFrame,
    *,
    n_bins: int = LEGACY_MURPHY_BINS,
) -> pd.DataFrame:
    """Summarize a validated forecast family to deterministic metric rows."""
    validate_forecast_panel(forecast_panel, surface="metric source forecast panel")
    rows: list[dict[str, object]] = []
    for keys, group in forecast_panel.groupby(list(METRIC_KEYS), sort=False):
        key_map = dict(zip(METRIC_KEYS, keys, strict=True))
        actuals = group.set_index("forecast_date")["actual"]
        probabilities = group.set_index("forecast_date")["probability"]
        y_true, y_prob = _validated_metric_arrays(
            actuals,
            probabilities,
            surface=f"metric group {key_map}",
        )
        decomposition = brier_score_decomposition(
            actuals,
            probabilities,
            n_bins=n_bins,
        )
        recession = y_true == 1.0
        expansion = y_true == 0.0
        rows.append(
            {
                **key_map,
                "num_forecasts": len(y_true),
                "num_positive_actuals": int(recession.sum()),
                "num_negative_actuals": int(expansion.sum()),
                "pr_auc": float(average_precision_score(y_true, y_prob)),
                "roc_auc": float(roc_auc_score(y_true, y_prob)),
                "brier": float(brier_score_loss(y_true, y_prob)),
                "brier_recession": float(np.mean((y_true[recession] - y_prob[recession]) ** 2)),
                "brier_expansion": float(np.mean((y_true[expansion] - y_prob[expansion]) ** 2)),
                **decomposition,
            }
        )
    return (
        pd.DataFrame(rows, columns=list(METRIC_COLUMNS))
        .sort_values(list(METRIC_KEYS), kind="mergesort")
        .reset_index(drop=True)
    )


def logit_transform(
    probabilities: pd.Series,
    *,
    clip_epsilon: float,
) -> pd.Series:
    """Clip a validated probability path and return index-preserving log odds."""
    if not np.isfinite(clip_epsilon) or not 0.0 < clip_epsilon < 0.5:
        raise ValueError("Logit clipping epsilon must be finite and in (0, 0.5)")
    require_probability(probabilities, surface="logit transform")
    clipped = probabilities.astype(float).clip(clip_epsilon, 1.0 - clip_epsilon)
    return np.log(clipped / (1.0 - clipped))


def encompassing_design(
    actuals: pd.Series,
    model_a: pd.Series,
    model_b: pd.Series,
    *,
    clip_epsilon: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build the exact common-date probit design without an inner-join fallback."""
    _validated_metric_arrays(actuals, model_a, surface="encompassing model A")
    _validated_metric_arrays(actuals, model_b, surface="encompassing model B")
    if not model_a.index.equals(model_b.index):
        raise ValueError("Encompassing probability indexes differ")
    design = pd.DataFrame(
        {
            "const": 1.0,
            "logit_a": logit_transform(model_a, clip_epsilon=clip_epsilon),
            "logit_b": logit_transform(model_b, clip_epsilon=clip_epsilon),
        },
        index=actuals.index,
    )
    return actuals.astype(float).copy(), design


@dataclass(frozen=True)
class EncompassingResult:
    coeff_a: float
    pval_a: float
    standard_error_a: float
    coeff_b: float
    pval_b: float
    standard_error_b: float
    observations: int
    covariance: EncompassingCovariance
    maxlags: int | None
    clip_epsilon: float

    def as_record(self) -> dict[str, object]:
        return {
            "coeff_a": self.coeff_a,
            "pval_a": self.pval_a,
            "standard_error_a": self.standard_error_a,
            "coeff_b": self.coeff_b,
            "pval_b": self.pval_b,
            "standard_error_b": self.standard_error_b,
            "observations": self.observations,
            "covariance": self.covariance.value,
            "maxlags": self.maxlags,
            "clip_epsilon": self.clip_epsilon,
        }


def encompassing_test(
    actuals: pd.Series,
    model_a: pd.Series,
    model_b: pd.Series,
    *,
    horizon: int,
    covariance: EncompassingCovariance,
    clip_epsilon: float,
) -> EncompassingResult:
    """Fit the paper's probit design under an explicit RF-007 covariance rule."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Encompassing horizon must be a positive integer")
    if not isinstance(covariance, EncompassingCovariance):
        raise ValueError(f"Unsupported encompassing covariance: {covariance}")
    outcome, design = encompassing_design(
        actuals,
        model_a,
        model_b,
        clip_epsilon=clip_epsilon,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=StatsmodelsConvergenceWarning)
            warnings.filterwarnings("error", category=PerfectSeparationWarning)
            if covariance is EncompassingCovariance.LEGACY_IID:
                fitted = sm.Probit(outcome, design).fit(disp=0)
                maxlags = None
            else:
                fitted = sm.Probit(outcome, design).fit(
                    disp=0,
                    cov_type="HAC",
                    cov_kwds={"maxlags": horizon},
                )
                maxlags = horizon
    except (
        np.linalg.LinAlgError,
        PerfectSeparationError,
        PerfectSeparationWarning,
        StatsmodelsConvergenceWarning,
    ) as exc:
        raise RuntimeError("Forecast-encompassing probit failed") from exc
    if not bool(fitted.mle_retvals.get("converged", False)):
        raise RuntimeError("Forecast-encompassing probit did not converge")
    values = np.array(
        [
            fitted.params["logit_a"],
            fitted.pvalues["logit_a"],
            fitted.bse["logit_a"],
            fitted.params["logit_b"],
            fitted.pvalues["logit_b"],
            fitted.bse["logit_b"],
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Forecast-encompassing probit returned nonfinite inference")
    return EncompassingResult(
        coeff_a=float(values[0]),
        pval_a=float(values[1]),
        standard_error_a=float(values[2]),
        coeff_b=float(values[3]),
        pval_b=float(values[4]),
        standard_error_b=float(values[5]),
        observations=len(outcome),
        covariance=covariance,
        maxlags=maxlags,
        clip_epsilon=float(clip_epsilon),
    )
