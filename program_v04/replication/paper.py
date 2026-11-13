"""Deterministic paper-facing summaries and figures from stored result panels."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HORIZONS
from .importances import (
    DegenerateImportancePolicy,
    build_episode_windows,
    summarize_sector_contributions,
    summarize_top_features,
)
from .metrics import LEGACY_LOGIT_CLIP, encompassing_test
from .predictors import VARIABLE_GROUPS
from .protocols import MainMethodProtocol
from .provenance import atomic_output_path
from .parity import validate_featured_cross_family_parity


def _metric_value(
    panel: pd.DataFrame,
    *,
    horizon: int,
    predictor_key: str,
    model_name: str,
    metric: str,
    data_variant: str = "revised_final",
) -> float:
    selected = panel[
        (panel["exercise"] == "revised")
        & (panel["data_variant"] == data_variant)
        & (panel["horizon"] == horizon)
        & (panel["predictor_key"] == predictor_key)
        & (panel["model_name"] == model_name)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Metric lookup must select one row: {predictor_key}/{model_name}, "
            f"h={horizon}, {metric}; found {len(selected)}"
        )
    value = float(selected.iloc[0][metric])
    if not np.isfinite(value):
        raise ValueError("Metric lookup returned a nonfinite value")
    return value


def _wide_metrics(
    panel: pd.DataFrame,
    rows: tuple[tuple[str, str, str], ...],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for label, predictor, model in rows:
        record: dict[str, object] = {"label": label}
        for horizon in HORIZONS:
            for metric in ("pr_auc", "brier"):
                record[f"{metric}_h{horizon}"] = _metric_value(
                    panel,
                    horizon=horizon,
                    predictor_key=predictor,
                    model_name=model,
                    metric=metric,
                )
        records.append(record)
    return pd.DataFrame(records)


def build_main_table02(
    main_metrics: pd.DataFrame,
    extended_metrics: pd.DataFrame,
    probit_metrics: pd.DataFrame,
    *,
    main_forecasts: pd.DataFrame,
    extended_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    validate_featured_cross_family_parity(
        main_forecasts,
        extended_forecasts,
        exercise="revised",
        data_variant="revised_final",
    )
    base = _wide_metrics(
        main_metrics,
        (
            ("Proposed ($Z_t$, Logit-L2)", "Zt_Disagg", "Logit_L2"),
            ("Benchmark ($X_t$, Logit-L2)", "Xt_Full", "Logit_L2"),
            ("Benchmark (PCA of $X_t$, Logit-L2)", "Xt_PCA", "Logit_L2"),
            ("Benchmark ($X_t$, XGBoost)", "Xt_Full", "XGBoost"),
        ),
    )
    forest = _wide_metrics(
        extended_metrics,
        (("Benchmark ($X_t$, Random Forest)", "Xt_Full", "RandomForest"),),
    )
    probit = _wide_metrics(
        probit_metrics,
        (("Benchmark (Term Spread Probit)", "Yield_Spread", "Probit"),),
    )
    return pd.concat((base, forest, probit), ignore_index=True)


def build_main_table03(
    main_metrics: pd.DataFrame,
    parameter_free_metrics: pd.DataFrame,
) -> pd.DataFrame:
    base = _wide_metrics(
        main_metrics,
        (
            ("Disaggregated ($Z_t$), Logit-L2", "Zt_Disagg", "Logit_L2"),
            ("Disaggregated ($Z_t$), XGBoost", "Zt_Disagg", "XGBoost"),
            ("Simple Average, Logit-L2", "Zt_Avg", "Logit_L2"),
            ("Simple Average, XGBoost", "Zt_Avg", "XGBoost"),
            ("PCA($Z_t$), Logit-L2", "Zt_PCA", "Logit_L2"),
            ("PCA($Z_t$), XGBoost", "Zt_PCA", "XGBoost"),
        ),
    )
    raw = _wide_metrics(
        parameter_free_metrics,
        (("Parameter-Free (share of $Z=1$)", "Zt_Share", "Parameter_Free"),),
    )
    return pd.concat((base, raw), ignore_index=True)


def _forecast_series(
    panel: pd.DataFrame,
    *,
    horizon: int,
    predictor: str,
    model: str,
    family: str = "main",
) -> pd.DataFrame:
    selected = panel[
        (panel["exercise"] == "revised")
        & (panel["data_variant"] == "revised_final")
        & (panel["forecast_family"] == family)
        & (panel["horizon"] == horizon)
        & (panel["predictor_key"] == predictor)
        & (panel["model_name"] == model)
    ].set_index("forecast_date")
    if selected.empty or not selected.index.is_unique:
        raise ValueError(f"Forecast selection is empty or duplicate: {predictor}/{model}, h={horizon}")
    return selected


def build_main_table04(
    forecasts: pd.DataFrame,
    *,
    extended_forecasts: pd.DataFrame,
    protocol: MainMethodProtocol,
) -> pd.DataFrame:
    validate_featured_cross_family_parity(
        forecasts,
        extended_forecasts,
        exercise="revised",
        data_variant="revised_final",
    )
    pairs = (
        ("$Z_t$ (Logit-L2)", "$X_t$ (Logit-L2)", "Zt_Disagg", "Logit_L2", "Xt_Full", "Logit_L2"),
        ("$Z_t$ (Logit-L2)", "$X_t$ (XGBoost)", "Zt_Disagg", "Logit_L2", "Xt_Full", "XGBoost"),
        ("$Z_t$ (Logit-L2)", "$X_t$ (Random Forest)", "Zt_Disagg", "Logit_L2", "Xt_Full", "RandomForest"),
        ("PCA($Z_t$) (Logit-L2)", "PCA($X_t$) (Logit-L2)", "Zt_PCA", "Logit_L2", "Xt_PCA", "Logit_L2"),
    )
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for label_a, label_b, predictor_a, model_a, predictor_b, model_b in pairs:
            left = _forecast_series(
                forecasts, horizon=horizon, predictor=predictor_a, model=model_a
            )
            right = _forecast_series(
                extended_forecasts if model_b == "RandomForest" else forecasts,
                horizon=horizon, predictor=predictor_b, model=model_b,
                family="extended" if model_b == "RandomForest" else "main",
            )
            if not left.index.equals(right.index) or not left["actual"].equals(right["actual"]):
                raise ValueError("Encompassing forecast paths are not exactly aligned")
            result = encompassing_test(
                left["actual"],
                left["probability"],
                right["probability"],
                horizon=horizon,
                covariance=protocol.encompassing_covariance,
                clip_epsilon=LEGACY_LOGIT_CLIP,
            )
            rows.append(
                {
                    "horizon": horizon,
                    "model_a": label_a,
                    "model_b": label_b,
                    "coeff_a": result.coeff_a,
                    "pval_a": result.pval_a,
                    "coeff_b": result.coeff_b,
                    "pval_b": result.pval_b,
                }
            )
    return pd.DataFrame(rows)


def build_main_table05(main_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, predictor, model in (
        ("Proposed", "Zt_Disagg", "Logit_L2"),
        ("Benchmark", "Xt_Full", "XGBoost"),
    ):
        row = {
            "model_block": label,
            "brier_recession": _metric_value(
                main_metrics, horizon=3, predictor_key=predictor, model_name=model, metric="brier_recession"
            ),
            "brier_expansion": _metric_value(
                main_metrics, horizon=3, predictor_key=predictor, model_name=model, metric="brier_expansion"
            ),
            "brier_full_sample": _metric_value(
                main_metrics, horizon=3, predictor_key=predictor, model_name=model, metric="brier"
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_main_table06(importances: pd.DataFrame) -> pd.DataFrame:
    kwargs = {
        "exercise": "revised",
        "data_variant": "revised_final",
        "forecast_family": "main",
        "horizon": 3,
        "top_n": 10,
        "degeneracy_policy": DegenerateImportancePolicy.EXCLUDE_RECORDED,
    }
    zt = summarize_top_features(
        importances,
        predictor_key="Zt_Disagg",
        model_name="Logit_L2",
        **kwargs,
    )
    xt = summarize_top_features(
        importances,
        predictor_key="Xt_Full",
        model_name="XGBoost",
        **kwargs,
    )
    if len(zt) != 10 or len(xt) != 10:
        raise ValueError("Table 6 requires exactly ten features from each framework")
    return pd.DataFrame(
        {
            "zt_variable": zt["display_name"],
            "zt_importance_share_pct": zt["importance_share_pct"],
            "xt_variable": xt["display_name"],
            "xt_importance_share_pct": xt["importance_share_pct"],
        }
    )


def build_main_table07(
    importances: pd.DataFrame,
    *,
    protocol: MainMethodProtocol,
) -> pd.DataFrame:
    windows = build_episode_windows(endpoint_policy=protocol.episode_endpoint_policy)
    kwargs = {
        "exercise": "revised",
        "data_variant": "revised_final",
        "forecast_family": "main",
        "horizon": 3,
        "windows": windows,
        "degeneracy_policy": DegenerateImportancePolicy.EXCLUDE_RECORDED,
        "incomplete_policy": protocol.incomplete_episode_policy,
    }
    frames: list[pd.DataFrame] = []
    for label, predictor, model in (
        ("At-Risk ($Z_t$) + Logit-L2", "Zt_Disagg", "Logit_L2"),
        ("Continuous ($X_t$) + XGBoost", "Xt_Full", "XGBoost"),
    ):
        summary = summarize_sector_contributions(
            importances,
            predictor_key=predictor,
            model_name=model,
            **kwargs,
        )
        pivot = (
            summary.pivot(index="sector", columns="episode_label", values="share_pct")
            .reindex(VARIABLE_GROUPS)
            .reset_index()
        )
        pivot.columns.name = None
        pivot.insert(0, "model_block", label)
        frames.append(pivot)
    return pd.concat(frames, ignore_index=True)


def render_generic_latex(frame: pd.DataFrame, *, provenance_comment: str) -> str:
    """Render a deterministic verification fragment at manuscript precision."""
    columns = "l" * max(len(frame.columns), 1)
    lines = [
        f"% {provenance_comment}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(str(column) for column in frame.columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if pd.isna(value):
                values.append("---")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _matplotlib() -> object:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "at_risk_replication_matplotlib")
    )
    os.environ.setdefault(
        "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "at_risk_replication_cache")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "savefig.dpi": 300,
        }
    )
    return plt


def _shade(axis: object, target: pd.Series, *, alpha: float = 0.2) -> None:
    active = False
    start: pd.Timestamp | None = None
    for date, value in target.items():
        if value == 1 and not active:
            start, active = pd.Timestamp(date), True
        elif value == 0 and active and start is not None:
            axis.axvspan(start, date, color="#b2182b", alpha=alpha, zorder=0)  # type: ignore[attr-defined]
            active = False
    if active and start is not None:
        axis.axvspan(start, target.index[-1], color="#b2182b", alpha=alpha, zorder=0)  # type: ignore[attr-defined]


def render_main_figure02(
    forecasts: pd.DataFrame,
    target: pd.Series,
    output_paths: tuple[Path, Path, Path],
) -> tuple[Path, Path, Path]:
    import matplotlib.dates as mdates

    plt = _matplotlib()
    anchor = (mdates.date2num(pd.Timestamp("1993-01-01")), 0.98)
    for horizon, output in zip(HORIZONS, output_paths, strict=True):
        zt = _forecast_series(
            forecasts, horizon=horizon, predictor="Zt_Disagg", model="Logit_L2"
        )
        xt = _forecast_series(
            forecasts, horizon=horizon, predictor="Xt_Full", model="XGBoost"
        )
        if not zt.index.equals(xt.index):
            raise ValueError("Figure 2 probability paths are not aligned")
        figure, axis = plt.subplots(figsize=(11, 3.6))
        _shade(axis, target)
        axis.plot(zt.index, zt["probability"], color="#2166ac", label="$Z_t$, Logit-L2")
        axis.plot(xt.index, xt["probability"], color="#1b7837", label="$X_t$, XGBoost")
        axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0, label="50% threshold")
        axis.set_ylim(0, 1)
        axis.set_title(f"Out-of-Sample Predicted Recession Probabilities (h={horizon})")
        axis.set_ylabel("Probability")
        axis.set_xlabel("Date")
        axis.set_xlim(zt.index.min(), zt.index.max())
        axis.legend(loc="upper left", bbox_to_anchor=anchor, bbox_transform=axis.transData, borderaxespad=0.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_output_path(output) as temporary:
            figure.savefig(temporary, format="pdf", bbox_inches="tight")
        plt.close(figure)
    return output_paths


def render_main_figure03(
    forecasts: pd.DataFrame,
    target: pd.Series,
    output_path: Path,
) -> Path:
    plt = _matplotlib()
    zt = _forecast_series(forecasts, horizon=3, predictor="Zt_Disagg", model="Logit_L2")
    xt = _forecast_series(forecasts, horizon=3, predictor="Xt_Full", model="XGBoost")
    if not zt.index.equals(xt.index):
        raise ValueError("Figure 3 probability paths are not aligned")
    difference = zt["probability"] - xt["probability"]
    figure, axis = plt.subplots(figsize=(11, 4.5))
    _shade(axis, target)
    axis.axhline(0, color="#666666", linewidth=1.0)
    axis.fill_between(difference.index, 0, difference, where=difference >= 0, color="#2166ac", alpha=0.45)
    axis.fill_between(difference.index, 0, difference, where=difference < 0, color="#1b7837", alpha=0.35)
    axis.plot(difference.index, difference, color="#333333", linewidth=0.9)
    axis.set_title("Forecast Disagreement: $Z_t$ Logit-L2 minus $X_t$ XGBoost (h=3)")
    axis.set_ylabel("Probability Difference")
    axis.set_xlabel("Date")
    axis.set_xlim(difference.index.min(), difference.index.max())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(output_path) as temporary:
        figure.savefig(temporary, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output_path
