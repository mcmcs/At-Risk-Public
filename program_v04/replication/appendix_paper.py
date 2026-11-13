"""Deterministic appendix tables and Figure 4 from stored result panels."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HORIZONS, PACKAGE_ROOT
from .provenance import atomic_output_path
from .robustness import (
    FEATURED_BENCHMARK_IDS,
    recession_month_spans,
)


def _metric(
    panel: pd.DataFrame,
    *,
    horizon: int,
    predictor: str,
    model: str,
    metric: str,
    data_variant: str = "revised_final",
) -> float:
    selected = panel[
        (panel["data_variant"] == data_variant)
        & (panel["horizon"] == horizon)
        & (panel["predictor_key"] == predictor)
        & (panel["model_name"] == model)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Appendix metric lookup must select one row: {predictor}/{model}, "
            f"h={horizon}, variant={data_variant}; found {len(selected)}"
        )
    value = float(selected.iloc[0][metric])
    if not np.isfinite(value):
        raise ValueError("Appendix metric lookup returned a nonfinite value")
    return value


def _wide(
    panel: pd.DataFrame,
    rows: tuple[tuple[str, str, str], ...],
    *,
    data_variant: str = "revised_final",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for label, predictor, model in rows:
        record: dict[str, object] = {"label": label}
        for horizon in HORIZONS:
            record[f"pr_auc_h{horizon}"] = _metric(
                panel,
                horizon=horizon,
                predictor=predictor,
                model=model,
                metric="pr_auc",
                data_variant=data_variant,
            )
            record[f"brier_h{horizon}"] = _metric(
                panel,
                horizon=horizon,
                predictor=predictor,
                model=model,
                metric="brier",
                data_variant=data_variant,
            )
        records.append(record)
    return pd.DataFrame(records)


def build_appendix_table08(main_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, predictor, model in (
        ("$Z_t$, Logit-L2", "Zt_Disagg", "Logit_L2"),
        ("$X_t$, Logit-L2", "Xt_Full", "Logit_L2"),
        ("PCA($X_t$), Logit-L2", "Xt_PCA", "Logit_L2"),
        ("$X_t$, XGBoost", "Xt_Full", "XGBoost"),
    ):
        row: dict[str, object] = {"label": label}
        for horizon in HORIZONS:
            row[f"roc_auc_h{horizon}"] = _metric(
                main_metrics,
                horizon=horizon,
                predictor=predictor,
                model=model,
                metric="roc_auc",
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_appendix_table09(
    main_metrics: pd.DataFrame, threshold_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for section, specs in (
        (
            "Disaggregated $Z_t$ + Logit-L2",
            (
                ("Global ($\\tau_g$)", main_metrics, "Zt_Disagg"),
                ("Sector ($\\tau_s$)", threshold_metrics, "Zt_Disagg_SecQ"),
                ("Variable ($\\tau_v$)", threshold_metrics, "Zt_Disagg_VarQ"),
            ),
        ),
        (
            "PCA($Z_t$) + Logit-L2",
            (
                ("Global ($\\tau_g$)", main_metrics, "Zt_PCA"),
                ("Sector ($\\tau_s$)", threshold_metrics, "Zt_PCA_SecQ"),
                ("Variable ($\\tau_v$)", threshold_metrics, "Zt_PCA_VarQ"),
            ),
        ),
    ):
        for label, panel, predictor in specs:
            record = _wide(panel, ((label, predictor, "Logit_L2"),)).iloc[0].to_dict()
            rows.append({"section": section, **record})
    return pd.DataFrame(rows)


def build_appendix_table10(
    main_metrics: pd.DataFrame, moving_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for section, base in (
        ("Disaggregated $Z_t$ + Logit-L2", "Zt_Disagg"),
        ("PCA($Z_t$) + Logit-L2", "Zt_PCA"),
    ):
        specs = [("$h_g=1$ (main spec.)", main_metrics, base)]
        specs.extend(
            (f"$h_g={window}$", moving_metrics, f"{base}_MA{window}")
            for window in (3, 6, 12)
        )
        for label, panel, predictor in specs:
            record = _wide(panel, ((label, predictor, "Logit_L2"),)).iloc[0].to_dict()
            rows.append({"section": section, **record})
    return pd.DataFrame(rows)


def build_appendix_table11(extended_metrics: pd.DataFrame) -> pd.DataFrame:
    models = ("Logit_L2", "Logit_L1", "ElasticNet", "XGBoost", "RandomForest", "MLP")
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for label, predictor in (("$Z_t$", "Zt_Disagg"), ("$X_t$", "Xt_Full")):
            row: dict[str, object] = {"horizon": horizon, "predictor_block": label}
            for model in models:
                row[model] = _metric(
                    extended_metrics,
                    horizon=horizon,
                    predictor=predictor,
                    model=model,
                    metric="pr_auc",
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_appendix_table12(
    main_metrics: pd.DataFrame, nolag_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows = (
        ("$Z_t$ (Logit-L2, with lags)", main_metrics, "Zt_Disagg"),
        ("$Z_t$ (Logit-L2, no lags)", nolag_metrics, "Zt_Disagg"),
        ("$X_t$ (Logit-L2, with lags)", main_metrics, "Xt_Full"),
        ("$X_t$ (Logit-L2, no lags)", nolag_metrics, "Xt_Full"),
    )
    return pd.concat(
        [_wide(panel, ((label, predictor, "Logit_L2"),)) for label, panel, predictor in rows],
        ignore_index=True,
    )


def build_appendix_table13(
    main_metrics: pd.DataFrame, parsimonious_metrics: pd.DataFrame
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for section, panel, specs in (
        (
            "Full Feature Space",
            main_metrics,
            (
                ("Continuous ($X_t$) + XGBoost", "Xt_Full", "XGBoost"),
                ('"At-Risk" ($Z_t$) + Logit-L2', "Zt_Disagg", "Logit_L2"),
            ),
        ),
        (
            "Parsimonious Models (10 Core Indicators)",
            parsimonious_metrics,
            (
                ("Continuous ($X_t$) + XGBoost", "Xt_Parsimonious", "XGBoost"),
                ('"At-Risk" ($Z_t$) + Logit-L2', "Zt_Parsimonious", "Logit_L2"),
            ),
        ),
    ):
        frame = _wide(panel, specs)
        frame.insert(0, "section", section)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_appendix_table14(
    main_metrics: pd.DataFrame, subset_metrics: pd.DataFrame
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for section, panel, specs in (
        (
            "Full Feature Space",
            main_metrics,
            (
                ("Continuous ($X_t$) + XGBoost", "Xt_Full", "XGBoost"),
                ('"At-Risk" ($Z_t$) + Logit-L2', "Zt_Disagg", "Logit_L2"),
            ),
        ),
        (
            "$S_t$ (8 Spreads)",
            subset_metrics,
            (
                ("Continuous ($X_t$) + XGBoost", "Xt_Spreads", "XGBoost"),
                ('"At-Risk" ($Z_t$) + Logit-L2', "Zt_Spreads", "Logit_L2"),
            ),
        ),
        (
            "$R_t$ (8 Real Economy Indicators)",
            subset_metrics,
            (
                ("Continuous ($X_t$) + XGBoost", "Xt_RealEcon", "XGBoost"),
                ('"At-Risk" ($Z_t$) + Logit-L2', "Zt_RealEcon", "Logit_L2"),
            ),
        ),
    ):
        frame = _wide(panel, specs)
        frame.insert(0, "section", section)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_appendix_table15(realtime_metrics: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("$Z_t$, Logit-L2", "Zt_Disagg", "Logit_L2"),
        ("PCA($Z_t$), Logit-L2", "Zt_PCA", "Logit_L2"),
        ("PCA($Z_t$), XGBoost", "Zt_PCA", "XGBoost"),
        ("$X_t$, Logit-L2", "Xt_Full", "Logit_L2"),
        ("$X_t$, XGBoost", "Xt_Full", "XGBoost"),
        ("$X_t$, Random Forest", "Xt_Full", "RandomForest"),
        ("PCA($X_t$), Logit-L2", "Xt_PCA", "Logit_L2"),
        ("PCA($X_t$), XGBoost", "Xt_PCA", "XGBoost"),
        ("PCA($X_t$), Random Forest", "Xt_PCA", "RandomForest"),
    )
    frames: list[pd.DataFrame] = []
    for section, variant in (
        ("Revised Data", "revised_common_window"),
        ("Real-Time Vintage Data", "realtime_vintage"),
    ):
        frame = _wide(realtime_metrics, specs, data_variant=variant)
        frame.insert(0, "section", section)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_derived_table(frame: pd.DataFrame, *, surface: str) -> pd.DataFrame:
    if frame.empty or frame.columns.has_duplicates or frame.isna().all(axis=1).any():
        raise ValueError(f"{surface} is empty or malformed")
    return frame.copy()


def render_cumulative_brier_figure(data: pd.DataFrame, output: Path) -> Path:
    """Render the h=3 paths; the manuscript supplies the caption and note."""
    os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_ROOT / "output" / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    required = {
        "horizon",
        "benchmark_id",
        "benchmark_label",
        "target_date",
        "actual",
        "cumulative_brier_advantage",
    }
    if data.empty or not required.issubset(data.columns):
        raise ValueError("Figure 4 source panel is empty or incomplete")
    horizon_data = data[data["horizon"] == 3]
    recession = (
        horizon_data[["target_date", "actual"]]
        .drop_duplicates()
        .sort_values("target_date")
    )
    if recession.empty:
        raise ValueError("Figure 4 is missing h=3")
    colors = {
        "xt_logit_l2": "#789CC4",
        "xt_xgboost": "#7CAB8F",
        "xt_random_forest": "#D3957F",
    }
    with plt.rc_context({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "grid.color": "#DADDE0",
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "axes.grid": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }):
        figure, axis = plt.subplots(figsize=(11, 5.7))
        axis.grid(True, axis="y")
        for start, end in recession_month_spans(recession["target_date"], recession["actual"]):
            axis.axvspan(start, end, color="#D9DDE2", alpha=0.55, linewidth=0, zorder=0)
        axis.axhline(0, color="#666666", linewidth=1.0, zorder=1)
        axis.axvline(pd.Timestamp("2020-01-01"), color="#777777", linestyle="--", linewidth=1.1, zorder=1)
        for benchmark_id in FEATURED_BENCHMARK_IDS:
            series = horizon_data[horizon_data["benchmark_id"] == benchmark_id].sort_values("target_date")
            if series.empty:
                raise ValueError(f"Figure 4 is missing {benchmark_id}, h=3")
            axis.plot(
                series["target_date"],
                series["cumulative_brier_advantage"],
                color=colors[benchmark_id],
                linewidth=3.0,
                label=str(series["benchmark_label"].iloc[0]).replace("Logit-L2", r"Logit-$\ell_2$"),
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=2,
            )
        axis.set_ylabel("Cumulative Brier-score\nadvantage", labelpad=12)
        axis.set_xlim(recession["target_date"].min(), recession["target_date"].max())
        axis.margins(y=0.10)
        axis.legend(
            loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False,
            ncol=3, fontsize=18, handlelength=2.3, columnspacing=1.1,
            borderaxespad=0,
        )
        axis.set_xlabel("Target month", labelpad=10)
        axis.xaxis.set_major_locator(mdates.YearLocator(5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        figure.subplots_adjust(left=0.15, right=0.985, top=0.84, bottom=0.17)
        output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_output_path(output) as temporary:
            figure.savefig(
                temporary,
                format="pdf",
                bbox_inches="tight",
                metadata={"Creator": "program_v04", "CreationDate": None, "ModDate": None},
            )
        plt.close(figure)
    return output


DOCUMENTARY_COLUMNS = {
    "ijf": ("row_order", "citation_key", "method", "data_sample", "predictor_treatment", "url"),
    "nonijf": ("row_order", "citation_key", "outlet_method", "data_sample", "predictor_treatment", "url"),
}


def load_documentary_rows(path: Path, *, survey: str) -> pd.DataFrame:
    if survey not in DOCUMENTARY_COLUMNS:
        raise ValueError(f"Unknown documentary survey: {survey}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = DOCUMENTARY_COLUMNS[survey]
    if tuple(frame.columns) != expected or frame.empty:
        raise ValueError(f"{survey} documentary source has an invalid schema")
    order = pd.to_numeric(frame["row_order"], errors="raise").astype(int)
    if list(order) != list(range(1, len(frame) + 1)):
        raise ValueError(f"{survey} documentary row order is not contiguous")
    for column in expected[1:]:
        if frame[column].eq("").any():
            raise ValueError(f"{survey} documentary source has blank {column}")
    if not frame["url"].str.startswith("https://").all():
        raise ValueError(f"{survey} documentary links must use HTTPS")
    return frame


def render_documentary_tex(frame: pd.DataFrame, *, survey: str) -> str:
    expected = DOCUMENTARY_COLUMNS[survey]
    if tuple(frame.columns) != expected:
        raise ValueError(f"{survey} documentary renderer received the wrong schema")
    method_field = "method" if survey == "ijf" else "outlet_method"
    lines = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        lines.append(
            rf"\cite{{{values['citation_key']}}} & {values[method_field]} & "
            rf"{values['data_sample']} & {values['predictor_treatment']} & "
            rf"\href{{{values['url']}}}{{link}} \\"
        )
    return "\n".join(lines) + "\n"
