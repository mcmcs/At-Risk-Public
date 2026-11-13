"""Audited Section 2 negative-spread and Sahm-rule illustration.

The frozen archived source accidentally resolves its repository root inside
the archive (RF-001).  This port reconstructs the same
estimands directly from the three declared raw inputs.  Core builders are
pure: callers own all filesystem writes.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from .config import GLOBAL_SEED, HORIZONS, PACKAGE_ROOT, RESULTS_ROOT, assert_canonical_generation_allowed
from .data import (
    InputRecord,
    load_fred_md_snapshot,
    load_target_snapshot,
    load_yield_spread_snapshot,
    manifest_by_id,
    verify_input,
)
from .provenance import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_output_path,
    build_run_metadata,
    fingerprint_file,
    validate_run_metadata,
    validate_result_dependency,
)


DISPLAY_START = pd.Timestamp("1960-01-01")
DISPLAY_END = pd.Timestamp("2024-12-01")
SAHM_MA_WINDOW = 3
SAHM_LOOKBACK = 12
SAHM_THRESHOLD = 0.5
SUMMARY_SIGNALS = ("yield_spread", "sahm_trigger")
SUMMARY_HORIZONS = (3, 6, 12)
SUMMARY_COLUMNS = ("signal", "horizon", "p_active_rec", "p_active_nonrec", "corr", "n")
SIGNAL_PANEL_COLUMNS = (
    "date",
    "yield_spread",
    "unrate",
    "unrate_ma3",
    "unrate_trailing_min12",
    "sahm_gap",
    "z_yield",
    "z_sahm",
    "recession",
)
SECTION2_INPUT_IDS = ("fred_md_final", "yield_spread", "usrecm_target_final")
SECTION2_RESULT_ROOT = RESULTS_ROOT / "main" / "section2"
SECTION2_SIGNAL_PATH = SECTION2_RESULT_ROOT / "signal_panel.parquet"
SECTION2_SUMMARY_PATH = SECTION2_RESULT_ROOT / "summary.parquet"
SECTION2_METADATA_PATH = SECTION2_RESULT_ROOT / "metadata.json"


@dataclass(frozen=True)
class Section2Data:
    signal_panel: pd.DataFrame
    summary: pd.DataFrame


def resolve_section2_inputs(
    *,
    mode: Literal["public", "development"] = "public",
    records: Mapping[str, InputRecord] | None = None,
) -> dict[str, Path]:
    """Resolve and checksum only the three declared Section 2 inputs."""
    registry = manifest_by_id() if records is None else dict(records)
    missing = sorted(set(SECTION2_INPUT_IDS) - set(registry))
    if missing:
        raise ValueError(f"Section 2 input records are missing: {missing}")
    return {
        input_id: verify_input(registry[input_id], mode=mode, verify_shape=True)
        for input_id in SECTION2_INPUT_IDS
    }


def load_section2_raw(paths: Mapping[str, Path]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Load raw term spread, full UNRATE history, and final recession target."""
    missing = sorted(set(SECTION2_INPUT_IDS) - set(paths))
    if missing:
        raise ValueError(f"Section 2 paths are missing: {missing}")

    spread = load_yield_spread_snapshot(paths["yield_spread"])
    fred, _tcodes = load_fred_md_snapshot(paths["fred_md_final"])
    if "UNRATE" not in fred:
        raise ValueError("Section 2 FRED-MD input must contain sasdate and UNRATE")
    unrate = fred["UNRATE"].rename("unrate")
    recession = load_target_snapshot(paths["usrecm_target_final"])
    return spread, unrate, recession


def sahm_components(unrate: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return the 3-month average, lagged 12-average minimum, and gap."""
    ma3 = unrate.rolling(SAHM_MA_WINDOW).mean()
    trailing_min = ma3.shift(1).rolling(SAHM_LOOKBACK).min()
    return ma3, trailing_min, ma3 - trailing_min


def _summary_row(signal: pd.Series, recession: pd.Series, *, name: str, horizon: int) -> dict[str, object]:
    evaluated = pd.concat(
        {"signal": signal.astype(float), "recession": recession.shift(-horizon)},
        axis=1,
    ).dropna()
    recession_rows = evaluated["recession"] == 1
    expansion_rows = evaluated["recession"] == 0
    if not recession_rows.any() or not expansion_rows.any():
        raise ValueError(f"Section 2 {name} h={horizon}: both target classes are required")
    return {
        "signal": name,
        "horizon": int(horizon),
        "p_active_rec": float(evaluated.loc[recession_rows, "signal"].mean()),
        "p_active_nonrec": float(evaluated.loc[expansion_rows, "signal"].mean()),
        "corr": float(evaluated["signal"].corr(evaluated["recession"])),
        "n": int(len(evaluated)),
    }


def validate_section2_data(data: Section2Data) -> None:
    panel = data.signal_panel
    summary = data.summary
    if tuple(panel.columns) != SIGNAL_PANEL_COLUMNS:
        raise ValueError(f"Section 2 panel schema mismatch: {tuple(panel.columns)}")
    if tuple(summary.columns) != SUMMARY_COLUMNS:
        raise ValueError(f"Section 2 summary schema mismatch: {tuple(summary.columns)}")
    expected_dates = pd.date_range(DISPLAY_START, DISPLAY_END, freq="MS")
    observed_dates = pd.DatetimeIndex(panel["date"])
    if not observed_dates.equals(expected_dates):
        raise ValueError("Section 2 signal panel must be gapless 1960-01 through 2024-12")
    if panel.drop(columns=["z_sahm", "unrate_ma3", "unrate_trailing_min12", "sahm_gap"]).isna().any().any():
        raise ValueError("Section 2 panel has unexpected missing values")
    for column in ("z_yield", "recession"):
        if not set(panel[column].unique()) <= {0.0, 1.0}:
            raise ValueError(f"Section 2 {column} is not binary")
    if int(panel["z_sahm"].isna().sum()) != 2 or panel["z_sahm"].iloc[2:].isna().any():
        raise ValueError("Section 2 Sahm signal must have exactly two leading missing values")
    if set(zip(summary["signal"], summary["horizon"], strict=True)) != {
        (signal, horizon) for signal in SUMMARY_SIGNALS for horizon in SUMMARY_HORIZONS
    }:
        raise ValueError("Section 2 summary signal/horizon coverage is incomplete")
    if summary["n"].le(0).any() or summary.drop(columns=["signal"]).isna().any().any():
        raise ValueError("Section 2 summary contains missing or empty estimates")


def build_section2_data(
    spread: pd.Series,
    unrate_history: pd.Series,
    recession: pd.Series,
) -> Section2Data:
    """Build the exact frozen Section 2 estimands on the display window."""
    display_index = pd.date_range(DISPLAY_START, DISPLAY_END, freq="MS")
    for name, series in (("yield_spread", spread), ("unrate", unrate_history), ("recession", recession)):
        if not series.index.is_monotonic_increasing or not series.index.is_unique:
            raise ValueError(f"Section 2 {name} index must be sorted and unique")
        missing = display_index.difference(series.index)
        if not missing.empty:
            raise ValueError(f"Section 2 {name} misses display months: {missing[:5].tolist()}")
        if series.reindex(display_index).isna().any():
            raise ValueError(f"Section 2 {name} has missing display-window values")
    if unrate_history.index.min() > pd.Timestamp("1959-01-01"):
        raise ValueError("Section 2 UNRATE history starts too late to initialize the Sahm rule")

    ma3_full, trailing_min_full, gap_full = sahm_components(unrate_history)
    spread_display = spread.reindex(display_index).astype(float)
    unrate_display = unrate_history.reindex(display_index).astype(float)
    recession_display = recession.reindex(display_index).astype(int)
    if not set(recession_display.unique()) <= {0, 1}:
        raise ValueError("Section 2 recession target is not binary")
    ma3 = ma3_full.reindex(display_index)
    trailing_min = trailing_min_full.reindex(display_index)
    gap = gap_full.reindex(display_index)
    z_yield = (spread_display < 0).astype(float)
    z_sahm = (gap >= SAHM_THRESHOLD).astype(float)
    z_sahm[gap.isna()] = np.nan

    panel = pd.DataFrame(
        {
            "date": display_index,
            "yield_spread": spread_display.to_numpy(),
            "unrate": unrate_display.to_numpy(),
            "unrate_ma3": ma3.to_numpy(),
            "unrate_trailing_min12": trailing_min.to_numpy(),
            "sahm_gap": gap.to_numpy(),
            "z_yield": z_yield.to_numpy(),
            "z_sahm": z_sahm.to_numpy(),
            "recession": recession_display.to_numpy(),
        },
        columns=SIGNAL_PANEL_COLUMNS,
    )
    panel_indexed = panel.set_index("date")
    summary = pd.DataFrame(
        [
            _summary_row(panel_indexed["z_yield"], panel_indexed["recession"], name="yield_spread", horizon=horizon)
            for horizon in SUMMARY_HORIZONS
        ]
        + [
            _summary_row(panel_indexed["z_sahm"], panel_indexed["recession"], name="sahm_trigger", horizon=horizon)
            for horizon in SUMMARY_HORIZONS
        ],
        columns=SUMMARY_COLUMNS,
    )
    data = Section2Data(signal_panel=panel, summary=summary)
    validate_section2_data(data)
    return data


def build_section2_from_declared_inputs(
    *, mode: Literal["public", "development"] = "public"
) -> Section2Data:
    paths = resolve_section2_inputs(mode=mode)
    return build_section2_data(*load_section2_raw(paths))


def write_section2_results(
    data: Section2Data,
    *,
    input_paths: Mapping[str, Path],
    command: tuple[str, ...],
    input_mode: Literal["public", "development"],
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, object]:
    """Atomically publish Section 2 panels, then their completion marker."""
    assert_canonical_generation_allowed()
    if started_at.tzinfo is None or ended_at.tzinfo is None:
        raise ValueError("Section 2 run timestamps must be timezone-aware")
    validate_section2_data(data)
    if set(input_paths) != set(SECTION2_INPUT_IDS):
        raise ValueError("Section 2 provenance must declare exactly its three raw inputs")
    atomic_write_parquet(SECTION2_SIGNAL_PATH, data.signal_panel)
    atomic_write_parquet(SECTION2_SUMMARY_PATH, data.summary)
    result_files = [
        fingerprint_file(SECTION2_SIGNAL_PATH, relative_to=RESULTS_ROOT.parent),
        fingerprint_file(SECTION2_SUMMARY_PATH, relative_to=RESULTS_ROOT.parent),
    ]
    metadata = build_run_metadata(
        profile="main",
        command=command,
        command_options={"input_mode": input_mode, "stage": "section2"},
        input_fingerprints=[
            fingerprint_file(path, relative_to=PACKAGE_ROOT.parent)
            for path in input_paths.values()
        ],
        requested_families=["section2"],
        completed_families=["section2"],
        requested_horizons=list(HORIZONS),
        completed_horizons=list(HORIZONS),
        seeds={"global": GLOBAL_SEED},
        start_time=started_at.astimezone(UTC),
        end_time=ended_at.astimezone(UTC),
        result_files=result_files,
        result_summaries={
            SECTION2_SIGNAL_PATH.name: {"rows": len(data.signal_panel), "columns": list(data.signal_panel)},
            SECTION2_SUMMARY_PATH.name: {"rows": len(data.summary), "columns": list(data.summary)},
        },
    )
    validate_run_metadata(metadata)
    atomic_write_json(SECTION2_METADATA_PATH, metadata)
    return metadata


def load_section2_results() -> Section2Data:
    """Load the complete fingerprint-matched Section 2 dependency bundle."""
    assert_canonical_generation_allowed()
    if not SECTION2_METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Section 2 result metadata is missing. Run: python -m program_v04.main.compute_main"
        )
    for path in (SECTION2_SIGNAL_PATH, SECTION2_SUMMARY_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Section 2 result dependency is missing: {path}. "
                "Run: python -m program_v04.main.compute_main"
            )
        validate_result_dependency(path, SECTION2_METADATA_PATH)
    data = Section2Data(
        signal_panel=pd.read_parquet(SECTION2_SIGNAL_PATH),
        summary=pd.read_parquet(SECTION2_SUMMARY_PATH),
    )
    validate_section2_data(data)
    return data


def render_summary_tex(summary: pd.DataFrame) -> str:
    """Render the exact active manuscript Table 1 fragment."""
    labels = {"yield_spread": "Yield spread", "sahm_trigger": "Sahm trigger"}
    lines = [
        "% Auto-generated by build_motivation_data.render_summary_tex.",
        "% Do not edit; regenerate via run_all.py.",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Signal & $h$ & $\Pr(z_t{=}1\mid y_{t+h}{=}1)$ & $\Pr(z_t{=}1\mid y_{t+h}{=}0)$ & $\mathrm{corr}(z_t,y_{t+h})$ \\",
        r"\midrule",
    ]
    for signal in SUMMARY_SIGNALS:
        for horizon in SUMMARY_HORIZONS:
            matches = summary[(summary["signal"] == signal) & (summary["horizon"] == horizon)]
            if len(matches) != 1:
                raise ValueError(f"Section 2 summary row missing or duplicate: {signal}, h={horizon}")
            row = matches.iloc[0]
            lines.append(
                f"{labels[signal]} & {horizon} & {row['p_active_rec']:.3f} & "
                f"{row['p_active_nonrec']:.3f} & {row['corr']:+.3f} \\\\"
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _shade_recessions(axis: object, recession: pd.Series, *, alpha: float) -> None:
    in_recession = False
    start: pd.Timestamp | None = None
    for date, value in recession.items():
        if value == 1 and not in_recession:
            start = date
            in_recession = True
        elif value == 0 and in_recession and start is not None:
            axis.axvspan(start, date, color="#b2182b", alpha=alpha, zorder=0)  # type: ignore[attr-defined]
            in_recession = False
    if in_recession and start is not None:
        axis.axvspan(start, recession.index[-1], color="#b2182b", alpha=alpha, zorder=0)  # type: ignore[attr-defined]


def _render_binary_strip(axis: object, signal: pd.Series) -> None:
    values = signal.fillna(0).astype(float).clip(0, 1)
    axis.fill_between(values.index, 0.0, values.values, step="post", color="#2b2b2b", linewidth=0)  # type: ignore[attr-defined]
    axis.set_ylim(0.0, 1.0)  # type: ignore[attr-defined]
    axis.set_yticks([])  # type: ignore[attr-defined]
    for name in ("left", "right", "top"):
        axis.spines[name].set_visible(False)  # type: ignore[attr-defined]


def render_figure(data: Section2Data, output_path: Path) -> Path:
    """Render the complete Figure 1 to a caller-selected path."""
    validate_section2_data(data)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "at_risk_replication_matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "at_risk_replication_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.grid": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "savefig.dpi": 300,
        }
    )
    panel = data.signal_panel.set_index("date")
    recession = panel["recession"]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 4.2),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.10, "wspace": 0.22},
    )
    yield_axis, sahm_axis = axes[0]
    z_yield_axis, z_sahm_axis = axes[1]

    _shade_recessions(yield_axis, recession, alpha=0.20)
    yield_axis.plot(panel.index, panel["yield_spread"], color="#2166ac", linewidth=1.8)
    yield_axis.axhline(0, color="black", linewidth=1.2)
    yield_axis.set_title("(a) Yield spread", loc="left", fontweight="bold")
    yield_axis.set_ylabel("Term spread (pp)")
    yield_axis.margins(x=0.0)
    yield_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    _shade_recessions(z_yield_axis, recession, alpha=0.10)
    _render_binary_strip(z_yield_axis, panel["z_yield"])
    z_yield_axis.set_ylabel(r"$z_t^{\mathrm{yld}}$", labelpad=8, fontsize=12)
    z_yield_axis.set_xlabel("Date")

    _shade_recessions(sahm_axis, recession, alpha=0.20)
    # Preserve the active Figure 1 display convention exactly: the plotted
    # minimum is recomputed after slicing to 1960, so its first 13 displayed
    # months are blank.  The Sahm estimand itself still uses pre-1960 history.
    display_trailing_min = panel["unrate_ma3"].shift(1).rolling(SAHM_LOOKBACK).min()
    sahm_axis.plot(panel.index, panel["unrate_ma3"], color="#2166ac", linewidth=1.8, label=r"$\bar u_t$ (3m MA)")
    sahm_axis.plot(panel.index, display_trailing_min, color="#92c5de", linewidth=1.8, linestyle="--", label=r"$m_t$ (trailing 12m min)")
    sahm_axis.set_title("(b) Sahm rule", loc="left", fontweight="bold")
    sahm_axis.set_ylabel("UNRATE (%, 3m MA)")
    sahm_axis.legend(loc="upper left", fontsize=11, frameon=False, ncol=1)
    sahm_axis.margins(x=0.0)
    sahm_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    _shade_recessions(z_sahm_axis, recession, alpha=0.10)
    _render_binary_strip(z_sahm_axis, panel["z_sahm"])
    z_sahm_axis.set_ylabel(r"$z_t^{\mathrm{Sahm}}$", labelpad=8, fontsize=12)
    z_sahm_axis.set_xlabel("Date")
    for axis in (z_yield_axis, z_sahm_axis):
        axis.xaxis.set_major_locator(mdates.YearLocator(10))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.margins(x=0.0)
        axis.tick_params(axis="x", labelsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(output_path) as temporary:
        figure.savefig(temporary, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output_path
