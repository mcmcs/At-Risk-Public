"""Shared exact-parity contracts for result families used by both profiles."""

from __future__ import annotations

import pandas as pd

from .data import FORECAST_KEYS, validate_forecast_panel


FEATURED_MODEL_SOURCES = {
    ("Zt_Disagg", "Logit_L2"): "main",
    ("Xt_Full", "Logit_L2"): "main",
    ("Xt_Full", "XGBoost"): "main",
    ("Xt_Full", "RandomForest"): "extended",
}
FEATURED_DUPLICATED_PAIRS = tuple(
    pair for pair, family in FEATURED_MODEL_SOURCES.items() if family == "main"
)
FEATURED_PARITY_KEYS = tuple(
    key for key in FORECAST_KEYS if key != "forecast_family"
)


def _require_surface_selector(exercise: str, data_variant: str) -> None:
    for field, value in (("exercise", exercise), ("data_variant", data_variant)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Featured-model {field} must be a nonempty string")


def validate_featured_cross_family_parity(
    main_forecasts: pd.DataFrame,
    extended_forecasts: pd.DataFrame,
    *,
    exercise: str,
    data_variant: str,
) -> None:
    """Require exact duplicated-model parity for one declared data surface."""
    _require_surface_selector(exercise, data_variant)
    validate_forecast_panel(main_forecasts, surface="featured main forecasts")
    validate_forecast_panel(
        extended_forecasts,
        surface="featured extended forecasts",
    )
    if not main_forecasts["forecast_family"].eq("main").all():
        raise ValueError("Featured main input contains a non-main family row")
    if not extended_forecasts["forecast_family"].eq("extended").all():
        raise ValueError("Featured extended input contains a non-extended family row")

    comparison_columns = [*FEATURED_PARITY_KEYS, "probability", "actual"]
    selected_surfaces = {
        "main": main_forecasts[
            (main_forecasts["exercise"] == exercise)
            & (main_forecasts["data_variant"] == data_variant)
        ],
        "extended": extended_forecasts[
            (extended_forecasts["exercise"] == exercise)
            & (extended_forecasts["data_variant"] == data_variant)
        ],
    }
    if any(frame.empty for frame in selected_surfaces.values()):
        raise ValueError("Featured parity surface selection is empty")

    for predictor_key, model_name in FEATURED_DUPLICATED_PAIRS:
        pair_name = f"{predictor_key}/{model_name}"
        selected: list[pd.DataFrame] = []
        for family in ("main", "extended"):
            pair = selected_surfaces[family][
                (selected_surfaces[family]["predictor_key"] == predictor_key)
                & (selected_surfaces[family]["model_name"] == model_name)
            ]
            if pair.empty:
                raise ValueError(
                    f"Featured parity is missing {pair_name} in {exercise}/{family}"
                )
            selected.append(
                pair[comparison_columns]
                .sort_values(list(FEATURED_PARITY_KEYS), kind="mergesort")
                .reset_index(drop=True)
            )
        try:
            pd.testing.assert_frame_equal(
                selected[0],
                selected[1],
                check_exact=True,
                check_like=False,
            )
        except AssertionError as exc:
            raise ValueError(
                f"Featured parity failed for {pair_name} between main and extended"
            ) from exc
