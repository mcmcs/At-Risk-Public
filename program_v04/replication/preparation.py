"""Declared-input revised-data preparation for all forecast families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from .data import (
    InputRecord,
    load_fred_md_snapshot,
    load_target_snapshot,
    load_yield_spread_snapshot,
    manifest_by_id,
    verify_input,
)
from .predictors import ALL_GROUPED_VARIABLES, VARS_TO_REMOVE
from .transforms import Code7Rule, transform_panel
from .validation import require_gapless_monthly_index


ANALYSIS_START = pd.Timestamp("1960-01-01")
ANALYSIS_END = pd.Timestamp("2024-12-01")
REVISED_INPUT_IDS = ("fred_md_final", "yield_spread", "usrecm_target_final")


@dataclass(frozen=True)
class RevisedData:
    """All normalized revised-data inputs needed by recursive families."""

    transformed: pd.DataFrame
    untransformed: pd.DataFrame
    yield_spread: pd.DataFrame
    target: pd.Series
    transformation_codes: pd.Series
    input_paths: Mapping[str, Path]
    code7_rule: Code7Rule

    def validate(self) -> None:
        expected_index = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="MS")
        for name, value in (
            ("transformed", self.transformed),
            ("untransformed", self.untransformed),
            ("yield_spread", self.yield_spread),
        ):
            if not value.index.equals(expected_index):
                raise ValueError(f"Revised {name} index must be gapless 1960-01 through 2024-12")
            require_gapless_monthly_index(value.index, surface=f"revised {name}")
            if value.columns.has_duplicates:
                raise ValueError(f"Revised {name} has duplicate columns")
        if not self.target.index.equals(expected_index):
            raise ValueError("Revised target index must be gapless 1960-01 through 2024-12")
        if self.target.isna().any() or not set(self.target.unique()) <= {0, 1}:
            raise ValueError("Revised target must be nonmissing and binary")
        if tuple(self.transformed.columns) != tuple(self.untransformed.columns):
            raise ValueError("Revised transformed and untransformed columns differ")
        if set(self.transformed.columns) != set(ALL_GROUPED_VARIABLES):
            missing = sorted(set(ALL_GROUPED_VARIABLES) - set(self.transformed.columns))
            extras = sorted(set(self.transformed.columns) - set(ALL_GROUPED_VARIABLES))
            raise ValueError(f"Revised predictor membership mismatch: missing={missing}, extras={extras}")
        if self.transformed.shape != (780, 122) or self.untransformed.shape != (780, 122):
            raise ValueError("Revised FRED-MD surfaces must each be 780 by 122")
        if list(self.yield_spread.columns) != ["YIELD_SPREAD"]:
            raise ValueError("Revised yield-spread frame must contain only YIELD_SPREAD")
        if self.yield_spread.isna().any().any():
            raise ValueError("Revised yield-spread frame contains missing values")
        if set(self.input_paths) != set(REVISED_INPUT_IDS):
            raise ValueError("Revised data must retain exactly its three input paths")
        if not isinstance(self.code7_rule, Code7Rule):
            raise ValueError("Revised data has no valid code-7 policy")
        if tuple(self.transformation_codes.index) != tuple(self.transformed.columns):
            raise ValueError("Revised transformation codes do not match ordered predictors")
        if self.transformation_codes.isna().any() or not np.isfinite(
            self.transformation_codes.to_numpy(dtype=float)
        ).all():
            raise ValueError("Revised transformation codes are incomplete")


def resolve_revised_inputs(
    *,
    mode: Literal["public", "development"] = "public",
    records: Mapping[str, InputRecord] | None = None,
) -> dict[str, Path]:
    """Resolve and checksum the complete revised-data raw input set."""
    registry = manifest_by_id() if records is None else dict(records)
    missing = sorted(set(REVISED_INPUT_IDS) - set(registry))
    if missing:
        raise ValueError(f"Revised-data input records are missing: {missing}")
    return {
        input_id: verify_input(registry[input_id], mode=mode, verify_shape=True)
        for input_id in REVISED_INPUT_IDS
    }


def prepare_revised_data(
    paths: Mapping[str, Path],
    *,
    code7_rule: Code7Rule,
) -> RevisedData:
    """Build normalized 1960--2024 revised inputs under an explicit code-7 rule."""
    if set(paths) != set(REVISED_INPUT_IDS):
        raise ValueError("Revised-data preparation requires exactly three declared inputs")
    if not isinstance(code7_rule, Code7Rule):
        raise ValueError("Revised-data preparation requires an explicit Code7Rule (RF-004)")
    raw, tcodes = load_fred_md_snapshot(paths["fred_md_final"])
    transformed = transform_panel(raw, tcodes, code7_rule=code7_rule)
    retained = tuple(column for column in raw.columns if column not in VARS_TO_REMOVE)
    if len(retained) != 122:
        raise ValueError(f"Expected 122 retained FRED-MD predictors, found {len(retained)}")
    analysis_index = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="MS")
    untransformed = raw.loc[:, list(retained)].reindex(analysis_index)
    transformed = transformed.loc[:, list(retained)].reindex(analysis_index)
    target = load_target_snapshot(paths["usrecm_target_final"]).reindex(analysis_index)
    target.name = "USRECM"
    spread = load_yield_spread_snapshot(paths["yield_spread"]).reindex(analysis_index)
    spread = spread.rename("YIELD_SPREAD").to_frame()
    data = RevisedData(
        transformed=transformed,
        untransformed=untransformed,
        yield_spread=spread,
        target=target,
        transformation_codes=tcodes.loc[list(retained)].copy(),
        input_paths=dict(paths),
        code7_rule=code7_rule,
    )
    data.validate()
    return data


def prepare_revised_from_declared_inputs(
    *,
    code7_rule: Code7Rule,
    mode: Literal["public", "development"] = "public",
) -> RevisedData:
    paths = resolve_revised_inputs(mode=mode)
    return prepare_revised_data(paths, code7_rule=code7_rule)

