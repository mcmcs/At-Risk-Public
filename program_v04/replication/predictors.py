"""Locked predictor registries, feature naming, and pure construction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from .thresholds import ThresholdCalibration
from .transforms import (
    RollingBoundary,
    TailInequality,
    generate_at_risk_states,
    generate_pca_factors,
    impute_training_only,
    scale_training_only,
)


VARIABLE_GROUPS: dict[str, tuple[str, ...]] = {
    "Output_Income": (
        "RPI", "W875RX1", "INDPRO", "IPFPNSS", "IPFINAL", "IPCONGD",
        "IPDCONGD", "IPNCONGD", "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT",
        "IPMANSICS", "IPB51222S", "IPFUELS", "CUMFNS",
    ),
    "Labor_Market": (
        "HWI", "HWIURATIO", "CLF16OV", "CE16OV", "UNRATE", "UEMPMEAN",
        "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV",
        "CLAIMSx", "PAYEMS", "USGOOD", "CES1021000001", "USCONS", "MANEMP",
        "DMANEMP", "NDMANEMP", "SRVPRD", "USTPU", "USWTRADE", "USTRADE",
        "USFIRE", "USGOVT", "CES0600000007", "AWOTMAN", "AWHMAN",
        "CES0600000008", "CES2000000008", "CES3000000008",
    ),
    "Housing": (
        "HOUST", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW", "PERMIT",
        "PERMITNE", "PERMITMW", "PERMITS", "PERMITW",
    ),
    "Consumption_Orders_Inventories": (
        "DPCERA3M086SBEA", "CMRMTSPLx", "RETAILx", "AMDMNOx", "AMDMUOx",
        "ANDENOx", "BUSINVx", "ISRATIOx",
    ),
    "Money_Credit": (
        "M1SL", "M2SL", "M2REAL", "BOGMBASE", "TOTRESNS", "NONBORRES",
        "BUSLOANS", "REALLN", "NONREVSL", "CONSPI", "DTCOLNVHFNM",
        "DTCTHFNM", "INVEST",
    ),
    "Interest_Rates_Spreads": (
        "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA",
        "BAA", "COMPAPFFx", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM",
        "T10YFFM", "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx",
        "EXCAUSx",
    ),
    "Prices": (
        "WPSFD49207", "WPSFD49502", "WPSID61", "WPSID62", "PPICMM",
        "CPIAUCSL", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC",
        "CUSR0000SAD", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2",
        "CUSR0000SA0L5", "PCEPI", "DDURRG3M086SBEA", "DNDGRG3M086SBEA",
        "DSERRG3M086SBEA",
    ),
    "Stock_Market": ("S&P 500", "S&P div yield", "S&P PE ratio", "VIXCLSx"),
}

COUNTER_CYCLICAL_VARS = frozenset(
    {
        "UNRATE", "UEMPMEAN", "UEMPLT5", "UEMP5TO14", "UEMP15OV",
        "UEMP15T26", "UEMP27OV", "CLAIMSx", "ISRATIOx", "VIXCLSx",
    }
)

VARS_TO_REMOVE = ("ACOGNO", "TWEXAFEGSMTHx", "UMCSENTx", "OILPRICEx")
PARSIMONIOUS_VARS = (
    "T10YFFM", "BAAFFM", "PAYEMS", "CLAIMSx", "UNRATE", "INDPRO", "HOUST",
    "RETAILx", "S&P 500", "M2REAL",
)
SPREADS_ONLY_VARS = (
    "T1YFFM", "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM", "TB3SMFFM",
    "TB6SMFFM", "COMPAPFFx",
)
REAL_ECONOMY_VARS = (
    "PAYEMS", "CLAIMSx", "UNRATE", "AWHMAN", "INDPRO", "RPI", "HOUST", "RETAILx",
)

DISPLAY_NAME_MAP = {
    "AAAFFM": "Aaa-FF spread",
    "AWHMAN": "Avg hrs: mfg",
    "BAAFFM": "Baa-FF spread",
    "COMPAPFFx": "CP-FF spread",
    "HOUST": "Starts: nonfarm",
    "M2REAL": "M2 (real)",
    "MANEMP": "Emp: mfg",
    "PAYEMS": "Emp: total",
    "TB3SMFFM": "3 mo-FF spread",
    "TB6SMFFM": "6 mo-FF spread",
    "T10YFFM": "10 yr-FF spread",
    "T1YFFM": "1 yr-FF spread",
    "T5YFFM": "5 yr-FF spread",
    "S&P 500": "S&P 500",
    "S&P div yield": "S&P div yield",
}

ALL_GROUPED_VARIABLES = tuple(
    variable for members in VARIABLE_GROUPS.values() for variable in members
)
if len(ALL_GROUPED_VARIABLES) != 122 or len(set(ALL_GROUPED_VARIABLES)) != 122:
    raise RuntimeError("The locked FRED-MD grouping must contain 122 unique variables")

VARIABLE_TO_SECTOR = {
    variable: sector for sector, members in VARIABLE_GROUPS.items() for variable in members
}


class PredictorType(StrEnum):
    ZT_DISAGG = "zt_disagg"
    ZT_AVG = "zt_avg"
    ZT_PCA = "zt_pca"
    XT_FULL = "xt_full"
    XT_PCA = "xt_pca"
    YIELD = "yield"


class ThresholdType(StrEnum):
    GLOBAL = "global"
    SECTOR = "sector"
    VARIABLE = "variable"


@dataclass(frozen=True)
class PredictorSpec:
    """Construction contract for one named predictor family member."""

    name: str
    predictor_type: PredictorType
    threshold_type: ThresholdType = ThresholdType.GLOBAL
    window_size: int = 1
    subset: tuple[str, ...] | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Predictor name is empty")
        if self.window_size < 1:
            raise ValueError(f"{self.name}: window_size must be positive")
        if self.subset is not None:
            if not self.subset or len(set(self.subset)) != len(self.subset):
                raise ValueError(f"{self.name}: subset must be nonempty and unique")


def add_lags(frame: pd.DataFrame, lags_to_add: tuple[int, ...]) -> pd.DataFrame:
    """Append contemporaneous predictors followed by stable positive lags."""
    if any(lag <= 0 for lag in lags_to_add) or len(set(lags_to_add)) != len(lags_to_add):
        raise ValueError("Lags must be unique positive integers")
    if frame.columns.has_duplicates:
        raise ValueError("Cannot lag a frame with duplicate feature names")
    pieces = [frame.copy()]
    for lag in lags_to_add:
        shifted = frame.shift(lag)
        shifted.columns = [f"{column}_lag{lag}" for column in frame.columns]
        pieces.append(shifted)
    result = pd.concat(pieces, axis=1)
    if result.columns.has_duplicates:
        raise ValueError("Lag construction produced duplicate feature names")
    return result


def strip_lag_suffix(feature_name: str) -> str:
    return re.sub(r"_lag\d+$", "", feature_name)


def infer_sector(variable_name: str) -> str:
    return VARIABLE_TO_SECTOR.get(strip_lag_suffix(variable_name), "Unmapped")


def to_display_name(variable_name: str) -> str:
    return DISPLAY_NAME_MAP.get(variable_name, variable_name)


def select_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, surface: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{surface}: missing declared predictors {missing}")
    return frame.loc[:, list(columns)].copy()


def modeling_frame(full_transformed: pd.DataFrame, spec: PredictorSpec) -> pd.DataFrame:
    """Apply a declared subset before any threshold calibration or fitting."""
    spec.validate()
    if full_transformed.empty:
        raise ValueError(f"{spec.name}: transformed predictor frame is empty")
    if spec.subset is not None:
        if spec.predictor_type in {
            PredictorType.ZT_DISAGG,
            PredictorType.ZT_AVG,
            PredictorType.ZT_PCA,
        }:
            requested = set(spec.subset)
            columns = tuple(variable for variable in ALL_GROUPED_VARIABLES if variable in requested)
            missing = requested - set(columns)
            if missing:
                raise KeyError(f"{spec.name}: missing declared predictors {sorted(missing)}")
            return select_columns(full_transformed, columns, surface=spec.name)
        return select_columns(full_transformed, spec.subset, surface=spec.name)
    if spec.predictor_type in {
        PredictorType.ZT_DISAGG,
        PredictorType.ZT_AVG,
        PredictorType.ZT_PCA,
    }:
        present = tuple(variable for variable in ALL_GROUPED_VARIABLES if variable in full_transformed)
        if not present:
            raise ValueError(f"{spec.name}: no grouped FRED-MD variables are present")
        return select_columns(full_transformed, present, surface=spec.name)
    return full_transformed.copy()


def build_predictor_data(
    spec: PredictorSpec,
    full_transformed: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    thresholds: ThresholdCalibration | None = None,
    rolling_boundary: RollingBoundary | None = None,
    tail_inequality: TailInequality | None = None,
    n_pca_factors: int = 8,
) -> pd.DataFrame:
    """Build one predictor on a recursive slice under explicit audited rules."""
    selected = modeling_frame(full_transformed, spec)
    if not selected.index.is_monotonic_increasing or not selected.index.is_unique:
        raise ValueError(f"{spec.name}: predictor index must be sorted and unique")
    train = selected.loc[:pd.Timestamp(train_end)]
    if train.empty:
        raise ValueError(f"{spec.name}: training slice through {train_end} is empty")

    if spec.predictor_type is PredictorType.YIELD:
        return selected
    if spec.predictor_type is PredictorType.XT_FULL:
        imputed = impute_training_only(train, selected)
        return scale_training_only(imputed.loc[train.index], imputed)
    if spec.predictor_type is PredictorType.XT_PCA:
        factors, _ = generate_pca_factors(
            train,
            selected,
            n_factors=n_pca_factors,
            standardize=True,
        )
        return factors

    if thresholds is None:
        raise ValueError(f"{spec.name}: at-risk predictors require calibrated thresholds")
    thresholds.validate()
    if thresholds.calibration_columns != tuple(selected.columns):
        raise ValueError(
            f"{spec.name}: threshold calibration columns do not match the ordered modeling frame"
        )
    for field, value in (
        ("predictor_end", thresholds.predictor_end),
        ("target_end", thresholds.target_end),
    ):
        if pd.Timestamp(value) > pd.Timestamp(train_end):
            raise ValueError(f"{spec.name}: threshold {field} exceeds train_end")
    if tail_inequality is None:
        raise ValueError(f"{spec.name}: at-risk predictors require TailInequality (RF-009)")
    if spec.window_size not in thresholds.global_by_window:
        raise KeyError(
            f"{spec.name}: missing exact global threshold for window {spec.window_size}"
        )
    if spec.threshold_type is ThresholdType.VARIABLE:
        eligible = [
            variable for variable in selected.columns
            if variable in thresholds.variable
        ]
    elif spec.threshold_type is ThresholdType.SECTOR:
        eligible = [
            variable for variable in selected.columns
            if VARIABLE_TO_SECTOR.get(variable) in thresholds.sector
        ]
    else:
        eligible = list(selected.columns)
    if not eligible:
        raise ValueError(
            f"{spec.name}: no predictors have a defined {spec.threshold_type.value} "
            "threshold in the training information set"
        )
    # Under per-variable available-row calibration, an early fold can contain
    # a series but no recession-period rank for that series. Such a variable
    # has no selected threshold and is unavailable to that fold; retaining it
    # would require either future recession data or an unselected fallback
    # estimand. Fit and validation use the same eligible ordered columns.
    selected = selected.loc[:, eligible]
    train = selected.loc[:pd.Timestamp(train_end)]
    imputed = impute_training_only(train, selected)
    train_imputed = imputed.loc[train.index]
    states = generate_at_risk_states(
        imputed,
        thresholds.global_by_window[spec.window_size],
        counter_cyclical=COUNTER_CYCLICAL_VARS,
        variable_to_sector=VARIABLE_TO_SECTOR,
        threshold_type=spec.threshold_type.value,
        sector_thresholds=thresholds.sector,
        variable_thresholds=thresholds.variable,
        window_size=spec.window_size,
        rolling_boundary=rolling_boundary,
        tail_inequality=tail_inequality,
        reference_df=train_imputed,
    )
    if spec.predictor_type is PredictorType.ZT_DISAGG:
        return states
    if spec.predictor_type is PredictorType.ZT_AVG:
        return states.mean(axis=1, skipna=False).to_frame(name="Zt_Avg")
    if spec.predictor_type is PredictorType.ZT_PCA:
        if spec.window_size > 1 and rolling_boundary is RollingBoundary.COMPLETE:
            complete = states.notna().all(axis=1)
            complete_train = states.loc[train.index].loc[complete.loc[train.index]]
            if complete_train.empty:
                raise ValueError(f"{spec.name}: no complete state rows remain for PCA training")
            complete_full = states.loc[complete]
            complete_factors, _ = generate_pca_factors(
                complete_train,
                complete_full,
                n_factors=n_pca_factors,
                standardize=False,
            )
            factors = complete_factors.reindex(states.index)
            return factors
        factors, _ = generate_pca_factors(
            states.loc[train.index],
            states,
            n_factors=n_pca_factors,
            standardize=False,
        )
        return factors
    raise ValueError(f"{spec.name}: unsupported predictor type {spec.predictor_type}")


PREDICTOR_SPECS: dict[str, PredictorSpec] = {
    "Zt_Disagg": PredictorSpec("Zt_Disagg", PredictorType.ZT_DISAGG),
    "Zt_Avg": PredictorSpec("Zt_Avg", PredictorType.ZT_AVG),
    "Zt_PCA": PredictorSpec("Zt_PCA", PredictorType.ZT_PCA),
    "Xt_Full": PredictorSpec("Xt_Full", PredictorType.XT_FULL),
    "Xt_PCA": PredictorSpec("Xt_PCA", PredictorType.XT_PCA),
    "Yield": PredictorSpec("Yield", PredictorType.YIELD),
    "Zt_Disagg_SecQ": PredictorSpec(
        "Zt_Disagg_SecQ", PredictorType.ZT_DISAGG, ThresholdType.SECTOR
    ),
    "Zt_Disagg_VarQ": PredictorSpec(
        "Zt_Disagg_VarQ", PredictorType.ZT_DISAGG, ThresholdType.VARIABLE
    ),
    "Zt_PCA_SecQ": PredictorSpec("Zt_PCA_SecQ", PredictorType.ZT_PCA, ThresholdType.SECTOR),
    "Zt_PCA_VarQ": PredictorSpec("Zt_PCA_VarQ", PredictorType.ZT_PCA, ThresholdType.VARIABLE),
    **{
        f"{base}_MA{window}": PredictorSpec(
            f"{base}_MA{window}",
            PredictorType.ZT_DISAGG if base == "Zt_Disagg" else PredictorType.ZT_PCA,
            window_size=window,
        )
        for window in (3, 6, 12)
        for base in ("Zt_Disagg", "Zt_PCA")
    },
    "Zt_Parsimonious": PredictorSpec(
        "Zt_Parsimonious", PredictorType.ZT_DISAGG, subset=PARSIMONIOUS_VARS
    ),
    "Xt_Parsimonious": PredictorSpec(
        "Xt_Parsimonious", PredictorType.XT_FULL, subset=PARSIMONIOUS_VARS
    ),
    "Yield_Univariate": PredictorSpec("Yield_Univariate", PredictorType.YIELD),
    "Zt_Spreads": PredictorSpec("Zt_Spreads", PredictorType.ZT_DISAGG, subset=SPREADS_ONLY_VARS),
    "Xt_Spreads": PredictorSpec("Xt_Spreads", PredictorType.XT_FULL, subset=SPREADS_ONLY_VARS),
    "Zt_RealEcon": PredictorSpec(
        "Zt_RealEcon", PredictorType.ZT_DISAGG, subset=REAL_ECONOMY_VARS
    ),
    "Xt_RealEcon": PredictorSpec("Xt_RealEcon", PredictorType.XT_FULL, subset=REAL_ECONOMY_VARS),
    "Zt_Disagg_ExpandingQ": PredictorSpec("Zt_Disagg_ExpandingQ", PredictorType.ZT_DISAGG),
}

for _spec in PREDICTOR_SPECS.values():
    _spec.validate()
