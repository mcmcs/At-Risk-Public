"""Monthly CRRA decisions, compound wealth, and certainty-equivalent returns.

Annualized scenario returns are converted geometrically to monthly rates.
Recorded h=3 forecasts select cash or investment for one target month only.
Cash earns zero; there are no deposits or overlapping holding periods.
The rule comes from a one-period CRRA comparison at given opening wealth.
It is applied mechanically to each h=3 forecast, not claimed to maximize
expected future wealth utility at the earlier origin when intervening wealth
is uncertain. The sample CE evaluates realized monthly returns
at common unit opening wealth, not each model's subjective expected utility.
Table 18 annualizes the monthly CE geometrically as (1 + CE_monthly)^12 - 1.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .data import validate_forecast_panel
from .robustness import FEATURED_MODEL_IDS, FEATURED_MODEL_SOURCES

WEALTH_MODELS = ("proposed", "xt_logit_l2")


@dataclass(frozen=True)
class WealthProtocol:
    horizon: int = 3
    initial_wealth: float = 100.0
    annual_expansion_return: float = 0.10
    annual_recession_return: float = -0.20
    gammas: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0, 20.0)

    def validate(self) -> None:
        values = (self.initial_wealth, self.annual_expansion_return,
                  self.annual_recession_return, *self.gammas)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Wealth parameters must be finite")
        if self.horizon != 3:
            raise ValueError("The wealth exercise is defined for h=3")
        if self.initial_wealth <= 0:
            raise ValueError("CRRA requires positive initial wealth")
        if not self.annual_expansion_return > 0 or not -1 < self.annual_recession_return < 0:
            raise ValueError("Require a positive expansion return and recession return in (-1,0)")
        if not self.gammas or len(set(self.gammas)) != len(self.gammas):
            raise ValueError("Risk-aversion values must be nonempty and unique")
        if any(value < 0 for value in self.gammas):
            raise ValueError("Risk aversion cannot be negative")

    @property
    def monthly_expansion_return(self) -> float:
        return math.expm1(math.log1p(self.annual_expansion_return) / 12)

    @property
    def monthly_recession_return(self) -> float:
        return math.expm1(math.log1p(self.annual_recession_return) / 12)


def crra_utility(wealth: np.ndarray | float, gamma: float) -> np.ndarray:
    """CRRA utility normalized to zero at wealth one, including log utility."""
    w = np.asarray(wealth, dtype=float)
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("Risk aversion must be finite and nonnegative")
    if not np.isfinite(w).all() or not (w > 0).all():
        raise ValueError("CRRA utility requires finite positive wealth")
    log_wealth = np.log(w)
    if gamma == 1:
        return log_wealth
    return np.expm1((1 - gamma) * log_wealth) / (1 - gamma)


def cash_threshold(gamma: float, expansion_return: float, recession_return: float) -> float:
    """Monthly probability cutoff for cash versus full investment."""
    if not (math.isfinite(expansion_return) and math.isfinite(recession_return)
            and expansion_return > 0 and -1 < recession_return < 0):
        raise ValueError("Monthly investment returns must straddle the zero cash return")
    up = float(crra_utility(1 + expansion_return, gamma))
    down = float(crra_utility(1 + recession_return, gamma))
    return up / (up - down)


def certainty_equivalent_return(gross_returns: np.ndarray, gamma: float) -> float:
    """Sample monthly CE from equally weighted realized gross returns."""
    gross = np.asarray(gross_returns, dtype=float)
    if gross.ndim != 1 or not len(gross):
        raise ValueError("Certainty equivalents require a nonempty return vector")
    mean_utility = float(crra_utility(gross, gamma).mean())
    if gamma == 1:
        return math.expm1(mean_utility)
    return math.expm1(math.log1p((1 - gamma) * mean_utility) / (1 - gamma))


def wealth_path(
    probabilities: np.ndarray, actual: np.ndarray, *, gamma: float,
    protocol: WealthProtocol,
) -> pd.DataFrame:
    """Select cash using forecasts only and settle one compounded monthly payoff."""
    protocol.validate()
    p, y = np.asarray(probabilities, dtype=float), np.asarray(actual, dtype=float)
    if p.ndim != 1 or p.shape != y.shape or not len(p):
        raise ValueError("Forecasts and outcomes require equal nonempty vectors")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Forecast probabilities must lie in [0,1]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Recession outcomes must be binary")
    expansion, recession = protocol.monthly_expansion_return, protocol.monthly_recession_return
    threshold = cash_threshold(gamma, expansion, recession)
    action = (p >= threshold).astype(int)  # One denotes cash.
    returns = (1 - action) * np.where(y == 1, recession, expansion)
    gross = 1 + returns
    closing = protocol.initial_wealth * np.cumprod(gross)
    opening = np.r_[protocol.initial_wealth, closing[:-1]]
    return pd.DataFrame({
        "probability": p, "actual": y.astype(int), "action": action,
        "decision_threshold": threshold, "simple_return": returns,
        "gross_return": gross, "normalized_utility": crra_utility(gross, gamma),
        "opening_wealth": opening, "wealth_change": closing - opening,
        "closing_wealth": closing,
    })


def build_decision_wealth(
    main_forecasts: pd.DataFrame,
    protocol: WealthProtocol = WealthProtocol(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol.validate()
    validate_forecast_panel(main_forecasts, surface="wealth main forecasts")
    if not main_forecasts.forecast_family.eq("main").all():
        raise ValueError("The X/Z Logit comparison requires the main forecast family")
    reference = None
    summaries, paths = [], []
    expansion, recession = protocol.monthly_expansion_return, protocol.monthly_recession_return
    for model_id in WEALTH_MODELS:
        predictor, model = FEATURED_MODEL_IDS[model_id]
        family = FEATURED_MODEL_SOURCES[(predictor, model)]
        selected = main_forecasts.loc[
            (main_forecasts.exercise == "revised") & (main_forecasts.data_variant == "revised_final")
            & (main_forecasts.forecast_family == family) & (main_forecasts.horizon == protocol.horizon)
            & (main_forecasts.predictor_key == predictor) & (main_forecasts.model_name == model),
            ["target_date", "origin_date", "probability", "actual"],
        ].sort_values("target_date").reset_index(drop=True)
        expected_origin = selected.target_date - pd.DateOffset(months=protocol.horizon)
        if not selected.origin_date.equals(expected_origin):
            raise ValueError("Decision dates must equal the recorded h=3 forecast origins")
        paired = selected[["target_date", "actual"]]
        if reference is None:
            reference = paired
        elif not paired.equals(reference):
            raise ValueError("The X/Z Logit comparison requires identical target dates and outcomes")
        for gamma in protocol.gammas:
            path = wealth_path(selected.probability, selected.actual, gamma=gamma, protocol=protocol)
            path["target_date"], path["origin_date"] = selected.target_date, selected.origin_date
            path["gamma"], path["model_id"], path["horizon"] = gamma, model_id, protocol.horizon
            paths.append(path)
            n, recession_count = len(path), int(path.actual.sum())
            always_invest = np.where(path.actual == 1, 1 + recession, 1 + expansion)
            summaries.append({
                "horizon": protocol.horizon, "gamma": gamma, "model_id": model_id,
                "predictor_key": predictor, "model_name": model, "source_family": family,
                "decision_threshold": float(path.decision_threshold.iloc[0]),
                "initial_wealth": protocol.initial_wealth,
                "annual_expansion_return": protocol.annual_expansion_return,
                "annual_recession_return": protocol.annual_recession_return,
                "monthly_expansion_return": expansion, "monthly_recession_return": recession,
                "target_month_count": n, "recession_month_count": recession_count,
                "target_start": selected.target_date.min(), "target_end": selected.target_date.max(),
                "cash_months": int(path.action.sum()),
                "false_positives": int(((path.action == 1) & (path.actual == 0)).sum()),
                "false_negatives": int(((path.action == 0) & (path.actual == 1)).sum()),
                "final_wealth": float(path.closing_wealth.iloc[-1]),
                "average_utility": float(path.normalized_utility.mean()),
                "certainty_equivalent_return": certainty_equivalent_return(path.gross_return, gamma),
                "always_invest_wealth": protocol.initial_wealth * math.exp(float(np.log(always_invest).sum())),
                "always_cash_wealth": protocol.initial_wealth,
                "always_invest_ce_return": certainty_equivalent_return(always_invest, gamma),
                "always_cash_ce_return": 0.0,
            })
    summary = pd.DataFrame(summaries).sort_values(["gamma", "model_id"]).reset_index(drop=True)
    for monthly_field, annual_field in (
        ("certainty_equivalent_return", "annualized_ce_return"),
        ("always_invest_ce_return", "always_invest_annualized_ce_return"),
        ("always_cash_ce_return", "always_cash_annualized_ce_return"),
    ):
        summary[annual_field] = np.expm1(12 * np.log1p(summary[monthly_field]))
    monthly = pd.concat(paths, ignore_index=True)
    validate_decision_wealth(summary, monthly, protocol)
    return summary, monthly


def validate_decision_wealth(
    summary: pd.DataFrame, paths: pd.DataFrame, protocol: WealthProtocol = WealthProtocol(),
) -> None:
    protocol.validate()
    expected = {(g, m) for g in protocol.gammas for m in WEALTH_MODELS}
    if len(summary) != len(expected) or set(zip(summary.gamma, summary.model_id)) != expected:
        raise ValueError("Wealth summary must contain both models at every gamma")
    if set(zip(paths.gamma, paths.model_id)) != expected:
        raise ValueError("Wealth paths must contain exactly the summary's model/gamma pairs")
    if paths.duplicated(["gamma", "model_id", "target_date"]).any():
        raise ValueError("Duplicate wealth observations")
    expansion, recession = protocol.monthly_expansion_return, protocol.monthly_recession_return
    for row in summary.itertuples(index=False):
        path = paths[(paths.gamma == row.gamma) & (paths.model_id == row.model_id)].sort_values("target_date")
        predictor, model = FEATURED_MODEL_IDS[row.model_id]
        if (row.predictor_key, row.model_name, row.source_family) != (
            predictor, model, FEATURED_MODEL_SOURCES[(predictor, model)]
        ):
            raise ValueError("Wealth summary model labels disagree with the recorded source mapping")
        if row.horizon != protocol.horizon or not (path.horizon == protocol.horizon).all():
            raise ValueError("Wealth summaries and paths require h=3")
        if len(path) != row.target_month_count or len(path) != 420 or int(path.actual.sum()) != 40:
            raise ValueError("Wealth accounts require the common 420-month, 40-recession sample")
        dates = pd.DatetimeIndex(path.target_date)
        if not dates.equals(pd.date_range("1990-01-01", "2024-12-01", freq="MS")):
            raise ValueError("Wealth accounts require the full 1990-2024 target calendar")
        if row.target_start != dates.min() or row.target_end != dates.max():
            raise ValueError("Summary target bounds disagree with monthly accounts")
        if not (path.origin_date == path.target_date - pd.DateOffset(months=3)).all():
            raise ValueError("Wealth path uses a decision from the wrong forecast origin")
        rebuilt = wealth_path(path.probability, path.actual, gamma=row.gamma, protocol=protocol)
        for column in rebuilt:
            if not np.allclose(path[column], rebuilt[column], rtol=0, atol=1e-12):
                raise ValueError(f"Wealth path disagrees with the fixed decision rule: {column}")
        false_positives = int(((path.action == 1) & (path.actual == 0)).sum())
        false_negatives = int(((path.action == 0) & (path.actual == 1)).sum())
        final_by_counts = protocol.initial_wealth * math.exp(
            (380 - false_positives) * math.log1p(expansion)
            + false_negatives * math.log1p(recession))
        always_invest = np.where(path.actual == 1, 1 + recession, 1 + expansion)
        for field, value in (
            ("initial_wealth", protocol.initial_wealth),
            ("annual_expansion_return", protocol.annual_expansion_return),
            ("annual_recession_return", protocol.annual_recession_return),
            ("monthly_expansion_return", expansion), ("monthly_recession_return", recession),
            ("decision_threshold", cash_threshold(row.gamma, expansion, recession)),
            ("recession_month_count", int(path.actual.sum())), ("cash_months", int(path.action.sum())),
            ("false_positives", false_positives), ("false_negatives", false_negatives),
            ("final_wealth", final_by_counts), ("average_utility", float(path.normalized_utility.mean())),
            ("certainty_equivalent_return", certainty_equivalent_return(path.gross_return, row.gamma)),
            ("always_invest_wealth", protocol.initial_wealth * math.exp(float(np.log(always_invest).sum()))),
            ("always_cash_wealth", protocol.initial_wealth),
            ("always_invest_ce_return", certainty_equivalent_return(always_invest, row.gamma)),
            ("always_cash_ce_return", 0.0),
        ):
            if not math.isclose(getattr(row, field), value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Wealth summary disagrees with monthly accounts: {field}")
        if not np.allclose(path.closing_wealth, path.opening_wealth * path.gross_return, rtol=1e-12, atol=1e-12):
            raise ValueError("Monthly wealth does not compound the reported return")
        if not np.allclose(path.opening_wealth.iloc[1:], path.closing_wealth.iloc[:-1], rtol=0, atol=1e-12):
            raise ValueError("Wealth does not carry forward between months")
        if row.final_wealth != path.closing_wealth.iloc[-1]:
            raise ValueError("Terminal wealth disagrees with the monthly path")
        ce_utility = float(crra_utility(1 + row.certainty_equivalent_return, row.gamma))
        if not math.isclose(ce_utility, row.average_utility, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Certainty equivalent does not reproduce average realized utility")
        for monthly_field, annual_field in (
            ("certainty_equivalent_return", "annualized_ce_return"),
            ("always_invest_ce_return", "always_invest_annualized_ce_return"),
            ("always_cash_ce_return", "always_cash_annualized_ce_return"),
        ):
            annualized = math.expm1(12 * math.log1p(getattr(row, monthly_field)))
            if not math.isclose(getattr(row, annual_field), annualized, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Annualized CE disagrees with monthly CE: {annual_field}")


def render_decision_wealth(summary: pd.DataFrame) -> str:
    rows = [r"\begin{tabular}{c r r r r r}", r"\toprule",
            r" & & \multicolumn{2}{c}{Terminal wealth} & \multicolumn{2}{c}{Annualized CE (\%)} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"$\gamma$ & Threshold (\%) & $Z_t$ & $X_t$ & $Z_t$ & $X_t$ \\",
            r"\midrule"]
    for gamma, group in summary.groupby("gamma", sort=True):
        values = group.set_index("model_id")
        z, x = values.loc["proposed"], values.loc["xt_logit_l2"]
        rows.append(f"{gamma:.1f} & {100*z.decision_threshold:.2f} & "
                    f"{z.final_wealth:.2f} & {x.final_wealth:.2f} & "
                    f"{100*z.annualized_ce_return:.2f} & "
                    f"{100*x.annualized_ce_return:.2f} " + r"\\")
    return "\n".join([*rows, r"\bottomrule", r"\end{tabular}", ""])
