"""Derive Appendix C.9 CRRA wealth and utility equivalents from recorded h=3 runs.

This command does not refit forecasts. It verifies the recorded input bytes
and preserves each producing run's source identity in the new metadata.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
import sys

import pandas as pd

from program_v04.replication.config import PACKAGE_ROOT, assert_canonical_generation_allowed
from program_v04.replication.decision_wealth import (
    WealthProtocol, build_decision_wealth, validate_decision_wealth,
)
from program_v04.replication.provenance import (
    fingerprint_file, locked_config_fingerprint, validate_run_metadata,
    write_result_bundle_with_metadata,
)


def load_recorded_forecasts(family: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Verify a saved input against its producing run, without relabeling that run.

The producer's source hash can differ from the new consumer's source hash.
The input's data hash, run signature, and locked forecasting config must match.
Full forecast regeneration continues to use the stricter source-current gate.
"""
    if family != "main":
        raise ValueError("The X/Z Logit comparison uses the recorded main run")
    path = PACKAGE_ROOT / "results" / "shared" / f"revised_{family}" / "forecasts.parquet"
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    validate_run_metadata(metadata)
    actual = asdict(fingerprint_file(path, relative_to=PACKAGE_ROOT))
    expected = {item["path"]: item for item in metadata["result_files"]}
    if expected.get(actual["path"]) != actual:
        raise ValueError(f"Saved forecast bytes differ from their producing run: {path}")
    if metadata["config_sha256"] != locked_config_fingerprint():
        raise ValueError("Recorded forecasting configuration differs from the locked configuration")
    return pd.read_parquet(path), {
        "family": family, "run_fingerprint": metadata["run_fingerprint"],
        "producer_source_sha256": metadata["source_sha256"],
        "input": actual,
    }


def compute_decision_wealth(command: tuple[str, ...] | None = None) -> pd.DataFrame:
    assert_canonical_generation_allowed()
    started = datetime.now(UTC)
    main, main_record = load_recorded_forecasts("main")
    protocol = WealthProtocol()
    summary, paths = build_decision_wealth(main, protocol)
    directory = PACKAGE_ROOT / "results" / "appendix" / "decision_wealth"
    frames = {directory / "summary.parquet": summary, directory / "wealth_paths.parquet": paths}
    inputs = [
        PACKAGE_ROOT / "results" / "shared" / "revised_main" / filename
        for filename in ("forecasts.parquet", "metadata.json")
    ]
    write_result_bundle_with_metadata(
        result_frames=frames,
        metadata_path=directory / "metadata.json",
        validators={
            directory / "summary.parquet": lambda frame: validate_decision_wealth(frame, paths, protocol),
            directory / "wealth_paths.parquet": lambda frame: validate_decision_wealth(summary, frame, protocol),
        },
        metadata_arguments={
            "profile": "appendix", "command": command or tuple(sys.argv),
            "command_options": {
                "family": "decision_wealth", "method": asdict(protocol),
                "input_policy": "recorded_forecast_bytes_verified_without_refitting",
                "input_runs": [main_record],
                "comparison": "Zt_Disagg_vs_Xt_Full_with_Logit_L2_in_both_cases",
                "forecast_timing": "recorded_origin_three_months_before_target",
                "decision_rule": "fixed_one_period_CRRA_cutoff_applied_to_recorded_h3_probability",
                "settlement_timing": "one_investment_return_settles_in_target_month",
                "annual_to_monthly_conversion": "(1+annual_return)**(1/12)-1",
                "wealth_rule": "W[t]=W[t-1]*(1+(1-cash[t])*state_return[t])",
                "assessment_scope": "realized_wealth_from_fixed_one_period_CRRA_rule",
                "certainty_equivalent": "u_inverse(mean(u(realized_gross_return)))-1",
                "reported_ce": "annualized_ce_return=(1+monthly_ce_return)**12-1",
                "utility_normalization": "common_unit_opening_wealth",
            },
            "input_fingerprints": [fingerprint_file(p, relative_to=PACKAGE_ROOT) for p in inputs],
            "requested_families": ["decision_wealth"], "completed_families": ["decision_wealth"],
            "requested_horizons": [3], "completed_horizons": [3],
            "seeds": {}, "start_time": started, "end_time": datetime.now(UTC), "events": (),
        },
    )
    return summary


def main() -> int:
    summary = compute_decision_wealth()
    print(summary[["gamma", "model_id", "decision_threshold", "final_wealth",
                   "annualized_ce_return"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
