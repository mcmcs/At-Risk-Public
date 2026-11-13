"""Generate appendix Table 16 (sampling uncertainty)."""

from __future__ import annotations

import pandas as pd

from program_v04.appendix.artifacts._shared import load_dependencies, write_table
from program_v04.replication.appendix_paper import validate_derived_table
from program_v04.replication.artifacts import artifact_cli_spec


def render_sampling_uncertainty(frame: pd.DataFrame) -> str:
    """Report primary 90% intervals and mark differences whose bounds exclude zero."""
    primary = frame[frame["is_primary"]]
    if primary.empty or not primary["confidence_level"].eq(0.90).all():
        raise ValueError("Table 16 requires primary 90% bootstrap intervals")
    lines = [r"\begin{tabular}{c l l r r r}", r"\toprule",
             r"$h$ & Benchmark & Metric & Difference & 5\% & 95\% \\", r"\midrule"]
    for index, (_, group) in enumerate(primary.groupby("horizon", sort=True)):
        if index:
            lines.append(r"\midrule")
        for row in group.itertuples(index=False):
            star = r"^{*}" if row.ci_lower > 0 or row.ci_upper < 0 else ""
            model = row.benchmark_label.replace("Logit-L2", r"Logit-$\ell_2$")
            metric = {"pr_auc": "PR AUC", "brier": "Brier"}[row.metric]
            lines.append(
                f"{row.horizon} & {model} & {metric} & ${row.point_estimate:.3f}{star}$ & "
                f"{row.ci_lower:.3f} & {row.ci_upper:.3f} " + r"\\"
            )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_sampling_uncertainty", argv)
    values, metadata = load_dependencies(spec)
    frame = validate_derived_table(
        values[spec.result_dependencies[0]], surface="Table 16"
    )
    write_table(spec, frame, metadata, tex=render_sampling_uncertainty(frame))
    print("Generated appendix Table 16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
