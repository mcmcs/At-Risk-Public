"""Generate appendix Table 17 (pre-2020 robustness)."""

from __future__ import annotations

import pandas as pd

from program_v04.appendix.artifacts._shared import load_dependencies, write_table
from program_v04.replication.appendix_paper import validate_derived_table
from program_v04.replication.artifacts import artifact_cli_spec


def render_pre2020_robustness(frame: pd.DataFrame) -> str:
    """Render the model comparisons; keep sample diagnostics in the CSV."""
    lines = [r"\begin{tabular}{c l l r r}", r"\toprule",
             r"$h$ & Window & Model & PR AUC & Brier \\", r"\midrule"]
    for index, (_, group) in enumerate(frame.groupby("horizon", sort=True)):
        if index:
            lines.append(r"\midrule")
        previous_window = None
        for row in group.itertuples(index=False):
            if previous_window is not None and row.window_id != previous_window:
                lines.append(r"\midrule")
            previous_window = row.window_id
            model = row.model_label.replace("Logit-L2", r"Logit-$\ell_2$")
            lines.append(
                f"{row.horizon} & {row.window_label} & {model} & "
                f"{row.pr_auc:.3f} & {row.brier:.3f} " + r"\\"
            )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_pre2020_robustness", argv)
    values, metadata = load_dependencies(spec)
    frame = validate_derived_table(
        values[spec.result_dependencies[0]], surface="Table 17"
    )
    write_table(spec, frame, metadata, tex=render_pre2020_robustness(frame))
    print("Generated appendix Table 17")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
