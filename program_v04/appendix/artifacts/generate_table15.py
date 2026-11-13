"""Generate appendix Table 15 (revised versus real-time data)."""

from __future__ import annotations

import pandas as pd

from program_v04.appendix.artifacts._shared import load_dependencies, write_table
from program_v04.replication.appendix_paper import build_appendix_table15
from program_v04.replication.artifacts import artifact_cli_spec


def render_realtime_table(frame: pd.DataFrame) -> str:
    """Render black numeric cells, marking within-variant leaders before rounding."""
    columns = [f"{metric}_h{h}" for metric in ("pr_auc", "brier") for h in (3, 6, 12)]
    lines = [
        r"\begin{tabular}{l ccc ccc}", r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{PR AUC}} & \multicolumn{3}{c}{\textbf{Brier Score}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"\textbf{Specification} & \textbf{$h=3$} & \textbf{$h=6$} & \textbf{$h=12$} & \textbf{$h=3$} & \textbf{$h=6$} & \textbf{$h=12$} \\",
    ]
    for section, group in frame.groupby("section", sort=False):
        lines.extend([r"\midrule", r"\multicolumn{7}{l}{\textit{" + section + r"}} \\"])
        best = {c: group[c].max() if c.startswith("pr_auc") else group[c].min() for c in columns}
        for _, row in group.iterrows():
            cells = []
            for c in columns:
                value = f"{row[c]:.3f}"
                cells.append(rf"\textbf{{{value}}}" if row[c] == best[c] else value)
            label = row['label'].replace('Logit-L2', r'Logit-$\ell_2$')
            lines.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def main(argv: list[str] | None = None) -> int:
    spec = artifact_cli_spec("tab_realtime", argv)
    values, metadata = load_dependencies(spec)
    frame = build_appendix_table15(values[spec.result_dependencies[0]])
    write_table(spec, frame, metadata, tex=render_realtime_table(frame))
    print("Generated appendix Table 15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
