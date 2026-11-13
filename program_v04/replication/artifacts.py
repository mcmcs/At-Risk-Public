"""Closed paper-artifact registry and manuscript-inventory validation.

Stable artifact identifiers and LaTeX labels are authoritative inside the
package.  Display numbers are manuscript properties: they are derived from
the current source order and checked against the expected public entry-point
names.  Figure 2 is one numbered artifact with three declared panel files.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import PACKAGE_ROOT


REPO_ROOT = PACKAGE_ROOT.parent
EMBEDDED_MANUSCRIPT_PATH = PACKAGE_ROOT / "docs" / "manuscript.tex"
MANUSCRIPT_PATH = (
    EMBEDDED_MANUSCRIPT_PATH
    if EMBEDDED_MANUSCRIPT_PATH.is_file()
    else REPO_ROOT / "draft" / "main.tex"
)
ArtifactKind = Literal["table", "figure"]
ArtifactProfile = Literal["main", "appendix"]
ArtifactClass = Literal["computational", "simulation", "documentary"]


@dataclass(frozen=True)
class ArtifactSpec:
    """One numbered manuscript float and its complete public contract."""

    artifact_id: str
    kind: ArtifactKind
    profile: ArtifactProfile
    artifact_class: ArtifactClass
    draft_label: str
    manuscript_section: str
    generator_module: str
    result_dependencies: tuple[str, ...]
    output_paths: tuple[str, ...]
    subfigure_labels: tuple[str, ...] = ()

    def validate(self) -> None:
        expected_prefix = "tab:" if self.kind == "table" else "fig:"
        if not self.draft_label.startswith(expected_prefix):
            raise ValueError(f"{self.artifact_id}: invalid {self.kind} label {self.draft_label!r}")
        if not self.generator_module.startswith(f"program_v04.{self.profile}.artifacts.generate_"):
            raise ValueError(f"{self.artifact_id}: generator is outside its profile")
        if not self.result_dependencies:
            raise ValueError(f"{self.artifact_id}: no result dependency declared")
        if not self.output_paths:
            raise ValueError(f"{self.artifact_id}: no output path declared")
        for value in (*self.result_dependencies, *self.output_paths):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{self.artifact_id}: unsafe package-relative path {value!r}")
        if self.kind == "table" and self.subfigure_labels:
            raise ValueError(f"{self.artifact_id}: a table cannot declare subfigure labels")


def _table(
    number: int,
    artifact_id: str,
    label: str,
    profile: ArtifactProfile,
    artifact_class: ArtifactClass,
    section: str,
    dependencies: tuple[str, ...],
    stem: str,
) -> ArtifactSpec:
    return ArtifactSpec(
        artifact_id=artifact_id,
        kind="table",
        profile=profile,
        artifact_class=artifact_class,
        draft_label=label,
        manuscript_section=section,
        generator_module=f"program_v04.{profile}.artifacts.generate_table{number:02d}",
        result_dependencies=dependencies,
        output_paths=(
            f"output/{profile}/tables/{stem}.csv",
            f"output/{profile}/tables/{stem}.tex",
        ),
    )


def _figure(
    number: int,
    artifact_id: str,
    label: str,
    profile: ArtifactProfile,
    artifact_class: ArtifactClass,
    section: str,
    dependencies: tuple[str, ...],
    outputs: tuple[str, ...],
    *,
    subfigure_labels: tuple[str, ...] = (),
) -> ArtifactSpec:
    return ArtifactSpec(
        artifact_id=artifact_id,
        kind="figure",
        profile=profile,
        artifact_class=artifact_class,
        draft_label=label,
        manuscript_section=section,
        generator_module=f"program_v04.{profile}.artifacts.generate_figure{number:02d}",
        result_dependencies=dependencies,
        output_paths=outputs,
        subfigure_labels=subfigure_labels,
    )


ARTIFACTS = (
    _figure(1, "fig_single_binary_examples", "fig:single_binary_examples", "main", "computational", "Illustrative Example", ("results/main/section2/signal_panel.parquet", "results/main/section2/summary.parquet", "results/main/section2/metadata.json"), ("output/main/figures/section2_binary_signals.pdf",)),
    _table(1, "tab_single_binary_summary", "tab:single_binary_summary", "main", "computational", "Illustrative Example", ("results/main/section2/summary.parquet", "results/main/section2/metadata.json"), "table01_single_binary_summary"),
    _table(2, "tab_zt_vs_xt", "tab:zt_vs_xt", "main", "computational", "Main Empirical Findings", ("results/shared/revised_main/metrics.parquet", "results/shared/revised_main/forecasts.parquet", "results/shared/revised_extended/metrics.parquet", "results/shared/revised_extended/forecasts.parquet", "results/main/revised_probit_spread/metrics.parquet"), "table02_zt_vs_benchmarks"),
    _table(3, "tab_aggregation_comparison", "tab:aggregation_comparison", "main", "computational", "Main Empirical Findings", ("results/shared/revised_main/metrics.parquet", "results/main/revised_parameter_free/metrics.parquet"), "table03_aggregation_comparison"),
    _table(4, "tab_encompassing_tests", "tab:encompassing_tests", "main", "computational", "Main Empirical Findings", ("results/shared/revised_main/forecasts.parquet", "results/shared/revised_extended/forecasts.parquet"), "table04_encompassing_tests"),
    _figure(2, "fig_main_probability_plots", "fig:main_probability_plots", "main", "computational", "Understanding Drivers", ("results/shared/revised_main/forecasts.parquet",), ("output/main/figures/Zt_vs_Xt_h3.pdf", "output/main/figures/Zt_vs_Xt_h6.pdf", "output/main/figures/Zt_vs_Xt_h12.pdf"), subfigure_labels=("fig:h3_prob_plot", "fig:h6_prob_plot", "fig:h12_prob_plot")),
    _table(5, "tab_sse_decomposition_h3", "tab:sse_decomposition_h3", "main", "computational", "Understanding Drivers", ("results/shared/revised_main/metrics.parquet",), "table05_mse_decomposition_h3"),
    _figure(3, "fig_disagreement_h3", "fig:disagreement_h3", "main", "computational", "Understanding Drivers", ("results/shared/revised_main/forecasts.parquet",), ("output/main/figures/forecast_disagreement_h3.pdf",)),
    _table(6, "tab_feature_importance", "tab:feature_importance", "main", "computational", "Understanding Drivers", ("results/shared/revised_main/importances.parquet",), "table06_feature_importance"),
    _table(7, "tab_sector_contribution", "tab:sector_contribution", "main", "computational", "Understanding Drivers", ("results/shared/revised_main/importances.parquet",), "table07_sector_contribution"),
    _table(8, "tab_zt_vs_xt_roc", "tab:zt_vs_xt_roc", "appendix", "computational", "Additional Metrics", ("results/shared/revised_main/metrics.parquet",), "table08_zt_vs_xt_roc"),
    _table(9, "tab_robustness_thresholds", "tab:robustness_thresholds", "appendix", "computational", "Robustness checks", ("results/shared/revised_main/metrics.parquet", "results/appendix/revised_threshold/metrics.parquet"), "table09_robustness_thresholds"),
    _table(10, "tab_robustness_ma", "tab:robustness_ma", "appendix", "computational", "Robustness checks", ("results/shared/revised_main/metrics.parquet", "results/appendix/revised_moving_average/metrics.parquet"), "table10_robustness_ma"),
    _table(11, "tab_extended_models", "tab:extended_models", "appendix", "computational", "Robustness checks", ("results/shared/revised_extended/metrics.parquet",), "table11_extended_models"),
    _table(12, "tab_zt_vs_xt_unlagged", "tab:zt_vs_xt_unlagged", "appendix", "computational", "Robustness checks", ("results/shared/revised_main/metrics.parquet", "results/appendix/revised_nolag/metrics.parquet"), "table12_zt_vs_xt_unlagged"),
    _table(13, "tab_subset", "tab:subset", "appendix", "computational", "Robustness checks", ("results/shared/revised_main/metrics.parquet", "results/appendix/revised_parsimonious/metrics.parquet"), "table13_subset"),
    _table(14, "tab_subset_extra", "tab:subset_extra", "appendix", "computational", "Robustness checks", ("results/shared/revised_main/metrics.parquet", "results/appendix/revised_alternative_subsets/metrics.parquet"), "table14_subset_extra"),
    _table(15, "tab_realtime", "tab:realtime", "appendix", "computational", "Robustness checks", ("results/appendix/realtime_main/metrics.parquet",), "table15_realtime"),
    _table(16, "tab_sampling_uncertainty", "tab:sampling_uncertainty", "appendix", "computational", "Robustness checks", ("results/appendix/derived/bootstrap.parquet",), "table16_sampling_uncertainty"),
    _table(17, "tab_pre2020_robustness", "tab:pre2020_robustness", "appendix", "computational", "Robustness checks", ("results/appendix/derived/pre2020.parquet",), "table17_pre2020_robustness"),
    _figure(4, "fig_cumulative_brier", "fig:cumulative_brier", "appendix", "computational", "Robustness checks", ("results/appendix/derived/cumulative_brier.parquet",), ("output/appendix/figures/cumulative_brier_differentials.pdf",)),
    _table(18, "tab_asymmetric_loss", "tab:asymmetric_loss", "appendix", "computational", "Robustness checks", ("results/appendix/decision_wealth/summary.parquet", "results/appendix/decision_wealth/wealth_paths.parquet"), "table18_asymmetric_loss"),
    _table(19, "tab_expanding_threshold", "tab:expanding_threshold", "appendix", "computational", "Robustness checks", ("results/shared/revised_main/metrics.parquet", "results/shared/revised_main/metadata.json", "results/appendix/revised_expanding_threshold/metrics.parquet", "results/appendix/revised_expanding_threshold/metadata.json"), "table19_expanding_threshold"),
    _table(20, "tab_mc_tailrisk", "tab:mc_tailrisk", "appendix", "simulation", "Robustness checks", ("results/appendix/simulation/metrics.parquet",), "table20_mc_tailrisk"),
    _table(21, "tab_lit_survey", "tab:lit_survey", "appendix", "documentary", "Predictor survey", ("data/documentary/literature_survey_ijf.csv",), "table21_lit_survey"),
    _table(22, "tab_lit_survey_nonijf", "tab:lit_survey_nonijf", "appendix", "documentary", "Predictor survey", ("data/documentary/literature_survey_nonijf.csv",), "table22_lit_survey_nonijf"),
)


@dataclass(frozen=True)
class ManuscriptFloat:
    kind: ArtifactKind
    number: int
    labels: tuple[str, ...]


_BEGIN_RE = re.compile(r"\\begin\{(table|figure)\}(?:\[[^]]*\])?")
_END_RE = re.compile(r"\\end\{(table|figure)\}")
_LABEL_RE = re.compile(r"\\label\{((?:tab|fig):[^}]+)\}")


def _without_comment(line: str) -> str:
    """Drop an unescaped LaTeX comment from one source line."""
    for index, character in enumerate(line):
        if character == "%" and (index == 0 or line[index - 1] != "\\"):
            return line[:index]
    return line


def parse_numbered_floats(path: Path = MANUSCRIPT_PATH) -> tuple[ManuscriptFloat, ...]:
    """Parse outer numbered table/figure environments in source order."""
    counters: dict[ArtifactKind, int] = {"table": 0, "figure": 0}
    active_kind: ArtifactKind | None = None
    active_labels: list[str] = []
    floats: list[ManuscriptFloat] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _without_comment(raw_line)
        begin = _BEGIN_RE.search(line)
        if begin:
            kind = begin.group(1)
            if active_kind is not None:
                raise ValueError(f"Nested numbered float is unsupported in {path}: {raw_line!r}")
            active_kind = kind  # type: ignore[assignment]
            counters[active_kind] += 1
            active_labels = []
        if active_kind is not None:
            active_labels.extend(_LABEL_RE.findall(line))
        end = _END_RE.search(line)
        if end and active_kind is not None:
            if end.group(1) != active_kind:
                raise ValueError(f"Mismatched float terminator in {path}: {raw_line!r}")
            floats.append(ManuscriptFloat(active_kind, counters[active_kind], tuple(active_labels)))
            active_kind = None
            active_labels = []
    if active_kind is not None:
        raise ValueError(f"Unclosed {active_kind} environment in {path}")
    return tuple(floats)


def artifact_registry(profile: ArtifactProfile | None = None) -> tuple[ArtifactSpec, ...]:
    selected = ARTIFACTS if profile is None else tuple(spec for spec in ARTIFACTS if spec.profile == profile)
    for spec in selected:
        spec.validate()
    return selected


def validate_artifact_registry(
    path: Path = MANUSCRIPT_PATH,
    *,
    profile: ArtifactProfile | None = None,
) -> dict[str, int]:
    """Validate the complete registry or one independently distributable profile."""
    registry = artifact_registry(profile)
    ids = [spec.artifact_id for spec in registry]
    labels = [spec.draft_label for spec in registry]
    generators = [spec.generator_module for spec in registry]
    if len(set(ids)) != len(ids) or len(set(labels)) != len(labels) or len(set(generators)) != len(generators):
        raise ValueError("Artifact registry ids, primary labels, and generators must be unique")

    floats = parse_numbered_floats(path)
    manuscript_primary: dict[str, ManuscriptFloat] = {}
    manuscript_all_labels: set[str] = set()
    for item in floats:
        matching = [label for label in item.labels if label.startswith("tab:" if item.kind == "table" else "fig:")]
        manuscript_all_labels.update(matching)
        primary = [label for label in matching if label in labels]
        if profile is not None and not primary:
            continue
        if len(primary) != 1:
            raise ValueError(
                f"Manuscript {item.kind} {item.number} must have exactly one registered primary label; "
                f"labels={matching}, registered_matches={primary}"
            )
        manuscript_primary[primary[0]] = item

    if set(labels) != set(manuscript_primary):
        missing = sorted(set(labels) - set(manuscript_primary))
        extra = sorted(set(manuscript_primary) - set(labels))
        raise ValueError(f"Artifact registry/manuscript mismatch: missing={missing}, extra={extra}")

    for spec in registry:
        item = manuscript_primary[spec.draft_label]
        expected_module = f"program_v04.{spec.profile}.artifacts.generate_{spec.kind}{item.number:02d}"
        if spec.generator_module != expected_module:
            raise ValueError(
                f"{spec.artifact_id}: manuscript number is {item.number}, so generator must be "
                f"{expected_module}; found {spec.generator_module}"
            )
        observed_subfigures = tuple(label for label in item.labels if label != spec.draft_label)
        if observed_subfigures != spec.subfigure_labels:
            raise ValueError(
                f"{spec.artifact_id}: subfigure-label mismatch; expected {spec.subfigure_labels}, "
                f"found {observed_subfigures}"
            )
    return {
        "artifacts": len(registry),
        "main": sum(spec.profile == "main" for spec in registry),
        "appendix": sum(spec.profile == "appendix" for spec in registry),
        "tables": sum(spec.kind == "table" for spec in registry),
        "figures": sum(spec.kind == "figure" for spec in registry),
        "manuscript_labels": (
            len(manuscript_all_labels)
            if profile is None
            else sum(1 + len(spec.subfigure_labels) for spec in registry)
        ),
    }


def artifact_by_id(artifact_id: str) -> ArtifactSpec:
    matches = [spec for spec in artifact_registry() if spec.artifact_id == artifact_id]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate artifact id: {artifact_id}")
    return matches[0]


def artifact_cli_spec(
    artifact_id: str,
    argv: list[str] | None = None,
) -> ArtifactSpec:
    """Parse a no-option artifact CLI and display its complete dependency contract."""
    spec = artifact_by_id(artifact_id)
    compute_module = (
        "program_v04.main.compute_main"
        if spec.profile == "main"
        else "program_v04.appendix.compute_appendix"
    )
    manuscript_number = int(spec.generator_module.rsplit("generate_", 1)[1][-2:])
    manuscript_name = f"{spec.kind.title()} {manuscript_number}"
    dependency_lines = "\n".join(f"  - {value}" for value in spec.result_dependencies)
    output_lines = "\n".join(f"  - {value}" for value in spec.output_paths)
    parser = argparse.ArgumentParser(
        description=(
            f"Generate manuscript {manuscript_name} {spec.draft_label} "
            f"({spec.artifact_id})."
        ),
        epilog=(
            "Required dependencies:\n"
            f"{dependency_lines}\n\n"
            "Output paths:\n"
            f"{output_lines}\n\n"
            "To compute a missing result dependency, run:\n"
            f"  python -m {compute_module} --input-mode public\n"
            "A missing versioned data dependency must be restored from the "
            "release package; compute commands do not recreate source data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args(argv)
    return spec
