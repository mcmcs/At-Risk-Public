from __future__ import annotations

import importlib
import sys
from pathlib import Path

import program_v04
from program_v04.replication.artifacts import artifact_registry, validate_artifact_registry
from program_v04.replication.config import FAMILY_REGISTRY, assert_canonical_generation_allowed


def main() -> None:
    installed_root = Path(sys.prefix).resolve()
    package_root = Path(program_v04.__file__).resolve().parent
    assert package_root.is_relative_to(installed_root)
    assert all(Path(path).resolve().is_relative_to(installed_root) for path in program_v04.__path__)
    specs = artifact_registry()
    assert len(specs) == 26
    assert sum(spec.profile == "main" for spec in specs) == 10
    assert sum(spec.profile == "appendix" for spec in specs) == 16
    summary = validate_artifact_registry()
    assert summary == {"artifacts": 26, "main": 10, "appendix": 16, "tables": 22, "figures": 4, "manuscript_labels": 29}
    assert len(FAMILY_REGISTRY) == 12
    for relative in (
        "docs/manuscript.tex",
        "docs/artifact_manifest.csv",
        "docs/ARTIFACTS.md",
        "data/documentary/README.md",
        "data/documentary/literature_survey_ijf.csv",
        "data/documentary/literature_survey_nonijf.csv",
    ):
        assert (package_root / relative).is_file(), relative
    for module in (
        "program_v04.main.compute_main",
        "program_v04.main.run_main",
        "program_v04.main.validate_main",
        "program_v04.appendix.compute_appendix",
        "program_v04.appendix.run_appendix",
        "program_v04.appendix.validate_appendix",
        "program_v04.run_all",
        "program_v04.validate_all",
    'program_v04.main.artifacts.generate_figure01',
    'program_v04.main.artifacts.generate_figure02',
    'program_v04.main.artifacts.generate_figure03',
    'program_v04.main.artifacts.generate_table01',
    'program_v04.main.artifacts.generate_table02',
    'program_v04.main.artifacts.generate_table03',
    'program_v04.main.artifacts.generate_table04',
    'program_v04.main.artifacts.generate_table05',
    'program_v04.main.artifacts.generate_table06',
    'program_v04.main.artifacts.generate_table07',
    'program_v04.appendix.artifacts.generate_figure04',
    'program_v04.appendix.artifacts.generate_table08',
    'program_v04.appendix.artifacts.generate_table09',
    'program_v04.appendix.artifacts.generate_table10',
    'program_v04.appendix.artifacts.generate_table11',
    'program_v04.appendix.artifacts.generate_table12',
    'program_v04.appendix.artifacts.generate_table13',
    'program_v04.appendix.artifacts.generate_table14',
    'program_v04.appendix.artifacts.generate_table15',
    'program_v04.appendix.artifacts.generate_table16',
    'program_v04.appendix.artifacts.generate_table17',
    'program_v04.appendix.artifacts.generate_table18',
    'program_v04.appendix.artifacts.generate_table19',
    'program_v04.appendix.artifacts.generate_table20',
    'program_v04.appendix.artifacts.generate_table21',
    'program_v04.appendix.artifacts.generate_table22'
    ):
        imported = importlib.import_module(module)
        assert callable(imported.main), module
        assert Path(imported.__file__).resolve().is_relative_to(installed_root), module
    assert_canonical_generation_allowed()


if __name__ == "__main__":
    main()
