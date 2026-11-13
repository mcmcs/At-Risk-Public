"""Build deterministic, physically pruned replication release profiles.

The main-only profile is constructed from the transitive internal import graph
of the main computation, artifact, and validation entry points.  It is not a
full source copy with an appendix directory hidden at runtime.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import io
import json
import os
import site
import shutil
import subprocess
import sys
import tempfile
import venv
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
ReleaseProfile = Literal["main-only", "full"]
ReleaseDataTreatment = Literal["public-no-data", "internal-bundled-data"]

RELEASE_DOCUMENTS = frozenset(
    {
        "ARTIFACTS.md",
        "artifact_manifest.csv",
    }
)

MAIN_ROOT_MODULES = (
    "program_v04",
    "program_v04.download_target",
    "program_v04.main",
    "program_v04.main.compute_main",
    "program_v04.main.run_main",
    "program_v04.main.validate_main",
)
MAIN_DECISION_IDS = frozenset(
    {
        "RF-004",
        "RF-007",
        "RF-009",
        "RF-010",
        "RF-011",
        "RF-012",
        "RF-013",
        "RF-014",
        "RF-015",
        "RF-016",
        "RF-017",
        "RF-021",
        "RF-022",
        "RF-025",
    }
)
APPENDIX_ONLY_MODULES = frozenset(
    {
        "program_v04.replication.appendix_paper",
        "program_v04.replication.realtime",
        "program_v04.replication.robustness",
        "program_v04.replication.simulation",
    }
)
MAIN_ARTIFACT_MODULES = tuple(
    f"program_v04.main.artifacts.generate_{kind}{number:02d}"
    for kind, numbers in (("figure", (1, 2, 3)), ("table", range(1, 8)))
    for number in numbers
)
FULL_ARTIFACT_MODULES = (
    *MAIN_ARTIFACT_MODULES,
    "program_v04.appendix.artifacts.generate_figure04",
    *(
        f"program_v04.appendix.artifacts.generate_table{number:02d}"
        for number in range(8, 23)
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_for_path(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("program_v04", *parts))


def _path_for_module(module: str) -> Path | None:
    if module == "program_v04":
        return PACKAGE_ROOT / "__init__.py"
    if not module.startswith("program_v04."):
        return None
    relative = Path(*module.split(".")[1:])
    module_path = PACKAGE_ROOT / relative.with_suffix(".py")
    if module_path.is_file():
        return module_path
    package_path = PACKAGE_ROOT / relative / "__init__.py"
    return package_path if package_path.is_file() else None


def _internal_imports(path: Path) -> set[str]:
    """Resolve static program_v04 imports made by one source module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = _module_for_path(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = "." * node.level + (node.module or "")
                try:
                    base = __import__("importlib.util").util.resolve_name(
                        relative, package
                    )
                except (ImportError, ValueError):
                    continue
            else:
                base = node.module or ""
            names = [base]
            names.extend(
                f"{base}.{alias.name}" for alias in node.names if alias.name != "*"
            )
        else:
            continue
        for name in names:
            if name.startswith("program_v04") and _path_for_module(name) is not None:
                result.add(name)
    return result


def main_import_closure() -> tuple[str, ...]:
    """Return the exact internal source closure of all public main entry points."""
    pending = [*MAIN_ROOT_MODULES, *MAIN_ARTIFACT_MODULES]
    seen: set[str] = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        path = _path_for_module(module)
        if path is None:
            raise FileNotFoundError(f"Main release root module is absent: {module}")
        seen.add(module)
        pending.extend(sorted(_internal_imports(path) - seen))
        parts = module.split(".")
        for stop in range(1, len(parts)):
            parent = ".".join(parts[:stop])
            if parent.startswith("program_v04") and _path_for_module(parent):
                pending.append(parent)
    forbidden = sorted(
        module
        for module in seen
        if module == "program_v04.appendix"
        or module.startswith("program_v04.appendix.")
        or module in APPENDIX_ONLY_MODULES
    )
    if forbidden:
        raise RuntimeError(
            "Main import graph reaches appendix-only production modules: "
            + ", ".join(forbidden)
        )
    return tuple(sorted(seen))


def _replace_node(source: str, node: ast.AST, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
    end = sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
    return source[:start] + replacement + source[end:]


def _assignment(source: str, name: str) -> ast.Assign:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one assignment to {name}; found {len(matches)}")
    return matches[0]


def _call_text_assignment(
    source: str,
    name: str,
    *,
    keep,
    rewrite=None,
) -> str:
    assignment = _assignment(source, name)
    if not isinstance(assignment.value, (ast.Tuple, ast.List)):
        raise TypeError(f"{name} must be a tuple/list literal")
    values = []
    for value in assignment.value.elts:
        if keep(value):
            values.append(rewrite(copy.deepcopy(value)) if rewrite else value)
    body = "\n".join(f"    {ast.unparse(value)}," for value in values)
    replacement = f"{name} = (\n{body}\n)"
    return _replace_node(source, assignment, replacement)


def _call_argument(value: ast.AST, position: int) -> str:
    if not isinstance(value, ast.Call) or len(value.args) <= position:
        raise TypeError("Expected a call with positional registry arguments")
    argument = value.args[position]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        raise TypeError("Expected a literal string registry argument")
    return argument.value


def _prune_config(source: str) -> str:
    source = _call_text_assignment(
        source,
        "METHOD_DECISIONS",
        keep=lambda value: _call_argument(value, 0) in MAIN_DECISION_IDS,
    )

    def keep_family(value: ast.AST) -> bool:
        return _call_argument(value, 2) in {"main", "shared"}

    def rewrite_family(value: ast.AST) -> ast.AST:
        if _call_argument(value, 2) == "shared":
            assert isinstance(value, ast.Call)
            value.args[5] = ast.Tuple(
                elts=[ast.Constant(value="main")], ctx=ast.Load()
            )
            ast.fix_missing_locations(value)
        return value

    source = _call_text_assignment(
        source,
        "RESULT_FAMILIES",
        keep=keep_family,
        rewrite=rewrite_family,
    )
    source = source.replace(
        'if spec.owner == "shared" and set(spec.consumers) != {"main", "appendix"}:',
        'if spec.owner == "shared" and set(spec.consumers) != {"main"}:',
    )
    for name in (
        "NOLAG_PAIRS",
        "PARSIMONIOUS_PAIRS",
        "REALTIME_MAIN_PAIRS",
        "REALTIME_EXTENDED_PAIRS",
    ):
        node = _assignment(source, name)
        source = _replace_node(source, node, "")
    return source


def _prune_artifacts(source: str) -> str:
    source = _call_text_assignment(
        source,
        "ARTIFACTS",
        keep=lambda value: _call_argument(value, 3) == "main",
    )
    return source.replace(
        'MANUSCRIPT_PATH = REPO_ROOT / "draft" / "main.tex"',
        'MANUSCRIPT_PATH = PACKAGE_ROOT / "docs" / "manuscript.tex"',
    )


def _release_artifacts_source(source: str) -> str:
    return source.replace(
        'MANUSCRIPT_PATH = REPO_ROOT / "draft" / "main.tex"',
        'MANUSCRIPT_PATH = PACKAGE_ROOT / "docs" / "manuscript.tex"',
    )


def _main_pyproject() -> str:
    dependencies = [
        "joblib==1.5.3",
        "matplotlib==3.10.8",
        "numpy==2.4.3",
        "pandas==3.0.1",
        "pyarrow==23.0.1",
        "scikit-learn==1.8.0",
        "scipy==1.17.1",
        "seaborn==0.13.2",
        "statsmodels==0.14.6",
        "tqdm==4.67.3",
        "xgboost==3.2.0",
    ]
    dependency_lines = "\n".join(f'  "{value}",' for value in dependencies)
    return f'''[build-system]
requires = ["setuptools==80.9.0", "wheel==0.45.1"]
build-backend = "setuptools.build_meta"

[project]
name = "at-risk-replication"
version = "0.4.0.dev0"
description = "Main-paper replication for At-Risk Transformation for U.S. Recession Prediction"
readme = "README.md"
requires-python = "==3.13.*"
dependencies = [
{dependency_lines}
]

[project.scripts]
at-risk-main = "program_v04.main.run_main:main"
at-risk-main-compute = "program_v04.main.compute_main:main"
at-risk-main-validate = "program_v04.main.validate_main:main"

[tool.setuptools]
package-dir = {{program_v04 = "."}}
packages = [
  "program_v04",
  "program_v04.replication",
  "program_v04.main",
  "program_v04.main.artifacts",
]
include-package-data = false

[tool.setuptools.package-data]
program_v04 = [
  "requirements.lock",
  "data/*.csv",
  "data/*.md",
  "docs/*.csv",
  "docs/*.md",
  "docs/*.tex",
  "docs/*.pdf",
]
'''


def _copy(path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_main_noncode(output: Path) -> None:
    for relative in ("requirements.lock", "uv.lock"):
        _copy(PACKAGE_ROOT / relative, output / relative)
    for path in sorted(PACKAGE_ROOT.glob("docs/*")):
        if path.is_file() and path.name in RELEASE_DOCUMENTS - {"artifact_manifest.csv"}:
            _copy(path, output / path.relative_to(PACKAGE_ROOT))
    with (PACKAGE_ROOT / "docs" / "artifact_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "paper_location" not in reader.fieldnames:
            raise ValueError("Artifact manifest has no paper_location column")
        rows = [row for row in reader if row["paper_location"] == "main"]
        if len(rows) != 10:
            raise ValueError(f"Expected ten main artifact-manifest rows; found {len(rows)}")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _write(output / "docs" / "artifact_manifest.csv", buffer.getvalue())
    with (PACKAGE_ROOT / "data" / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "profile" not in reader.fieldnames:
            raise ValueError("Input manifest has no profile column")
        rows = [row for row in reader if row["profile"] in {"main", "shared"}]
        if len(rows) != 3:
            raise ValueError(f"Expected three main/shared input rows; found {len(rows)}")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _write(output / "data" / "manifest.csv", buffer.getvalue())
    _copy(REPO_ROOT / "draft" / "main.tex", output / "docs" / "manuscript.tex")


def _main_release_test() -> str:
    modules = ",\n    ".join(repr(module) for module in MAIN_ARTIFACT_MODULES)
    return f'''from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import program_v04
from program_v04.replication.artifacts import artifact_registry, validate_artifact_registry
from program_v04.replication.config import FAMILY_REGISTRY, assert_canonical_generation_allowed


def main() -> None:
    installed_root = Path(sys.prefix).resolve()
    assert Path(program_v04.__file__).resolve().is_relative_to(installed_root)
    assert all(Path(path).resolve().is_relative_to(installed_root) for path in program_v04.__path__)
    specs = artifact_registry()
    assert len(specs) == 10
    assert all(spec.profile == "main" for spec in specs)
    assert all("appendix" not in value for spec in specs for value in (*spec.result_dependencies, *spec.output_paths, spec.generator_module))
    summary = validate_artifact_registry(profile="main")
    assert summary == {{"artifacts": 10, "main": 10, "appendix": 0, "tables": 7, "figures": 3, "manuscript_labels": 13}}
    assert len(FAMILY_REGISTRY) == 4
    assert all(spec.owner in {{"main", "shared"}} for spec in FAMILY_REGISTRY.values())
    assert all(spec.consumers == ("main",) for spec in FAMILY_REGISTRY.values())
    assert all("appendix" not in spec.result_subdirectory for spec in FAMILY_REGISTRY.values())
    for module in (
        "program_v04.appendix",
        "program_v04.replication.appendix_paper",
        "program_v04.replication.realtime",
        "program_v04.replication.robustness",
        "program_v04.replication.simulation",
    ):
        try:
            found = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            found = None
        assert found is None, module
    for module in (
    {modules}
    ):
        imported = importlib.import_module(module)
        assert callable(imported.main)
        assert Path(imported.__file__).resolve().is_relative_to(installed_root)
    assert_canonical_generation_allowed()


if __name__ == "__main__":
    main()
'''


def _full_release_test() -> str:
    modules = ",\n    ".join(repr(module) for module in FULL_ARTIFACT_MODULES)
    return f'''from __future__ import annotations

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
    assert summary == {{"artifacts": 26, "main": 10, "appendix": 16, "tables": 22, "figures": 4, "manuscript_labels": 29}}
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
    {modules}
    ):
        imported = importlib.import_module(module)
        assert callable(imported.main), module
        assert Path(imported.__file__).resolve().is_relative_to(installed_root), module
    assert_canonical_generation_allowed()


if __name__ == "__main__":
    main()
'''


def _release_manifest(
    output: Path,
    profile: ReleaseProfile,
    modules: tuple[str, ...],
    data_treatment: ReleaseDataTreatment,
) -> None:
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "RELEASE_PROFILE.json"
    )
    payload = {
        "schema_version": 1,
        "profile": profile,
        "data_treatment": data_treatment,
        "modules": list(modules),
        "files": [
            {
                "path": str(path.relative_to(output)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    _write(
        output / "RELEASE_PROFILE.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _release_data_note(data_treatment: ReleaseDataTreatment) -> str:
    if data_treatment == "internal-bundled-data":
        return (
            "This internal journal-replication edition includes every empirical "
            "input at the exact path declared in `data/manifest.csv`; no separate "
            "download or placement step is required. The bundled snapshots are for "
            "confidential replication use and must be removed from any public copy."
        )
    return (
        "This public-facing edition intentionally excludes the empirical snapshots. "
        "Obtain the inputs described in `data/README.md` and place them at the exact "
        "`public_path` locations in `data/manifest.csv` before running computation. "
        "The USRECM target can be downloaded and verified with "
        "`python -m program_v04.download_target`. Other inputs still require "
        "the snapshots described in the manifest."
    )


def _render_reader_files(
    output: Path,
    profile: ReleaseProfile,
    data_treatment: ReleaseDataTreatment,
) -> None:
    readme_name = (
        "README_main_release.md" if profile == "main-only" else "README_full_release.md"
    )
    note = _release_data_note(data_treatment)
    templates = {
        output / "README.md": PACKAGE_ROOT / "docs" / readme_name,
        output / "data" / "README.md": PACKAGE_ROOT / "docs" / "DATA_README_release.md",
    }
    for target, template in templates.items():
        content = template.read_text(encoding="utf-8")
        if "{{DATA_NOTE}}" not in content:
            raise ValueError(f"Release template lacks DATA_NOTE marker: {template}")
        _write(target, content.replace("{{DATA_NOTE}}", note))


def _bundle_declared_inputs(output: Path) -> int:
    """Copy checksum-verified development snapshots to public package paths."""
    with (output / "data" / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    copied = 0
    for row in rows:
        source_path = (REPO_ROOT / row["development_source_path"]).resolve()
        target_path = output / row["public_path"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Declared internal input is missing: {source_path}")
        if source_path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Declared input size mismatch: {source_path}")
        if _sha256(source_path) != row["sha256"]:
            raise ValueError(f"Declared input checksum mismatch: {source_path}")
        _copy(source_path, target_path)
        copied += 1
    return copied


def _build_main(output: Path) -> tuple[str, ...]:
    modules = main_import_closure()
    for module in modules:
        source_path = _path_for_module(module)
        assert source_path is not None
        relative = source_path.relative_to(PACKAGE_ROOT)
        source = source_path.read_text(encoding="utf-8")
        if module == "program_v04.replication.config":
            source = _prune_config(source)
        elif module == "program_v04.replication.artifacts":
            source = _prune_artifacts(source)
        _write(output / relative, source)
    _copy_main_noncode(output)
    _write(output / "pyproject.toml", _main_pyproject())
    _write(output / "release_tests" / "test_main_release.py", _main_release_test())
    return modules


def _full_release_files() -> list[Path]:
    result: set[Path] = set()
    for pattern in (
        "*.py",
        "replication/*.py",
        "main/**/*.py",
        "appendix/**/*.py",
        "data/**/*.csv",
        "data/**/*.md",
    ):
        result.update(PACKAGE_ROOT.glob(pattern))
    result.update(PACKAGE_ROOT / "docs" / name for name in RELEASE_DOCUMENTS)
    for name in ("README.md", "requirements.lock", "pyproject.toml", "uv.lock"):
        result.add(PACKAGE_ROOT / name)
    return sorted(
        path
        for path in result
        if path.is_file()
        and "__pycache__" not in path.parts
        and (
            path.parent != PACKAGE_ROOT / "docs"
            or path.name in RELEASE_DOCUMENTS
        )
    )


def _build_full(output: Path) -> tuple[str, ...]:
    for path in _full_release_files():
        relative = path.relative_to(PACKAGE_ROOT)
        if relative == Path("replication/artifacts.py"):
            _write(
                output / relative,
                _release_artifacts_source(path.read_text(encoding="utf-8")),
            )
        else:
            _copy(path, output / relative)
    _copy(REPO_ROOT / "draft" / "main.tex", output / "docs" / "manuscript.tex")
    _write(output / "release_tests" / "test_full_release.py", _full_release_test())
    return tuple(
        sorted(
            _module_for_path(path)
            for path in _full_release_files()
            if path.suffix == ".py" and path.is_relative_to(PACKAGE_ROOT)
        )
    )


def validate_main_only_tree(output: Path) -> dict[str, int]:
    """Statically prove the physical and import-graph main-only boundary."""
    prohibited_paths = [
        output / "appendix",
        *(output / "replication" / f"{name}.py" for name in ("appendix_paper", "realtime", "robustness", "simulation")),
        output / "run_all.py",
        output / "data" / "documentary",
        output / "results" / "appendix",
        output / "output" / "appendix",
    ]
    present = [str(path.relative_to(output)) for path in prohibited_paths if path.exists()]
    if present:
        raise ValueError(f"Main-only release retains appendix paths: {present}")
    modules = {
        _module_for_candidate_path(path, output)
        for path in output.rglob("*.py")
        if "release_tests" not in path.parts
    }
    dangling: list[tuple[str, str]] = []
    forbidden: list[tuple[str, str]] = []
    for path in sorted(output.rglob("*.py")):
        if "release_tests" in path.parts:
            continue
        for imported in _candidate_internal_imports(path, output):
            if imported.startswith("program_v04.appendix") or imported in APPENDIX_ONLY_MODULES:
                forbidden.append((_module_for_candidate_path(path, output), imported))
            elif imported not in modules and not any(
                module.startswith(imported + ".") for module in modules
            ):
                dangling.append((_module_for_candidate_path(path, output), imported))
    if forbidden:
        raise ValueError(f"Main-only import graph reaches appendix: {forbidden}")
    if dangling:
        raise ValueError(f"Main-only import graph has missing internal modules: {dangling}")
    manifest_path = output / "docs" / "artifact_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 10 or any(row.get("paper_location") != "main" for row in rows):
        raise ValueError("Main-only artifact manifest retains non-main rows")
    with (output / "data" / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        input_rows = list(csv.DictReader(source))
    if len(input_rows) != 3 or any(
        row.get("profile") not in {"main", "shared"} for row in input_rows
    ):
        raise ValueError("Main-only input manifest retains non-main rows")
    readme = (output / "README.md").read_text(encoding="utf-8")
    prohibited_readme_text = (
        "program_v04.appendix",
        "Figure 4",
        "Tables 8",
        "full profile",
    )
    retained = [value for value in prohibited_readme_text if value in readme]
    if retained:
        raise ValueError(f"Main-only README retains full/appendix instructions: {retained}")
    return {
        "python_modules": len(modules),
        "files": sum(path.is_file() for path in output.rglob("*")),
    }


def _module_for_candidate_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("program_v04", *parts))


def _candidate_internal_imports(path: Path, root: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = _module_for_candidate_path(path, root)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = (
                __import__("importlib.util").util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
                if node.level
                else node.module or ""
            )
            names = [base]
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                relative = Path(*candidate.split(".")[1:])
                if (
                    (root / relative.with_suffix(".py")).is_file()
                    or (root / relative / "__init__.py").is_file()
                ):
                    names.append(candidate)
        else:
            continue
        imports.update(name for name in names if name.startswith("program_v04"))
    return imports


@contextmanager
def _installed_candidate(
    output: Path,
) -> Iterator[tuple[Path, Path, dict[str, str]]]:
    """Yield an isolated venv containing the candidate wheel over locked dependencies."""
    with tempfile.TemporaryDirectory(prefix="at_risk_release_verify_") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
        python = environment / "bin" / "python"
        dependency_sites = [
            Path(path)
            for path in site.getsitepackages()
            if Path(path).is_dir()
        ]
        if not dependency_sites:
            raise RuntimeError("No locked parent environment site-packages were found")
        nested_sites = sorted((environment / "lib").glob("python*/site-packages"))
        if len(nested_sites) != 1:
            raise RuntimeError(f"Expected one verifier site-packages path: {nested_sites}")
        _write(
            nested_sites[0] / "locked_dependency_layer.pth",
            "\n".join(str(path) for path in dependency_sites) + "\n",
        )
        uv_executable = shutil.which("uv")
        if uv_executable is None:
            raise RuntimeError("The documented uv executable is required to verify a release")
        subprocess.run(
            [
                uv_executable,
                "build",
                "--quiet",
                "--wheel",
                "--no-build-logs",
                "--no-create-gitignore",
                "--out-dir",
                str(root / "wheelhouse"),
                str(output),
            ],
            cwd=root,
            check=True,
        )
        wheels = sorted((root / "wheelhouse").glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one release wheel; found {wheels}")
        pip_environment = {
            **os.environ,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_WARN_SCRIPT_LOCATION": "1",
        }
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--ignore-installed",
                "--no-deps",
                str(wheels[0]),
            ],
            cwd=root,
            check=True,
            env=pip_environment,
        )
        yield root, python, pip_environment


def _install_editable(
    output: Path,
    *,
    root: Path,
    python: Path,
    pip_environment: dict[str, str],
) -> None:
    """Expose source-placed public inputs for a post-decision workflow run."""
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--ignore-installed",
            "--no-build-isolation",
            "--no-deps",
            "--editable",
            str(output),
        ],
        cwd=root,
        check=True,
        env=pip_environment,
    )


def verify_main_install(output: Path, *, run_workflow: bool, input_mode: str) -> None:
    """Install the pruned candidate and exercise every main public surface."""
    validate_main_only_tree(output)
    with _installed_candidate(output) as (root, python, pip_environment):
        subprocess.run(
            [str(python), str(output / "release_tests" / "test_main_release.py")],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for module in (
            "program_v04.main.compute_main",
            "program_v04.main.run_main",
            "program_v04.main.validate_main",
            *MAIN_ARTIFACT_MODULES,
        ):
            subprocess.run(
                [str(python), "-m", module, "--help"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        if run_workflow:
            _install_editable(
                output,
                root=root,
                python=python,
                pip_environment=pip_environment,
            )
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "program_v04.main.run_main",
                    "--input-mode",
                    input_mode,
                ],
                cwd=root,
                check=True,
            )


def verify_full_install(output: Path, *, run_workflow: bool, input_mode: str) -> None:
    """Install the full candidate and exercise every public package surface."""
    test_path = output / "release_tests" / "test_full_release.py"
    if not test_path.is_file():
        raise FileNotFoundError(f"Full release test is missing: {test_path}")
    with _installed_candidate(output) as (root, python, pip_environment):
        subprocess.run(
            [str(python), str(test_path)],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for module in (
            "program_v04.main.compute_main",
            "program_v04.main.run_main",
            "program_v04.main.validate_main",
            "program_v04.appendix.compute_appendix",
            "program_v04.appendix.run_appendix",
            "program_v04.appendix.validate_appendix",
            "program_v04.run_all",
            "program_v04.validate_all",
            *FULL_ARTIFACT_MODULES,
        ):
            subprocess.run(
                [str(python), "-m", module, "--help"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        if run_workflow:
            _install_editable(
                output,
                root=root,
                python=python,
                pip_environment=pip_environment,
            )
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "program_v04.run_all",
                    "--input-mode",
                    input_mode,
                ],
                cwd=root,
                check=True,
            )


def build_release(
    profile: ReleaseProfile,
    output: Path,
    data_treatment: ReleaseDataTreatment = "public-no-data",
) -> dict[str, object]:
    """Construct one new release tree, refusing to overwrite any existing path."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Release output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        modules = _build_main(staging) if profile == "main-only" else _build_full(staging)
        _render_reader_files(staging, profile, data_treatment)
        bundled_inputs = (
            _bundle_declared_inputs(staging)
            if data_treatment == "internal-bundled-data"
            else 0
        )
        _release_manifest(staging, profile, modules, data_treatment)
        if profile == "main-only":
            validate_main_only_tree(staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "profile": profile,
        "output": str(output),
        "modules": len(modules),
        "files": sum(path.is_file() for path in output.rglob("*")),
        "data_treatment": data_treatment,
        "bundled_inputs": bundled_inputs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("main-only", "full"), required=True)
    parser.add_argument(
        "--data-treatment",
        choices=("public-no-data", "internal-bundled-data"),
        default="public-no-data",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Install the selected profile in a temporary venv and test every entry point.",
    )
    parser.add_argument(
        "--run-workflow",
        action="store_true",
        help="After verification, execute the full canonical main workflow.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("public", "development"),
        default="public",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_workflow and not args.verify:
        raise SystemExit("--run-workflow requires --verify")
    summary = build_release(args.profile, args.output, args.data_treatment)
    if args.verify:
        verifier = verify_main_install if args.profile == "main-only" else verify_full_install
        verifier(
            args.output.resolve(),
            run_workflow=args.run_workflow,
            input_mode=args.input_mode,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
