"""E1-T01 AC4/AC5: layer rule and profile rule, enforced by an AST scan (overview §4.3).

Layer rule: `model, statespace, engine, estimation, inference` must not import anything from
`regimekit.data` except `regimekit.data.contracts`.

Profile rule: no core package (`config, data, model, statespace, engine, estimation, inference,
evaluation, persistence`) may import `regimekit.finance`; only `cli/` and `project.py` may.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "regimekit"

LAYER_RULE_PACKAGES = ("model", "statespace", "engine", "estimation", "inference")
PROFILE_RULE_PACKAGES = (
    "config",
    "data",
    "model",
    "statespace",
    "engine",
    "estimation",
    "inference",
    "evaluation",
    "persistence",
)


def _iter_python_files(*package_names: str) -> list[Path]:
    files = []
    for name in package_names:
        files.extend((SRC / name).rglob("*.py"))
    return files


def _find_disallowed_imports(
    files: list[Path], *, disallowed_prefix: str, allowed_exact: str | None = None
) -> list[str]:
    parent, _, child = disallowed_prefix.rpartition(".")
    violations = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == allowed_exact:
                        continue
                    if alias.name == disallowed_prefix or alias.name.startswith(disallowed_prefix + "."):
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == allowed_exact:
                    continue
                if module == disallowed_prefix:
                    for alias in node.names:
                        if allowed_exact is not None and f"{module}.{alias.name}" == allowed_exact:
                            continue
                        violations.append(f"{path}: from {module} import {alias.name}")
                elif module.startswith(disallowed_prefix + "."):
                    violations.append(f"{path}: from {module} import ...")
                elif module == parent:
                    # e.g. `from regimekit import data` importing the disallowed package itself
                    for alias in node.names:
                        if alias.name == child:
                            violations.append(f"{path}: from {module} import {alias.name}")
    return violations


def test_e1_t01_layer_rule_no_data_imports_beyond_contracts() -> None:
    """`model/statespace/engine/estimation/inference` only import `regimekit.data.contracts`."""
    files = _iter_python_files(*LAYER_RULE_PACKAGES)
    violations = _find_disallowed_imports(
        files, disallowed_prefix="regimekit.data", allowed_exact="regimekit.data.contracts"
    )
    assert not violations, "layer rule violated:\n" + "\n".join(violations)


def test_e1_t01_profile_rule_no_core_package_imports_finance() -> None:
    """No core package imports `regimekit.finance` (`cli/` and `project.py` are exempt)."""
    files = _iter_python_files(*PROFILE_RULE_PACKAGES)
    violations = _find_disallowed_imports(files, disallowed_prefix="regimekit.finance")
    assert not violations, "profile rule violated:\n" + "\n".join(violations)


def test_e1_t01_layer_rule_scanner_catches_a_synthetic_violation(tmp_path: Path) -> None:
    """Sanity check: the scanner actually flags a disallowed `regimekit.data.cache` import."""
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("from regimekit.data import cache\n")
    violations = _find_disallowed_imports(
        [bad_file], disallowed_prefix="regimekit.data", allowed_exact="regimekit.data.contracts"
    )
    assert violations


def test_e1_t01_profile_rule_scanner_catches_a_synthetic_violation(tmp_path: Path) -> None:
    """Sanity check: the scanner actually flags a disallowed `regimekit.finance` import."""
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("import regimekit.finance.providers\n")
    violations = _find_disallowed_imports([bad_file], disallowed_prefix="regimekit.finance")
    assert violations
