"""E10-T01: neutral-name scanner, finance-import rule and allow-lists (FR-G01, FR-G08)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.helpers.neutral_names import SRC, entry_of, scan_source, scan_tree

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NEUTRAL_ALLOWLIST = FIXTURES / "neutral_names_allowlist.txt"
IMPORT_ALLOWLIST = FIXTURES / "finance_import_allowlist.txt"
TAG = re.compile(r"E10-T(02b|03|04a|04b|05|06|07a|07b)")
SCOPE_PACKAGES = (
    "config",
    "data",
    "model",
    "statespace",
    "engine",
    "estimation",
    "inference",
    "evaluation",
    "persistence",
    "online",
    "selection",
    "cli",
    "utils",
)


def _read_allowlist(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entry, _, tag = line.partition("#")
        entries[entry.strip()] = tag.strip()
    return entries


def _terms(source: str) -> set[str]:
    return {term for _, term in scan_source(source)}


# --- scanner on synthetic snippets ---------------------------------------------------------------


def test_e10_t01_scanner_flags_identifiers() -> None:
    """Scanner flags identifiers."""
    assert _terms("def price(): ...") == {"price"}
    assert _terms("log_return = 1") == {"log_return"}
    assert _terms("class TickerSet: ...") == {"ticker"}
    assert _terms("f(bars_per_day=1)") == {"bars_per_day"}
    assert _terms("def f(annualized): ...") == {"annualiz"}


def test_e10_t01_scanner_flags_strings_comments_and_docstrings() -> None:
    """Scanner flags strings comments and docstrings."""
    assert _terms('x = "the sharpe ratio"') == {"sharpe"}
    assert _terms("x = 1  # trading days") == {"trading"}
    assert _terms('def f():\n    """Max drawdown of the path."""\n') == {"drawdown"}
    assert _terms("d = {'symbol': 1}") == {"symbol"}


def test_e10_t01_scanner_flags_literal_252() -> None:
    """Scanner flags literal 252."""
    assert _terms("x = 1 / 252") == {"252"}
    assert _terms("x = 253") == set()


def test_e10_t01_scanner_ignores_returns_section_and_keyword() -> None:
    """Scanner ignores returns section and keyword."""
    doc = 'def f():\n    """Summary.\n\n    Returns:\n        A value.\n    """\n    return 1\n'
    assert _terms(doc) == set()
    assert _terms('def f():\n    """Returns the mask."""\n') == set()
    assert _terms("def f(returns): ...") == {"returns"}


def test_e10_t01_scanner_flags_returns_as_a_noun() -> None:
    """The noun "returns" after a determiner or preposition is a hit; the verb is not."""
    for text in ("the returns", "a returns array", "(log) returns", "and returns", "whose returns"):
        assert _terms(f"x = {text!r}") == {"returns"}, text
    for text in ("it returns x", "Also returns y", "which returns z"):
        assert _terms(f"x = {text!r}") == set(), text


def test_e10_t01_scanner_matches_whole_words_only() -> None:
    """Scanner matches whole words only."""
    assert _terms("surprise = 1\npricey_level = 2") == set()
    assert _terms("symbolic = 1") == set()


def test_e10_t01_scanner_reports_line_and_path_for_engine_price(tmp_path: Path) -> None:
    """AC2: `def price(): ...` in engine/base.py is reported as `engine/base.py:<line>:price`."""
    original = (SRC / "engine" / "base.py").read_text(encoding="utf-8")
    lines = original.count("\n") + 2
    hits = [f"engine/base.py:{ln}:{t}" for ln, t in scan_source(original + "\ndef price(): ...\n")]
    assert f"engine/base.py:{lines}:price" in hits
    assert not [h for h in hits if h.endswith(":price") and not h.startswith(f"engine/base.py:{lines}:")]


# --- the real tree ---------------------------------------------------------------------------------


def test_e10_t01_every_hit_is_allow_listed_and_no_entry_is_stale() -> None:
    """AC3 (closed by E10-T07b): the core scope has no finance term at all, and there is no allow-list."""
    hits = [entry_of(h) for h in scan_tree()]
    assert not hits, "finance terms in core modules (fix the code):\n" + "\n".join(sorted(set(hits)))


def _finance_imports(path: Path, package: tuple[str, ...] | None = None) -> list[str]:
    pkg = list(package) if package is not None else ["hsmmkit", *path.relative_to(SRC).parts[:-1]]
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.startswith("hsmmkit.finance")]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)]
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            if module.startswith("hsmmkit.finance"):
                found.append(module)
            elif module == "hsmmkit":
                found += [f"hsmmkit.{a.name}" for a in node.names if a.name == "finance"]
    return found


def _scope_python_files() -> list[Path]:
    files = [f for name in SCOPE_PACKAGES for f in sorted((SRC / name).rglob("*.py"))]
    return [*files, SRC / "project.py", SRC / "__init__.py"]


def test_e10_t01_no_finance_import_beyond_the_allow_list() -> None:
    """The core scope (including `cli/`, `project.py`, `__init__.py`) imports `hsmmkit.finance` nowhere."""
    found = {
        f"{p.relative_to(SRC).as_posix()}:{m}" for p in _scope_python_files() for m in _finance_imports(p)
    }
    assert not found, "finance imports in the core: " + ", ".join(sorted(found))


def test_e10_t01_finance_import_scanner_catches_absolute_and_relative(tmp_path: Path) -> None:
    """Finance import scanner catches absolute and relative."""
    sample = tmp_path / "probe.py"
    sample.write_text(
        "import hsmmkit.finance.providers\nfrom ..finance import clocks\nfrom hsmmkit import finance\n"
    )
    found = _finance_imports(sample, ("hsmmkit", "cli"))
    assert found == ["hsmmkit.finance.providers", "hsmmkit.finance", "hsmmkit.finance"]


@pytest.mark.parametrize("name", ["neutral_names_allowlist.txt", "finance_import_allowlist.txt"])
def test_e10_t01_allow_list_files_exist(name: str) -> None:
    """E10-T07b deleted both allow-list files; they must not come back."""
    assert not (FIXTURES / name).exists()
