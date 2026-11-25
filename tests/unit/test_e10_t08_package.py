"""E10-T08: the extracted package is named `hsmmkit` and carries no application or provider names."""

from __future__ import annotations

import importlib.metadata as md
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("regimekit", "yfinance", "fredapi", "pandas_datareader")


def test_e10_t08_import_pulls_in_no_application_or_provider_module() -> None:
    """Importing hsmmkit loads no application or provider module."""
    code = (
        "import hsmmkit, sys; "
        f"bad=[m for m in sys.modules if m.startswith({FORBIDDEN!r})]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_e10_t08_metadata_names_hsmmkit() -> None:
    """Distribution metadata and console script are named hsmmkit."""
    meta = md.metadata("hsmmkit")
    assert meta["Name"] == "hsmmkit"
    assert meta["Version"] == "0.1.0"
    assert not any("regimekit" in r.lower() for r in (md.requires("hsmmkit") or []))
    scripts = {
        e.name for e in md.entry_points(group="console_scripts") if e.dist and e.dist.name == "hsmmkit"
    }
    assert scripts == {"hsmmkit"}


def test_e10_t08_error_and_env_names() -> None:
    """The error hierarchy is renamed."""
    import hsmmkit
    from hsmmkit.utils import errors

    assert issubclass(errors.ConfigError, errors.HsmmkitError)
    assert not hasattr(errors, "RegimekitError")
    assert hsmmkit.__name__ == "hsmmkit"


def test_e10_t08_no_regimekit_strings_in_sources_and_guides() -> None:
    """Sources and guides mention regimekit only as the example extension."""
    pattern = re.compile("regimekit", re.IGNORECASE)
    hits: list[str] = []
    for base in (ROOT / "src", ROOT / "docs" / "guide"):
        for p in base.rglob("*"):
            if p.suffix not in {".py", ".md", ".yaml", ".yml", ".toml", ".j2"} or not p.is_file():
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if (
                    pattern.search(line)
                    and "example extension" not in line.lower()
                    and "regimekit" not in _ok(line)
                ):
                    hits.append(f"{p.relative_to(ROOT)}:{n}")
    assert not hits, hits


def _ok(line: str) -> str:
    """Lines that merely point at the downstream application (the allowed 'example extension' mention)."""
    return "regimekit" if "example extension" in line.lower() else ""


@pytest.mark.parametrize("name", ["HSMMKIT_TYPECHECK", "HSMMKIT_RECOVERY_T"])
def test_e10_t08_renamed_env_vars_documented(name: str) -> None:
    """Renamed environment variables are documented and used."""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8") + (ROOT / "noxfile.py").read_text(
        encoding="utf-8"
    )
    assert name in text


def test_e10_t08_wheel_imports_without_dev_dependencies() -> None:
    """`beartype` is a dev dependency: importing hsmmkit must not need it (wheel smoke install)."""
    code = (
        "import sys; sys.modules['beartype'] = None; import hsmmkit, hsmmkit.utils.typing as t; "
        "f = t.typechecked(lambda x: x); assert f(1) == 1"
    )
    env = {k: v for k, v in os.environ.items() if k != "HSMMKIT_TYPECHECK"}
    subprocess.run([sys.executable, "-I", "-c", code], check=True, env=env)
