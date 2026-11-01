"""E8-T04: release tooling is configured (structure only; nothing is tagged, built or published here)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
_YAML = YAML(typ="safe")


def _workflow(name: str) -> dict[str, Any]:
    return _YAML.load(ROOT / ".github" / "workflows" / name)  # type: ignore[no-any-return]


def test_e8_t04_release_workflow_builds_on_a_version_tag_and_publishes_to_github() -> None:
    """AC2: a `v*` tag triggers `uv build` and a GitHub Release with the artifacts; PyPI is opt-in."""
    doc = _workflow("release.yml")
    assert doc["on"]["push"]["tags"] == ["v*"]
    build_steps = doc["jobs"]["build"]["steps"]
    assert any(s.get("run") == "uv build" for s in build_steps)
    assert any("dist" in str(s.get("with", {}).get("path", "")) for s in build_steps)
    release = doc["jobs"]["github-release"]
    assert release["permissions"]["contents"] == "write"
    assert "startsWith(github.ref, 'refs/tags/v')" in release["if"]
    assert any("gh release create" in s.get("run", "") for s in release["steps"])
    pypi = doc["jobs"]["pypi"]
    assert "publish_pypi" in pypi["if"]
    assert doc["on"]["workflow_dispatch"]["inputs"]["publish_pypi"]["default"] is False


def test_e8_t04_release_workflow_checks_the_tag_against_the_package_version() -> None:
    """A tag that disagrees with `project.version` fails before anything is built."""
    steps = _workflow("release.yml")["jobs"]["build"]["steps"]
    checks = [s for s in steps if "must match" in s.get("name", "")]
    assert checks
    assert "project" in checks[0]["run"]
    assert "github.ref_name" in checks[0]["run"]


def test_e8_t04_nightly_runs_the_tutorial_notebook() -> None:
    """The notebook (E8-T03 AC3) executes in CI through `nox -s examples`."""
    steps = _workflow("nightly.yml")["jobs"]["examples"]["steps"]
    assert any("nox -s examples" in s.get("run", "") for s in steps)


def test_e8_t04_semantic_release_is_configured_for_zero_ver_conventional_commits() -> None:
    """Conventional commits, `v{version}` tags, `0.y.z` kept until the factorial extension."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["semantic_release"]
    assert cfg["tag_format"] == "v{version}"
    assert cfg["commit_parser"] == "conventional"
    assert cfg["allow_zero_version"] is True
    assert cfg["major_on_zero"] is False
    assert cfg["version_toml"] == ["pyproject.toml:project.version"]


def test_e8_t04_towncrier_collects_the_fragments(tmp_path: Path) -> None:
    """A draft build in a scratch copy renders a fragment of each kind under its section, writing nothing.

    (The real fragments are consumed by `towncrier build` when a release is cut, so the test brings its own.)
    """
    shutil.copy(ROOT / "towncrier.toml", tmp_path / "towncrier.toml")
    (tmp_path / "changelog.d").mkdir()
    shutil.copy(ROOT / "changelog.d" / "_template.md.j2", tmp_path / "changelog.d" / "_template.md.j2")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n<!-- towncrier release notes start -->\n", encoding="utf-8"
    )
    (tmp_path / "changelog.d" / "E0-T01.feature.md").write_text("Add a thing.\n", encoding="utf-8")
    (tmp_path / "changelog.d" / "E0-T02.bugfix.md").write_text("Fix a thing.\n", encoding="utf-8")
    before = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    done = subprocess.run(
        ["towncrier", "build", "--draft", "--version", "9.9.9"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONUTF8="1"),
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "## 9.9.9" in done.stdout
    assert "### Features" in done.stdout
    assert "### Bug fixes" in done.stdout
    assert "- E0-T01: Add a thing." in done.stdout
    assert (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8") == before


def test_e8_t04_release_checklist_lists_the_four_success_criteria() -> None:
    """AC3: `docs/releases/0.1.0.md` has one row per section-1.4 criterion and records the v0.1.0 tag."""
    text = (ROOT / "docs" / "releases" / "0.1.0.md").read_text(encoding="utf-8")
    for number in ("| 1 |", "| 2 |", "| 3 |", "| 4 |"):
        assert number in text
    assert "**released.**" in text
    assert "`v0.1.0`" in text
    assert "weekly full run" in text.lower()
    assert text.count("actions/runs/") >= 4  # the CI runs that prove the criteria
    assert "pending" not in text.lower()
