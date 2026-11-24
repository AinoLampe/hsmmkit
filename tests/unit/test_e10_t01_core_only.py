"""E10-T01 AC4: the `HSMM_CORE_ONLY=1` finder blocks `regimekit.finance` and the provider packages."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNIPPET = (
    "import sys; sys.path.insert(0, 'tests'); import conftest\n"
    "try:\n    import {module}\nexcept ImportError as exc:\n    print('BLOCKED', exc)\n"
    "else:\n    print('IMPORTED')\n"
)


def _run(module: str, *, core_only: bool) -> str:
    env = {k: v for k, v in os.environ.items() if k != "HSMM_CORE_ONLY"}
    if core_only:
        env["HSMM_CORE_ONLY"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", SNIPPET.format(module=module)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "module",
    [
        "regimekit.finance",
        "regimekit.finance.providers",
        "yfinance",
        "fredapi",
        "pandas_datareader",
        "curl_cffi",
    ],
)
def test_e10_t01_core_only_blocks_finance_and_provider_packages(module: str) -> None:
    """Core only blocks finance and provider packages."""
    assert _run(module, core_only=True).startswith("BLOCKED finance blocked")


def test_e10_t01_without_the_flag_finance_imports() -> None:
    """Without the flag finance imports."""
    assert _run("regimekit.finance", core_only=False) == "IMPORTED"


def test_e10_t01_core_only_keeps_core_packages_importable() -> None:
    """Core only keeps core packages importable."""
    assert _run("regimekit.engine", core_only=True) == "IMPORTED"


def test_e10_t01_pr_workflow_has_a_core_only_job_that_uploads_failing_ids() -> None:
    """Pr workflow has a core only job that uploads failing ids."""
    from ruamel.yaml import YAML

    doc = YAML(typ="safe").load(ROOT / ".github" / "workflows" / "pr.yml")
    job = doc["jobs"]["core-only"]
    assert job["continue-on-error"] is True  # required only from E10-T07b
    assert any("nox -s core" in str(s.get("run", "")) for s in job["steps"])
    assert any(s.get("uses", "").startswith("actions/upload-artifact") for s in job["steps"])


def test_e10_t01_adr_0014_is_proposed() -> None:
    """Adr 0014 is proposed."""
    text = (ROOT / "docs" / "adr" / "0014-two-repo-split.md").read_text(encoding="utf-8")
    assert text.rstrip().splitlines()[-1].startswith("proposed")
    assert "## Status" in text
