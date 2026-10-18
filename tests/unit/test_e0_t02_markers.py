"""E0-T02: pytest markers are registered and excluded from the default run."""

import subprocess
import sys

import pytest


def test_e0_t02_markers_registered(pytestconfig: pytest.Config) -> None:
    """`slow`, `live`, `benchmark` and `gpu` markers are registered (no "unknown marker" warning)."""
    registered = {line.split(":", 1)[0].strip() for line in pytestconfig.getini("markers")}
    assert {"slow", "live", "benchmark", "gpu"} <= registered


def test_e0_t02_addopts_deselects_marked_suites(pytestconfig: pytest.Config) -> None:
    """`addopts` carries the `not slow and not live and not benchmark and not gpu` filter."""
    addopts = " ".join(pytestconfig.getini("addopts"))
    for marker in ("slow", "live", "benchmark", "gpu"):
        assert f"not {marker}" in addopts


def test_e0_t02_slow_marked_test_deselected_by_default() -> None:
    """A `@pytest.mark.slow` test is collected only when `-m slow` is passed explicitly."""
    fixture = "tests/fixtures/test_dummy_slow.py"
    default_run = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", fixture],
        capture_output=True,
        text=True,
        cwd=".",
        check=False,
    )
    assert "test_dummy_slow" not in default_run.stdout
    assert "no tests ran" in default_run.stdout or "1 deselected" in default_run.stdout

    slow_run = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "slow", fixture],
        capture_output=True,
        text=True,
        cwd=".",
        check=False,
    )
    assert "test_dummy_slow" in slow_run.stdout
