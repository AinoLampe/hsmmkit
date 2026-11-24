"""E0-T01: every package in overview §4.2 must import cleanly."""

import importlib

import pytest

PACKAGES = [
    "regimekit",
    "regimekit.config",
    "regimekit.data",
    pytest.param("regimekit.finance", marks=pytest.mark.finance),
    pytest.param("regimekit.finance.providers", marks=pytest.mark.finance),
    "regimekit.model",
    "regimekit.model.params",
    "regimekit.model.emission",
    "regimekit.model.duration",
    "regimekit.statespace",
    "regimekit.engine",
    "regimekit.estimation",
    "regimekit.inference",
    "regimekit.evaluation",
    "regimekit.evaluation.metrics",
    "regimekit.persistence",
    "regimekit.cli",
    "regimekit.utils",
]


@pytest.mark.parametrize("dotted_name", PACKAGES)
def test_e0_t01_package_imports(dotted_name: str) -> None:
    """Each package listed in overview §4.2 is importable."""
    importlib.import_module(dotted_name)


def test_e0_t01_version_exposed() -> None:
    """`regimekit.__version__` is a non-empty string read from package metadata."""
    import regimekit

    assert isinstance(regimekit.__version__, str)
    assert regimekit.__version__
