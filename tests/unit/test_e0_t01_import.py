"""E0-T01: every package in overview §4.2 must import cleanly."""

import importlib

import pytest

PACKAGES = [
    "hsmmkit",
    "hsmmkit.config",
    "hsmmkit.data",
    "hsmmkit.model",
    "hsmmkit.model.params",
    "hsmmkit.model.emission",
    "hsmmkit.model.duration",
    "hsmmkit.statespace",
    "hsmmkit.engine",
    "hsmmkit.estimation",
    "hsmmkit.inference",
    "hsmmkit.evaluation",
    "hsmmkit.evaluation.metrics",
    "hsmmkit.persistence",
    "hsmmkit.cli",
    "hsmmkit.utils",
]


@pytest.mark.parametrize("dotted_name", PACKAGES)
def test_e0_t01_package_imports(dotted_name: str) -> None:
    """Each package listed in overview §4.2 is importable."""
    importlib.import_module(dotted_name)


def test_e0_t01_version_exposed() -> None:
    """`hsmmkit.__version__` is a non-empty string read from package metadata."""
    import hsmmkit

    assert isinstance(hsmmkit.__version__, str)
    assert hsmmkit.__version__
