"""regimekit: JAX-based hidden semi-Markov models for market-regime detection and forecasting."""

import importlib
from importlib.metadata import version as _version

from regimekit.utils.jax_setup import setup as _setup

__version__ = _version("regimekit")

_setup()

__all__ = ["Project", "__version__"]

_FACADE = "regimekit.finance.project"  # the application package that provides `Project`


def __getattr__(name: str) -> object:
    """`regimekit.Project` is the finance facade (`FinanceProject`), imported on first use (0.2.x API).

    The core never imports the application package statically; the facade is looked up by name.
    """
    if name == "Project":
        return importlib.import_module(_FACADE).FinanceProject
    raise AttributeError(f"module 'regimekit' has no attribute {name!r}")
