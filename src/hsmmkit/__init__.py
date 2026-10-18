"""regimekit: JAX-based hidden semi-Markov models for market-regime detection and forecasting."""

from importlib.metadata import version as _version

from regimekit.utils.jax_setup import setup as _setup

__version__ = _version("regimekit")

_setup()

__all__ = ["__version__"]
