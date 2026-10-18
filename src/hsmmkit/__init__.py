"""regimekit: JAX-based hidden semi-Markov models for market-regime detection and forecasting."""

from importlib.metadata import version as _version

from regimekit.utils.jax_setup import setup as _setup

__version__ = _version("regimekit")

_setup()

from regimekit.project import Project  # noqa: E402  (after x64 setup)

__all__ = ["Project", "__version__"]
