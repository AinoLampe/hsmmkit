"""hsmmkit: domain-neutral hidden semi-Markov models (JAX)."""

from importlib.metadata import version as _version

from hsmmkit.utils.jax_setup import setup as _setup

__version__ = _version("hsmmkit")

_setup()

from hsmmkit.project import Project  # noqa: E402  (after x64 setup)

__all__ = ["Project", "__version__"]
