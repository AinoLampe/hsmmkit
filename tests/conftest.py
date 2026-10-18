"""Shared pytest configuration.

Sets `REGIMEKIT_TYPECHECK=1` so `@jaxtyped(typechecker=beartype)` is active during tests, imports
`regimekit` before any JAX array is created (x64 must be enabled first, see `utils.jax_setup`,
E0-T05), and registers a `deadline=None` Hypothesis profile (first-call JIT compilation exceeds the
default deadline) and provides the `gpu_device` fixture (E4-T07).
"""

import os

os.environ.setdefault("REGIMEKIT_TYPECHECK", "1")

import jax
import pytest
from hypothesis import HealthCheck, settings

import regimekit

settings.register_profile(
    "regimekit",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("regimekit")

_ = regimekit.__version__  # touch the import so linters don't flag it as unused


@pytest.fixture(scope="session")
def gpu_device() -> "jax.Device":
    """The first JAX GPU device; skips (never fails) when no GPU backend exists.

    `jax.devices("gpu")` raises `RuntimeError` when the backend is absent (e.g. `JAX_PLATFORMS=cpu`)
    and returns an empty list on some builds; both mean "no GPU" (CLAUDE.md §5).
    """
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("no GPU backend available")
    return devices[0]
