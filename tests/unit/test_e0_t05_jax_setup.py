"""E0-T05 AC1: x64 enforcement and device selection in `utils/jax_setup.setup`.

`jax.config.update("jax_enable_x64", ...)` is fully dynamic in the installed JAX version (0.11.x):
calling `setup()` after a prior `jax.config.update("jax_enable_x64", False)` in the same process
re-enables x64 successfully (verified empirically; see `test_e0_t05_setup_succeeds_after_prior_x64_disable`
below and the STATUS.md finding for E0-T05), so the ticket's literal "disable x64, then import
hsmmkit raises" scenario cannot be reproduced that way. The defensive guard itself — raising
`NumericalError` when the post-`update` dtype check fails — is instead verified directly by forcing
the failure via monkeypatch, which is the scenario the guard exists to catch (e.g. a future JAX
version, or a wrapper library, that ignores or overrides the config update).
"""

import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from hsmmkit.utils.errors import NumericalError
from hsmmkit.utils.jax_setup import setup


def test_e0_t05_setup_raises_when_x64_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """`setup()` raises `NumericalError` if x64 is not actually enabled after the config update."""
    import hsmmkit.utils.jax_setup as jax_setup_module

    class _FakeZeros:
        dtype = jnp.float32

    monkeypatch.setattr(jax_setup_module.jnp, "zeros", lambda *_a, **_k: _FakeZeros())
    with pytest.raises(NumericalError, match="x64 disabled"):
        setup(device="cpu")


def test_e0_t05_setup_raises_for_gpu_without_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """`setup(device="gpu")` raises `NumericalError` when no GPU backend is available."""
    import hsmmkit.utils.jax_setup as jax_setup_module

    monkeypatch.setattr(jax_setup_module, "_has_gpu", lambda: False)
    with pytest.raises(NumericalError, match="no GPU backend"):
        setup(device="gpu")


def test_e0_t05_setup_auto_falls_back_to_cpu_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """`setup(device="auto")` resolves to CPU (and succeeds) when no GPU backend exists."""
    import hsmmkit.utils.jax_setup as jax_setup_module

    monkeypatch.setattr(jax_setup_module, "_has_gpu", lambda: False)
    setup(device="auto")
    assert jnp.zeros(1).dtype == jnp.float64


def test_e0_t05_setup_succeeds_after_prior_x64_disable() -> None:
    """Document current behavior: a prior in-process `update(False)` does not survive `setup()`."""
    jax.config.update("jax_enable_x64", False)
    setup(device="cpu")
    assert jnp.zeros(1).dtype == jnp.float64


def test_e0_t05_import_in_fresh_subprocess_ends_with_x64_enabled() -> None:
    """A fresh subprocess that disables x64 before importing `hsmmkit` ends up with x64 enabled."""
    code = (
        "import jax; jax.config.update('jax_enable_x64', False)\n"
        "import hsmmkit\n"
        "import jax.numpy as jnp\n"
        "assert jnp.zeros(1).dtype == jnp.float64\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
