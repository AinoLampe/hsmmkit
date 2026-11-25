"""JAX runtime bootstrap: x64 precision and device selection (overview §4.5)."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from hsmmkit.utils.errors import NumericalError


def _has_gpu() -> bool:
    """Return whether a JAX GPU backend is available, without letting the probe raise."""
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


def setup(device: Literal["cpu", "gpu", "auto"] = "auto") -> None:
    """Enable x64 precision and select the JAX device backend.

    Args:
        device: `"cpu"`, `"gpu"`, or `"auto"` (GPU if available, else CPU).

    Raises:
        NumericalError: if x64 could not be enabled (an earlier JAX initialization elsewhere in
            the process disabled it), or if `device="gpu"` was requested but no GPU backend exists.
    """
    want_gpu = device == "gpu" or (device == "auto" and _has_gpu())
    resolved: Literal["cpu", "gpu"] = "gpu" if want_gpu else "cpu"

    if resolved == "gpu" and not _has_gpu():
        raise NumericalError(
            "device='gpu' requested but no GPU backend is available; install the `gpu` extra "
            "(`uv sync --extra gpu`) on a CUDA-capable Linux host, or pass device='cpu'/'auto'."
        )

    jax.config.update("jax_platforms", resolved)  # type: ignore[no-untyped-call]
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

    if jnp.zeros(1).dtype != jnp.float64:
        raise NumericalError(
            "x64 disabled: jax_enable_x64 could not be enabled, likely because JAX was already "
            "initialized elsewhere in this process with x64 turned off; hsmmkit requires "
            "float64 throughout."
        )
