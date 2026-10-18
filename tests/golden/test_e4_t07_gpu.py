"""E4-T07: GPU variants of the golden checks (`-m gpu`); they skip cleanly on CPU-only hosts."""

from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np
import pytest

from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.engine.explicit_duration import ExplicitDurationEngine
from tests.helpers.engine_fixtures import make_explicit_scenario, make_scenario

pytestmark = pytest.mark.gpu


def on_device[T](device: jax.Device, run: Callable[[], T]) -> T:
    """Run `run()` with `device` as the default device (inputs are placed by JAX)."""
    with jax.default_device(device):
        return run()


def test_e4_t07_gpu_expanded_forward_backward_matches_cpu(gpu_device: jax.Device) -> None:
    """Expanded forward-backward on the GPU agrees with the CPU to 1e-8 (Appendix D)."""
    sc = make_scenario(4, [2, 3, 1, 2], 1, 300, n_cov_pi=2, n_cov_d=2, seed=5)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    cpu = on_device(jax.devices("cpu")[0], lambda: ExpandedHMMEngine().forward_backward(*args))
    gpu = on_device(gpu_device, lambda: ExpandedHMMEngine().forward_backward(*args))
    assert abs(float(gpu.loglik) - float(cpu.loglik)) <= 1e-8 * abs(float(cpu.loglik))
    np.testing.assert_allclose(
        np.asarray(gpu.smoothed),
        np.asarray(cpu.smoothed),
        rtol=0,
        atol=1e-8,
    )


def test_e4_t07_gpu_parallel_scan_matches_cpu_sequential(gpu_device: jax.Device) -> None:
    """The associative-scan smoother on the GPU agrees with the sequential CPU scan to 1e-8."""
    sc = make_scenario(3, [2, 2, 2], 1, 300, seed=8)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    cpu = on_device(jax.devices("cpu")[0], lambda: ExpandedHMMEngine().forward_backward(*args))
    gpu = on_device(gpu_device, lambda: ExpandedHMMEngine(parallel_scan=True).forward_backward(*args))
    np.testing.assert_allclose(
        np.asarray(gpu.smoothed),
        np.asarray(cpu.smoothed),
        rtol=0,
        atol=1e-8,
    )


def test_e4_t07_gpu_explicit_forward_backward_matches_cpu(gpu_device: jax.Device) -> None:
    """Explicit-duration forward-backward on the GPU agrees with the CPU to 1e-8."""
    sc = make_explicit_scenario(3, 15, 200, family="gamma", seed=3)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    cpu = on_device(jax.devices("cpu")[0], lambda: ExplicitDurationEngine().forward_backward(*args))
    gpu = on_device(gpu_device, lambda: ExplicitDurationEngine().forward_backward(*args))
    np.testing.assert_allclose(
        np.asarray(gpu.smoothed),
        np.asarray(cpu.smoothed),
        rtol=0,
        atol=1e-8,
    )
