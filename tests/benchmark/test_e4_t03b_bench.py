"""E4-T03b AC5: jitted forward-backward <= 100 ms at NFR-02 size (`-m benchmark`)."""

from __future__ import annotations

import jax
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from hsmmkit.engine.expanded_hmm import _forward_backward_core
from tests.helpers.engine_fixtures import make_scenario

pytestmark = pytest.mark.benchmark


def test_e4_t03b_forward_backward_time_budget(benchmark: BenchmarkFixture) -> None:
    """Median jitted forward_backward (T=5000, N=4, r=5, 2+2 covariates) stays within 100 ms."""
    sc = make_scenario(4, [5, 5, 5, 5], 1, 5000, n_cov_pi=2, n_cov_d=2, seed=0)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    jax.block_until_ready(_forward_backward_core(*args))  # compile outside the timing
    benchmark.pedantic(lambda: jax.block_until_ready(_forward_backward_core(*args)), rounds=20, iterations=1)
    median_ms = benchmark.stats["median"] * 1e3
    print(f"E4-T03b forward_backward median: {median_ms:.1f} ms")
    assert median_ms <= 100.0
