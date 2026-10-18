"""E4-T03a AC5: jitted forward <= 50 ms for T=5000, N=4, r=5, order 1, 2 covariates (`-m benchmark`)."""

from __future__ import annotations

import jax
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from regimekit.engine.expanded_hmm import _forward_core
from tests.helpers.engine_fixtures import make_scenario

pytestmark = pytest.mark.benchmark


def test_e4_t03a_forward_time_budget(benchmark: BenchmarkFixture) -> None:
    """Median jitted forward time stays within the 50 ms budget (compile excluded)."""
    sc = make_scenario(4, [5, 5, 5, 5], 1, 5000, n_cov_pi=2, n_cov_d=2, seed=0)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    jax.block_until_ready(_forward_core(*args))  # compile outside the timing
    benchmark.pedantic(lambda: jax.block_until_ready(_forward_core(*args)), rounds=20, iterations=1)
    median_ms = benchmark.stats["median"] * 1e3
    print(f"E4-T03a forward median: {median_ms:.1f} ms")
    assert median_ms <= 50.0
