"""E4-T06a AC5: explicit-duration forward-backward <= 2 s at T=5000, N=4, D_max=250 (`-m benchmark`)."""

from __future__ import annotations

import jax
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from regimekit.engine.explicit_duration import _forward_backward_core, _loglik_core
from tests.helpers.engine_fixtures import make_explicit_scenario

pytestmark = pytest.mark.benchmark


def test_e4_t06a_forward_backward_time_budget(benchmark: BenchmarkFixture) -> None:
    """Median jitted forward-backward stays within the 2 s budget (compile excluded)."""
    sc = make_explicit_scenario(4, 250, 5000, family="negbin", seed=0, mean_steps=60.0)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    jax.block_until_ready(_forward_backward_core(*args))
    benchmark.pedantic(lambda: jax.block_until_ready(_forward_backward_core(*args)), rounds=5, iterations=1)
    median_s = benchmark.stats["median"]
    print(f"E4-T06a explicit forward-backward median: {median_s * 1e3:.0f} ms")
    assert median_s <= 2.0


def test_e4_t06a_loglik_time_budget(benchmark: BenchmarkFixture) -> None:
    """The likelihood-only path (used by estimation) is well inside the same budget."""
    sc = make_explicit_scenario(4, 250, 5000, family="negbin", seed=0, mean_steps=60.0)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    jax.block_until_ready(_loglik_core(*args))
    benchmark.pedantic(lambda: jax.block_until_ready(_loglik_core(*args)), rounds=5, iterations=1)
    median_s = benchmark.stats["median"]
    print(f"E4-T06a explicit loglik median: {median_s * 1e3:.0f} ms")
    assert median_s <= 2.0
