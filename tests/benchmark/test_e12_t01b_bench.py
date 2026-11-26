"""E12-T01b AC5 (NFR-12): batched forward-backward `B = 64, T = 1000, S = 12` <= 300 ms (`-m benchmark`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from hsmmkit.data.batch import PaddedBatch
from hsmmkit.engine.batch import _batched_core
from tests.helpers.engine_fixtures import make_scenario

pytestmark = pytest.mark.benchmark

B, T = 64, 1000


def test_e12_t01b_batched_forward_backward_time_budget(benchmark: BenchmarkFixture) -> None:
    """Median batched forward-backward (64 sequences of 1000 steps, `S = 12`) within 300 ms."""
    sc = make_scenario(4, [3, 3, 3, 3], 1, T, seed=0)
    rng = np.random.default_rng(0)
    pb = PaddedBatch(
        observations=jnp.asarray(rng.normal(0.0, 0.01, size=(B, T, 1))),
        mask=jnp.ones((B, T, 1), dtype=bool),
        valid=jnp.ones((B, T), dtype=bool),
        dt=jnp.full((B, T), float(sc.data.nominal_dt)),
        cov_transition=jnp.zeros((B, T, 0)),
        cov_duration=jnp.zeros((B, T, 0)),
        cov_emission=jnp.zeros((B, T, 0)),
        labels=None,
        lengths=jnp.full((B,), T, dtype=jnp.int32),
        group_index=jnp.zeros((B,), dtype=jnp.int32),
        nominal_dt=float(sc.data.nominal_dt),
        clock=sc.data.clock,
    )
    args = (sc.theta, sc.spec, sc.space, pb, "fb")
    jax.block_until_ready(_batched_core(*args))  # compile outside the timing
    benchmark.pedantic(lambda: jax.block_until_ready(_batched_core(*args)), rounds=10, iterations=1)
    median_ms = benchmark.stats["median"] * 1e3
    print(f"E12-T01b batched forward_backward median: {median_ms:.1f} ms (B={B}, T={T})")
    assert sc.space.n_expanded == 12
    assert median_ms <= 300.0
