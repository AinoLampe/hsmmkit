"""E13-T02a AC5 (NFR-12): one EM iteration at `B = 64, T = 1000, N = 4, S = 12` <= 400 ms (`-m benchmark`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from hsmmkit.data.batch import PaddedBatch
from hsmmkit.estimation.em import _em_step
from tests.helpers.engine_fixtures import make_scenario

pytestmark = pytest.mark.benchmark

B, T = 64, 1000


def test_e13_t02a_one_em_iteration_time_budget(benchmark: BenchmarkFixture) -> None:
    """Median E-step + closed-form M-step over 64 sequences of 1000 steps (`S = 12`), compile excluded."""
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
    theta = {
        k: v for k, v in sc.theta.items() if not k.startswith(("transition.a", "transition.b", "duration.c"))
    }
    args = (theta, sc.spec, sc.space, pb, False)
    jax.block_until_ready(_em_step(*args))  # compile outside the timing
    benchmark.pedantic(lambda: jax.block_until_ready(_em_step(*args)), rounds=5, iterations=1)
    median_ms = benchmark.stats["median"] * 1e3
    print(f"E13-T02a EM iteration median: {median_ms:.1f} ms (B={B}, T={T}, S={sc.space.n_expanded})")
    assert sc.space.n_expanded == 12
    assert median_ms <= 400.0
