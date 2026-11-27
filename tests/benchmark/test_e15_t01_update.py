"""E15-T01 AC4: one jitted `update` <= 0.2 ms at S = 12 on CPU (`-m benchmark`)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from hsmmkit.online import ExpandedOnlineFilter
from tests.helpers.engine_fixtures import fit_of, make_scenario

pytestmark = pytest.mark.benchmark


def test_e15_t01_update_time_budget(benchmark: BenchmarkFixture) -> None:
    """Median per-step `update` at N = 3, r = 4 (S = 12) with covariates, compile excluded."""
    sc = make_scenario(3, [4, 4, 4], 1, 64, n_cov_pi=2, n_cov_d=2, seed=0)
    assert sc.space.n_expanded == 12
    filt = ExpandedOnlineFilter(fit_of(sc))  # type: ignore[arg-type]
    y = np.asarray(sc.data.observations)[0]
    z = {"transition": np.asarray(sc.data.cov_transition)[0], "duration": np.asarray(sc.data.cov_duration)[0]}
    state = filt.start()
    for _ in range(3):  # compile + first-step branch
        state, _ = filt.update(state, y, 1 / 252, z)

    def step() -> None:
        nonlocal state
        state, _ = filt.update(state, y, 1 / 252, z)

    benchmark.pedantic(step, rounds=500, iterations=1, warmup_rounds=20)
    median_ms = benchmark.stats["median"] * 1e3
    print(f"E15-T01 update median: {median_ms:.3f} ms (S=12)")
    assert median_ms <= 0.2
