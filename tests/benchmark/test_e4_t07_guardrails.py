"""E4-T07 performance guardrails (NFR-02), run with `-m benchmark`.

Instance: `T=5000, N=4, NB r=5, order 1, 2 covariates`, GBM. Compile time is excluded (one warm-up
round). Numbers of the committed baseline come from `pytest -m benchmark --benchmark-json=bench.json`
(see `benchmarks/baseline.json`); the nightly job fails on a > 30 % regression.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.engine.explicit_duration import ExplicitDurationEngine
from tests.helpers.engine_fixtures import make_explicit_scenario, make_scenario

pytestmark = pytest.mark.benchmark

EXPANDED_FB_BUDGET_S = 0.100
EXPLICIT_FB_BUDGET_S = 2.0


def _time(benchmark: BenchmarkFixture, fn: object, rounds: int = 10) -> float:
    """Median seconds of `fn()` after one warm-up call (compile excluded)."""
    benchmark.pedantic(lambda: jax.block_until_ready(fn()), rounds=rounds, iterations=1, warmup_rounds=1)  # type: ignore[operator]
    return float(benchmark.stats["median"])


def test_e4_t07_expanded_forward_backward_budget(benchmark: BenchmarkFixture) -> None:
    """Expanded engine `forward_backward` (public method) <= 100 ms."""
    sc = make_scenario(4, [5, 5, 5, 5], 1, 5000, n_cov_pi=2, n_cov_d=2, seed=0)
    engine = ExpandedHMMEngine()
    median = _time(benchmark, lambda: engine.forward_backward(sc.theta, sc.spec, sc.space, sc.data))
    print(f"E4-T07 expanded forward_backward median: {median * 1e3:.1f} ms")
    assert median <= EXPANDED_FB_BUDGET_S


def test_e4_t07_expanded_viterbi_time(benchmark: BenchmarkFixture) -> None:
    """Expanded Viterbi on the same instance (recorded for the baseline, budget: the FB budget)."""
    sc = make_scenario(4, [5, 5, 5, 5], 1, 5000, n_cov_pi=2, n_cov_d=2, seed=0)
    engine = ExpandedHMMEngine()
    median = _time(benchmark, lambda: engine.viterbi(sc.theta, sc.spec, sc.space, sc.data))
    print(f"E4-T07 expanded viterbi median: {median * 1e3:.1f} ms")
    assert median <= EXPANDED_FB_BUDGET_S


def test_e4_t07_explicit_forward_backward_budget(benchmark: BenchmarkFixture) -> None:
    """Explicit-duration `forward_backward` at `D_max = 250` <= 2 s."""
    sc = make_explicit_scenario(4, 250, 5000, family="negbin", seed=0, mean_steps=60.0)
    engine = ExplicitDurationEngine()
    median = _time(benchmark, lambda: engine.forward_backward(sc.theta, sc.spec, sc.space, sc.data), rounds=5)
    print(f"E4-T07 explicit forward_backward median: {median * 1e3:.0f} ms")
    assert median <= EXPLICIT_FB_BUDGET_S


def test_e4_t07_explicit_viterbi_time(benchmark: BenchmarkFixture) -> None:
    """Explicit-duration segmental Viterbi at `D_max = 250`, same 2 s budget."""
    sc = make_explicit_scenario(4, 250, 5000, family="negbin", seed=0, mean_steps=60.0)
    engine = ExplicitDurationEngine()
    median = _time(benchmark, lambda: engine.viterbi(sc.theta, sc.spec, sc.space, sc.data), rounds=5)
    print(f"E4-T07 explicit viterbi median: {median * 1e3:.0f} ms")
    assert median <= EXPLICIT_FB_BUDGET_S


def test_e4_t07_parallel_scan_forward_backward_time(benchmark: BenchmarkFixture) -> None:
    """Associative-scan smoother (opt-in; `S = 20`, `T S^2 = 2e6`). Recorded only: it pays off on GPUs."""
    sc = make_scenario(4, [5, 5, 5, 5], 1, 5000, n_cov_pi=2, n_cov_d=2, seed=0)
    engine = ExpandedHMMEngine(parallel_scan=True)
    median = _time(benchmark, lambda: engine.forward_backward(sc.theta, sc.spec, sc.space, sc.data), rounds=5)
    print(f"E4-T07 parallel-scan forward_backward median: {median * 1e3:.1f} ms")
    assert np.isfinite(median)
