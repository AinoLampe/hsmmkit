"""E5-T02 AC4 performance guardrail (NFR-02), run with `-m benchmark`.

Instance: `T=5000, N=3` GBM/geometric (transition covariates force the 3rd state), order 1, expanded
engine -- the same NFR-02 problem size as E4-T07's engine benchmarks. One L-BFGS run capped at 500
iterations must complete within 60 s; the number is recorded in the benchmark baseline, not gated there
(this ticket doesn't own `benchmarks/baseline.json`, E4-T07 does).
"""

from __future__ import annotations

import jax
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation.objective import make_objective
from hsmmkit.estimation.optimizers import minimize
from hsmmkit.utils.logging import get_logger
from tests.helpers.objective_fixtures import make_case

pytestmark = pytest.mark.benchmark

FIT_BUDGET_S = 60.0
LOG = get_logger("bench_e5_t02")


def test_e5_t02_lbfgs_fit_completes_within_budget(benchmark: BenchmarkFixture) -> None:
    """AC4: one L-BFGS fit, <= 500 iterations, at NFR-02 size completes <= 60 s."""
    case = make_case("gbm", "geometric", 1, cov_tr=True, t_len=5000, seed=0)
    objective = make_objective(case.spec, case.space, case.engine, case.manager, case.data, use_priors=True)
    cfg = EstimationSection(optimizer="lbfgs", max_iter=500)

    def run() -> object:
        record = minimize(objective, case.u0, cfg, LOG)
        return jax.block_until_ready(record.u)

    benchmark.pedantic(run, rounds=1, iterations=1, warmup_rounds=1)
    median = float(benchmark.stats["median"])
    print(f"E5-T02 lbfgs fit (T=5000, N=3, <=500 iter) median: {median:.2f} s")
    assert median <= FIT_BUDGET_S
