"""E12-T02 benchmark (NFR-12 size): one batched MAP objective + gradient evaluation (`-m benchmark`).

Recorded, no hard budget (the EM budget belongs to E13-T02a): `B = 64`, `T = 1000`, `S = 12` expanded states
(4 regimes with negative-binomial `r = 3` durations).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from hsmmkit.config.schema import parse_project_config
from hsmmkit.engine.batch import BatchedEngine
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.estimation.objective import make_objective
from hsmmkit.inference.simulate import simulate_batch
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.statespace.builders import build_expanded
from tests.helpers.core_config import core_raw

pytestmark = pytest.mark.benchmark

B, T = 64, 1000


def test_e12_t02_batched_objective_value_and_grad(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    """Median of `value_and_grad` of the batch objective at NFR-12 size is recorded and finite."""
    raw = core_raw(tmp_path)
    raw["model"]["n_states"] = 4
    raw["model"]["emission"]["params"] = {
        "mean": {"mode": "free", "init": [-2.0, -0.5, 0.5, 2.0]},
        "scale": {"mode": "free", "init": [0.5, 0.8, 1.1, 1.4], "constraint": "increasing"},
    }
    raw["model"]["duration"] = {
        "family": "negbin",
        "r": [3, 3, 3, 3],
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [10.0] * 4}},
    }
    spec = build_model_spec_from(parse_project_config(raw), BuildInputs(1, {}, "steps", 1.0))
    space = build_expanded(spec)
    assert space.n_expanded == 12
    manager = ParamManager(spec.param_specs)
    theta = dict(manager.unflatten(manager.init_u()))
    pb = simulate_batch(theta, spec, [T] * B, jax.random.key(0)).padded()
    objective = make_objective(spec, space, BatchedEngine(ExpandedHMMEngine()), manager, pb, use_priors=True)
    u = manager.init_u()
    value, grad = objective.value_and_grad(u)  # compile outside the timing
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))
    benchmark.pedantic(lambda: jax.block_until_ready(objective.value_and_grad(u)), rounds=5, iterations=1)
    median_ms = benchmark.stats["median"] * 1e3
    print(f"E12-T02 batched objective value_and_grad median: {median_ms:.1f} ms (B={B}, T={T}, S=12)")
