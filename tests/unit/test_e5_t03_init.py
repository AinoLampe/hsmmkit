"""E5-T03: initialization strategies (`estimation/init.py`)."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.estimation.init import lattice_jitter, random, rolling_kmeans
from regimekit.estimation.objective import make_objective
from regimekit.model.params.manager import Params
from regimekit.utils.rng import key_for
from tests.helpers.objective_fixtures import Case, make_case


def _finite_objective(case: Case, theta: Params) -> float:
    objective = make_objective(case.spec, case.space, case.engine, case.manager, case.data, use_priors=False)
    u = case.manager.flatten(theta)
    return float(objective.value(u))


@pytest.mark.parametrize(
    ("emission", "duration", "order", "kwargs"),
    [
        ("gbm", "negbin", 1, {}),
        ("gbm", "negbin", 2, {}),
        ("gbm", "negbin", 1, {"cov_em": True, "cov_du": True}),
        ("gbm", "negbin", 2, {"cov_tr": True}),
        ("gbm", "gamma", 1, {"cov_du": True}),
        ("student_t", "negbin", 1, {}),
    ],
)
def test_e5_t03_every_strategy_gives_a_finite_objective(
    emission: str, duration: str, order: int, kwargs: dict[str, object]
) -> None:
    """AC1: each strategy yields `u` for which `objective.value(u)` is finite (GBM/NB, order 1/2, covs)."""
    case = make_case(emission, duration, order, t_len=200, seed=3, **kwargs)  # type: ignore[arg-type]
    key = key_for(0, 1)

    theta_roll = rolling_kmeans(case.data, case.spec, key)
    theta_jitter = lattice_jitter(case.spec, key)
    theta_random = random(case.data, case.spec, key)

    assert np.isfinite(_finite_objective(case, theta_roll))
    assert np.isfinite(_finite_objective(case, theta_jitter))
    assert np.isfinite(_finite_objective(case, theta_random))


def test_e5_t03_lattice_jitter_keeps_nu_above_two() -> None:
    """`lattice_jitter`'s `nu_gt_2` branch must never jitter `nu` below its `> 2` domain floor."""
    case = make_case("student_t", "negbin", 1, t_len=60, seed=4)
    for i in range(30):
        theta = lattice_jitter(case.spec, key_for(0, i))
        assert bool(np.all(np.asarray(theta["emission.nu"]) > 2.0))
        assert np.isfinite(_finite_objective(case, theta))


def _simulate_two_regime_gbm(
    sigma: tuple[float, float], n_segments: int, seg_len: tuple[int, int], seed: int, dt: float
) -> np.ndarray:
    """Alternating-regime GBM log-returns (mu=0), for a case k-means should recover cleanly."""
    rng = np.random.default_rng(seed)
    chunks = []
    regime = 0
    for _ in range(n_segments):
        length = int(rng.integers(*seg_len))
        chunks.append(rng.normal(0.0, sigma[regime] * np.sqrt(dt), size=length))
        regime = 1 - regime
    return np.concatenate(chunks)


def test_e5_t03_rolling_kmeans_recovers_sigma_and_ordering() -> None:
    """AC2: on a 2-regime series with sigma=(0.10, 0.40), recovers sigma within 30% and the ordering."""
    true_sigma = (0.10, 0.40)
    dt = 1.0 / 252.0
    y = _simulate_two_regime_gbm(true_sigma, n_segments=60, seg_len=(40, 90), seed=11, dt=dt)

    case = make_case("gbm", "geometric", 1, t_len=y.shape[0], seed=0)
    data = dataclasses.replace(case.data, observations=jnp.asarray(y[:, None]))

    theta = rolling_kmeans(data, case.spec, key_for(0, 1))
    sigma_hat = np.asarray(theta["emission.scale"])

    assert sigma_hat.shape == (2,)
    assert sigma_hat[0] < sigma_hat[1]  # correct ordering: low-vol regime first
    assert abs(sigma_hat[0] - true_sigma[0]) / true_sigma[0] <= 0.30
    assert abs(sigma_hat[1] - true_sigma[1]) / true_sigma[1] <= 0.30
