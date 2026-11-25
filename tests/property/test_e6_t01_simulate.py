"""E6-T01: statistical recovery properties of `simulate_paths` (ACs 1-3, 5)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pytest

from hsmmkit.inference.simulate import SimResult, segments_of, simulate_paths
from hsmmkit.model.params.manager import Params
from hsmmkit.utils.rng import key_for
from tests.helpers.objective_fixtures import Case, make_case

N_PATHS = 30
T_LEN = 3000


@dataclass
class _BigSim:
    case: Case
    theta: Params
    result: SimResult
    segments: list[list[tuple[int, int, int]]]


@pytest.fixture(scope="module")
def big_sim() -> _BigSim:
    """One large no-covariate simulation, shared by the sojourn/jump/return-moment checks."""
    case = make_case("gbm", "negbin", 1, t_len=T_LEN, seed=0)
    theta = case.manager.unflatten(case.u0)
    result = simulate_paths(
        key_for(0, 1), theta, case.spec, case.space, case.data.dt, None, None, None, n_paths=N_PATHS
    )
    return _BigSim(case=case, theta=theta, result=result, segments=segments_of(result.regimes))


def _segment_lengths_by_regime(segs: list[list[tuple[int, int, int]]], n_states: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {i: [] for i in range(n_states)}
    for path_segs in segs:
        for start, end, regime in path_segs:
            out[regime].append(end - start + 1)
    return out


def test_e6_t01_sojourn_mean_within_3se_and_minimum_equals_r(big_sim: _BigSim) -> None:
    """AC1: empirical mean sojourn is within 3 SE of `r_i / a_i`; the minimum sojourn equals `r_i`."""
    case, theta = big_sim.case, big_sim.theta
    n = case.spec.n_states
    lengths = _segment_lengths_by_regime(big_sim.segments, n)
    nominal_dt = case.data.nominal_dt
    mean_steps = np.asarray(case.spec.duration.mean_sojourn(theta, nominal_dt)) / nominal_dt
    r = case.spec.duration.r

    for i in range(n):
        assert len(lengths[i]) >= 2000 // n, "fixture needs >= 2000 segments total"
        arr = np.asarray(lengths[i], dtype=np.float64)
        se = arr.std(ddof=1) / np.sqrt(arr.size)
        assert abs(arr.mean() - mean_steps[i]) <= 3 * se
        assert arr.min() == r[i]


def test_e6_t01_jump_frequencies_within_3se_of_pi(big_sim: _BigSim) -> None:
    """AC2: empirical run-boundary jump frequencies are within 3 SE of `theta["transition.pi_rows"]`."""
    case, theta = big_sim.case, big_sim.theta
    n = case.spec.n_states
    true_pi = np.zeros((n, n))
    pi_rows = np.asarray(theta["transition.pi_rows"])
    for i in range(n):
        targets = [j for j in range(n) if j != i]
        true_pi[i, targets] = pi_rows[i]

    counts = np.zeros((n, n))
    for path_segs in big_sim.segments:
        regimes = [r for _, _, r in path_segs]
        for a, b in itertools.pairwise(regimes):
            counts[a, b] += 1

    for i in range(n):
        n_i = counts[i].sum()
        assert n_i >= 200, "fixture needs enough boundary transitions from every state"
        for j in range(n):
            if i == j:
                continue
            p_hat = counts[i, j] / n_i
            p = true_pi[i, j]
            se = np.sqrt(max(p * (1 - p), 1e-9) / n_i)
            assert abs(p_hat - p) <= 3 * se + 1e-6


def test_e6_t01_return_moments_within_3se_of_gbm_analytic(big_sim: _BigSim) -> None:
    """AC3: per-regime return mean/variance are within 3 SE of the GBM analytic values."""
    case, theta, result = big_sim.case, big_sim.theta, big_sim.result
    n = case.spec.n_states
    dt = float(case.data.nominal_dt)
    y = np.asarray(result.values)[:, :, 0].ravel()
    regimes = np.asarray(result.regimes).ravel()
    mu = np.asarray(theta["emission.mean"])[:, 0]
    sigma = np.asarray(theta["emission.scale"])[:, 0]

    for i in range(n):
        y_i = y[regimes == i]
        assert y_i.size >= 500
        mean = (mu[i] - 0.5 * sigma[i] ** 2) * dt
        var = sigma[i] ** 2 * dt
        sd = np.sqrt(var)
        se_mean = sd / np.sqrt(y_i.size)
        se_var = var * np.sqrt(2.0 / y_i.size)
        assert abs(y_i.mean() - mean) <= 3 * se_mean
        assert abs(y_i.var() - var) <= 3 * se_var


def test_e6_t01_higher_duration_covariate_shortens_sojourns() -> None:
    """AC5: with `duration.c > 0`, a higher constant `z_d` level shortens the mean sojourn (monotone)."""
    case = make_case("gbm", "geometric", 1, t_len=T_LEN, seed=1, cov_du=True)
    theta = dict(case.manager.unflatten(case.u0))
    theta["duration.c"] = theta["duration.c"] * 0.0 + 1.0  # duration.c > 0, uniformly across states

    mean_lengths = []
    for z_level in (-1.0, 1.0):
        z_d = case.data.dt[:, None] * 0.0 + z_level
        result = simulate_paths(
            key_for(0, 2), theta, case.spec, case.space, case.data.dt, None, z_d, None, n_paths=N_PATHS
        )
        segs = segments_of(result.regimes)
        lengths = [end - start + 1 for path in segs for start, end, _ in path]
        mean_lengths.append(float(np.mean(lengths)))

    assert mean_lengths[1] < mean_lengths[0]
