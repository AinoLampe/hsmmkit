"""E13-T08 spike: online EM tracks a drifting sigma and is exact for frozen parameters (slow marker)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.estep import expanded_estep
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.statespace.builders import build_expanded
from tests.helpers import em_models

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "spikes"))
import online_em as oem

pytestmark = pytest.mark.slow


def test_e13_t08_online_em_tracks_a_drifting_sigma() -> None:
    """AC2: mean tracking error of the drifting `sigma_1` < 10 % after burn-in (T = 20 000, +50 %)."""
    y, _, sigma1 = oem.simulate_drift(20_000, seed=0)
    path, online = oem.track(y, oem.initial_params(), kappa=0.7, burn_in=500)
    err = np.abs(path[2000:, 1] - sigma1[2000:]) / sigma1[2000:]
    assert err.mean() < 0.10
    assert abs(path[-1, 0] - 0.5) / 0.5 < 0.15  # the constant regime stays put
    assert np.isfinite(online.loglik)
    # a 1/t schedule (no forgetting) cannot follow the drift: that is why kappa < 1
    slow, _ = oem.track(y, oem.initial_params(), kappa=1.0, burn_in=500)
    err_slow = np.abs(slow[2000:, 1] - sigma1[2000:]) / sigma1[2000:]
    assert err_slow.mean() > err.mean()


def test_e13_t08_frozen_parameters_and_a_one_over_t_schedule_reproduce_the_offline_estep(
    tmp_path: Path,
) -> None:
    """AC2: `t S_t` equals the offline expected statistics (NumPy and `expanded_estep`, within 1 %)."""
    spec, theta = em_models.build(tmp_path, "gaussian", "geometric")
    data = em_models.dataset("gaussian", 600, 3)
    y = np.asarray(data.observations)[:, 0]
    mu = np.asarray(theta["emission.mean"]).reshape(-1)
    sigma = np.asarray(theta["emission.scale"]).reshape(-1)
    rate = np.asarray(theta["duration.rate"]).reshape(-1)
    stay = np.exp(-rate)
    a = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
    params = oem.Params(np.array([0.5, 0.5]), a, mu, sigma)
    online = oem.OnlineEM(params, kappa=1.0, update=False)
    for v in y:
        online.step(float(v))
    total = online.t * online.stats
    np.testing.assert_allclose(total, oem.offline_estats(params, y), rtol=1e-8)
    stats = expanded_estep(
        ExpandedHMMEngine(), theta, spec, build_expanded(spec), SequenceBatch.of(data).padded()
    )
    n = 2
    gamma = np.asarray(stats.gamma_regime)[0, : y.shape[0]]
    np.testing.assert_allclose(total[n * n : n * n + n], gamma.sum(axis=0), rtol=0.01)
    np.testing.assert_allclose(total[n * n + n : n * n + 2 * n], gamma.T @ y, rtol=0.01, atol=1e-6)
    np.testing.assert_allclose(total[n * n + 2 * n :], gamma.T @ (y * y), rtol=0.01)
    counts = total[: n * n].reshape(n, n)
    np.testing.assert_allclose(np.diag(counts), np.asarray(stats.stay), rtol=0.01)
    np.testing.assert_allclose(counts.sum(axis=1) - np.diag(counts), np.asarray(stats.advance), rtol=0.01)


def test_e13_t08_a_stationary_run_recovers_the_true_parameters_with_kappa_one() -> None:
    """With kappa = 1 and no drift the one-pass estimate is within 5 % of the truth."""
    y, _, _ = oem.simulate_drift(20_000, seed=1, drift=0.0)
    path, online = oem.track(y, oem.initial_params(), kappa=1.0, burn_in=500)
    np.testing.assert_allclose(path[-1], [0.5, 1.2], rtol=0.05)
    np.testing.assert_allclose(online.theta.mu, [0.0, 1.5], atol=0.1)
