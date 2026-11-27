"""E15-T01: the online filter reproduces the offline forward pass bitwise (AC1, AC3)."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.online import ExpandedOnlineFilter
from hsmmkit.utils.errors import ConfigError
from tests.helpers.engine_fixtures import fit_of, make_scenario

T = 5000


@pytest.fixture(scope="module")
def case() -> dict[str, Any]:
    """The scenario, its offline forward-backward posterior and the masked, irregular data."""
    sc = make_scenario(3, [3, 3, 3], 1, T, n_cov_pi=2, n_cov_d=2, seed=11)
    rng = np.random.default_rng(5)
    mask = np.ones((T, 1), dtype=bool)
    mask[rng.random(T) < 0.03] = False
    mask[0] = True
    data = dataclasses.replace(sc.data, mask=jnp.asarray(mask))
    # a few irregular gaps
    dt = np.asarray(data.dt).copy()
    dt[rng.random(T) < 0.1] *= 3
    data = dataclasses.replace(data, dt=jnp.asarray(dt))
    post = ExpandedHMMEngine().forward_backward(sc.theta, sc.spec, sc.space, data)
    return {"sc": sc, "data": data, "post": post, "fit": fit_of(sc), "mask": mask, "dt": dt}


def _ys(case: dict[str, Any]) -> np.ndarray:
    y = np.asarray(case["data"].observations).copy()
    y[~case["mask"]] = np.nan
    return y


def _z(case: dict[str, Any], sl: slice) -> dict[str, np.ndarray]:
    d = case["data"]
    return {"transition": np.asarray(d.cov_transition)[sl], "duration": np.asarray(d.cov_duration)[sl]}


def test_e15_t01_update_many_filtered_and_lognorm_equal_offline_bitwise(case: dict[str, Any]) -> None:
    """E15-T01: update many filtered and lognorm equal offline bitwise."""
    filt = ExpandedOnlineFilter(case["fit"])
    state, filtered = filt.update_many(filt.start(), _ys(case), case["dt"], _z(case, slice(None)))
    off = case["post"]
    np.testing.assert_array_equal(np.asarray(filtered), np.asarray(off.regime_filtered()))
    np.testing.assert_array_equal(np.asarray(state.log_norm), np.asarray(off.loglik))
    np.testing.assert_array_equal(np.asarray(state.log_alpha), np.asarray(off.log_alpha[-1]))
    assert state.step == T


def test_e15_t01_stepwise_update_equals_offline_bitwise(case: dict[str, Any]) -> None:
    """E15-T01: stepwise update equals offline bitwise."""
    filt = ExpandedOnlineFilter(case["fit"])
    y = _ys(case)
    off = case["post"]
    off_f = np.asarray(off.regime_filtered())
    state = filt.start()
    n = 600  # per-step python loop; the long replay is the scan test above
    for t in range(n):
        z = {k: v[0] for k, v in _z(case, slice(t, t + 1)).items()}
        state, f = filt.update(state, y[t], case["dt"][t], z)
        np.testing.assert_array_equal(np.asarray(f), off_f[t])
        np.testing.assert_array_equal(np.asarray(state.log_alpha), np.asarray(off.log_alpha[t]))


def test_e15_t01_warm_start_equals_offline_prefix(case: dict[str, Any]) -> None:
    """`start(initial)` runs the offline forward on a prefix; continuing equals the full offline pass."""
    filt = ExpandedOnlineFilter(case["fit"])
    k = 1200
    state = filt.start(case["data"].window(0, k))
    state, filtered = filt.update_many(state, _ys(case)[k:], case["dt"][k:], _z(case, slice(k, None)))
    off = case["post"]
    np.testing.assert_array_equal(np.asarray(filtered), np.asarray(off.regime_filtered())[k:])
    np.testing.assert_array_equal(np.asarray(state.log_norm), np.asarray(off.loglik))


def test_e15_t01_offline_forward_is_unchanged_by_the_forward_step_refactor(case: dict[str, Any]) -> None:
    """AC3: the offline recursion equals an explicit per-step re-computation with the shared body."""
    from hsmmkit.engine.expanded_hmm import (  # type: ignore[attr-defined]
        _expanded_log_emis,
        forward_step,
        initial_log_rho,
    )

    sc, data = case["sc"], case["data"]
    ell = _expanded_log_emis(sc.theta, sc.spec, sc.space, data)
    alpha = initial_log_rho(sc.spec, sc.space, sc.theta, data.nominal_dt) + ell[0]
    rows = [alpha]
    z_pi, z_d = np.asarray(data.cov_transition), np.asarray(data.cov_duration)
    for t in range(1, 300):
        alpha = forward_step(
            sc.space, sc.theta, False, alpha, ell[t], z_pi[t], z_d[t], data.dt[t], jnp.zeros((), jnp.int8)
        )
        rows.append(alpha)
    off, _ = ExpandedHMMEngine().forward(sc.theta, sc.spec, sc.space, data)
    np.testing.assert_allclose(
        np.stack([np.asarray(r) for r in rows]), np.asarray(off[:300]), rtol=0, atol=1e-12
    )


def test_e15_t01_masked_first_observation_and_nan_rows(case: dict[str, Any]) -> None:
    """E15-T01: masked first observation and nan rows."""
    filt = ExpandedOnlineFilter(case["fit"])
    state = filt.start()
    z = {k: v[0] for k, v in _z(case, slice(0, 1)).items()}
    state, f = filt.update(state, [np.nan], 1 / 252, z)
    assert np.isclose(float(jnp.sum(f)), 1.0)
    assert state.step == 1


def test_e15_t01_the_explicit_engine_is_rejected() -> None:
    """E15-T01: the explicit engine is rejected."""
    from tests.helpers.engine_fixtures import make_explicit_scenario

    sc = make_explicit_scenario(3, 12, 50)
    with pytest.raises(ConfigError, match="requires engine: expanded"):
        ExpandedOnlineFilter(fit_of(sc))
