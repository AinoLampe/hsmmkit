"""E6-T01: `simulate_paths`/`simulate_dataset` -- shapes, determinism, `init_state` override."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.inference.simulate import segments_of, simulate_dataset, simulate_paths
from hsmmkit.utils.errors import NumericalError
from hsmmkit.utils.rng import key_for
from tests.helpers.objective_fixtures import make_case


def test_e6_t01_shapes_and_regime_range() -> None:
    """`states`/`regimes`/`returns` have the documented shapes; regimes stay within `[0, N)`."""
    case = make_case("gbm", "geometric", 1, t_len=120, seed=0)
    theta = case.manager.unflatten(case.u0)
    n_paths = 4

    result = simulate_paths(
        key_for(0, 1), theta, case.spec, case.space, case.data.dt, None, None, None, n_paths
    )

    assert result.states.shape == (n_paths, 120)
    assert result.regimes.shape == (n_paths, 120)
    assert result.values.shape == (n_paths, 120, 1)
    regimes = np.asarray(result.regimes)
    assert regimes.min() >= 0
    assert regimes.max() < case.spec.n_states
    assert np.all(np.isfinite(np.asarray(result.values)))


def test_e6_t01_same_key_bitwise_identical_different_key_differs() -> None:
    """AC4: the same key gives a bitwise identical `SimResult`; a different key differs."""
    case = make_case("gbm", "negbin", 1, t_len=200, seed=1)
    theta = case.manager.unflatten(case.u0)
    args = (theta, case.spec, case.space, case.data.dt, None, None, None)

    r1 = simulate_paths(key_for(0, 1), *args, n_paths=6)
    r2 = simulate_paths(key_for(0, 1), *args, n_paths=6)
    r3 = simulate_paths(key_for(0, 2), *args, n_paths=6)

    assert np.array_equal(np.asarray(r1.states), np.asarray(r2.states))
    assert np.array_equal(np.asarray(r1.values), np.asarray(r2.values))
    assert not np.array_equal(np.asarray(r1.states), np.asarray(r3.states))


def test_e6_t01_init_state_overrides_the_initial_draw() -> None:
    """A given `init_state` is used verbatim, skipping the `initial_log_rho` draw."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=2)
    theta = case.manager.unflatten(case.u0)
    phase1 = case.space.phase1_idx
    init_state = jnp.asarray([phase1[0], phase1[0], phase1[-1]])

    result = simulate_paths(
        key_for(0, 1),
        theta,
        case.spec,
        case.space,
        case.data.dt,
        None,
        None,
        None,
        n_paths=3,
        init_state=init_state,
    )

    assert np.array_equal(np.asarray(result.states)[:, 0], np.asarray(init_state))


def test_e6_t01_segments_of_matches_manual_run_length_encoding() -> None:
    """`segments_of` gives one `(start, end, regime)` (end inclusive) list per path, RLE-correct."""
    regimes = jnp.asarray([[0, 0, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])

    segs = segments_of(regimes)

    assert segs[0] == [(0, 1, 0), (2, 4, 1), (5, 5, 0)]
    assert segs[1] == [(0, 5, 1)]


def test_e6_t01_non_finite_returns_raise_numerical_error() -> None:
    """CLAUDE.md §4 invariant 5: a non-finite sample must not leave `simulate_paths`."""
    case = make_case("gbm", "geometric", 1, t_len=20, seed=4)
    theta = dict(case.manager.unflatten(case.u0))
    theta["emission.scale"] = jnp.asarray(theta["emission.scale"]) * jnp.inf

    with pytest.raises(NumericalError, match="values"):
        simulate_paths(key_for(0, 1), theta, case.spec, case.space, case.data.dt, None, None, None, n_paths=2)


def test_e6_t01_simulate_dataset_round_trip() -> None:
    """`simulate_dataset` returns a `PreparedDataset` shaped like the input `dt`, fully observed."""
    case = make_case("gbm", "geometric", 1, t_len=50, seed=3)
    theta = case.manager.unflatten(case.u0)

    ds = simulate_dataset(key_for(0, 1), theta, case.spec, case.space, case.data.dt, clock="steps")

    assert ds.observations.shape == (50, 1)
    assert bool(jnp.all(ds.mask))
    assert ds.clock == "steps"
    assert np.isfinite(np.asarray(ds.observations)).all()
