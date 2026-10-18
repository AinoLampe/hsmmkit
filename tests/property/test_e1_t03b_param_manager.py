"""E1-T03b: `ParamManager` round trips, mode semantics, Appendix B counts, and jit/grad."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from regimekit.model.params import NormalPrior, ParamManager, ParamSpec
from regimekit.utils.errors import ConfigError


def _spec_free(name: str, n: int, domain: str = "positive") -> ParamSpec:
    return ParamSpec(name=name, shape=(n,), domain=domain, mode="free", init=np.ones(n))


# --- AC3: Appendix B parameter counts ---------------------------------------------------------


def test_e1_t03b_n_free_matches_appendix_b_order1_gbm_negbin() -> None:
    """N=4, GBM + NB order 1, full jump matrix, no covariates -> n_free == 20."""
    n = 4
    specs = (
        ParamSpec(
            name="transition.jump",
            shape=(n, n - 1),
            domain="simplex_ref",
            mode="free",
            init=np.full((n, n - 1), 1.0 / (n - 1)),
        ),
        _spec_free("duration.rate", n),
        _spec_free("emission.mu", n, domain="real"),
        _spec_free("emission.sigma", n),
    )
    manager = ParamManager(specs)
    assert manager.n_free == 20


# --- AC1: round trip -------------------------------------------------------------------------


def _mixed_mode_specs() -> tuple[ParamSpec, ...]:
    """One spec per mode (free, fixed, tied x2, bounded, prior), mixing all five."""
    return (
        ParamSpec(name="a.free", shape=(3,), domain="positive", mode="free", init=np.array([0.5, 1.0, 2.0])),
        ParamSpec(name="a.fixed", shape=(3,), domain="real", mode="fixed", value=np.array([-0.3, 0.0, 0.2])),
        ParamSpec(
            name="a.tied",
            shape=(3,),
            domain="positive",
            mode="tied",
            tie_group="g1",
            init=np.array([1.0, 1.0, 1.0]),
        ),
        ParamSpec(
            name="a.tied_other",
            shape=(2,),
            domain="positive",
            mode="tied",
            tie_group="g1",
            init=np.array([1.0, 1.0]),
        ),
        ParamSpec(
            name="a.bounded",
            shape=(3,),
            domain="interval",
            mode="bounded",
            bounds=(-1.0, 1.0),
            init=np.array([0.1, -0.2, 0.3]),
        ),
        ParamSpec(
            name="a.prior",
            shape=(2,),
            domain="real",
            mode="prior",
            init=np.zeros(2),
            prior=NormalPrior(sd=1.0),
        ),
    )


@given(seed=st.integers(min_value=0, max_value=10_000))
def test_e1_t03b_round_trip_u_to_theta_to_u(seed: int) -> None:
    """`flatten(unflatten(u)) == u` for random `u`, within 1e-9."""
    manager = ParamManager(_mixed_mode_specs())
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.uniform(-2.0, 2.0, size=manager.n_free))
    theta = manager.unflatten(u)
    u_back = manager.flatten(theta)
    np.testing.assert_allclose(np.asarray(u_back), np.asarray(u), atol=1e-9, rtol=1e-9)


@given(seed=st.integers(min_value=0, max_value=10_000))
def test_e1_t03b_round_trip_theta_to_u_to_theta(seed: int) -> None:
    """`unflatten(flatten(theta)) == theta` for `theta` produced by `unflatten` (a valid theta)."""
    manager = ParamManager(_mixed_mode_specs())
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.uniform(-2.0, 2.0, size=manager.n_free))
    theta = manager.unflatten(u)
    theta_back = manager.unflatten(manager.flatten(theta))
    for key in theta:
        np.testing.assert_allclose(np.asarray(theta_back[key]), np.asarray(theta[key]), atol=1e-9)


# --- AC2: fixed absent from u; tied shares exactly one coordinate ----------------------------


def test_e1_t03b_fixed_entries_absent_from_u_and_unchanged() -> None:
    """`fixed` entries contribute no `u` coordinates and are returned unchanged by `unflatten`."""
    manager = ParamManager(_mixed_mode_specs())
    assert not any("a.fixed" in name for name in manager.names())
    u = manager.init_u()
    theta = manager.unflatten(u)
    np.testing.assert_array_equal(np.asarray(theta["a.fixed"]), np.array([-0.3, 0.0, 0.2]))


def test_e1_t03b_scalar_shape_fixed_and_free_specs() -> None:
    """A scalar (`shape=()`) spec works for both `mode='fixed'` and `mode='free'`."""
    specs = (
        ParamSpec(name="scalar.fixed", shape=(), domain="real", mode="fixed", value=np.array(0.5)),
        ParamSpec(name="scalar.free", shape=(), domain="positive", mode="free", init=np.array(1.5)),
    )
    manager = ParamManager(specs)
    theta = manager.unflatten(manager.init_u())
    assert float(theta["scalar.fixed"]) == pytest.approx(0.5)
    assert float(theta["scalar.free"]) == pytest.approx(1.5)


def test_e1_t03b_tied_entries_share_exactly_one_coordinate() -> None:
    """Changing the shared `tied` coordinate changes every group member identically."""
    manager = ParamManager(_mixed_mode_specs())
    names = manager.names()
    tied_indices = [i for i, name in enumerate(names) if "tied:g1" in name]
    assert len(tied_indices) == 1  # exactly one u coordinate for the whole group

    u = manager.init_u()
    u_perturbed = u.at[tied_indices[0]].add(0.7)
    theta = manager.unflatten(u_perturbed)
    tied_values = jnp.concatenate([theta["a.tied"], theta["a.tied_other"]])
    assert bool(jnp.all(tied_values == tied_values[0]))  # all group members identical

    theta_base = manager.unflatten(u)
    assert not bool(jnp.any(theta["a.tied"] == theta_base["a.tied"]))  # actually changed


def test_e1_t03b_tied_group_with_incompatible_shapes_raises_config_error() -> None:
    """Two specs sharing a `tie_group` with different domains/cell shapes raise `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager(
            (
                ParamSpec(
                    name="m.tied_a",
                    shape=(2,),
                    domain="positive",
                    mode="tied",
                    tie_group="mismatch",
                    init=np.ones(2),
                ),
                ParamSpec(
                    name="m.tied_b",
                    shape=(2, 3),
                    domain="simplex_ref",
                    mode="tied",
                    tie_group="mismatch",
                    init=np.full((2, 3), 1.0 / 3),
                ),
            )
        )


# --- AC4: log_det_jac / log_prior jittable, differentiable; no-prior case is exactly 0 -------


def test_e1_t03b_log_prior_is_exactly_zero_without_priors() -> None:
    """`log_prior` is exactly 0 when no spec has `mode == 'prior'`."""
    manager = ParamManager((_spec_free("x", 4), _spec_free("y", 2, domain="real")))
    theta = manager.unflatten(manager.init_u())
    assert manager.log_prior(theta) == 0.0


def test_e1_t03b_log_det_jac_and_log_prior_are_jittable_and_differentiable() -> None:
    """`log_det_jac`/`log_prior` are jittable, and `jax.grad` through them is finite."""
    manager = ParamManager(_mixed_mode_specs())
    u0 = manager.init_u()

    log_det = jax.jit(manager.log_det_jac)(u0)
    assert jnp.isfinite(log_det)
    grad_det = jax.grad(jax.jit(manager.log_det_jac))(u0)
    assert bool(jnp.all(jnp.isfinite(grad_det)))

    def log_prior_of_u(u: jax.Array) -> jax.Array:
        return manager.log_prior(manager.unflatten(u))

    value = jax.jit(log_prior_of_u)(u0)
    assert jnp.isfinite(value)
    grad_prior = jax.grad(jax.jit(log_prior_of_u))(u0)
    assert bool(jnp.all(jnp.isfinite(grad_prior)))


# --- AC5: constraint=increasing produces strictly increasing theta for any u -----------------


@given(u=st.lists(st.floats(min_value=-6.0, max_value=6.0, allow_nan=False), min_size=4, max_size=4))
def test_e1_t03b_increasing_constraint_is_strictly_increasing(u: list[float]) -> None:
    """`constraint='increasing'` produces a strictly increasing `theta` for any `u`."""
    spec = ParamSpec(
        name="emission.sigma",
        shape=(4,),
        domain="positive",
        mode="free",
        init=np.array([0.1, 0.2, 0.3, 0.4]),
        constraint="increasing",
    )
    manager = ParamManager((spec,))
    theta = manager.unflatten(jnp.asarray(u))["emission.sigma"]
    assert bool(jnp.all(jnp.diff(theta) > 0))


def test_e1_t03b_increasing_constraint_round_trips_and_names() -> None:
    """`flatten`/`unflatten` round-trip an `increasing`-constrained spec; `names()` labels it."""
    spec = ParamSpec(
        name="emission.sigma",
        shape=(4,),
        domain="positive",
        mode="free",
        init=np.array([0.1, 0.2, 0.3, 0.4]),
        constraint="increasing",
    )
    manager = ParamManager((spec,))
    assert manager.names() == [f"emission.sigma[increasing]:{i}" for i in range(4)]

    u0 = manager.init_u()
    theta = manager.unflatten(u0)
    np.testing.assert_allclose(np.asarray(theta["emission.sigma"]), spec.init, atol=1e-9)
    u_back = manager.flatten(theta)
    np.testing.assert_allclose(np.asarray(u_back), np.asarray(u0), atol=1e-9)


def test_e1_t03b_increasing_with_prior_mode_computes_and_contributes_log_prior() -> None:
    """`constraint='increasing'` combined with `mode='prior'` round-trips and adds a prior term."""
    spec = ParamSpec(
        name="emission.sigma",
        shape=(3,),
        domain="positive",
        mode="prior",
        init=np.array([0.1, 0.2, 0.3]),
        prior=NormalPrior(sd=1.0),
        constraint="increasing",
    )
    manager = ParamManager((spec,))
    u0 = manager.init_u()
    theta = manager.unflatten(u0)
    assert bool(jnp.all(jnp.diff(theta["emission.sigma"]) > 0))
    np.testing.assert_allclose(np.asarray(manager.flatten(theta)), np.asarray(u0), atol=1e-9)
    assert manager.log_prior(theta) != 0.0


def test_e1_t03b_increasing_with_bounded_mode_computes() -> None:
    """`constraint='increasing'` combined with `mode='bounded'` uses the interval bijector."""
    spec = ParamSpec(
        name="transition.coef",
        shape=(3,),
        domain="interval",
        mode="bounded",
        bounds=(-1.0, 1.0),
        init=np.array([-0.5, 0.0, 0.5]),
        constraint="increasing",
    )
    manager = ParamManager((spec,))
    u0 = manager.init_u()
    theta = manager.unflatten(u0)
    assert bool(jnp.all(jnp.diff(theta["transition.coef"]) > 0))
    assert bool(jnp.all(theta["transition.coef"] > -1.0))
    assert bool(jnp.all(theta["transition.coef"] < 1.0))
    np.testing.assert_allclose(np.asarray(manager.flatten(theta)), np.asarray(u0), atol=1e-9)


# --- Error cases: fixed without value; increasing + per-state fixed --------------------------


def test_e1_t03b_fixed_without_value_raises_config_error() -> None:
    """`mode='fixed'` without `value` raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager((ParamSpec(name="a", shape=(2,), domain="real", mode="fixed"),))


def test_e1_t03b_increasing_with_fixed_raises_config_error() -> None:
    """`constraint='increasing'` combined with `mode='fixed'` raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager(
            (
                ParamSpec(
                    name="b",
                    shape=(2,),
                    domain="positive",
                    mode="fixed",
                    value=np.ones(2),
                    constraint="increasing",
                ),
            )
        )


def test_e1_t03b_increasing_with_per_state_mode_raises_config_error() -> None:
    """`constraint='increasing'` combined with a per-state mode tuple raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager(
            (
                ParamSpec(
                    name="c",
                    shape=(2,),
                    domain="positive",
                    mode=("free", "free"),
                    init=np.ones(2),
                    constraint="increasing",
                ),
            )
        )


def test_e1_t03b_bounded_without_interval_domain_raises_config_error() -> None:
    """`mode='bounded'` without `domain='interval'`/`bounds` raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager((ParamSpec(name="d", shape=(2,), domain="real", mode="bounded"),))


def test_e1_t03b_prior_mode_without_prior_raises_config_error() -> None:
    """`mode='prior'` without a `prior` set raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager((ParamSpec(name="e", shape=(2,), domain="real", mode="prior"),))


def test_e1_t03b_tied_without_tie_group_raises_config_error() -> None:
    """`mode='tied'` without `tie_group` raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager((ParamSpec(name="f", shape=(2,), domain="real", mode="tied"),))


def test_e1_t03b_interval_domain_without_bounds_raises_config_error() -> None:
    """`domain='interval'` without `bounds` raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager((ParamSpec(name="g", shape=(2,), domain="interval", mode="free", init=np.zeros(2)),))


def test_e1_t03b_unknown_domain_raises_config_error() -> None:
    """An unrecognized `domain` raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager(
            (ParamSpec(name="h", shape=(2,), domain="bogus", mode="free", init=np.zeros(2)),)  # type: ignore[arg-type]
        )


def test_e1_t03b_mode_tuple_length_mismatch_raises_config_error() -> None:
    """A per-state `mode` tuple whose length doesn't match the leading shape raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager(
            (ParamSpec(name="i", shape=(3,), domain="real", mode=("free", "free"), init=np.zeros(3)),)
        )


def test_e1_t03b_increasing_wrong_ndim_raises_config_error() -> None:
    """`constraint='increasing'` on a spec whose shape isn't 1-D raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager(
            (
                ParamSpec(
                    name="j",
                    shape=(2, 2),
                    domain="positive",
                    mode="free",
                    init=np.ones((2, 2)),
                    constraint="increasing",
                ),
            )
        )


def test_e1_t03b_increasing_prior_without_prior_raises_config_error() -> None:
    """`constraint='increasing'` with `mode='prior'` but no `prior` set raises `ConfigError`."""
    with pytest.raises(ConfigError):
        ParamManager(
            (
                ParamSpec(
                    name="k",
                    shape=(3,),
                    domain="positive",
                    mode="prior",
                    init=np.ones(3),
                    constraint="increasing",
                ),
            )
        )
