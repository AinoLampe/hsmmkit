"""E4-T02a property checks: hypothesis S-formula, jit safety of StateSpace, initial_log_rho."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from regimekit.model.duration.negbin import NegBinDuration
from regimekit.model.emission.gbm import GBMFamily
from regimekit.model.spec import ModelSpec
from regimekit.model.transition import JumpTransition
from regimekit.statespace.builders import build_expanded, initial_log_rho


def _spec(n_states: int, order: int, duration: object, initial: str = "uniform") -> ModelSpec:
    return ModelSpec(
        n_states=n_states,
        n_series=1,
        emission=GBMFamily(),
        duration=duration,  # type: ignore[arg-type]
        transition=JumpTransition(),
        order=order,
        duration_conditioning="current",
        engine="expanded",
        d_max=10,
        initial=initial,  # type: ignore[arg-type]
        param_specs=(),
    )


@given(
    n_states=st.integers(2, 4),
    order=st.integers(1, 2),
    data=st.data(),
)
def test_e4_t02a_s_formula_property(n_states: int, order: int, data: st.DataObject) -> None:
    """`S` matches the closed-form formula for random `N`/`order`/`r_i`."""
    r = data.draw(st.lists(st.integers(1, 3), min_size=n_states, max_size=n_states))
    duration = NegBinDuration(r=tuple(r))
    space = build_expanded(_spec(n_states, order, duration))
    expected = sum(r) if order == 1 else (n_states - 1) * sum(r)
    assert space.n_expanded == expected


def test_e4_t02a_order_greater_than_2_warns() -> None:
    """`order > 2` triggers a `warnings.warn` with the actual `S`."""
    duration = NegBinDuration(r=(2, 2, 2))
    with pytest.warns(UserWarning, match="order=3"):
        build_expanded(_spec(3, 3, duration))


# --- jit safety: StateSpace (with its static MemoryLayout) is hashable/jit-safe ------------------


def test_e4_t02a_state_space_jittable() -> None:
    """`StateSpace` (incl. its static `layout`) is jittable as an actual jit argument, no crash."""
    duration = NegBinDuration(r=(2, 3))
    space = build_expanded(_spec(2, 1, duration))

    @jax.jit
    def f(x: jax.Array, sp: object) -> jax.Array:
        return x + sp.n_expanded  # type: ignore[attr-defined]

    assert float(f(jnp.array(1.0), space)) == pytest.approx(1.0 + space.n_expanded)


def test_e4_t02a_state_space_no_recompile_across_equal_layouts() -> None:
    """A function static-closing over `MemoryLayout` does not retrace across equal instances.

    Two distinct-but-structurally-equal `MemoryLayout` instances exercise its content-based
    `__hash__`/`__eq__` (CLAUDE.md §5). `StateSpace.transition_fn` (E4-T02b) is a fresh Python
    closure per `build_expanded` call, so it -- unlike `layout` -- is not expected to compare equal
    across instances; AC3 for that (E4-T02b) is single-`StateSpace` recompile-freedom, tested in
    `tests/property/test_e4_t02b_transitions.py`.
    """
    duration = NegBinDuration(r=(2, 3))
    space1 = build_expanded(_spec(2, 1, duration))
    space2 = build_expanded(_spec(2, 1, duration))
    assert space1.layout is not space2.layout

    @functools.partial(jax.jit, static_argnums=0)
    def f(layout: object) -> jax.Array:
        return jnp.asarray(layout.n_states)  # type: ignore[attr-defined]

    f(space1.layout)
    f(space2.layout)
    assert f._cache_size() == 1


def test_e4_t02a_state_space_hash_and_eq_consistent() -> None:
    """Two `StateSpace`s built from the same spec are structurally equal (and hash the same layout)."""
    duration = NegBinDuration(r=(2, 3))
    space1 = build_expanded(_spec(2, 1, duration))
    space2 = build_expanded(_spec(2, 1, duration))
    assert space1.layout == space2.layout
    assert hash(space1.layout) == hash(space2.layout)


# --- initial_log_rho ------------------------------------------------------------------------------


def test_e4_t02a_initial_log_rho_uniform() -> None:
    """`uniform`: `-inf` off `phase==1`, uniform mass (sums to 1 after `exp`) on it."""
    duration = NegBinDuration(r=(2, 3))
    spec = _spec(2, 1, duration, initial="uniform")
    space = build_expanded(spec)
    log_rho = np.asarray(initial_log_rho(spec, space, {}))
    phase_of = np.asarray(space.index.phase_of)

    assert np.all(np.isneginf(log_rho[phase_of != 1]))
    assert np.all(np.isfinite(log_rho[phase_of == 1]))
    np.testing.assert_allclose(np.exp(log_rho[phase_of == 1]).sum(), 1.0, atol=1e-12)


def test_e4_t02a_initial_log_rho_free() -> None:
    """`free`: `theta["initial.logits"]` (already a simplex) is scattered onto `phase==1` states."""
    duration = NegBinDuration(r=(2, 3))
    spec = _spec(2, 1, duration, initial="free")
    space = build_expanded(spec)
    n_phase1 = space.phase1_idx.shape[0]
    probs = jnp.full((n_phase1,), 1.0 / n_phase1)
    log_rho = np.asarray(initial_log_rho(spec, space, {"initial.logits": probs}))
    phase_of = np.asarray(space.index.phase_of)

    np.testing.assert_allclose(np.exp(log_rho[phase_of == 1]).sum(), 1.0, atol=1e-12)
    assert np.all(np.isneginf(log_rho[phase_of != 1]))


def test_e4_t02a_initial_log_rho_unknown_mode_raises() -> None:
    """An unrecognized `spec.initial` value raises `ConfigError` (`stationary` is real, E4-T02b)."""
    from regimekit.utils.errors import ConfigError

    duration = NegBinDuration(r=(2, 3))
    spec = _spec(2, 1, duration, initial="bogus")
    space = build_expanded(spec)
    with pytest.raises(ConfigError):
        initial_log_rho(spec, space, {})


def test_e4_t02a_initial_log_rho_jittable_and_differentiable() -> None:
    """`initial_log_rho` (free mode) is jittable and differentiable in `theta`."""
    duration = NegBinDuration(r=(2, 3))
    spec = _spec(2, 1, duration, initial="free")
    space = build_expanded(spec)
    n_phase1 = space.phase1_idx.shape[0]

    def total(probs: jax.Array) -> jax.Array:
        return jnp.sum(jnp.exp(initial_log_rho(spec, space, {"initial.logits": probs})))

    jitted = jax.jit(total)
    probs0 = jnp.full((n_phase1,), 1.0 / n_phase1)
    assert float(jitted(probs0)) == pytest.approx(1.0, abs=1e-6)
    grad = jax.grad(total)(probs0)
    assert np.all(np.isfinite(np.asarray(grad)))
