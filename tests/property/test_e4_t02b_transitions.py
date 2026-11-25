"""E4-T02b property checks: independent-reference A_t, no-retrace jit, differentiability."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import softmax

from hsmmkit.model.duration.negbin import NegBinDuration
from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.spec import ModelSpec
from hsmmkit.model.transition import JumpTransition, build_memory_layout
from hsmmkit.statespace.builders import build_expanded


def _spec(
    n_states: int,
    order: int,
    duration: object,
    transition: object,
    *,
    duration_conditioning: str = "current",
    initial: str = "uniform",
) -> ModelSpec:
    return ModelSpec(
        n_states=n_states,
        n_series=1,
        emission=GaussianFamily(scale_by_dt=True, ito_correction=True),
        duration=duration,  # type: ignore[arg-type]
        transition=transition,  # type: ignore[arg-type]
        order=order,
        duration_conditioning=duration_conditioning,  # type: ignore[arg-type]
        engine="expanded",
        d_max=10,
        initial=initial,  # type: ignore[arg-type]
        param_specs=(),
    )


def _ref_a(
    n_states: int,
    order: int,
    r: list[int],
    rate: np.ndarray,
    pi_a: np.ndarray | None,
    pi_b: np.ndarray | None,
    z_pi: np.ndarray | None,
    c: np.ndarray | None,
    z_d: np.ndarray | None,
    dt: float,
    pi_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Independent NumPy transcription of the stay/advance/exit `A_t` construction."""
    layout = build_memory_layout(n_states, order)
    tuples = layout.tuples
    current_of = layout.current_of
    r_rows = tuples.shape[0]

    lam = rate.copy()
    if c is not None and z_d is not None:
        lam = lam * np.exp(c @ z_d)
    a_of_current = 1.0 - np.exp(-lam * dt)  # (N,)

    def targets_of(current: int) -> list[int]:
        return [j for j in range(n_states) if j != current]

    def pi_row(row_idx: int) -> np.ndarray:
        if pi_a is None:
            assert pi_rows is not None
            return np.asarray(pi_rows[row_idx])
        logits_nonref = pi_a[row_idx] + (pi_b[row_idx] @ z_pi if z_pi is not None else 0.0)
        logits = np.concatenate([[0.0], logits_nonref])
        return softmax(logits)

    states = []
    row_of = []
    row_start = np.empty(r_rows, dtype=np.int64)
    s = 0
    for row_idx in range(r_rows):
        current = int(current_of[row_idx])
        row_start[row_idx] = s
        for phase in range(1, r[current] + 1):
            states.append((current, phase))
            row_of.append(row_idx)
            s += 1
    n_expanded = s

    def shift(memory: tuple[int, ...], current: int) -> tuple[int, ...]:
        if order == 1:
            return ()
        return (*memory[1:], current)

    memories = tuples[:, :-1]
    row_lookup = {tuple(row): idx for idx, row in enumerate(tuples)}

    a_mat = np.zeros((n_expanded, n_expanded))
    for state_idx, (current, phase) in enumerate(states):
        row_idx = row_of[state_idx]
        a_i = a_of_current[current]
        if phase < r[current]:
            a_mat[state_idx, state_idx] = 1.0 - a_i
            a_mat[state_idx, state_idx + 1] = a_i
            continue
        a_mat[state_idx, state_idx] = 1.0 - a_i
        pi = pi_row(row_idx)
        memory_r = tuple(int(m) for m in memories[row_idx])
        shifted = shift(memory_r, current)
        for k, target in enumerate(targets_of(current)):
            entry_row = row_lookup[(*shifted, target)]
            entry_state = row_start[entry_row]
            a_mat[state_idx, entry_state] += a_i * pi[k]
    return a_mat


@given(
    n_states=st.integers(2, 4),
    order=st.integers(1, 2),
    data=st.data(),
)
def test_e4_t02b_matches_independent_reference_no_covariates(
    n_states: int, order: int, data: st.DataObject
) -> None:
    """`A_t` matches an independent NumPy reference, <= 1e-12, random N/r_i/order, no covariates."""
    r = data.draw(st.lists(st.integers(1, 3), min_size=n_states, max_size=n_states))
    rate = np.asarray(data.draw(st.lists(st.floats(0.01, 0.3), min_size=n_states, max_size=n_states)))
    dt = 1.0 / 252.0

    duration = NegBinDuration(r=tuple(r))
    transition = JumpTransition()
    spec = _spec(n_states, order, duration, transition)
    space = build_expanded(spec)

    r_rows = space.layout.tuples.shape[0]
    rng = np.random.default_rng(0)
    pi_rows = rng.dirichlet(np.ones(n_states - 1), size=r_rows)
    theta = {"duration.rate": jnp.asarray(rate), "transition.pi_rows": jnp.asarray(pi_rows)}

    a, _ = space.transition_fn(theta, None, None, jnp.asarray(dt))
    ref = _ref_a(n_states, order, r, rate, None, None, None, None, None, dt, pi_rows=pi_rows)

    np.testing.assert_allclose(np.asarray(a), ref, atol=1e-12)


def test_e4_t02b_matches_independent_reference_with_covariates() -> None:
    """`A_t` matches the independent reference with duration/transition covariates."""
    n_states, order, r = 3, 1, [2, 1, 3]
    rate = np.array([0.05, 0.1, 0.2])
    rng = np.random.default_rng(1)
    c = rng.normal(size=(n_states, 1))
    z_d = rng.normal(size=(1,))
    dt = 1.0 / 252.0

    duration = NegBinDuration(r=tuple(r), cov_idx=(0,))
    transition = JumpTransition(cov_idx=(0,))
    spec = _spec(n_states, order, duration, transition)
    space = build_expanded(spec)

    r_rows = space.layout.tuples.shape[0]
    pi_a = rng.normal(size=(r_rows, n_states - 2))
    pi_b = rng.normal(size=(r_rows, n_states - 2, 1))
    z_pi = rng.normal(size=(1,))

    theta = {
        "duration.rate": jnp.asarray(rate),
        "duration.c": jnp.asarray(c),
        "transition.a": jnp.asarray(pi_a),
        "transition.b": jnp.asarray(pi_b),
    }
    a, _ = space.transition_fn(theta, jnp.asarray(z_pi), jnp.asarray(z_d), jnp.asarray(dt))
    ref = _ref_a(n_states, order, r, rate, pi_a, pi_b, z_pi, c, z_d, dt)

    np.testing.assert_allclose(np.asarray(a), ref, atol=1e-12)


# --- AC3: jit compiles once per StateSpace, no retrace across theta/z -----------------------------


def test_e4_t02b_no_retrace_across_theta_and_z() -> None:
    """`jax.jit(transition_fn)` compiles once; different `theta`/`z` values don't retrace."""
    duration = NegBinDuration(r=(2, 3))
    spec = _spec(2, 1, duration, JumpTransition())
    space = build_expanded(spec)

    jitted = jax.jit(space.transition_fn)
    theta1 = {"duration.rate": jnp.array([0.1, 0.2]), "transition.pi_rows": jnp.array([[1.0], [1.0]])}
    theta2 = {"duration.rate": jnp.array([0.3, 0.5]), "transition.pi_rows": jnp.array([[1.0], [1.0]])}

    jitted(theta1, None, None, jnp.array(1.0 / 252.0))
    jitted(theta2, None, None, jnp.array(2.0 / 252.0))
    assert jitted._cache_size() == 1


# --- AC5: differentiable ---------------------------------------------------------------------------


def test_e4_t02b_differentiable_wrt_theta() -> None:
    """`jax.grad` of `sum(A_t)` w.r.t. `theta` is finite."""
    duration = NegBinDuration(r=(2, 3))
    spec = _spec(2, 1, duration, JumpTransition())
    space = build_expanded(spec)

    def total(theta: dict[str, jax.Array]) -> jax.Array:
        a, _ = space.transition_fn(theta, None, None, jnp.array(1.0 / 252.0))
        return jnp.sum(a)

    theta = {"duration.rate": jnp.array([0.1, 0.2]), "transition.pi_rows": jnp.array([[1.0], [1.0]])}
    grads = jax.grad(total)(theta)
    assert np.all(np.isfinite(np.asarray(grads["duration.rate"])))
    assert np.all(np.isfinite(np.asarray(grads["transition.pi_rows"])))


# --- rate_index_of / n_rates shape guard ("pair" duration conditioning) --------------------------


def test_e4_t02b_transition_fn_rejects_mismatched_duration_rate_shape() -> None:
    """A `theta['duration.rate']` shaped for the wrong `duration_conditioning` raises `DataError`."""
    from hsmmkit.utils.errors import DataError

    duration = NegBinDuration(r=(2, 2, 2))
    spec = _spec(3, 2, duration, JumpTransition(), duration_conditioning="pair")
    space = build_expanded(spec)
    assert space.n_rates == 6

    wrong_shape_theta = {
        "duration.rate": jnp.full((3,), 0.1),  # sized for "current" (N=3), not "pair" (n_rates=6)
        "transition.pi_rows": jnp.full((space.layout.tuples.shape[0], 1), 1.0),
    }
    with pytest.raises(DataError):
        space.transition_fn(wrong_shape_theta, None, None, jnp.array(0.1))


# --- stationary initial distribution: sign-robust, differentiable --------------------------------


@given(n_states=st.integers(2, 5), seed=st.integers(0, 2**31 - 1))
def test_e4_t02b_stationary_distribution_never_nan(n_states: int, seed: int) -> None:
    """Power-iteration stationary distribution is never NaN, regardless of eigenvector sign quirks."""
    from hsmmkit.statespace.builders import _stationary_distribution

    rng = np.random.default_rng(seed)
    a_bar = rng.dirichlet(np.ones(n_states), size=n_states)
    vec = np.asarray(_stationary_distribution(jnp.asarray(a_bar)))
    assert np.all(np.isfinite(vec))
    assert np.all(vec >= -1e-9)
    np.testing.assert_allclose(vec.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(vec @ a_bar, vec, atol=1e-4)


def test_e4_t02b_stationary_initial_distribution_differentiable() -> None:
    """`jax.grad` through the `stationary` initial distribution is finite (power iteration, not eig)."""
    from hsmmkit.statespace.builders import initial_log_rho

    duration = NegBinDuration(r=(2, 3))
    spec = _spec(2, 1, duration, JumpTransition(), initial="stationary")
    space = build_expanded(spec)

    def total(rate: jax.Array) -> jax.Array:
        theta = {"duration.rate": rate, "transition.pi_rows": jnp.array([[1.0], [1.0]])}
        return jnp.sum(jnp.exp(initial_log_rho(spec, space, theta, nominal_dt=1.0 / 252.0)))

    grad = jax.grad(total)(jnp.array([0.1, 0.2]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_e4_t02b_stationary_with_transition_covariates_uses_zero_vector_not_none() -> None:
    """`stationary` evaluates covariate-logit transitions at `z=0`, not a shape-mismatch crash."""
    from hsmmkit.statespace.builders import initial_log_rho

    duration = NegBinDuration(r=(2, 3))
    transition = JumpTransition(cov_idx=(0,))
    spec = _spec(2, 1, duration, transition, initial="stationary")
    space = build_expanded(spec)
    r_rows = space.layout.tuples.shape[0]
    theta = {
        "duration.rate": jnp.array([0.1, 0.2]),
        "transition.a": jnp.zeros((r_rows, 0)),
        "transition.b": jnp.zeros((r_rows, 0, 1)),
    }
    log_pi = initial_log_rho(spec, space, theta, nominal_dt=1.0 / 252.0)
    assert np.all(np.isfinite(np.asarray(log_pi)))
