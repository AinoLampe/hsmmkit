"""E3-T05: `JumpTransition` -- row normalization, covariate link, generator mode, memory layout."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.model.params import ParamManager
from regimekit.model.transition import (
    JumpTransition,
    build_memory_layout,
    transition_matrix_generator,
)


def _uniform_theta(n_states: int, order: int) -> dict[str, jnp.ndarray]:
    layout = build_memory_layout(n_states, order)
    r = layout.tuples.shape[0]
    return {"transition.pi_rows": jnp.full((r, n_states - 1), 1.0 / (n_states - 1))}


# --- AC1: rows sum to 1, zero on current regime -------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("n_states", [2, 3, 4])
def test_e3_t05_rows_sum_to_one_and_zero_on_current(n_states: int, order: int) -> None:
    """Rows sum to 1 within 1e-12, exactly zero on the current regime, for orders 1/2, N in {2,3,4}."""
    family = JumpTransition()
    layout = build_memory_layout(n_states, order)
    theta = _uniform_theta(n_states, order)
    out = np.asarray(family.jump_probs(theta, None, layout))[0]  # (R, N)

    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    for r, current in enumerate(layout.current_of):
        assert out[r, current] == 0.0


# --- AC2: no covariates => constant over t -------------------------------------------------------


def test_e3_t05_no_covariates_constant_over_t() -> None:
    """No covariates: `jump_probs` is constant over `t`."""
    family = JumpTransition()
    layout = build_memory_layout(3, 1)
    theta = _uniform_theta(3, 1)
    z_pi = jnp.zeros((5, 0))
    out = np.asarray(family.jump_probs(theta, z_pi, layout))
    for t in range(1, 5):
        np.testing.assert_array_equal(out[t], out[0])


# --- AC3: parameter counts match Appendix B -------------------------------------------------------


@pytest.mark.parametrize(
    ("n_states", "order", "expected"),
    [
        (2, 1, 2 * 0),
        (3, 1, 3 * 1),
        (4, 1, 4 * 2),
        (3, 2, 3 * 2 * 1),
        (4, 2, 4 * 3 * 2),
    ],
)
def test_e3_t05_param_count_matches_appendix_b(n_states: int, order: int, expected: int) -> None:
    """`ParamManager.n_free` for the transition specs equals Appendix B's `N(N-2)`/`N(N-1)(N-2)`."""
    family = JumpTransition()
    specs = family.param_specs(n_states, order, ())
    manager = ParamManager(specs)
    assert manager.n_free == expected


# --- AC4: Appendix E golden (expm vs first-order) -------------------------------------------------


def test_e3_t05_appendix_e_generator_golden() -> None:
    """`lambda=(2,6)`, weekly `dt=1/52`: `expm(Q dt)` off-diagonals vs first-order `lambda*dt`.

    Concept Appendix E states `0.0357 / 0.1070`; the mathematically exact 2-state CTMC value
    (verified independently two ways -- `scipy.linalg.expm` and the closed-form stationary-
    distribution formula `pi_j (1 - exp(-(lambda_0+lambda_1) dt))` -- agree to 15+ digits) is
    `0.035649... / 0.106947...`, which rounds to `0.0356 / 0.1069`. This is a last-digit rounding
    slip in the concept text (logged in STATUS -> Spec fixes), not a semantic disagreement; the
    test asserts the verified-correct value.
    """
    theta = {"transition.pi_rows": jnp.array([[1.0], [1.0]])}
    lam = jnp.array([2.0, 6.0])
    dt = jnp.array([1.0 / 52.0])
    p = np.asarray(transition_matrix_generator(theta, lam, dt))[0]

    assert round(float(p[0, 1]), 4) == pytest.approx(0.0356)
    assert round(float(p[1, 0]), 4) == pytest.approx(0.1069)

    first_order = np.array([2.0, 6.0]) / 52.0
    assert round(first_order[0], 4) == pytest.approx(0.0385)
    assert round(first_order[1], 4) == pytest.approx(0.1154)

    diff01 = abs(float(p[0, 1]) - first_order[0])
    diff10 = abs(float(p[1, 0]) - first_order[1])
    lam_max_dt = 6.0 / 52.0
    assert diff01 <= lam_max_dt**2
    assert diff10 <= lam_max_dt**2


# --- AC5: order-2 layout excludes i=j; current_of correct -----------------------------------------


def test_e3_t05_order2_layout_excludes_equal_tuples_and_current_of_correct() -> None:
    """Order-2 layout excludes tuples with `i = j`; `current_of` is each tuple's last element."""
    layout = build_memory_layout(3, 2)
    assert layout.tuples.shape == (3 * 2, 2)
    assert not np.any(layout.tuples[:, 0] == layout.tuples[:, 1])
    np.testing.assert_array_equal(layout.current_of, layout.tuples[:, -1])

    expected = {(i, j) for i in range(3) for j in range(3) if i != j}
    actual = {tuple(row) for row in layout.tuples}
    assert actual == expected


def test_e3_t05_order1_layout() -> None:
    """Order-1 layout: `R = N`, tuple `(i,)`, `current_of = arange(N)`."""
    layout = build_memory_layout(4, 1)
    assert layout.tuples.shape == (4, 1)
    np.testing.assert_array_equal(layout.current_of, np.arange(4))
