"""E5-T01 AC1: `jax.grad` of the MAP objective vs central finite differences over the model grid."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from regimekit.estimation.objective import check_finite, make_objective
from tests.helpers.objective_fixtures import make_case

COV_FLAGS = list(itertools.product([False, True], repeat=3))  # (transition, duration, emission)
GRID = [
    (emission, duration, order, flags)
    for emission in ("gbm", "student_t")
    for duration, order in (("geometric", 1), ("geometric", 2), ("negbin", 1), ("negbin", 2), ("gamma", 1))
    for flags in COV_FLAGS
]

STEP = 1e-6
REL = 1e-5
NOISE_FLOOR = (
    1e-7  # FD noise: a ~1e2 log-likelihood in float64 at step 1e-6 (1e-8 fails 6 of 80 cases on noise alone)
)


def _id(param: tuple[str, str, int, tuple[bool, bool, bool]]) -> str:
    emission, duration, order, (tr, du, em) = param
    tags = "".join(t for t, on in (("T", tr), ("D", du), ("E", em)) if on) or "-"
    return f"{emission}-{duration}-o{order}-{tags}"


@pytest.mark.parametrize("param", GRID, ids=[_id(p) for p in GRID])
def test_e5_t01_gradient_matches_central_finite_differences(
    param: tuple[str, str, int, tuple[bool, bool, bool]],
) -> None:
    """Every coordinate: |grad - fd| <= 1e-5 |fd| + noise floor (step 1e-6), `T=40`.

    `N=2`, or `N=3` with transition covariates (`N=2` has no free logit to attach them to).
    """
    emission, duration, order, (tr, du, em) = param
    case = make_case(
        emission,
        duration,
        order,
        cov_tr=tr,
        cov_du=du,
        cov_em=em,
        priors=True,
        dirichlet_alpha=1.5 if tr else None,
        seed=11,
    )
    objective = make_objective(case.spec, case.space, case.engine, case.manager, case.data, True)
    value, grad = objective.value_and_grad(case.u0)
    check_finite(value, grad, case.u0, case.manager.names())
    grad_np = np.asarray(grad)
    assert grad_np.shape == (case.manager.n_free,)
    assert case.manager.n_free > 0

    names = case.manager.names()
    for i in range(case.manager.n_free):
        plus = float(objective.value(case.u0.at[i].add(STEP)))
        minus = float(objective.value(case.u0.at[i].add(-STEP)))
        fd = (plus - minus) / (2 * STEP)
        assert abs(grad_np[i] - fd) <= REL * abs(fd) + NOISE_FLOOR, (names[i], grad_np[i], fd)


def test_e5_t01_grid_covers_the_ticket_combinations() -> None:
    """{GBM, Student-t} x {geometric, negbin, gamma-explicit} x {order 1, 2 (expanded)} x 8 covariate sets."""
    assert len(GRID) == 2 * 5 * 8
    assert {g[1] for g in GRID} == {"geometric", "negbin", "gamma"}
    assert all(order == 1 for _, duration, order, _ in GRID if duration == "gamma")
