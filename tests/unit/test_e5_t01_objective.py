"""E5-T01: objective decomposition, guards and compilation reuse."""

from __future__ import annotations

import dataclasses
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.estimation import objective as obj
from hsmmkit.estimation.objective import check_finite, make_objective
from hsmmkit.utils.errors import NumericalError
from tests.helpers.engine_fixtures import DT, make_dataset
from tests.helpers.objective_fixtures import Case, make_case


def build(case: Case, use_priors: bool) -> obj.Objective:
    """`make_objective` over a case."""
    return make_objective(case.spec, case.space, case.engine, case.manager, case.data, use_priors)


@pytest.mark.parametrize(
    ("emission", "duration", "order"),
    [("gbm", "negbin", 1), ("student_t", "geometric", 2), ("gbm", "gamma", 1)],
)
def test_e5_t01_mle_value_equals_negative_loglik_bitwise(emission: str, duration: str, order: int) -> None:
    """AC2: `use_priors=False` gives exactly `-engine.loglik(theta(u))`."""
    case = make_case(emission, duration, order, cov_em=True, cov_du=True, seed=3)
    objective = build(case, use_priors=False)
    theta = case.manager.unflatten(case.u0)
    expected = -case.engine.loglik(theta, case.spec, case.space, case.data)
    got = objective.value(case.u0)
    assert float(got) == float(expected)
    assert np.asarray(got).shape == ()
    value, grad = objective.value_and_grad(case.u0)
    assert float(value) == float(got)
    assert grad.shape == (objective.n_free,)


@pytest.mark.parametrize(
    ("emission", "duration", "order", "kwargs"),
    [
        ("gbm", "negbin", 1, {}),
        ("student_t", "geometric", 2, {"cov_em": True}),
        ("gbm", "gamma", 1, {"cov_du": True}),
        ("gbm", "geometric", 1, {"cov_tr": True, "dirichlet_alpha": 1.5}),
    ],
)
def test_e5_t01_map_value_decomposes_into_loglik_prior_and_log_det(
    emission: str, duration: str, order: int, kwargs: dict[str, object]
) -> None:
    """AC3: `J = -loglik - log_prior - log_det_jac`, with the transition's extra Dirichlet term included."""
    case = make_case(emission, duration, order, priors=True, seed=5, **kwargs)  # type: ignore[arg-type]
    theta = case.manager.unflatten(case.u0)
    loglik = float(case.engine.loglik(theta, case.spec, case.space, case.data))
    log_prior = float(case.manager.log_prior(theta))
    extra = getattr(case.spec.transition, "log_prior_extra", None)
    if extra is not None and case.spec.transition.dirichlet_alpha is not None:  # type: ignore[attr-defined]
        extra_term = float(extra(theta, case.space.layout))
        assert extra_term != 0.0
        log_prior += extra_term
    log_det = float(case.manager.log_det_jac(case.u0))
    assert log_prior != 0.0
    assert log_det != 0.0

    mle = float(build(case, use_priors=False).value(case.u0))
    mapv = float(build(case, use_priors=True).value(case.u0))
    assert mle == pytest.approx(-loglik, abs=1e-12)
    assert mapv == pytest.approx(-loglik - log_prior - log_det, abs=1e-10)
    assert mapv - mle == pytest.approx(-log_prior - log_det, abs=1e-10)


def test_e5_t01_check_finite_names_the_first_offending_coordinate() -> None:
    """AC4: a crafted gradient with `inf` at coordinate 3 is reported by name."""
    case = make_case("gbm", "negbin", 1, seed=1)
    names = case.manager.names()
    assert len(names) == case.manager.n_free
    grad = np.zeros(case.manager.n_free)
    grad[3] = np.inf
    grad[5] = np.nan
    with pytest.raises(NumericalError) as exc:
        check_finite(jnp.asarray(1.0), jnp.asarray(grad), case.u0, names)
    assert exc.value.path == names[3]
    assert names[3] in str(exc.value)
    assert "2 of" in str(exc.value)


def test_e5_t01_check_finite_passes_finite_and_flags_a_bad_value() -> None:
    """Finite `(J, g)` pass; a non-finite value with a finite gradient names the objective."""
    case = make_case("gbm", "negbin", 1, seed=1)
    grad = jnp.zeros(case.manager.n_free)
    check_finite(jnp.asarray(2.0), grad, case.u0, case.manager.names())
    with pytest.raises(NumericalError) as exc:
        check_finite(jnp.asarray(jnp.inf), grad, case.u0, case.manager.names())
    assert exc.value.path == "objective"


def test_e5_t01_a_crafted_u_that_overflows_is_caught_by_check_finite() -> None:
    """A `u` sending every `sigma` to infinity gives a non-finite result that `check_finite` rejects."""
    case = make_case("gbm", "negbin", 1, seed=1)
    objective = build(case, use_priors=True)
    names = case.manager.names()
    u = case.u0
    for i, name in enumerate(names):
        if name.startswith("emission.scale"):
            u = u.at[i].set(jnp.inf)  # sigma = +inf in every regime
    value, grad = objective.value_and_grad(u)
    assert not (np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(grad))))
    with pytest.raises(NumericalError):
        check_finite(value, grad, u, names)


def test_e5_t01_second_call_with_new_u_and_new_data_does_not_recompile() -> None:
    """AC5: same-shape `u` and `data` (even with different timestamps/statistics) reuse one compilation."""
    case = make_case("gbm", "negbin", 1, seed=2)
    objective = build(case, use_priors=True)
    objective.value_and_grad(case.u0)
    objective.value(case.u0)
    n_vg, n_v = obj._value_and_grad._cache_size(), obj._value._cache_size()

    rng = np.random.default_rng(9)
    other = make_dataset(rng.normal(0.0, 0.01, size=(40, 1)), np.full(40, DT))
    later = dataclasses.replace(
        other,
        times=other.times + np.timedelta64(1000, "D"),
        snapshot_ids=("another-snapshot",),
        prep_stats=dataclasses.replace(other.prep_stats, train_window=("1999-01-01", "2000-01-01")),
    )
    v1, g1 = objective.value_and_grad(case.u0 + 0.1, data=later)
    v2 = objective.value(case.u0 - 0.1, data=other)
    assert obj._value_and_grad._cache_size() == n_vg
    assert obj._value._cache_size() == n_v
    assert np.isfinite(float(v1))
    assert np.all(np.isfinite(np.asarray(g1)))
    assert np.isfinite(float(v2))
    # the override really changes the value (it is not silently using the bound data)
    assert float(v1) != float(objective.value(case.u0 + 0.1))


def test_e5_t01_different_shapes_do_recompile_but_stay_correct() -> None:
    """A dataset with another `T` compiles a new specialization (sanity check of the cache probe)."""
    case = make_case("gbm", "negbin", 1, seed=2)
    objective = build(case, use_priors=False)
    objective.value(case.u0)
    before = obj._value._cache_size()
    longer = make_dataset(np.zeros((55, 1)), np.full(55, DT))
    objective.value(case.u0, data=longer)
    assert obj._value._cache_size() == before + 1


def test_e5_t01_explicit_engine_irregular_grid_warns_once_at_construction() -> None:
    """The explicit engine cannot warn under jit, so `make_objective` warns (once) for an irregular grid."""
    case = make_case("gbm", "gamma", 1, seed=1)
    rng = np.random.default_rng(0)
    irregular = make_dataset(
        rng.normal(0.0, 0.01, size=(40, 1)), DT * rng.integers(1, 4, size=40).astype(float)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        objective = make_objective(case.spec, case.space, case.engine, case.manager, irregular, True)
        objective.value_and_grad(case.u0)
        objective.value_and_grad(case.u0)
    assert sum("regular grid" in str(w.message) for w in caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build(make_case("gbm", "negbin", 1, seed=1), True)  # expanded engine: no warning


def test_e5_t01_objective_is_a_pytree_over_space_and_data() -> None:
    """`space` and `data` are leaves (dynamic); spec/engine/manager are static."""
    import jax

    case = make_case("gbm", "negbin", 1, seed=2)
    objective = build(case, use_priors=True)
    leaves = jax.tree_util.tree_leaves(objective)
    assert any(getattr(x, "shape", None) == (40, 1) for x in leaves)  # observations
    assert objective.n_free == case.manager.n_free
