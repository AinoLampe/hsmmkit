"""E5-T02: `minimize` convergence, non-finite handling, determinism."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax

from regimekit.config.schema import EstimationSection
from regimekit.estimation.objective import make_objective
from regimekit.estimation.optimizers import _ADAM_LR, RunRecord, minimize
from regimekit.utils.logging import get_logger
from tests.helpers.objective_fixtures import make_case

LOG = get_logger("test_e5_t02")


def _cfg(**overrides: object) -> EstimationSection:
    return EstimationSection(**overrides)  # type: ignore[arg-type]


def test_e5_t02_lbfgs_converges_near_long_adam_run() -> None:
    """AC1: on a 2-state GBM/geometric problem, lbfgs (<=200 iter) matches a long adam run within 1e-6.

    Adam's cosine-decay-to-zero schedule makes `|delta J| <= tol_f` trivially true near the end of *any*
    fixed-length run regardless of `max_iter` (the step size itself is what is vanishing, not the
    distance to the optimum), so the long reference run disables early stopping (`tol_f`/`tol_g` near
    machine precision) and simply runs the full schedule; verified separately that this run is stable
    under a further 3x more iterations (same value to 1e-9), i.e. it has actually reached the optimum,
    not merely frozen by a decayed-to-zero learning rate. lbfgs's own default `tol_f` (1e-9, scaled by
    `|J| ~ 1600`) stops it a few 1e-6 short of that optimum by AC1's own tolerance -- tightened here to
    1e-11 to actually test "does lbfgs's *trajectory* reach the optimum within 200 iterations", which is
    what the criterion is about.
    """
    case = make_case("gbm", "geometric", 1, t_len=500, seed=1)
    objective = make_objective(case.spec, case.space, case.engine, case.manager, case.data, use_priors=False)

    lbfgs_record = minimize(objective, case.u0, _cfg(optimizer="lbfgs", max_iter=200, tol_f=1e-11), LOG)
    adam_record = minimize(
        objective, case.u0, _cfg(optimizer="adam", max_iter=100_000, tol_f=1e-14, tol_g=1e-10), LOG
    )

    assert lbfgs_record.n_iter <= 200
    assert abs(lbfgs_record.objective - adam_record.objective) <= 1e-6


def test_e5_t02_max_iter_returns_not_converged() -> None:
    """AC3: `max_iter=3` stops after exactly 3 iterations with `converged=False, reason="max_iter"`."""
    case = make_case("gbm", "geometric", 1, t_len=60, seed=2)
    objective = make_objective(case.spec, case.space, case.engine, case.manager, case.data, use_priors=False)

    record = minimize(objective, case.u0, _cfg(optimizer="lbfgs", max_iter=3), LOG)

    assert record.converged is False
    assert record.reason == "max_iter"
    assert record.n_iter == 3


def test_e5_t02_two_runs_give_bitwise_identical_u() -> None:
    """AC5: two runs with identical inputs give bitwise identical `u` (no randomness in `minimize`)."""
    case = make_case("gbm", "geometric", 1, t_len=60, seed=3)
    objective = make_objective(case.spec, case.space, case.engine, case.manager, case.data, use_priors=False)
    cfg = _cfg(optimizer="lbfgs", max_iter=50)

    r1 = minimize(objective, case.u0, cfg, LOG)
    r2 = minimize(objective, case.u0, cfg, LOG)

    assert np.array_equal(np.asarray(r1.u), np.asarray(r2.u))
    assert r1.objective == r2.objective
    assert r1.n_iter == r2.n_iter


def _rosenbrock(u: jnp.ndarray) -> jnp.ndarray:
    """The banana function: even lbfgs needs dozens of iterations on this, unlike a plain quadratic."""
    return jnp.sum(100.0 * (u[1:] - u[:-1] ** 2) ** 2 + (1.0 - u[:-1]) ** 2)


@dataclasses.dataclass
class _FakeObjective:
    """A duck-typed stand-in for `Objective` whose `value_and_grad` goes non-finite at one exact `u`.

    The trigger is a pure, jit-traceable function of `u` (equality with a captured constant array), not
    a Python-side call counter: `minimize`'s per-iteration step is fused into one `jax.jit`-ed function
    (for speed, see `_make_step`), and a Python side effect inside a jitted function only fires once, at
    trace time -- a call-counting double would silently never trigger on replay. `blow_up_u=None` (via
    the all-NaN sentinel) never matches, so the double behaves like a plain Rosenbrock objective.

    Used with `optimizer="adam"` (no line search): lbfgs's zoom line search actively backtracks away
    from a candidate step where `value_fn` returns a non-finite value (robust, correct behaviour), so a
    poisoned-at-one-point double exercises a *different* trajectory once poisoned, not a clean abort at a
    known point -- `minimize`'s own non-finite bookkeeping is exercised directly and unambiguously with a
    plain gradient step instead.
    """

    blow_up_u: jnp.ndarray

    def _matches(self, u: jnp.ndarray) -> jnp.ndarray:
        # a tolerance, not exact equality: `blow_up_u` is discovered by replaying the same update
        # sequence through a *different* (unfused, eager) code path than `minimize`'s own fused
        # `jax.jit`-ed step, so the two can differ by a few ULPs while being "the same" point.
        return jnp.allclose(u, self.blow_up_u, atol=1e-8, rtol=1e-8)

    def value(self, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(self._matches(u), jnp.inf, _rosenbrock(u))

    def value_and_grad(self, u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        value, grad = jax.value_and_grad(_rosenbrock)(u)
        value = jnp.where(self._matches(u), jnp.inf, value)
        return value, grad

    def loglik(self, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(0.0)


_NEVER = jnp.full(3, jnp.nan)  # `u == nan` is always False: a `blow_up_u` that never matches


def _adam_trajectory(objective: _FakeObjective, u0: jnp.ndarray, max_iter: int, n_steps: int) -> jnp.ndarray:
    """`u` after `n_steps` adam updates under the same schedule `minimize(..., max_iter=max_iter)` uses.

    `cfg.max_iter` sets *both* the loop length and the cosine-decay schedule's length (see
    `_make_optimizer`), so two `minimize()` calls with different `max_iter` follow different learning
    rate schedules from step 1 -- their trajectories cannot be spliced together. This replicates
    `minimize`'s adam step exactly (same schedule, same update order) to read off an intermediate `u`
    from *one* fixed schedule instead.
    """
    schedule = optax.cosine_decay_schedule(init_value=_ADAM_LR, decay_steps=max(max_iter, 1), alpha=0.0)
    opt = optax.adam(learning_rate=schedule)
    state = opt.init(u0)
    u = u0
    _, grad = objective.value_and_grad(u)
    for _ in range(n_steps):
        updates, state = opt.update(grad, state, u)
        u = optax.apply_updates(u, updates)
        _, grad = objective.value_and_grad(u)
    return u


def test_e5_t02_non_finite_aborts_at_offending_iteration() -> None:
    """AC2: an objective going non-finite mid-run aborts with the *previous* accepted iterate's `u`."""
    u0 = jnp.zeros(3)
    # unsatisfiable (negative) tolerances: the run must not stop early via tol_f/tol_g before the
    # crafted point is reached.
    never_converge = _cfg(optimizer="adam", max_iter=50, tol_f=-1.0, tol_g=-1.0)

    harmless = _FakeObjective(blow_up_u=_NEVER)
    u4 = _adam_trajectory(harmless, u0, max_iter=50, n_steps=4)
    u5 = _adam_trajectory(harmless, u0, max_iter=50, n_steps=5)
    assert not np.array_equal(np.asarray(u4), np.asarray(u5)), "fixture needs >= 5 distinct iterates"

    poisoned = _FakeObjective(blow_up_u=u5)
    record = minimize(poisoned, u0, never_converge, LOG)

    assert record.reason == "non_finite"
    assert record.converged is False
    assert record.n_iter == 4
    # not bitwise: `u4` comes from a separate, unfused replay (see `_adam_trajectory`), which can differ
    # from `minimize`'s own fused `jax.jit`-ed step by a few ULPs on the same mathematical trajectory.
    assert np.allclose(np.asarray(record.u), np.asarray(u4), atol=1e-8, rtol=1e-8)


def test_e5_t02_non_finite_start_reports_distinct_reason() -> None:
    """A non-finite `u0` is reported as `"non_finite_start"`, distinct from a mid-run abort."""
    u0 = jnp.zeros(3)
    fake = _FakeObjective(blow_up_u=u0)

    record = minimize(fake, u0, _cfg(optimizer="adam", max_iter=50), LOG)

    assert record.reason == "non_finite_start"
    assert record.n_iter == 0
    assert np.array_equal(np.asarray(record.u), np.asarray(u0))


def test_e5_t02_run_record_reports_loglik_separately_from_objective() -> None:
    """`RunRecord.loglik` is the plain float `ell(theta(u))`, not `J(u)`."""
    case = make_case("gbm", "geometric", 1, t_len=60, seed=4)
    objective = make_objective(case.spec, case.space, case.engine, case.manager, case.data, use_priors=False)

    record = minimize(objective, case.u0, _cfg(optimizer="lbfgs", max_iter=50), LOG)

    assert isinstance(record, RunRecord)
    assert isinstance(record.loglik, float)
    # use_priors=False means J == -loglik bitwise, so this must hold exactly
    assert record.objective == -record.loglik
