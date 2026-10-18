"""Single-start optimization of the objective (E5-T02): L-BFGS + zoom line search, Adam fallback.

Concept refs: §4.8 Optimizers, §5.4 (`FitResult.restarts`). `minimize` drives one run from one starting
point `u0` to a `RunRecord`; multi-start and initialization strategy are E5-T03's job.

Convergence: `|delta J| <= tol_f * max(1, |J|)` or `||grad J||_inf <= tol_g`, else `max_iter` reached.
Adam's convergence test is only evaluated every 10 iterations (its per-step objective estimate is noisier
than L-BFGS's line-search-verified one). A non-finite value or gradient aborts the run without raising;
the caller (E5-T03) decides what to do with a `converged=False` record.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import optax

from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.config.schema import EstimationSection
    from regimekit.estimation.objective import Objective

_ADAM_LR = 1.0e-2
_LOG_EVERY = 25
_ADAM_CHECK_EVERY = 10


class _ObjectiveLike(Protocol):
    """The slice of `Objective` `minimize` needs (also satisfied by test doubles)."""

    def value(self, u: Array) -> Array: ...
    def value_and_grad(self, u: Array) -> tuple[Array, Array]: ...
    def loglik(self, u: Array) -> Array: ...


@dataclass(frozen=True)
class RunRecord:
    """The outcome of one optimization run from one starting point.

    Attributes:
        objective: `J(u)` at the returned `u`.
        loglik: `ell(theta(u))` at the returned `u` (Python float, unlike `Objective`'s array).
        n_iter: Number of accepted iterations (0 if `u0` itself was non-finite).
        converged: Whether a convergence test (not `max_iter` or a non-finite abort) stopped the run.
        wall_s: Wall-clock seconds spent in `minimize`.
        reason: One of `"tol_f"`, `"tol_g"`, `"max_iter"`, `"non_finite"`, `"non_finite_start"`.
        u: The returned point (last finite point reached).
    """

    objective: float
    loglik: float
    n_iter: int
    converged: bool
    wall_s: float
    reason: str
    u: Array


def _all_finite(*arrays: Array) -> bool:
    return all(bool(jnp.all(jnp.isfinite(a))) for a in arrays)


def _make_optimizer(cfg: EstimationSection) -> optax.GradientTransformationExtraArgs:
    if cfg.optimizer == "lbfgs":
        linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=20)
        return optax.lbfgs(linesearch=linesearch)
    schedule = optax.cosine_decay_schedule(init_value=_ADAM_LR, decay_steps=max(cfg.max_iter, 1), alpha=0.0)
    return optax.adam(learning_rate=schedule)


_Step = Callable[[Array, Array, Array, optax.OptState], tuple[Array, Array, Array, optax.OptState]]


def _make_step(
    objective: _ObjectiveLike, opt: optax.GradientTransformationExtraArgs, is_lbfgs: bool
) -> _Step:
    """One jitted iteration: optimizer update + apply + re-evaluate `(value, grad)` at the new point.

    Fusing this into a single `jax.jit` call (rather than dispatching `opt.update`,
    `optax.apply_updates` and `objective.value_and_grad` as separate eager JAX calls) is what keeps the
    Python loop fast: L-BFGS's zoom line search alone issues several objective evaluations per outer
    iteration, and unfused eager dispatch of each one dominates the wall clock.
    """

    def value_fn(u: Array) -> Array:
        return objective.value(u)

    # For lbfgs, fetch (value, grad) from the post-update state when the line search already computed
    # them at the accepted point (optax's own idiom for this optimizer), instead of unconditionally
    # recomputing via a fresh `objective.value_and_grad`.
    value_and_grad_from_state = optax.value_and_grad_from_state(value_fn)

    @jax.jit
    def step(
        u_prev: Array, value_prev: Array, grad_prev: Array, state: optax.OptState
    ) -> tuple[Array, Array, Array, optax.OptState]:
        if is_lbfgs:
            updates, new_state = opt.update(
                grad_prev, state, u_prev, value=value_prev, grad=grad_prev, value_fn=value_fn
            )
            u = optax.apply_updates(u_prev, updates)
            value, grad = value_and_grad_from_state(u, state=new_state)
        else:
            updates, new_state = opt.update(grad_prev, state, u_prev)
            u = optax.apply_updates(u_prev, updates)
            value, grad = objective.value_and_grad(u)
        return u, value, grad, new_state

    return step


def _final_loglik(objective: _ObjectiveLike, u: Array) -> float:
    return float(objective.loglik(u))


def minimize(
    objective: Objective,
    u0: Array,
    cfg: EstimationSection,
    log: object,
) -> RunRecord:
    """Minimize `objective.value` from `u0`, following `cfg.optimizer`'s schedule and tolerances.

    Never raises for a non-finite value/gradient: it reports `reason="non_finite"` (or
    `"non_finite_start"` when `u0` itself is bad) and returns the last point known to be finite. `log`
    is a structlog-style bound logger; a progress event (`iter`, `J`, `grad_norm`, `elapsed`) is emitted
    every `k` iterations (`_LOG_EVERY`, default 25).
    """
    start = time.perf_counter()
    is_lbfgs = cfg.optimizer == "lbfgs"
    opt = _make_optimizer(cfg)

    value, grad = objective.value_and_grad(u0)
    if not _all_finite(value, grad):
        return RunRecord(
            objective=float("nan"),
            loglik=float("nan"),
            n_iter=0,
            converged=False,
            wall_s=time.perf_counter() - start,
            reason="non_finite_start",
            u=u0,
        )

    step = _make_step(objective, opt, is_lbfgs)
    state = opt.init(u0)
    u_prev, value_prev, grad_prev = u0, value, grad
    u, converged, reason, n_iter = u0, False, "max_iter", 0

    for i in range(1, cfg.max_iter + 1):
        u, value, grad, state = step(u_prev, value_prev, grad_prev, state)

        if not _all_finite(value, grad):
            u, n_iter, reason = u_prev, i - 1, "non_finite"
            value = value_prev
            break

        n_iter = i
        d_obj = abs(float(value) - float(value_prev))
        g_inf = float(jnp.max(jnp.abs(grad)))
        u_prev, value_prev, grad_prev = u, value, grad

        check_now = is_lbfgs or i % _ADAM_CHECK_EVERY == 0
        if check_now:
            if d_obj <= cfg.tol_f * max(1.0, abs(float(value))):
                converged, reason = True, "tol_f"
                break
            if g_inf <= cfg.tol_g:
                converged, reason = True, "tol_g"
                break

        if i % _LOG_EVERY == 0:
            log.info(  # type: ignore[attr-defined]
                "estimation.optimize.progress",
                iter=i,
                J=float(value),
                grad_norm=g_inf,
                elapsed=time.perf_counter() - start,
            )

    return RunRecord(
        objective=float(value),
        loglik=_final_loglik(objective, u),
        n_iter=n_iter,
        converged=converged,
        wall_s=time.perf_counter() - start,
        reason=reason,
        u=u,
    )
