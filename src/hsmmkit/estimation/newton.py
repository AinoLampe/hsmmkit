"""Damped Newton minimization with a BFGS fallback (E13-T03b).

`newton_minimize` minimizes a smooth `fun(u)` over a few unconstrained coordinates (the M-step of a parametric
duration family): modified Newton steps (the Hessian eigenvalues are replaced by `max(|lambda|, lm)`, so every
step is a descent direction; `lm = 1e-6`) with step halving until the objective decreases, at most 30
iterations. If Newton makes no progress from a non-stationary start (a singular, non-finite or badly scaled
Hessian), BFGS (`jax.scipy.optimize`) runs from the start instead. The result never has a higher objective
than
the start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.optimize as jopt

if TYPE_CHECKING:
    from collections.abc import Callable

    from hsmmkit.utils.typing import Array

__all__ = ["newton_minimize"]

_MAX_HALVINGS = 30


def _newton(fun: Callable[[Array], Array], u0: Array, max_iter: int, lm: float) -> tuple[Array, Array]:
    grad = jax.grad(fun)
    hess: Callable[[Array], Array] = jax.hessian(fun)
    try:  # some special functions (the incomplete gamma) have no second derivative: central differences
        jax.eval_shape(hess, u0)
    except NotImplementedError:

        def hess(u: Array) -> Array:
            h = 1e-5 * jnp.maximum(jnp.abs(u), 1.0)
            cols = [
                (grad(u + h[i] * e) - grad(u - h[i] * e)) / (2.0 * h[i])
                for i, e in enumerate(jnp.eye(u.size))
            ]
            return jnp.stack(cols, axis=1)

    def direction(u: Array) -> Array:
        g, h = grad(u), hess(u)
        h = jnp.where(jnp.isfinite(h), h, 0.0)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        lam, vec = jnp.linalg.eigh(0.5 * (h + h.T))
        lam = jnp.maximum(jnp.abs(lam), lm)
        step: Array = -(vec @ ((vec.T @ g) / lam))
        return step

    def body(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        u, f, it, _ = state
        delta = direction(u)

        def halve(c: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            t, _, k = c
            t = 0.5 * t
            return t, fun(u + t * delta), k + 1

        def not_better(c: tuple[Array, Array, Array]) -> Array:
            _, ft, k = c
            return ((~jnp.isfinite(ft)) | (ft >= f)) & (k < _MAX_HALVINGS)

        f1 = fun(u + delta)
        t, ft, _k = jax.lax.while_loop(not_better, halve, (jnp.ones(()), f1, jnp.zeros((), jnp.int32)))
        ok = jnp.isfinite(ft) & (ft < f)
        u_new = jnp.where(ok, u + t * delta, u)
        f_new = jnp.where(ok, ft, f)
        done = (~ok) | (jnp.abs(f - f_new) <= 1e-13 * (1.0 + jnp.abs(f)))
        return u_new, f_new, it + 1, done

    def cond(state: tuple[Array, Array, Array, Array]) -> Array:
        return (state[2] < max_iter) & ~state[3]

    f0 = fun(u0)
    u, f, _, _ = jax.lax.while_loop(cond, body, (u0, f0, jnp.zeros((), jnp.int32), jnp.zeros((), bool)))
    return u, f


def newton_minimize(
    fun: Callable[[Array], Array],
    u0: Array,
    *,
    max_iter: int = 30,
    lm: float = 1e-6,
    fallback_iter: int = 100,
) -> tuple[Array, Array, Array]:
    """`(u, fun(u), used_fallback)`: damped Newton from `u0`, BFGS if it stalls (never worse than `u0`)."""
    f0 = fun(u0)
    u_n, f_n = _newton(fun, u0, max_iter, lm)
    gnorm = jnp.max(jnp.abs(jnp.where(jnp.isfinite(jax.grad(fun)(u0)), jax.grad(fun)(u0), 0.0)))
    stalled = ~(f_n < f0) & (gnorm > 1e-6)

    def run_bfgs(_: None) -> tuple[Array, Array]:
        res = jopt.minimize(fun, u0, method="BFGS", options={"maxiter": fallback_iter})
        better = jnp.isfinite(res.fun) & (res.fun <= f0)
        return jnp.where(better, res.x, u0), jnp.where(better, res.fun, f0)

    u, f = jax.lax.cond(stalled, run_bfgs, lambda _: (u_n, f_n), None)
    return u, f, stalled
