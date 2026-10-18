"""Augmented-posterior utilities: phase/memory marginals, expected remaining/elapsed steps (E4-T04).

Concept refs: §4.7.1, §5.4. All functions are pure and jittable (`Posterior` is a pytree; `spec` is
closed over/static). Regime-conditional expectations return 0 (not NaN) where a regime has no
posterior mass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.engine.base import Posterior
    from regimekit.model.params import Params
    from regimekit.model.spec import ModelSpec


def collapse_to_regimes(x: Array, collapse: Array) -> Array:
    """`(T, S) @ C (S, N) -> (T, N)`: sum a per-state quantity over each regime's states."""
    return x @ collapse


def phase_marginals(post: Posterior) -> Array:
    """`(T, N, r_max)`: smoothed mass per `(regime, phase)`, summed over memory; zero for `phase > r_i`."""
    index = post.index
    n_regimes = index.collapse.shape[1]
    max_phase = _max_phase(index.phase_of)
    regime_one_hot = jax.nn.one_hot(index.regime_of, n_regimes)  # (S, N)
    phase_one_hot = jax.nn.one_hot(index.phase_of - 1, max_phase)  # (S, r_max)
    indicator = regime_one_hot[:, :, None] * phase_one_hot[:, None, :]  # (S, N, r_max)
    return jnp.einsum("ts,snp->tnp", post.smoothed, indicator)


def _max_phase(phase_of: Array) -> int:
    """Static `r_max`; `Posterior.index` is pytree aux data, so `phase_of` is concrete even under jit."""
    return int(np.max(np.asarray(phase_of)))


def memory_marginals(post: Posterior) -> Array:
    """`(T, R)` smoothed mass per memory-tuple row (order >= 2), else `(T, 1)` (all mass)."""
    index = post.index
    if index.order < 2:
        return jnp.sum(post.smoothed, axis=-1, keepdims=True)
    n_rows = index.n_states * (index.n_states - 1) ** (index.order - 1)
    row_one_hot = jax.nn.one_hot(index.row_of_state, n_rows)  # (S, R)
    return post.smoothed @ row_one_hot


def _phase_expectation(
    post: Posterior, spec: ModelSpec, theta: Params, dt: Array, z_d: Array | None, elapsed: bool
) -> Array:
    phases = phase_marginals(post)  # (T, N, r_max)
    n_regimes, r_max = phases.shape[1], phases.shape[2]
    r = jnp.asarray([spec.duration.n_phases(i) for i in range(n_regimes)])  # (N,)
    c = jnp.arange(1, r_max + 1)[None, :]  # (1, r_max)
    steps = (c - 1) if elapsed else (r[:, None] - c + 1)  # (N, r_max) phases still to go / already done
    a = spec.duration.advance_prob(theta, dt, z_d)  # (T, N)
    a_safe = jnp.where(a > 0, a, 1.0)  # double-where: a == 0 must not poison gradients
    weighted = jnp.sum(phases * steps[None, :, :], axis=-1) / a_safe  # (T, N)
    regime_mass = jnp.sum(phases, axis=-1)
    safe_mass = jnp.where(regime_mass > 0, regime_mass, 1.0)
    return jnp.where(regime_mass > 0, weighted / safe_mass, 0.0)


def expected_remaining_steps(
    post: Posterior, spec: ModelSpec, theta: Params, dt: Array, z_d: Array | None = None
) -> Array:
    """`(T, N)`: `E[remaining steps in the regime | regime i at t]` for the phase-type (NB) chain.

    In phase `c` of `r_i`, `r_i - c + 1` phases remain, each lasting `1 / a_i(t)` steps in
    expectation (constant-rate approximation at the step's own `a_i(t)`), weighted by the
    smoothed phase marginals conditional on regime `i`.
    """
    return _phase_expectation(post, spec, theta, dt, z_d, elapsed=False)


def expected_elapsed_steps(
    post: Posterior, spec: ModelSpec, theta: Params, dt: Array, z_d: Array | None = None
) -> Array:
    """`(T, N)`: phase-based **lower bound** on the elapsed steps in the current regime.

    `(c - 1) / a_i(t)` for phase `c` (the time already spent in earlier phases; time spent inside
    phase `c` itself is not counted), conditional on regime `i`. Exact elapsed-duration posteriors
    come from the explicit-duration engine (`Posterior.elapsed_posterior`, E4-T06a): a run-length
    recursion on regime marginals is not exact for an HSMM.
    """
    return _phase_expectation(post, spec, theta, dt, z_d, elapsed=True)
