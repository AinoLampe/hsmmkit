"""Duration-dependent jump probabilities in the explicit-duration engine (E12-T05, FR-DU05).

`pi_ij(d) ∝ Omega_ij exp(a_ij + b_ij . z_t + c_ij log d)` is evaluated with the duration `d` of the segment
being left. The model is a first-order Markov chain on `(regime, elapsed)` states `s = j D_max + e - 1`:

    (i, e) -> (i, e + 1)  with probability  1 - h_i(e)
    (i, e) -> (j, 1)      with probability  h_i(e) pi_ij(e)        h_i(e) = p_i(e) / P(duration >= e)

(`h_i(D_max) = 1`: the pmf is truncated and renormalized at `D_max`, exactly as in the segmental engine), with
emissions gathered by regime. The forward-backward, Viterbi and log-likelihood recursions of the expanded HMM
engine therefore apply unchanged; right-censoring of the last segment is the absence of an end constraint.
The cost is `O(N^2 D_max^2)` per step (a dense `(N D_max)^2` transition matrix) against `O(N^2)` for the
constant-`pi` segmental recursion, which stays the path of every model with `duration_dependent_jumps: false`.
The state layout and the `Posterior` are those of the explicit engine (`(T, N D_max)`).
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.engine.base import Posterior
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.statespace.builders import build_explicit
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

_NEG_INF = -jnp.inf
MAX_STATES_SQUARED = 5e7


@lru_cache(maxsize=16)
def dd_space(spec: ModelSpec) -> StateSpace:
    """The explicit `(regime, elapsed)` state space with the duration-dependent transition function."""
    if spec.duration.cov_idx if hasattr(spec.duration, "cov_idx") else False:
        raise ConfigError(
            "duration-dependent jumps do not combine with entry-time duration covariates",
            path="model.duration.covariates",
        )
    base = build_explicit(spec)
    n, d_max = spec.n_states, spec.d_max
    layout = base.layout
    regimes = np.repeat(np.arange(n), d_max - 1)  # static numpy index arrays: nothing traced is captured
    elapsed = np.tile(np.arange(d_max - 1), n)

    def transition_fn(
        theta: Params, z_pi_t: Array | None, z_d_t: Array | None, dt_t: Array
    ) -> tuple[Array, Array]:
        del z_d_t
        lp = jnp.asarray(spec.duration.log_pmf(theta, d_max, dt_t, None))  # type: ignore[arg-type]  # (N, D)
        log_surv = jax.lax.cumlogsumexp(lp, axis=1, reverse=True)
        finite = log_surv > -jnp.inf
        hazard = jnp.where(finite, jnp.exp(lp - jnp.where(finite, log_surv, 0.0)), 1.0)
        hazard = hazard.at[:, d_max - 1].set(1.0)  # (N, D)
        durations = jnp.arange(1, d_max + 1, dtype=jnp.float64)
        pi_d = spec.transition.jump_probs_dur(theta, z_pi_t, layout, durations)  # type: ignore[attr-defined]
        a = jnp.zeros((n, d_max, n, d_max))
        # exits: (i, e) -> (j, 1)
        a = a.at[:, :, :, 0].set(hazard[:, :, None] * jnp.transpose(pi_d, (1, 0, 2)))
        # continuation: (i, e) -> (i, e + 1)
        a = a.at[regimes, elapsed, regimes, elapsed + 1].set((1.0 - hazard[:, : d_max - 1]).reshape(-1))
        mat = a.reshape(n * d_max, n * d_max)
        safe = jnp.where(mat > 0, mat, 1.0)
        return mat, jnp.where(mat > 0, jnp.log(safe), _NEG_INF)

    return dataclasses.replace(base, transition_fn=transition_fn)


def _view(spec: ModelSpec, data: PreparedDataset) -> tuple[StateSpace, PreparedDataset]:
    """The dd state space and `data` on the nominal grid (the explicit engine ignores irregular `dt`)."""
    s = spec.n_states * spec.d_max
    if s * s > MAX_STATES_SQUARED or data.observations.shape[0] * s > MAX_STATES_SQUARED:
        raise NumericalError(
            f"duration-dependent jumps need a dense ({s} x {s}) transition and a (T, {s}) posterior: "
            f"too large (limit {MAX_STATES_SQUARED:.0e} cells); lower d_max",
            path="model.d_max",
        )
    return dd_space(spec), dataclasses.replace(data, dt=jnp.full_like(data.dt, data.nominal_dt))


def forward(theta: Params, spec: ModelSpec, data: PreparedDataset) -> tuple[Array, Array]:
    """`(log_alpha (T, N D_max), loglik)`."""
    space, d = _view(spec, data)
    return ExpandedHMMEngine().forward(theta, spec, space, d)


def loglik(theta: Params, spec: ModelSpec, data: PreparedDataset) -> Array:
    """The right-censored log-likelihood."""
    space, d = _view(spec, data)
    return ExpandedHMMEngine().loglik(theta, spec, space, d)


def forward_backward(theta: Params, spec: ModelSpec, data: PreparedDataset) -> Posterior:
    """Posterior over `(regime, elapsed)` with the explicit-engine tag."""
    space, d = _view(spec, data)
    post = ExpandedHMMEngine().forward_backward(theta, spec, space, d)
    return dataclasses.replace(post, engine="explicit")


def viterbi(theta: Params, spec: ModelSpec, data: PreparedDataset) -> Array:
    """Most likely `(regime, elapsed)` path `(T,)`, state `j D_max + e - 1`."""
    space, d = _view(spec, data)
    return ExpandedHMMEngine().viterbi(theta, spec, space, d)
