"""Viterbi decoding on the expanded state space and regime-segment run-length encoding (E4-T05).

Concept refs: §4.7.4. Max-product in log space: `delta_t(s) = ell_t(s) + max_{s'} [delta_{t-1}(s') +
log A_t(s', s)]`, argmax `psi_t (T, S)` int32 kept for the backtrack (`T*S` memory, never `(T, S, S)`).
Ties break to the lowest index (`argmax` semantics), matching `tests/reference/ref_hmm.viterbi`.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.engine.constraints import check_feasible
from hsmmkit.engine.expanded_hmm import _emission_inputs, _log_a
from hsmmkit.utils.errors import NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace


@partial(jax.jit, static_argnames=("spec",))
def _viterbi_core(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array]:
    """`(path (T,), best log-probability)`."""
    log_rho, ell_0, _, xs = _emission_inputs(theta, spec, space, data)
    delta_0 = log_rho + ell_0

    active = data.boundaries is not None

    def forward_step(delta_prev: Array, xs_t: tuple[Array, ...]) -> tuple[Array, Array]:
        ell_t, z_pi_t, z_d_t, dt_t, code_t = xs_t
        log_a = _log_a(space, theta, z_pi_t, z_d_t, dt_t, code_t, active)
        scores = delta_prev[:, None] + log_a  # (S_prev, S)
        psi_t = jnp.argmax(scores, axis=0).astype(jnp.int32)
        delta_t = ell_t + jnp.max(scores, axis=0)
        return delta_t, psi_t

    delta_last, psi = jax.lax.scan(forward_step, delta_0, xs)  # psi[t-1] = psi_t, t = 1..T-1

    def back_step(s_next: Array, psi_t: Array) -> tuple[Array, Array]:
        s_t = psi_t[s_next]
        return s_t, s_t

    s_last = jnp.argmax(delta_last).astype(jnp.int32)
    _, earlier = jax.lax.scan(back_step, s_last, psi, reverse=True)  # earlier[t] = path[t], t = 0..T-2
    return jnp.concatenate([earlier, s_last[None]], axis=0), jnp.max(delta_last)


def viterbi_path(theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
    """Most likely expanded-state path `(T,)` (int32 state ids).

    The regime path is `space.index.regime_of[path]`.

    Raises:
        NumericalError: if the best path has non-finite log-probability (checked outside jit).
    """
    path: Array
    best: Array
    path, best = _viterbi_core(theta, spec, space, data)
    check_feasible(best, data, None)
    if not isinstance(best, jax.core.Tracer) and not bool(jnp.isfinite(best)):
        raise NumericalError("non-finite best-path log-probability in Viterbi", path="viterbi")
    return path


def segments(regime_path: np.ndarray | Array) -> list[tuple[int, int, int]]:
    """Run-length encoding of a regime path as `(start, end, regime)`; `end` is inclusive."""
    path = np.asarray(regime_path)
    if path.ndim != 1:
        raise NumericalError("regime path must be one-dimensional", path="regime_path")
    if path.size == 0:
        return []
    change = np.flatnonzero(path[1:] != path[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change - 1, [path.size - 1]])
    return [(int(a), int(b), int(path[a])) for a, b in zip(starts, ends, strict=True)]
