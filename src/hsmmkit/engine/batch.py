"""Batched engines: forward-backward, log-likelihood and Viterbi over a `PaddedBatch` (E12-T01b, ADR-0021).

Every function `jax.vmap`s a per-sequence recursion over the batch axis of a `PaddedBatch`; `spec`/`space`
are static, so there is one compiled program per `(B, T_max)` bucket. Padding semantics (`valid == False`):

* **expanded engine**: the emission log-density of a padded step is 0 and its transition is the identity, so
  `alpha` is constant on the padding, `logsumexp alpha_{T_max} = loglik_b` and `beta` is 0 on the padding;
* **explicit engine**: the emission cumulative sums are constant on the padding (masked steps contribute 0)
  and the right-censored final segment ends at the true length `T_b` (`_Inputs.length`); segments that would
  start in the padding have survival term `-inf`.

Empty padding sequences (`length == 0`) contribute exactly 0 to the log-likelihood and 0 posterior mass.
With `initial.logits` of shape `(G, S_phase1)` (`initial_groups: per_group`, E12-T02) sequence `b` starts from
row `group_index[b]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.data.contracts import PreparedDataset, PrepStats
from hsmmkit.engine.base import Posterior
from hsmmkit.engine.expanded_hmm import (
    ExpandedHMMEngine,
    _emission_inputs,
    _safe_logsumexp,
)
from hsmmkit.engine.explicit_duration import (
    ExplicitDurationEngine,
    _forward_backward_core,
    _loglik_core,
    _viterbi_core,
)
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateIndex, StateSpace

__all__ = [
    "BatchedEngine",
    "BatchedPosterior",
    "batched_forward_backward",
    "batched_loglik",
    "batched_viterbi",
]

_STATS = PrepStats(
    cov_mean=np.zeros(0), cov_std=np.ones(0), train_window=("", ""), cov_names=(), transforms={}, lags={}
)


def _dataset_view(pb: PaddedBatch) -> PreparedDataset:
    """A `PreparedDataset` whose array leaves carry a leading batch axis (for `vmap`)."""
    t_max = pb.observations.shape[1]
    times = np.arange(t_max).astype("datetime64[D]").astype("datetime64[ns]")

    def block(x: Array) -> Array | None:
        return None if x.shape[-1] == 0 else x

    return PreparedDataset(
        times=times,
        observations=pb.observations,
        dt=pb.dt,
        mask=pb.mask,
        cov_transition=block(pb.cov_transition),
        cov_duration=block(pb.cov_duration),
        cov_emission=block(pb.cov_emission),
        cov_names={},
        clock=pb.clock,
        nominal_dt=pb.nominal_dt,
        prep_stats=_STATS,
        snapshot_ids=(),
        labels=pb.labels,
    )


# --- expanded engine ------------------------------------------------------------------------------------


def _expanded_inputs(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset, valid: Array
) -> tuple[Array, Array, tuple[Array, Array, Array, Array, Array], Array]:
    """`(alpha_0, log_eye, xs, ell)`: padded steps have emission 0 and (in the scans) identity transitions."""
    log_rho, _, ell, xs = _emission_inputs(theta, spec, space, data)
    ell = jnp.where(valid[:, None], ell, 0.0)
    _, z_pi, z_d, dt = xs[0], xs[1], xs[2], xs[3]
    s = ell.shape[1]
    log_eye = jnp.where(jnp.eye(s, dtype=bool), 0.0, -jnp.inf)
    return log_rho + ell[0], log_eye, (ell[1:], z_pi, z_d, dt, valid[1:]), ell


def _expanded_forward(
    theta: Params,
    space: StateSpace,
    alpha_0: Array,
    log_eye: Array,
    xs: Any,  # noqa: ANN401
) -> Array:
    def step(alpha_prev: Array, x: Any) -> tuple[Array, Array]:  # noqa: ANN401
        ell_t, z_pi_t, z_d_t, dt_t, v = x
        _, log_a = space.transition_fn(theta, z_pi_t, z_d_t, dt_t)
        log_a = jnp.where(v, log_a, log_eye)
        alpha_t = ell_t + _safe_logsumexp(alpha_prev[:, None] + log_a, axis=0)
        return alpha_t, alpha_t

    _, rest = jax.lax.scan(step, alpha_0, xs)
    return jnp.concatenate([alpha_0[None, :], rest], axis=0)


def _expanded_one(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset, valid: Array, kind: str
) -> tuple[Array, ...]:
    alpha_0, log_eye, xs, _ = _expanded_inputs(theta, spec, space, data, valid)
    log_alpha = _expanded_forward(theta, space, alpha_0, log_eye, xs)
    loglik = _safe_logsumexp(log_alpha[-1], axis=0)
    if kind == "loglik":
        return (loglik,)
    if kind == "viterbi":
        return (_expanded_viterbi(theta, space, alpha_0, log_eye, xs),)
    s = log_alpha.shape[1]

    def back(beta_next: Array, x: Any) -> tuple[Array, Array]:  # noqa: ANN401
        ell_next, z_pi_next, z_d_next, dt_next, v_next = x
        _, log_a = space.transition_fn(theta, z_pi_next, z_d_next, dt_next)
        log_a = jnp.where(v_next, log_a, log_eye)
        beta_t = _safe_logsumexp(log_a + (ell_next + beta_next)[None, :], axis=1)
        return beta_t, beta_t

    beta_last = jnp.zeros((s,))
    _, rest = jax.lax.scan(back, beta_last, xs, reverse=True)
    log_beta = jnp.concatenate([rest, beta_last[None, :]], axis=0)
    log_gamma = log_alpha + log_beta
    norm = _safe_logsumexp(log_gamma, axis=1)[:, None]
    finite = jnp.isfinite(log_gamma)
    smoothed = jnp.where(finite, jnp.exp(jnp.where(finite, log_gamma - norm, 0.0)), 0.0)
    return log_alpha, log_beta, smoothed, loglik


def _expanded_viterbi(theta: Params, space: StateSpace, alpha_0: Array, log_eye: Array, xs: Any) -> Array:  # noqa: ANN401
    def step(delta_prev: Array, x: Any) -> tuple[Array, Array]:  # noqa: ANN401
        ell_t, z_pi_t, z_d_t, dt_t, v = x
        _, log_a = space.transition_fn(theta, z_pi_t, z_d_t, dt_t)
        log_a = jnp.where(v, log_a, log_eye)
        scores = delta_prev[:, None] + log_a
        return ell_t + jnp.max(scores, axis=0), jnp.argmax(scores, axis=0).astype(jnp.int32)

    delta_last, psi = jax.lax.scan(step, alpha_0, xs)

    def back_step(s_next: Array, psi_t: Array) -> tuple[Array, Array]:
        s_t = psi_t[s_next]
        return s_t, s_t

    s_last = jnp.argmax(delta_last).astype(jnp.int32)
    _, earlier = jax.lax.scan(back_step, s_last, psi, reverse=True)
    return jnp.concatenate([earlier, s_last[None]], axis=0)


# --- explicit engine ---------------------------------------------------------------------------------------


def _explicit_one(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset, length: Array, kind: str
) -> tuple[Array, ...]:
    eff = jnp.maximum(length, 1)  # an empty sequence is evaluated as a 1-step one and zeroed afterwards
    if kind == "loglik":
        return (_loglik_core(theta, spec, space, data, eff),)
    if kind == "viterbi":
        path, _ = _viterbi_core(theta, spec, space, data, eff)
        return (path,)
    log_alpha, log_beta, smoothed, loglik, _ = _forward_backward_core(theta, spec, space, data, None, eff)
    return log_alpha, log_beta, smoothed, loglik


# --- batched drivers -----------------------------------------------------------------------


@partial(jax.jit, static_argnames=("spec", "kind"))
def _batched_core(
    theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch, kind: str
) -> tuple[Array, ...]:
    data = _dataset_view(pb)

    grouped = "initial.logits" in theta and theta["initial.logits"].ndim == 2

    def theta_of(g: Array) -> Params:
        """The parameters of one sequence: its group's row of `initial.logits` when there are groups."""
        return {**theta, "initial.logits": theta["initial.logits"][g]} if grouped else theta

    def explicit_fn(d: PreparedDataset, n: Array, g: Array) -> tuple[Array, ...]:
        return _explicit_one(theta_of(g), spec, space, d, n, kind)

    def expanded_fn(d: PreparedDataset, v: Array, g: Array) -> tuple[Array, ...]:
        return _expanded_one(theta_of(g), spec, space, d, v, kind)

    if spec.engine == "explicit":
        out = jax.vmap(explicit_fn)(data, pb.lengths, pb.group_index)
    else:
        out = jax.vmap(expanded_fn)(data, pb.valid, pb.group_index)
    empty = pb.lengths == 0
    if kind == "loglik":
        return (jnp.where(empty, 0.0, out[0]),)
    if kind == "viterbi":
        return (jnp.where(pb.valid, out[0], -1),)
    log_alpha, log_beta, smoothed, loglik = out
    keep = pb.valid[:, :, None]
    return (
        jnp.where(keep, log_alpha, -jnp.inf),
        jnp.where(keep, log_beta, -jnp.inf),
        jnp.where(keep, smoothed, 0.0),
        jnp.where(empty, 0.0, loglik),
    )


def _check_all_finite(loglik: Array) -> None:
    """Public-boundary guard: every per-sequence log-likelihood is finite."""
    if not bool(jnp.all(jnp.isfinite(loglik))):
        raise NumericalError("non-finite log-likelihood from the batched recursion", path="loglik")


def _check(spec: ModelSpec, engine: object) -> None:
    expected = ExplicitDurationEngine if spec.engine == "explicit" else ExpandedHMMEngine
    if not isinstance(engine, expected):
        raise ConfigError(
            f"spec.engine is {spec.engine!r} but the engine passed is a {type(engine).__name__}",
            path="model.engine",
        )
    if spec.engine == "explicit" and spec.duration_dependent_jumps:
        raise ConfigError(
            "batched inference with duration-dependent jumps is not supported yet",
            path="model.duration.duration_dependent_jumps",
        )


def batched_loglik(
    engine: object, params: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch
) -> Array:
    """`(B_pad,)` per-sequence log-likelihoods (0 for the empty padding sequences); the total is their sum.

    Raises:
        NumericalError: a non-finite log-likelihood of a real sequence (checked outside jit).
    """
    _check(spec, engine)
    (out,) = _batched_core(params, spec, space, pb, "loglik")
    if not isinstance(out, jax.core.Tracer):
        _check_all_finite(out)
    loglik: Array = out
    return loglik


class BatchedEngine(eqx.Module):
    """Adapter that gives a single-sequence engine the `loglik(theta, spec, space, data)` of a `PaddedBatch`.

    `data` is a `PaddedBatch` and `loglik` the sum of the per-sequence log-likelihoods (the log-likelihood
    of independent sequences), so `estimation.objective.Objective` and the optimizers work unchanged.
    """

    inner: object = eqx.field(static=True)

    def loglik(self, theta: Params, spec: ModelSpec, space: StateSpace, data: PaddedBatch) -> Array:
        """Total log-likelihood `sum_b loglik_b` (0-d)."""
        return jnp.sum(batched_loglik(self.inner, theta, spec, space, data))


@dataclass(frozen=True)
class BatchedPosterior:
    """Forward-backward output of a batch: arrays `(B, T_max, S)`, `loglik (B,)`; padding is `-inf` / 0."""

    log_alpha: Array
    log_beta: Array
    smoothed: Array
    loglik: Array
    index: StateIndex
    lengths: np.ndarray
    engine: Literal["expanded", "explicit"] = "expanded"

    def unpad(self) -> tuple[Posterior, ...]:
        """One Part I `Posterior` per real sequence, cut to its own length."""
        return tuple(
            Posterior(
                log_alpha=self.log_alpha[b, :n],
                log_beta=self.log_beta[b, :n],
                smoothed=self.smoothed[b, :n],
                loglik=self.loglik[b],
                index=self.index,
                engine=self.engine,
            )
            for b, n in enumerate(self.lengths)
        )


def batched_forward_backward(
    engine: object, params: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch
) -> BatchedPosterior:
    """Smoothed posteriors of every sequence; `unpad()` gives the single-sequence `Posterior`s."""
    _check(spec, engine)
    log_alpha, log_beta, smoothed, loglik = _batched_core(params, spec, space, pb, "fb")
    if not isinstance(loglik, jax.core.Tracer):
        _check_all_finite(loglik)
        if bool(jnp.any(jnp.isnan(log_alpha))):
            raise NumericalError("NaN in log_alpha from the batched forward-backward", path="log_alpha")
    n_real = int(np.count_nonzero(np.asarray(pb.lengths)))
    return BatchedPosterior(
        log_alpha=log_alpha,
        log_beta=log_beta,
        smoothed=smoothed,
        loglik=loglik,
        index=space.index,
        lengths=np.asarray(pb.lengths)[:n_real],
        engine="explicit" if spec.engine == "explicit" else "expanded",
    )


def batched_viterbi(
    engine: object, params: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch
) -> Array:
    """`(B_pad, T_max)` most likely expanded-state paths, `-1` on the padding."""
    _check(spec, engine)
    (out,) = _batched_core(params, spec, space, pb, "viterbi")
    paths: Array = out
    return paths
