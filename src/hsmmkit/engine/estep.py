"""Expanded-engine E-step: expected counts and statistics from one forward-backward pass (E13-T01a).

Concept refs: Part II §14.6 (E-step), §16.3 item 10; Part I §4.7.2. Everything EM and VB need that is not a
family sufficient statistic is accumulated *inside* a scan, so the pairwise posteriors
`xi_t(s, s') = exp(alpha_{t-1}(s) + log A_t(s, s') + ell_t(s') + beta_t(s') - loglik)` never exist for all `t`
at once: the reverse scan carries `beta_t`, forms the `(S, S)` matrix of the current step, reduces it with the
static index arrays of the state space and adds it to the carry. The forward pass stores `alpha (T, S)`;
emissions are stored per regime `(T, N)` and gathered by `regime_of` inside the scans, so no `(T, S, S)` array
and no second `(T, S)` table (beta, emissions) is kept. Memory is `O(T S + T N + S^2)` per sequence.

Definitions (sums over the valid transitions `t = 1 .. T_b - 1` of each sequence `b`, then over `b`):

* `stay[k]`: `xi_t(s, s)` summed over the states `s` of rate row `k` (`rate_index_of`);
* `advance[k]`: the other mass leaving `s` -- the phase advance, and the exits of the last phase;
* `occupancy[k]`: `gamma_{t-1}(s)` summed over the same states, `t - 1 = 0 .. T_b - 2`, so that
  `stay + advance = occupancy`;
* `jump[r, j]`: `xi_t(s, entry_state(r, j))` over the last-phase states `s` of jump-matrix row `r`
  (`row_of_state`);
* `gamma1[g]`: the posterior `gamma_0(s)` of the first step summed over the sequences of initial group `g`;
* `gamma_regime[b, t]`: regime marginals (padding steps are 0), for the family `sufficient_stats`.

Tempering (`beta_temper in (0, 1]`, E13-T02b) runs the whole pass on `beta * ell`; `loglik` is always the
untempered log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.engine.base import InferenceEngine
from hsmmkit.engine.batch import _dataset_view
from hsmmkit.engine.constraints import apply_transition_mask
from hsmmkit.engine.expanded_hmm import _safe_logsumexp, regime_log_emis
from hsmmkit.statespace.builders import initial_log_rho
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

__all__ = ["ExpandedEStats", "expanded_estep"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "loglik",
        "gamma_regime",
        "gamma1",
        "stay",
        "advance",
        "occupancy",
        "jump",
        "stay_t",
        "advance_t",
        "jump_t",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class ExpandedEStats:
    """Expected counts of one E-step (a pytree).

    Attributes:
        loglik: `(B,)` log-likelihood per sequence (untempered; 0 for empty padding sequences).
        gamma_regime: `(B, T_max, N)` regime marginals, 0 on padding.
        gamma1: `(G, S)` first-step posterior summed per initial group.
        stay: `(R_rate,)` expected self-transitions per duration-rate row.
        advance: `(R_rate,)` expected phase advances plus exits per duration-rate row.
        occupancy: `(R_rate,)` expected steps spent in the states of each rate row before a transition.
        jump: `(R, N)` expected exits from jump-matrix row `r` into regime `j`.
        stay_t, advance_t: `(B, T_max, R_rate)` the per-step terms of `stay` / `advance` (only with
            `keep_per_t`; row `t` is the transition into step `t`, row 0 is 0).
        jump_t: `(B, T_max, R, N)` the per-step terms of `jump` (only with `keep_per_t`).
    """

    loglik: Array
    gamma_regime: Array
    gamma1: Array
    stay: Array
    advance: Array
    occupancy: Array
    jump: Array
    stay_t: Array | None = None
    advance_t: Array | None = None
    jump_t: Array | None = None


def _regime_emissions(theta: Params, spec: ModelSpec, data: Any, valid: Array) -> Array:  # noqa: ANN401
    """`(T, N)` per-regime emission log-terms (labels applied, padding 0)."""
    t_len = data.observations.shape[0]
    z_e = jnp.zeros((t_len, 0)) if data.cov_emission is None else data.cov_emission
    ell = regime_log_emis(theta, spec, data.observations, data.mask, data.dt, z_e)
    if data.labels is not None:
        ell = jnp.where(data.labels, ell, -jnp.inf)  # same mask as `label_log_mask` at the regime level
    return jnp.where(valid[:, None], ell, 0.0)


def _estep_one(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    data: Any,  # noqa: ANN401
    valid: Array,
    code: Array,
    beta: Array,
    *,
    active: bool,
    tempered: bool,
    keep_per_t: bool,
) -> tuple[Array, ...]:
    """One sequence: `(loglik, gamma_regime, gamma_0, stay, advance, occupancy, jump, per-t buffers)`."""
    idx = space.index
    n_state = space.n_expanded
    n_rate = space.n_rates
    n_row, n_reg = space.entry_state_of.shape
    t_len = valid.shape[0]
    z_pi = jnp.zeros((t_len, 0)) if data.cov_transition is None else data.cov_transition
    z_d = jnp.zeros((t_len, 0)) if data.cov_duration is None else data.cov_duration
    log_rho = initial_log_rho(spec, space, theta, data.nominal_dt)
    log_eye = jnp.where(jnp.eye(n_state, dtype=bool), 0.0, -jnp.inf)
    ell_reg = _regime_emissions(theta, spec, data, valid)  # (T, N), untempered
    regime_of = idx.regime_of
    steps = jnp.arange(1, t_len)

    def log_a_at(t: Array) -> Array:
        log_a = space.transition_fn(theta, z_pi[t], z_d[t], data.dt[t])[1]
        if active:
            log_a = apply_transition_mask(log_a, regime_of[:, None] != regime_of[None, :], code[t])
        return jnp.where(valid[t], log_a, log_eye)

    def row_of(ell_table: Array, t: Array | int, scaled: bool) -> Array:
        """The `(S,)` emission row of step `t`, scaled by `beta` when tempering (no scaled table)."""
        row = ell_table[t]
        return (row * beta if scaled and tempered else row)[regime_of]

    def forward(ell_table: Array) -> Array:
        """`(T, S)` forward vectors, written in place into one buffer (no stacked-and-concatenated copy)."""
        alpha_0 = log_rho + row_of(ell_table, 0, True)

        def step(carry: tuple[Array, Array], t: Array) -> tuple[tuple[Array, Array], None]:
            alpha_prev, buf = carry
            alpha_t = row_of(ell_table, t, True) + _safe_logsumexp(alpha_prev[:, None] + log_a_at(t), axis=0)
            return (alpha_t, buf.at[t].set(alpha_t)), None

        buf0 = jnp.zeros((t_len, n_state)).at[0].set(alpha_0)
        (_, buf), _ = jax.lax.scan(step, (alpha_0, buf0), steps)
        return buf

    log_alpha = forward(ell_reg)
    ll_use = _safe_logsumexp(log_alpha[-1], axis=0)

    def forward_loglik(ell_table: Array) -> Array:
        """The log-likelihood alone: a carry-only forward pass (no `(T, S)` buffer)."""

        def step(alpha_prev: Array, t: Array) -> tuple[Array, None]:
            alpha_t = row_of(ell_table, t, False) + _safe_logsumexp(alpha_prev[:, None] + log_a_at(t), axis=0)
            return alpha_t, None

        alpha_last, _ = jax.lax.scan(step, log_rho + row_of(ell_table, 0, False), steps)
        return _safe_logsumexp(alpha_last, axis=0)

    ll = forward_loglik(ell_reg) if tempered else ll_use

    last = space.is_last_phase[:, None] & (space.exit_targets >= 0)  # (S, N) exits that exist
    targets = jnp.maximum(space.exit_targets, 0)
    rows = jnp.arange(n_state)

    def reduce(xi: Array) -> tuple[Array, Array, Array]:
        diag = jnp.diagonal(xi)
        leave = jnp.maximum(jnp.sum(xi, axis=1) - diag, 0.0)  # rounding must not give a negative count
        stay = jnp.zeros(n_rate).at[space.rate_index_of].add(diag)
        adv = jnp.zeros(n_rate).at[space.rate_index_of].add(leave)
        exits = jnp.where(last, xi[rows[:, None], targets], 0.0)  # (S, N)
        jump = jnp.zeros((n_row, n_reg)).at[idx.row_of_state].add(exits)
        return stay, adv, jump

    def norm(log_gamma: Array) -> Array:
        finite = jnp.isfinite(log_gamma)
        return jnp.where(finite, jnp.exp(jnp.where(finite, log_gamma - ll_use, 0.0)), 0.0)

    def back(carry: tuple[Any, ...], t: Array) -> tuple[tuple[Any, ...], None]:
        beta_t, stay, adv, occ, jmp, gbuf, st_t, ad_t, jt_t = carry
        alpha_prev, alpha_t, v = log_alpha[t - 1], log_alpha[t], valid[t]
        log_a = log_a_at(t)
        ell_s = row_of(ell_reg, t, True)
        log_xi = alpha_prev[:, None] + log_a + (ell_s + beta_t)[None, :] - ll_use
        ok = jnp.isfinite(log_xi) & v
        xi = jnp.where(ok, jnp.exp(jnp.where(ok, log_xi, 0.0)), 0.0)
        d_stay, d_adv, d_jmp = reduce(xi)
        beta_prev = _safe_logsumexp(log_a + (ell_s + beta_t)[None, :], axis=1)
        gamma_t = norm(alpha_t + beta_t)  # (S,), the posterior of step t
        gamma_prev = norm(alpha_prev + beta_prev)
        occ = occ + jnp.zeros(n_rate).at[space.rate_index_of].add(jnp.where(v, gamma_prev, 0.0))
        gbuf = gbuf.at[t].set(jnp.where(v, gamma_t, 0.0) @ idx.collapse)
        if keep_per_t:
            st_t, ad_t, jt_t = st_t.at[t].set(d_stay), ad_t.at[t].set(d_adv), jt_t.at[t].set(d_jmp)
        return (beta_prev, stay + d_stay, adv + d_adv, occ, jmp + d_jmp, gbuf, st_t, ad_t, jt_t), None

    zeros = jnp.zeros(n_rate)
    n_t = t_len if keep_per_t else 0
    init: Any = (
        jnp.zeros(n_state), zeros, zeros, zeros, jnp.zeros((n_row, n_reg)), jnp.zeros((t_len, n_reg)),
        jnp.zeros((n_t, n_rate)), jnp.zeros((n_t, n_rate)), jnp.zeros((n_t, n_row, n_reg)),
    )  # fmt: skip
    (beta_0, stay, adv, occ, jmp, gamma_reg, st_t, ad_t, jt_t), _ = jax.lax.scan(
        back, init, steps, reverse=True
    )
    gamma_0 = norm(log_alpha[0] + beta_0)
    gamma_reg = gamma_reg.at[0].set(jnp.where(valid[0], gamma_0 @ idx.collapse, 0.0))
    return ll, gamma_reg, gamma_0, stay, adv, occ, jmp, st_t, ad_t, jt_t


@partial(jax.jit, static_argnames=("spec", "tempered", "keep_per_t"))
def _estep_batch(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    beta: Array,
    tempered: bool,
    keep_per_t: bool = False,
) -> ExpandedEStats:
    data = _dataset_view(pb)
    grouped = "initial.logits" in theta and theta["initial.logits"].ndim == 2
    active = pb.boundary_code is not None
    codes = jnp.zeros(pb.valid.shape, dtype=jnp.int8) if pb.boundary_code is None else pb.boundary_code

    def one(d: Any, v: Array, g: Array, c: Array) -> tuple[Array, ...]:  # noqa: ANN401
        th = {**theta, "initial.logits": theta["initial.logits"][g]} if grouped else theta
        return _estep_one(
            th, spec, space, d, v, c, beta, active=active, tempered=tempered, keep_per_t=keep_per_t
        )

    ll, gamma_reg, gamma_0, stay, adv, occ, jmp, st_t, ad_t, jt_t = jax.vmap(one)(
        data, pb.valid, pb.group_index, codes
    )
    nonempty = pb.lengths > 0
    n_groups = spec.n_groups
    keep = nonempty.astype(gamma_0.dtype)
    gamma1 = jax.ops.segment_sum(gamma_0 * keep[:, None], pb.group_index, num_segments=n_groups)

    def total(x: Array) -> Array:
        return jnp.sum(x * keep.reshape((-1,) + (1,) * (x.ndim - 1)), axis=0)

    return ExpandedEStats(
        loglik=jnp.where(nonempty, ll, 0.0),
        gamma_regime=gamma_reg,
        gamma1=gamma1,
        stay=total(stay),
        advance=total(adv),
        occupancy=total(occ),
        jump=total(jmp),
        stay_t=(st_t * keep[:, None, None]) if keep_per_t else None,
        advance_t=(ad_t * keep[:, None, None]) if keep_per_t else None,
        jump_t=(jt_t * keep[:, None, None, None]) if keep_per_t else None,
    )


def expanded_estep(
    engine: InferenceEngine,
    params: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    *,
    beta_temper: float = 1.0,
    keep_per_t: bool = False,
) -> ExpandedEStats:
    """The expected counts of `pb` under `params` (one forward pass, one fused backward / accumulation scan).

    Args:
        engine: the expanded engine (`ExpandedHMMEngine`); the explicit engine has its own E-step (E13-T01b).
        params: constrained parameters.
        spec: model structure (`engine: expanded`, segment memory).
        space: `build_expanded(spec)`.
        pb: the padded batch (`SequenceBatch.padded()`; a single dataset is a `B = 1` batch).
        beta_temper: `beta in (0, 1]`; the E-step runs on `beta * ell` (tempered EM); `loglik` stays raw.
        keep_per_t: also return the per-step counts `stay_t`, `advance_t`, `jump_t` (memory
            `O(B T (R_rate + R N))`; ECM of covariate / irregular-step blocks needs them).

    Raises:
        ConfigError: explicit engine, time-step memory, or `beta_temper` outside `(0, 1]`.
    """
    if spec.engine != "expanded":
        raise ConfigError(
            "expanded_estep needs engine: expanded (the explicit engine: E13-T01b)", path="model.engine"
        )
    if spec.memory_mode != "segments":
        raise ConfigError("expanded_estep needs segment memory (not memory_mode: steps)", path="model")
    if not 0.0 < beta_temper <= 1.0:
        raise ConfigError(f"beta_temper must be in (0, 1], got {beta_temper}", path="estimation.em.tempering")
    stats: ExpandedEStats = _estep_batch(
        params, spec, space, pb, jnp.asarray(beta_temper), beta_temper != 1.0, keep_per_t
    )
    live = np.asarray(pb.lengths) > 0
    ll = np.asarray(stats.loglik)[live]
    if not np.all(np.isfinite(ll)):
        bad = int(np.flatnonzero(~np.isfinite(ll))[0])
        raise NumericalError(
            f"non-finite log-likelihood in the E-step (sequence {bad}): the labels or boundaries rule out "
            "every path, or the parameters give zero probability",
            path="loglik",
        )
    return stats
