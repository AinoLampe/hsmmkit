"""Path simulation from any model/parameters (E6-T01): `simulate_paths`, `simulate_dataset`.

Concept refs: §5.5, §6.3 item 2 (`simulate_paths` is a JIT boundary), §6.4.2. States advance one step
per `jax.lax.scan` iteration using `space.transition_fn` (the same per-step `A_t` builder the engines
use, CLAUDE.md §3.7's timing convention: `A_t` moves index `t-1` to `t` using covariate/`dt` row `t`).
Observations are then drawn in one vectorized call per path via `spec.emission.sample`, which already
takes whole `(T,)` `regime`/`dt`/`z` arrays (not a per-step call) -- the ticket's per-step narrative
describes the *semantics*, not literally one `emission.sample` call per step.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.data.contracts import PreparedDataset, PrepStats, lag_names, own_lag_columns
from hsmmkit.engine.viterbi import segments
from hsmmkit.statespace.builders import initial_log_rho
from hsmmkit.utils.errors import ConfigError, NumericalError

if TYPE_CHECKING:
    from hsmmkit.data.batch import SequenceBatch
    from hsmmkit.model.params.manager import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace
    from hsmmkit.utils.typing import Array


def _block(x: Array | None, t: int) -> Array:
    """A covariate block, or `(T, 0)` zeros when absent (CLAUDE.md §3.7)."""
    return jnp.zeros((t, 0)) if x is None else x


def _check_finite(values: Array) -> None:
    """Public-boundary guard (CLAUDE.md §4): no NaN/Inf may leave `simulate_paths`."""
    if not bool(jnp.all(jnp.isfinite(values))):
        raise NumericalError("non-finite value in simulated values", path="values")


@dataclass(frozen=True)
class SimResult:
    """`n_paths` simulated paths from `simulate_paths`."""

    states: Array  # (n_paths, T) int, expanded state index
    regimes: Array  # (n_paths, T) int, regime index
    values: Array  # (n_paths, T, M)
    log_rho: Array  # (S,) initial log-distribution actually used (for tests/diagnostics)


def _simulate_one_path(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    dt: Array,
    z_pi: Array,
    z_d: Array,
    z_e: Array,
    init_state: Array,
) -> tuple[Array, Array, Array]:
    """One path: `(states (T,), regimes (T,), values (T, M))`."""
    if spec.duration_dependent_jumps:
        raise ConfigError(
            "simulating duration-dependent jumps is not supported yet",
            path="model.duration.duration_dependent_jumps",
        )
    t_len = dt.shape[0]
    step_key, emit_key = jax.random.split(key)

    def step(s_prev: Array, xs_t: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
        z_pi_t, z_d_t, dt_t, key_t = xs_t
        _, log_a = space.transition_fn(theta, z_pi_t, z_d_t, dt_t)
        # `categorical`'s output dtype need not match `s_prev`'s (e.g. index arrays are int32,
        # `categorical` defaults to int64 under x64) -- the scan carry's dtype must stay fixed.
        s_t = jax.random.categorical(key_t, log_a[s_prev]).astype(s_prev.dtype)
        return s_t, s_t

    if t_len > 1:
        step_keys = jax.random.split(step_key, t_len - 1)
        xs = (z_pi[1:], z_d[1:], dt[1:], step_keys)
        _, rest = jax.lax.scan(step, init_state, xs)
        states = jnp.concatenate([init_state[None], rest])
    else:
        states = init_state[None]

    regimes = space.index.regime_of[states]
    if hasattr(spec.emission, "sample_path"):
        # Autoregressive emissions (`msar`) feed each draw back into the next lags.
        values = spec.emission.sample_path(emit_key, regimes, dt, theta)
    else:
        values = spec.emission.sample(emit_key, regimes, dt, theta, z_e)
    return states, regimes, values


@partial(jax.jit, static_argnames=("spec", "n_paths"))
def _simulate_core(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    dt: Array,
    z_pi: Array,
    z_d: Array,
    z_e: Array,
    n_paths: int,
    init_state: Array,
) -> tuple[Array, Array, Array]:
    path_keys = jax.random.split(key, n_paths)
    in_axes = (0, None, None, None, None, None, None, None, 0)
    return jax.vmap(_simulate_one_path, in_axes=in_axes)(
        path_keys, theta, spec, space, dt, z_pi, z_d, z_e, init_state
    )


def simulate_paths(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    dt: Array,
    z_pi: Array | None,
    z_d: Array | None,
    z_e: Array | None,
    n_paths: int,
    init_state: Array | None = None,
) -> SimResult:
    """Sample `n_paths` independent expanded-state paths and observations, deterministic per `key`.

    Without `init_state`, the initial expanded state is drawn from `initial_log_rho(spec, space,
    theta)` (uniform/free: `phase == 1` support only; stationary: every state) using `nominal_dt =
    median(dt)`. `init_state (n_paths,)` (forecasting: a filtered-posterior sample) skips that draw.
    `spec`/`n_paths` are static; the state-path scan and emission sampling are jitted together
    (`_simulate_core`).
    """
    t_len = dt.shape[0]
    z_pi_b, z_d_b, z_e_b = _block(z_pi, t_len), _block(z_d, t_len), _block(z_e, t_len)
    log_rho = initial_log_rho(spec, space, theta, float(jnp.median(dt)))

    if init_state is None:
        init_key, key = jax.random.split(key)
        logits = jnp.broadcast_to(log_rho, (n_paths, log_rho.shape[0]))
        init_state = jax.random.categorical(init_key, logits)
    init_state = jnp.asarray(init_state, dtype=space.index.regime_of.dtype)

    states, regimes, values = _simulate_core(
        key, theta, spec, space, dt, z_pi_b, z_d_b, z_e_b, n_paths, init_state
    )
    _check_finite(values)
    return SimResult(states=states, regimes=regimes, values=values, log_rho=log_rho)


def segments_of(regimes: Array | np.ndarray) -> list[list[tuple[int, int, int]]]:
    """`[segments(regimes[i]) for i in range(n_paths)]` (NumPy, outside jit; `end` inclusive)."""
    arr = np.asarray(regimes)
    return [segments(arr[i]) for i in range(arr.shape[0])]


def simulate_dataset(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    dt: Array,
    z_pi: Array | None = None,
    z_d: Array | None = None,
    z_e: Array | None = None,
    *,
    clock: str,
    nominal_dt: float | None = None,
) -> PreparedDataset:
    """A single simulated path (`n_paths=1`) wrapped as a synthetic `PreparedDataset` (tests/recovery).

    `mask` is all-`True` (nothing missing); `PrepStats` is the zero/identity "no transform applied"
    default, matching a dataset that was never run through `Preparer`.
    """
    result = simulate_paths(key, theta, spec, space, dt, z_pi, z_d, z_e, n_paths=1)
    y = result.values[0]
    t_len = y.shape[0]
    mask = jnp.ones(y.shape, dtype=bool)
    emission_names: tuple[str, ...] = ()
    p = int(getattr(spec.emission, "requires_own_lags", 0))
    if p:  # an autoregressive emission: the own lags are emission covariates and the first `p` steps masked
        lags, ok = own_lag_columns(np.asarray(y), np.ones(y.shape, dtype=bool), p)
        z_e = jnp.asarray(lags) if z_e is None else jnp.concatenate([z_e, jnp.asarray(lags)], axis=1)
        emission_names = lag_names(tuple(f"y{m}" for m in range(y.shape[1])), p)
        mask = jnp.asarray(np.broadcast_to(ok[:, None], y.shape))
    stats = PrepStats(
        cov_mean=np.zeros(0), cov_std=np.ones(0), train_window=("", ""), cov_names=(), transforms={}, lags={}
    )
    return PreparedDataset(
        times=np.arange(t_len).astype("datetime64[D]").astype("datetime64[ns]"),
        observations=y,
        dt=dt,
        mask=mask,
        cov_transition=z_pi,
        cov_duration=z_d,
        cov_emission=z_e,
        cov_names={"transition": (), "duration": (), "emission": emission_names},
        clock=clock,
        nominal_dt=float(jnp.median(dt)) if nominal_dt is None else nominal_dt,
        prep_stats=stats,
        snapshot_ids=(),
    )


def simulate_batch(
    params: Params,
    spec: ModelSpec,
    lengths: Sequence[int],
    key: Array,
    groups: Sequence[str] | None = None,
    *,
    clock: str = "steps",
    nominal_dt: float = 1.0,
) -> SequenceBatch:
    """Simulate one sequence of each length in `lengths` as a `SequenceBatch` (E12-T02).

    Sequence `b` is drawn with `jax.random.fold_in(key, b)` on a regular grid of step `nominal_dt`, so the
    result is deterministic per `key` and independent of the other sequences. With `groups` (one name per
    sequence; the model must have `initial.logits` of shape `(G, S_phase1)`) sequence `b` starts from the
    row of its group (`sorted(set(groups))` indexes the rows, as `SequenceBatch.group_names` does) and
    carries `group = groups[b]`. Covariate-free models only.

    Raises:
        ConfigError: `groups` given for a model without per-group initial distributions (or the wrong
            number of groups), or a model with covariates.
    """
    from hsmmkit.data.batch import (
        SequenceBatch,
    )
    from hsmmkit.statespace.builders import build_expanded, build_explicit

    lengths = [int(n) for n in lengths]
    if any(n < 1 for n in lengths):
        raise ConfigError("every sequence length must be >= 1", path="lengths")
    grouped = "initial.logits" in params and params["initial.logits"].ndim == 2
    names: tuple[str, ...] = ()
    if groups is not None:
        if not grouped:
            raise ConfigError("groups need a model with initial_groups: per_group", path="groups")
        if len(groups) != len(lengths):
            raise ConfigError("groups and lengths differ in length", path="groups")
        names = tuple(sorted(set(groups)))
        if len(names) != params["initial.logits"].shape[0]:
            raise ConfigError(
                f"the model has {params['initial.logits'].shape[0]} groups, groups has {len(names)}",
                path="groups",
            )
    elif grouped:
        raise ConfigError("a per_group model needs groups= to simulate", path="groups")
    space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
    datasets = []
    for b, n in enumerate(lengths):
        theta = params
        if groups is not None:
            theta = {**params, "initial.logits": params["initial.logits"][names.index(groups[b])]}
        ds = simulate_dataset(
            jax.random.fold_in(key, b),
            theta,
            spec,
            space,
            jnp.full((n,), float(nominal_dt)),
            clock=clock,
            nominal_dt=float(nominal_dt),
        )
        datasets.append(dataclasses.replace(ds, group=None if groups is None else str(groups[b])))
    return SequenceBatch(tuple(datasets), (), tuple(str(b) for b in range(len(lengths))))
