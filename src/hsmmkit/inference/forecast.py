"""Monte-Carlo forecasting from the filtered augmented posterior (E6-T04a sampling, E6-T04b aggregation).

Concept refs: §4.9 algorithm steps 1-2, §5.5. The origin state of every path is drawn from
`softmax(log_alpha_T)`; step `k = 1..h` then applies `A_{T+k}` (built from covariate row `k-1` of the
covariate path and `dt_future[k-1]`, CLAUDE.md §3.7 -- `cov_paths[:, 0]` is the row already known at
the origin) and draws the return from the new regime's emission.

Expanded engine: the state-path scan of `simulate_paths` (E6-T01) is reused unchanged. A dummy row 0
(a copy of row 1) plays the role of the origin step, so the scan applies `A_{T+1}, ...` from the sampled
origin state; that row is dropped from the outputs. Covariates are per path (vmap over the path axis).

Explicit engine: state `(regime, elapsed)`. The origin's total sojourn `d` is drawn from the regime's
pmf restricted to `d >= elapsed` (the segment has already run `elapsed` steps, right-censored at `T`);
the segment ends after exactly `d` steps, then a new regime is drawn from the jump row and a fresh
duration from its pmf. That is the hazard simulation `h_j(e)` written as a duration draw (the two are the
same distribution). Durations use `nominal_dt` (regular-grid assumption, ADR-0007); emissions use the
real `dt_future`.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from regimekit.inference.covariate_forecasters import ColumnMap, _unified_covariates
from regimekit.inference.results import ForecastResult
from regimekit.inference.simulate import _simulate_one_path
from regimekit.utils.clock import future_grid
from regimekit.utils.errors import ConfigError, NumericalError
from regimekit.utils.rng import FORECAST_COVARIATES, FORECAST_PATHS, PURPOSE_FORECAST, key_for

if TYPE_CHECKING:
    from regimekit.config.schema import ForecastSection
    from regimekit.data.contracts import PreparedDataset
    from regimekit.engine.base import InferenceEngine, Posterior
    from regimekit.estimation.results import FitResult
    from regimekit.inference.covariate_forecasters import CovariateForecaster
    from regimekit.model.params.manager import Params
    from regimekit.model.spec import ModelSpec
    from regimekit.statespace.index import StateSpace
    from regimekit.utils.typing import Array

MAX_SAMPLE_FLOATS = 2e8


class ForecastPaths(NamedTuple):
    """`n_sim` future paths of horizon `h`."""

    states: Array  # (n_sim, h) int, augmented state index at each future step
    regimes: Array  # (n_sim, h) int
    returns: Array  # (n_sim, h, M)


def _blocks(
    cov_paths: Array | None, colmap: ColumnMap | None, n_sim: int, h: int
) -> tuple[Array, Array, Array]:
    """Per-entry-point covariate blocks `(n_sim, h, K_*)`; absent blocks are zeros of width 0."""
    empty = jnp.zeros((n_sim, h, 0))
    if cov_paths is None:
        return empty, empty, empty
    if colmap is None:
        raise ConfigError("cov_paths needs a ColumnMap to split it by entry point", path="forecast")
    if cov_paths.shape[:2] != (n_sim, h):
        raise ConfigError(
            f"cov_paths must have shape (n_sim={n_sim}, h={h}, K), got {cov_paths.shape}",
            path="forecast.covariate_forecaster",
        )
    parts = colmap.split(cov_paths)
    z_pi, z_d, z_e = (parts[k] for k in ("transition", "duration", "emission"))
    return (
        empty if z_pi is None else z_pi,
        empty if z_d is None else z_d,
        empty if z_e is None else z_e,
    )


def _check_inputs(log_alpha_t: Array, dt_future: Array, n_sim: int) -> None:
    if n_sim < 1:
        raise ConfigError(f"n_sim must be >= 1, got {n_sim}", path="forecast.n_sim")
    if dt_future.ndim != 1 or dt_future.shape[0] < 1:
        raise ConfigError("dt_future must be a non-empty (h,) array", path="forecast.horizon")
    if log_alpha_t.ndim != 1:
        raise ConfigError("log_alpha_T must be the (S,) last row of log_alpha", path="log_alpha")


def _check_state_count(log_alpha_t: Array, space: StateSpace) -> None:
    if log_alpha_t.shape[0] != space.n_expanded:
        raise ConfigError(
            f"log_alpha_T has {log_alpha_t.shape[0]} states, the state space has {space.n_expanded}",
            path="log_alpha",
        )


def _check_finite(returns: Array) -> None:
    """Public-boundary guard (CLAUDE.md §4): no NaN/Inf may leave the sampler."""
    if not bool(jnp.all(jnp.isfinite(returns))):
        raise NumericalError("non-finite value in forecast returns", path="returns")


def _prepend(x: Array) -> Array:
    """Copy of row 0 in front, so the simulate scan's untransitioned row 0 is the origin step."""
    return jnp.concatenate([x[:1], x], axis=0)


def _expanded_path(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    dt: Array,
    z_pi: Array,
    z_d: Array,
    z_e: Array,
    s0: Array,
) -> tuple[Array, Array, Array]:
    states, regimes, returns = _simulate_one_path(
        key, theta, spec, space, _prepend(dt), _prepend(z_pi), _prepend(z_d), _prepend(z_e), s0
    )
    return states[1:], regimes[1:], returns[1:]


@partial(jax.jit, static_argnames=("spec", "n_sim"))
def _expanded_core(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    log_alpha_t: Array,
    dt: Array,
    z_pi: Array,
    z_d: Array,
    z_e: Array,
    n_sim: int,
) -> tuple[Array, Array, Array]:
    origin_key, path_key = jax.random.split(key)
    logits = jnp.broadcast_to(log_alpha_t, (n_sim, log_alpha_t.shape[0]))
    s0 = jax.random.categorical(origin_key, logits).astype(space.index.regime_of.dtype)
    keys = jax.random.split(path_key, n_sim)
    in_axes = (0, None, None, None, None, 0, 0, 0, 0)
    return jax.vmap(_expanded_path, in_axes=in_axes)(keys, theta, spec, space, dt, z_pi, z_d, z_e, s0)


def forecast_paths(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    log_alpha_t: Array,
    dt_future: Array,
    n_sim: int,
    cov_paths: Array | None = None,
    colmap: ColumnMap | None = None,
) -> ForecastPaths:
    """Sample `n_sim` paths of `h = len(dt_future)` steps from `softmax(log_alpha_t)` (expanded engine).

    Args:
        key: PRNG key; the result is deterministic per key.
        theta: parameters.
        spec: model structure (static).
        space: the expanded `StateSpace` the posterior lives in.
        log_alpha_t: `(S,)` last row of the filtered `log_alpha` at the origin.
        dt_future: `(h,)` `Δ` of the future steps (`utils.clock.future_grid`).
        n_sim: number of paths.
        cov_paths: `(n_sim, h, K)` future covariates in `colmap` order (row 0 = the row known at the
            origin), or `None` for a model without covariates.
        colmap: unified-column -> entry-point mapping of `cov_paths`.

    Returns:
        `ForecastPaths(states (n_sim, h), regimes (n_sim, h), returns (n_sim, h, M))`.

    Raises:
        ConfigError: inconsistent shapes or a non-positive `n_sim`.
        NumericalError: a non-finite simulated return.
    """
    _check_inputs(log_alpha_t, dt_future, n_sim)
    _check_state_count(log_alpha_t, space)
    h = int(dt_future.shape[0])
    z_pi, z_d, z_e = _blocks(cov_paths, colmap, n_sim, h)
    out = _expanded_core(key, theta, spec, space, log_alpha_t, dt_future, z_pi, z_d, z_e, n_sim)
    _check_finite(out[2])
    return ForecastPaths(*out)


def _explicit_path(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    dt: Array,
    z_pi: Array,
    z_d: Array,
    z_e: Array,
    j0: Array,
    e0: Array,
    z_d_origin: Array,
    nominal_dt: float,
) -> tuple[Array, Array, Array]:
    d_max, family = spec.d_max, spec.duration
    has_entry = bool(getattr(family, "cov_idx", ()))
    const_lp = None if has_entry else family.log_pmf(theta, d_max, nominal_dt, None)

    def pmf(z: Array) -> Array:
        return family.log_pmf(theta, d_max, nominal_dt, z) if has_entry else const_lp  # type: ignore[return-value]

    h = dt.shape[0]
    k_dur, k_steps, k_emit = jax.random.split(key, 3)
    support = jnp.arange(1, d_max + 1)
    masked = jnp.where(support >= e0, pmf(z_d_origin)[j0], -jnp.inf)
    d0 = (jax.random.categorical(k_dur, masked) + 1).astype(jnp.int64)
    d0 = jnp.where(jnp.any(jnp.isfinite(masked)), d0, e0.astype(jnp.int64))  # no mass left: exit now

    def step(
        carry: tuple[Array, Array, Array], xs: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array]]:
        j, e, d = carry
        z_pi_k, z_d_k, key_k = xs
        key_j, key_d = jax.random.split(key_k)
        ended = e >= d
        pi = spec.transition.jump_probs_t(theta, z_pi_k, space.layout)[j]
        log_pi = jnp.where(pi > 0, jnp.log(jnp.where(pi > 0, pi, 1.0)), -jnp.inf)
        j_new = jax.random.categorical(key_j, log_pi).astype(jnp.int64)
        d_new = (jax.random.categorical(key_d, pmf(z_d_k)[j_new]) + 1).astype(jnp.int64)
        j2 = jnp.where(ended, j_new, j)
        e2 = jnp.where(ended, jnp.int64(1), e + 1)
        d2 = jnp.where(ended, d_new, d)
        state = j2 * d_max + jnp.minimum(e2, d_max) - 1
        return (j2, e2, d2), (state, j2)

    carry0 = (j0.astype(jnp.int64), e0.astype(jnp.int64), d0)
    _, (states, regimes) = jax.lax.scan(step, carry0, (z_pi, z_d, jax.random.split(k_steps, h)))
    returns = spec.emission.sample(k_emit, regimes, dt, theta, z_e)
    return states, regimes, returns


@partial(jax.jit, static_argnames=("spec", "n_sim", "nominal_dt"))
def _explicit_core(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    log_alpha_t: Array,
    dt: Array,
    z_pi: Array,
    z_d: Array,
    z_e: Array,
    z_d_hist: Array,
    n_sim: int,
    nominal_dt: float,
) -> tuple[Array, Array, Array]:
    origin_key, path_key = jax.random.split(key)
    logits = jnp.broadcast_to(log_alpha_t, (n_sim, log_alpha_t.shape[0]))
    s0 = jax.random.categorical(origin_key, logits)
    j0 = space.index.regime_of[s0]
    e0 = space.index.phase_of[s0]
    t_len = z_d_hist.shape[0]
    z_d_origin = z_d_hist[jnp.clip(t_len - e0, 0, t_len - 1)]  # (n_sim, K_d): row where the segment began
    keys = jax.random.split(path_key, n_sim)
    in_axes = (0, None, None, None, None, 0, 0, 0, 0, 0, 0, None)
    return jax.vmap(_explicit_path, in_axes=in_axes)(
        keys, theta, spec, space, dt, z_pi, z_d, z_e, j0, e0, z_d_origin, nominal_dt
    )


def forecast_paths_explicit(
    key: Array,
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    log_alpha_t: Array,
    dt_future: Array,
    n_sim: int,
    nominal_dt: float,
    cov_paths: Array | None = None,
    colmap: ColumnMap | None = None,
    z_d_hist: Array | None = None,
) -> ForecastPaths:
    """`forecast_paths` for the explicit-duration engine: origin `(regime, elapsed)`, then duration draws.

    `log_alpha_t` is the `(N * D_max,)` last row of the explicit engine's filtered `log_alpha` (state
    `j * D_max + e - 1`). `z_d_hist (T, K_d)` is the observed duration-covariate block: with entry-time
    duration covariates the origin segment's pmf uses the row at which that segment began
    (`T - elapsed`). `states` are augmented indices `j * D_max + min(e, D_max) - 1`.

    Raises:
        ConfigError: inconsistent shapes, `spec.engine != "explicit"` or a non-positive `n_sim`.
        NumericalError: a non-finite simulated return.
    """
    if spec.engine != "explicit":
        raise ConfigError("forecast_paths_explicit needs an explicit-engine spec", path="model.engine")
    _check_inputs(log_alpha_t, dt_future, n_sim)
    _check_state_count(log_alpha_t, space)
    h = int(dt_future.shape[0])
    z_pi, z_d, z_e = _blocks(cov_paths, colmap, n_sim, h)
    needs_entry = bool(getattr(spec.duration, "cov_idx", ()))
    if needs_entry and (z_d.shape[-1] == 0 or z_d_hist is None or z_d_hist.shape[0] == 0):
        raise ConfigError(
            "the duration family has entry-time covariates: pass cov_paths and z_d_hist",
            path="forecast.covariate_forecaster",
        )
    hist = jnp.zeros((1, 0)) if z_d_hist is None or z_d_hist.shape[0] == 0 else z_d_hist
    out = _explicit_core(
        key, theta, spec, space, log_alpha_t, dt_future, z_pi, z_d, z_e, hist, n_sim, float(nominal_dt)
    )
    _check_finite(out[2])
    return ForecastPaths(*out)


class MonteCarloForecaster:
    """`forecast(fit, data, horizon, n_sim, seed) -> ForecastResult` (E6-T04b).

    Runs the engine's filter on `data`, draws `n_sim` paths from the filtered posterior at the origin
    (`forecast_paths` / `forecast_paths_explicit`) and aggregates them: regime probabilities, per-step
    samples, cumulative-return quantiles and moments. Deterministic per `(fit, data, horizon, n_sim, seed)`.

    Covariates. The row for the first forecast step is not part of `data` (the dataset ends at the last
    observed step), so the last *observed* covariate row stands in for `last` -- the value `freeze`
    repeats and the other forecasters start from. Multi-step forecasts with covariates need a
    `forecaster`; a one-step forecast does not.

    Args:
        spec: model structure.
        space: the state space `spec.engine` calls for (`build_expanded` / `build_explicit`).
        engine: the matching inference engine.
        forecaster: covariate forecaster, or `None` for covariate-free models and `horizon == 1`.
        cfg: `forecast` config section (quantile levels).
        clock_options: the options of the dataset's clock (`data.clock_options`).
    """

    def __init__(
        self,
        spec: ModelSpec,
        space: StateSpace,
        engine: InferenceEngine,
        forecaster: CovariateForecaster | None,
        cfg: ForecastSection,
        clock_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Store the model structure, engine, covariate forecaster and forecast config."""
        self.spec = spec
        self.space = space
        self.engine = engine
        self.forecaster = forecaster
        self.cfg = cfg
        self.clock_options = dict(clock_options or {})

    def _origin_log_alpha(self, fit: FitResult, data: PreparedDataset, posterior: Posterior | None) -> Array:
        if posterior is None:
            log_alpha, _ = self.engine.forward(fit.params, self.spec, self.space, data)
        else:
            log_alpha = posterior.log_alpha
        if log_alpha.shape[-1] != self.space.n_expanded:
            raise ConfigError(
                "the posterior is not over the full state space (chunked explicit posterior?)",
                path="posterior",
            )
        return log_alpha[-1]

    def _cov_paths(
        self, data: PreparedDataset, horizon: int, n_sim: int, seed: int
    ) -> tuple[Array | None, ColumnMap | None, str]:
        n_cov = len(data.prep_stats.cov_names)
        if n_cov == 0:
            return None, None, "none"
        last = jnp.asarray(_unified_covariates(data)[-1])
        colmap = ColumnMap.from_dataset(data)
        if horizon == 1:
            return jnp.broadcast_to(last, (n_sim, 1, n_cov)), colmap, "none"
        if self.forecaster is None:
            raise ConfigError(
                "horizon > 1 with covariates requires a covariate forecaster",
                path="forecast.covariate_forecaster",
            )
        fitted = self.forecaster.fit(data)
        key = key_for(seed, PURPOSE_FORECAST, FORECAST_COVARIATES)
        return fitted.sample_paths(key, horizon, n_sim, last), colmap, fitted.name

    def forecast(
        self,
        fit: FitResult,
        data: PreparedDataset,
        horizon: int,
        n_sim: int,
        seed: int,
        posterior: Posterior | None = None,
    ) -> ForecastResult:
        """Forecast `horizon` steps from the end of `data` (`posterior` reuses a supplied filter pass).

        Raises:
            ConfigError: `horizon < 1`, `n_sim < 1`, or covariates with `horizon > 1` and no forecaster.
            NumericalError: `n_sim * horizon * M > 2e8` (use fewer simulations) or non-finite output.
        """
        if horizon < 1:
            raise ConfigError(f"horizon must be >= 1, got {horizon}", path="forecast.horizon")
        if n_sim < 1:
            raise ConfigError(f"n_sim must be >= 1, got {n_sim}", path="forecast.n_sim")
        m = self.spec.n_series
        if n_sim * horizon * m > MAX_SAMPLE_FLOATS:
            raise NumericalError(
                f"n_sim * horizon * M = {n_sim * horizon * m:.3g} > {MAX_SAMPLE_FLOATS:.0e} floats: "
                "use fewer simulations or a shorter horizon",
                path="forecast.n_sim",
            )

        times = np.asarray(data.times).astype("datetime64[ns]")
        horizon_times, dt_np = future_grid(times[-1], horizon, data.clock, self.clock_options, times)
        dt_future = jnp.asarray(dt_np)

        log_alpha_t = self._origin_log_alpha(fit, data, posterior)
        cov_paths, colmap, forecaster_name = self._cov_paths(data, horizon, n_sim, seed)
        key = key_for(seed, PURPOSE_FORECAST, FORECAST_PATHS)
        if self.spec.engine == "explicit":
            paths = forecast_paths_explicit(
                key,
                fit.params,
                self.spec,
                self.space,
                log_alpha_t,
                dt_future,
                n_sim,
                data.nominal_dt,
                cov_paths,
                colmap,
                data.cov_duration,
            )
        else:
            paths = forecast_paths(
                key, fit.params, self.spec, self.space, log_alpha_t, dt_future, n_sim, cov_paths, colmap
            )

        samples = paths.returns  # (n_sim, h, M)
        regime_probs = jax.nn.one_hot(paths.regimes, self.spec.n_states).mean(axis=0)  # (h, N)
        cum = jnp.cumsum(samples, axis=1)
        levels = tuple(float(q) for q in self.cfg.quantiles)
        cum_quantiles = jnp.transpose(jnp.quantile(cum, jnp.asarray(levels), axis=0), (1, 0, 2))  # (h, Q, M)
        mean, var = cum.mean(axis=0), cum.var(axis=0)
        for name, arr in (("regime_probs", regime_probs), ("cum_quantiles", cum_quantiles), ("var", var)):
            if not bool(jnp.all(jnp.isfinite(arr))):
                raise NumericalError(f"non-finite {name} in forecast", path=name)

        return ForecastResult(
            origin_time=times[-1],
            horizon_times=horizon_times,
            regime_probs=regime_probs,
            samples=samples,
            cum_quantiles=cum_quantiles,
            mean=mean,
            var=var,
            quantile_levels=levels,
            covariate_forecaster=forecaster_name,
            seed=seed,
        )
