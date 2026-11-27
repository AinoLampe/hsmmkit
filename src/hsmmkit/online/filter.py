"""`ExpandedOnlineFilter`: one-observation-at-a-time filtering, bitwise equal to the offline pass (E15-T01).

Concept refs: Part II §14.11, §15.3; NFR-15. The per-step body is `engine.expanded_hmm.forward_step`, the same
function the offline forward scan runs, and the emission row of one observation comes from the same
`expanded_log_emis` code with `T = 1`, so the filtered probabilities and the running log-likelihood equal
the offline results bitwise. `theta`, `space` are arguments of the jitted step (never closed-over constants)
and `spec` is static, exactly as in the offline engine; the step is compiled once per fit.

Only the expanded engine with segment memory and families without a recurrent emission carry can filter
online; anything else raises `ConfigError` at construction.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.engine.base import Posterior, filtered_row
from hsmmkit.engine.expanded_hmm import (
    ExpandedHMMEngine,
    _safe_logsumexp,
    expanded_log_emis,
    forward_step,
)
from hsmmkit.online.alerts import AlertEvent, AlertRule, evaluate, rules_from_config
from hsmmkit.online.buffer import RingBuffer, buffer_from_prefix, empty_buffer, push, smooth_window
from hsmmkit.online.state import OnlineState, spec_hash
from hsmmkit.statespace.builders import build_expanded, initial_log_rho
from hsmmkit.utils.clock import get_clock, parse_clock_options
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.inference.covariate_forecasters import CovariateForecaster
    from hsmmkit.inference.results import ForecastResult
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

Step = tuple[Any, ...]
"""`(state', filtered[, smoothed][, events])`: the lagged smoothed marginals when the filter has `lag > 0`,
the list of `AlertEvent`s (possibly empty) when it has alert rules (in this order)."""

Covariates = Mapping[str, Any]
"""Covariate rows of one step: keys `transition`, `duration`, `emission`, each `(K,)` (missing: width 0)."""


def _normalize_time(value: Any) -> Any:  # noqa: ANN401
    """A timestamp as `datetime64[ns]` (datetime-like, `pandas.Timestamp`, string); numbers stay numbers."""
    if value is None or isinstance(value, (int, float, np.integer, np.floating)):
        return value
    return np.datetime64(value, "ns")


def _row(z: Covariates | None, key: str, width: int | None) -> np.ndarray:
    value = None if z is None else z.get(key)
    if value is None:
        if width:
            raise ConfigError(f"the model needs a covariate row of width {width}", path=f"z.{key}")
        return np.zeros((0,))
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if width is not None and arr.shape[0] != width:
        raise ConfigError(f"expected a covariate row of width {width}, got {arr.shape[0]}", path=f"z.{key}")
    return arr


def _remaining_steps(
    theta: Params, spec: ModelSpec, space: StateSpace, log_alpha: Array, dt: Array, z_d: Array
) -> Array:
    """`(N,)` expected remaining steps of the current segment under the filtered distribution.

    The array form of `engine.augmented.expected_remaining_steps` for one step: in phase `c` of `r_i`,
    `r_i - c + 1` phases remain, each lasting `1 / a_i` steps; weighted by `P(phase | regime i, y_1:t)`.
    """
    index = space.index
    probs = jax.nn.softmax(log_alpha)  # (S,)
    r = jnp.asarray([spec.duration.n_phases(i) for i in range(spec.n_states)])
    steps = (r[index.regime_of] - index.phase_of + 1).astype(probs.dtype)  # (S,)
    a = spec.duration.advance_prob(theta, dt[None], z_d[None, :])[0]  # (N,)
    a_safe = jnp.where(a > 0, a, 1.0)
    mass = probs @ index.collapse
    safe_mass = jnp.where(mass > 0, mass, 1.0)
    weighted = ((probs * steps) @ index.collapse) / a_safe
    return jnp.where(mass > 0, weighted / safe_mass, 0.0)


def _step_fn(
    theta: Params,
    space: StateSpace,
    log_alpha: Array,
    t: Array,
    y: Array,
    dt: Array,
    z_pi: Array,
    z_d: Array,
    z_e: Array,
    buffer: RingBuffer | None,
    *,
    spec: ModelSpec,
    lag: int,
    with_remaining: bool = False,
) -> tuple[Array, Array, Array, Array, RingBuffer | None, Array | None, Array | None]:
    """`(log_alpha', log_norm', t', filtered, buffer', smoothed, remaining)`.

    The first observation (`t == 0`) starts from `log_rho`; `buffer`/`smoothed` are `None` without a lag and
    `remaining` (`(N,)` expected remaining steps of the current segment, E15-T03) without duration alerts.
    """
    mask = ~jnp.isnan(y)
    ell = expanded_log_emis(theta, spec, space, y[None, :], mask[None, :], dt[None], z_e[None, :])[0]
    # `log_alpha` holds `log_rho` before the first observation (`start`), so the same
    # `forward_step` input is used for t == 0; its result is replaced by `log_rho + ell` there.
    alpha_next = forward_step(
        space, theta, False, log_alpha, ell, z_pi, z_d, dt, jnp.zeros((), dtype=jnp.int8)
    )
    alpha_first = log_alpha + ell
    new = jnp.where(t == 0, alpha_first, alpha_next)
    log_norm = _safe_logsumexp(new, axis=0)
    filtered = filtered_row(new, space.index.collapse)
    remaining = None
    if with_remaining:
        remaining = _remaining_steps(theta, spec, space, new, dt, z_d)
    if buffer is None:
        return new, log_norm, t + 1, filtered, None, None, remaining
    buffer = push(buffer, ell, z_pi, z_d, dt, new)
    return new, log_norm, t + 1, filtered, buffer, smooth_window(theta, space, buffer, lag), remaining


def _steps_fn(
    theta: Params,
    space: StateSpace,
    log_alpha: Array,
    t: Array,
    ys: Array,
    dts: Array,
    z_pis: Array,
    z_ds: Array,
    z_es: Array,
    buffer: RingBuffer | None,
    *,
    spec: ModelSpec,
    lag: int,
    with_remaining: bool = False,
) -> tuple[Array, Array, Array, Array, RingBuffer | None, Array | None, Array | None]:
    def body(carry: tuple[Any, ...], xs: tuple[Array, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        alpha, tt, buf = carry
        y, dt, z_pi, z_d, z_e = xs
        alpha, _, tt, filtered, buf, smoothed, remaining = _step_fn(
            theta,
            space,
            alpha,
            tt,
            y,
            dt,
            z_pi,
            z_d,
            z_e,
            buf,
            spec=spec,
            lag=lag,
            with_remaining=with_remaining,
        )
        return (alpha, tt, buf), (filtered, smoothed, remaining)

    (alpha, tt, buf), (filtered, smoothed, remaining) = jax.lax.scan(
        body, (log_alpha, t, buffer), (ys, dts, z_pis, z_ds, z_es)
    )
    return alpha, _safe_logsumexp(alpha, axis=0), tt, filtered, buf, smoothed, remaining


class ExpandedOnlineFilter:
    """Streaming filter of a fitted model on the expanded engine.

    Args:
        fit: the fitted model.
        group: the initial-distribution group of the stream when the fit has per-group initial
            distributions (`model.initial_groups: per_group`); an unseen group uses the mean of the group rows
            (with a warning), as `Project.score` does.
        nominal_dt: the median step of the fitting grid (`PreparedDataset.nominal_dt`). Needed for
            a cold start (`start()`) with a `stationary` initial distribution and as the `dt` of the first
            observation in `update_from_time`; `start(initial)` takes `initial.nominal_dt`.

    Raises:
        ConfigError: `engine: explicit`, time-step memory, a recurrent emission carry, or a missing `group`.
    """

    def __init__(
        self,
        fit: FitResult,
        *,
        group: str | None = None,
        nominal_dt: float | None = None,
        lag: int | None = None,
        alerts: Sequence[Mapping[str, Any] | Any] | None = None,
    ) -> None:
        """Validate the model, build the state space and freeze the parameters used by every step."""
        spec = fit.spec
        if spec.engine != "expanded":
            raise ConfigError("online filtering requires engine: expanded", path="model.engine")
        if spec.memory_mode != "segments":
            raise ConfigError("online filtering needs segment memory (not memory_mode: steps)", path="model")
        if spec.emission.init_carry() is not None:
            raise ConfigError("online filtering does not support recurrent emission carries", path="model")
        self.fit = fit
        self.spec = spec
        self.space = build_expanded(spec)
        self.nominal_dt = nominal_dt
        self.theta: Params = dict(fit.params)
        if spec.n_groups > 1:
            self.theta["initial.logits"] = self._group_row(fit, group)
        self.hash = spec_hash(fit)
        # `spec` is bound once (hashing an equinox module costs ~40 us per call); compiled once per filter
        if lag is None:
            lag = int((fit.config_resolved.get("online") or {}).get("lag", 0))
        if lag < 0:
            raise ConfigError(f"lag must be >= 0, got {lag}", path="online.lag")
        self.lag = lag
        if alerts is None:
            alerts = (fit.config_resolved.get("online") or {}).get("alerts") or []
        self.rules: tuple[AlertRule, ...] = rules_from_config(alerts)
        for rule in self.rules:
            if not 0 <= rule.regime < spec.n_states:
                raise ConfigError(
                    f"regime must be < n_states={spec.n_states}", path=f"online.alerts.{rule.name}"
                )
            if rule.use == "smoothed" and lag == 0:
                raise ConfigError("use: smoothed needs a lag > 0", path=f"online.alerts.{rule.name}.use")
        has_duration = any(r.kind == "duration" for r in self.rules)
        self._step = jax.jit(partial(_step_fn, spec=spec, lag=lag, with_remaining=has_duration))
        self._steps = jax.jit(partial(_steps_fn, spec=spec, lag=lag, with_remaining=has_duration))
        self._has_duration = has_duration
        self._smooth: dict[int, Any] = {}  # lag -> jitted smoother
        self._w_pi = self.theta["transition.b"].shape[-1] if "transition.b" in self.theta else 0
        self._w_d = self.theta["duration.c"].shape[-1] if "duration.c" in self.theta else 0
        self._clock = str(fit.config_resolved.get("data", {}).get("clock", "steps"))
        self._clock_options = fit.config_resolved.get("data", {}).get("clock_options", {})

    @staticmethod
    def _group_row(fit: FitResult, group: str | None) -> Array:
        rows = np.asarray(fit.params["initial.logits"])
        known = list(fit.diagnostics.get("initial_groups") or [])
        if len(known) != rows.shape[0]:
            raise ConfigError(
                "the fit has per-group initial distributions but no group names "
                "(diagnostics['initial_groups'])",
                path="diagnostics.initial_groups",
            )
        if group is None:
            raise ConfigError(f"the fit has per-group initial distributions: pass group= one of {known}")
        if group in known:
            return jnp.asarray(rows[known.index(group)])
        warnings.warn(
            f"initial-distribution group {group!r} was not seen in the fit: using the mean of the group rows",
            UserWarning,
            stacklevel=3,
        )
        return jnp.asarray(rows.mean(axis=0))

    # -- starting -------------------------------------------------------------------------

    def _empty_alerts(self) -> np.ndarray | None:
        return np.zeros(len(self.rules), dtype=bool) if self.rules else None

    def start(self, initial: PreparedDataset | None = None) -> OnlineState:
        """The state before the first observation, or warm-started by the offline forward on `initial`."""
        if initial is None:
            if self.nominal_dt is None and self.spec.initial == "stationary":
                raise ConfigError(
                    "a cold start with a stationary initial distribution needs nominal_dt "
                    "(ExpandedOnlineFilter(..., nominal_dt=...)) or a warm start",
                    path="nominal_dt",
                )
            log_rho = initial_log_rho(
                self.spec, self.space, self.theta, 1.0 if self.nominal_dt is None else self.nominal_dt
            )
            return OnlineState(
                log_alpha=log_rho,
                log_norm=jnp.zeros(()),
                t=jnp.zeros((), dtype=jnp.int64),
                carry=None,
                last_time=None,
                buffer=self._empty_buffer(),
                spec_hash=self.hash,
                alerts=self._empty_alerts(),
            )
        if initial.observations.shape[0] == 0:
            raise ConfigError("the warm-start dataset is empty", path="initial")
        self.nominal_dt = float(initial.nominal_dt)
        log_alpha, loglik = ExpandedHMMEngine().forward(self.theta, self.spec, self.space, initial)
        times = np.asarray(initial.times)
        return OnlineState(
            log_alpha=log_alpha[-1],
            log_norm=loglik,
            t=jnp.asarray(initial.observations.shape[0], dtype=jnp.int64),
            carry=None,
            last_time=_normalize_time(times[-1]),
            buffer=None
            if self.lag == 0
            else buffer_from_prefix(self.theta, self.spec, self.space, initial, log_alpha, self.lag),
            spec_hash=self.hash,
            alerts=self._empty_alerts(),
        )

    def _empty_buffer(self) -> RingBuffer | None:
        if self.lag == 0:
            return None
        return empty_buffer(self.lag, self.space.n_expanded, self._w_pi, self._w_d)

    # -- updating -------------------------------------------------------------------------

    def _check_state(self, state: OnlineState) -> None:
        have = None if state.buffer is None else state.buffer.lag
        if have != (self.lag or None):
            raise ConfigError(
                f"the state carries a fixed-lag buffer of lag {have}, the filter has lag {self.lag}",
                path="state.buffer",
            )
        n_alerts = None if state.alerts is None else int(np.asarray(state.alerts).shape[0])
        if n_alerts != (len(self.rules) or None):
            raise ConfigError(
                f"the state has {n_alerts} alert flags, the filter has {len(self.rules)} rules",
                path="state.alerts",
            )
        if state.spec_hash != self.hash:
            raise ConfigError(
                f"the state belongs to another model (spec_hash {state.spec_hash} != {self.hash})",
                path="state",
            )

    def _inputs(self, y: ArrayLike, dt: float, z: Covariates | None) -> tuple[np.ndarray, ...]:
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_arr.shape[0] != self.spec.n_series:
            raise ConfigError(f"expected {self.spec.n_series} series, got {y_arr.shape[0]}", path="y")
        dt_arr = np.asarray(dt, dtype=np.float64)
        if not np.isfinite(dt_arr) or dt_arr <= 0:
            raise ConfigError(f"dt must be positive and finite, got {float(dt_arr)}", path="dt")
        # NumPy inputs go straight to the jitted step (cheaper than one `jnp.asarray` dispatch each)
        return (
            y_arr,
            dt_arr,
            _row(z, "transition", self._w_pi),
            _row(z, "duration", self._w_d),
            _row(z, "emission", None),
        )

    def _clock_remaining(self, steps: np.ndarray | None, dt: float) -> np.ndarray | None:
        """Expected remaining steps -> clock units (`nominal_dt`, else the current step's `dt`)."""
        if steps is None:
            return None
        return steps * (dt if self.nominal_dt is None else self.nominal_dt)

    def _result(
        self, state: OnlineState, filtered: Array, smoothed: Array | None, events: list[AlertEvent]
    ) -> Step:
        out: list[Any] = [state, filtered]
        if smoothed is not None:
            out.append(smoothed)
        if self.rules:
            out.append(events)
        return tuple(out)

    def update(
        self,
        state: OnlineState,
        y: ArrayLike,
        dt: float,
        z: Covariates | None = None,
        *,
        time: Any = None,  # noqa: ANN401
    ) -> Step:
        """Absorb one observation `y (M,)` (`NaN` = unobserved) after a gap `dt`; `z` the lagged rows.

        Returns:
            `(state', filtered)` with `filtered (N,)` the filtered regime probabilities; with `lag > 0`
            `smoothed_lagged (N,)` follows (the smoothed marginal of step `t - lag` given the data up to `t`,
            `NaN` until `lag + 1` observations were seen); with alert rules the list of `AlertEvent`s of this
            step comes last. `time` stamps the events (default: the step index).

        Raises:
            NumericalError: a non-finite log-likelihood (every regime impossible).
        """
        self._check_state(state)
        ins = self._inputs(y, dt, z)
        alpha, log_norm, t, filtered, buffer, smoothed, remaining = self._step(
            self.theta,
            self.space,
            state.log_alpha,
            state.t,
            *ins,
            state.buffer,
        )
        if not math.isfinite(float(log_norm)):
            raise NumericalError(f"non-finite running log-likelihood at step {int(t)}", path="state.log_norm")
        new = dataclasses.replace(state, log_alpha=alpha, log_norm=log_norm, t=t, buffer=buffer, carry=None)
        events: list[AlertEvent] = []
        if self.rules:
            rem = self._clock_remaining(None if remaining is None else np.asarray(remaining), float(ins[1]))
            active, events = evaluate(
                self.rules,
                np.asarray(state.alerts),
                int(t),
                int(t) if time is None else _normalize_time(time),
                np.asarray(filtered),
                None if smoothed is None else np.asarray(smoothed),
                rem,
            )
            new = dataclasses.replace(new, alerts=active)
        return self._result(new, filtered, smoothed, events)

    def update_many(
        self,
        state: OnlineState,
        ys: ArrayLike,
        dts: ArrayLike,
        zs: Covariates | None = None,
        *,
        times: ArrayLike | None = None,
    ) -> Step:
        """`lax.scan` of `update` over `T'` observations: `ys (T', M)`, `dts (T',)`, `zs` blocks `(T', K)`.

        Returns:
            `(state', filtered)` with `filtered (T', N)`; with `lag > 0` also `smoothed (T', N)` (row `i`:
            step `t_i - lag` given data up to `t_i`, `NaN` while the buffer fills); with alert rules the
            ordered list of `AlertEvent`s of all steps comes last (`times (T',)` stamps them).
        """
        self._check_state(state)
        ys_arr = np.asarray(ys, dtype=np.float64)
        n = ys_arr.shape[0]
        if ys_arr.ndim != 2 or ys_arr.shape[1] != self.spec.n_series:
            raise ConfigError(f"expected ys of shape (T', {self.spec.n_series})", path="ys")
        dts_arr = np.asarray(dts, dtype=np.float64).reshape(-1)
        if dts_arr.shape[0] != n or not np.all(np.isfinite(dts_arr) & (dts_arr > 0)):
            raise ConfigError("dts must be T' positive finite values", path="dts")

        def block(key: str, width: int | None) -> np.ndarray:
            v = None if zs is None else zs.get(key)
            if v is None:
                if width:
                    raise ConfigError(f"the model needs covariate rows of width {width}", path=f"zs.{key}")
                return np.zeros((n, 0))
            arr = np.asarray(v, dtype=np.float64).reshape(n, -1)
            if width is not None and arr.shape[1] != width:
                raise ConfigError(f"expected width {width}, got {arr.shape[1]}", path=f"zs.{key}")
            return arr

        if times is not None and np.asarray(times).shape[0] != n:
            raise ConfigError("times must have T' entries", path="times")
        alpha, log_norm, t, filtered, buffer, smoothed, remaining = self._steps(
            self.theta,
            self.space,
            state.log_alpha,
            state.t,
            jnp.asarray(ys_arr),
            jnp.asarray(dts_arr),
            jnp.asarray(block("transition", self._w_pi)),
            jnp.asarray(block("duration", self._w_d)),
            jnp.asarray(block("emission", None)),
            state.buffer,
        )
        if n and not bool(jnp.isfinite(log_norm)):
            raise NumericalError("non-finite running log-likelihood", path="state.log_norm")
        new = dataclasses.replace(
            state,
            log_alpha=alpha,
            log_norm=log_norm if n else state.log_norm,
            t=t,
            buffer=buffer,
            carry=None,
        )
        events: list[AlertEvent] = []
        if self.rules:
            stamps = None if times is None else np.asarray(times)
            f_np = np.asarray(filtered)
            s_np = None if smoothed is None else np.asarray(smoothed)
            r_np = None if remaining is None else np.asarray(remaining)
            active = np.asarray(state.alerts)
            t0 = int(state.t)
            for i in range(n):
                rem = self._clock_remaining(None if r_np is None else r_np[i], float(dts_arr[i]))
                active, ev = evaluate(
                    self.rules,
                    active,
                    t0 + i + 1,
                    t0 + i + 1 if stamps is None else _normalize_time(stamps[i]),
                    f_np[i],
                    None if s_np is None else s_np[i],
                    rem,
                )
                events.extend(ev)
            new = dataclasses.replace(new, alerts=active)
        return self._result(new, filtered, smoothed, events)

    def smoothed_at(self, state: OnlineState, lag: int | None = None) -> Array:
        """Smoothed regime marginals `(N,)` of step `t - lag` given the data up to `t` (`NaN` while filling).

        Equals the offline smoother run on the prefix `y_{1:t}` (`data.window(0, t + 1)`), and the full-data
        smoother at `t - lag` when `t` is the last step. `lag <= self.lag`; cost `O(lag S^2)`.
        """
        self._check_state(state)
        use = self.lag if lag is None else lag
        if state.buffer is None or not 0 < use <= self.lag:
            raise ConfigError(f"fixed-lag smoothing needs 0 < lag <= {self.lag} (filter lag)", path="lag")
        if use not in self._smooth:
            self._smooth[use] = jax.jit(partial(smooth_window, lag=use))
        out: Array = self._smooth[use](self.theta, self.space, state.buffer)
        return out

    def forecast(
        self,
        state: OnlineState,
        horizon: int,
        n_sim: int,
        data: PreparedDataset,
        *,
        covariate_forecaster: CovariateForecaster | None = None,
        seed: int = 0,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> ForecastResult:
        """Monte-Carlo forecast from the live state: `softmax(state.log_alpha)` is the origin distribution.

        Identical to `MonteCarloForecaster.forecast` on the offline filter pass up to the same step (same
        `seed`, bitwise samples): only the origin vector differs in how it was obtained.

        Args:
            state: the stream state at the forecast origin.
            horizon: steps to forecast.
            n_sim: number of simulated paths.
            data: the prepared history that ends at the origin (its last row gives the origin time, the clock
                and the last covariate row; covariate forecasters are fitted on it).
            covariate_forecaster: needed for `horizon > 1` with covariates.
            seed: RNG seed.
            quantiles: cumulative-return quantile levels.
        """
        from hsmmkit.config.schema import ForecastSection
        from hsmmkit.inference.forecast import MonteCarloForecaster

        self._check_state(state)
        if int(state.t) == 0:
            raise ConfigError("forecasting needs at least one observation", path="state.t")
        cfg = ForecastSection(horizon=horizon, n_sim=n_sim, quantiles=list(quantiles))
        mc = MonteCarloForecaster(
            self.spec, self.space, ExpandedHMMEngine(), covariate_forecaster, cfg, self._clock_options
        )
        zeros = jnp.zeros((1, self.space.n_expanded))
        origin = Posterior(state.log_alpha[None, :], zeros, zeros, state.log_norm, self.space.index)
        return mc.forecast(self.fit, data, horizon, n_sim, seed, posterior=origin)

    def update_from_time(
        self,
        state: OnlineState,
        time: Any,  # noqa: ANN401
        y: ArrayLike,
        z: Covariates | None = None,
    ) -> Step:
        """`update` with `dt` computed from `state.last_time` and `time` by the configured clock.

        The first observation of a stream without a warm start has no predecessor: its `dt` is `nominal_dt`
        (it enters the emission, e.g. a family with `scale_by_dt`). The returned `last_time` is `time`; events
        carry it as their `time`.
        """
        if state.last_time is None:
            if self.nominal_dt is None:
                raise ConfigError(
                    "the first observation of a cold-started stream needs nominal_dt as its dt "
                    "(ExpandedOnlineFilter(..., nominal_dt=...))",
                    path="nominal_dt",
                )
            dt: Any = self.nominal_dt
        else:
            clk = get_clock(self._clock)
            opts = parse_clock_options(self._clock, self._clock_options)
            pair = np.asarray([state.last_time, _normalize_time(time)])
            dt = float(clk.dt(pair, 1, opts)[0])
        stamp = _normalize_time(time)
        new, *rest = self.update(state, y, dt, z, time=stamp)
        return (dataclasses.replace(new, last_time=stamp), *rest)
