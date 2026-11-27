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

import math
import warnings
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.engine.base import filtered_row
from hsmmkit.engine.expanded_hmm import (
    ExpandedHMMEngine,
    _safe_logsumexp,
    expanded_log_emis,
    forward_step,
)
from hsmmkit.online.state import OnlineState, spec_hash
from hsmmkit.statespace.builders import build_expanded, initial_log_rho
from hsmmkit.utils.clock import get_clock, parse_clock_options
from hsmmkit.utils.errors import ConfigError, NumericalError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace
    from hsmmkit.utils.typing import Array

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
    *,
    spec: ModelSpec,
) -> tuple[Array, Array, Array, Array]:
    """`(log_alpha', log_norm', t', filtered)`; the first observation (`t == 0`) starts from `log_rho`."""
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
    return new, log_norm, t + 1, filtered


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
    *,
    spec: ModelSpec,
) -> tuple[Array, Array, Array, Array]:
    def body(carry: tuple[Array, Array], xs: tuple[Array, ...]) -> tuple[tuple[Array, Array], Array]:
        alpha, tt = carry
        alpha, _, tt, filtered = _step_fn(theta, space, alpha, tt, *xs, spec=spec)
        return (alpha, tt), filtered

    (alpha, tt), filtered = jax.lax.scan(body, (log_alpha, t), (ys, dts, z_pis, z_ds, z_es))
    return alpha, _safe_logsumexp(alpha, axis=0), tt, filtered


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

    def __init__(self, fit: FitResult, *, group: str | None = None, nominal_dt: float | None = None) -> None:
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
        self._step = jax.jit(partial(_step_fn, spec=spec))
        self._steps = jax.jit(partial(_steps_fn, spec=spec))
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
                buffer=None,
                spec_hash=self.hash,
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
            buffer=None,
            spec_hash=self.hash,
        )

    # -- updating -------------------------------------------------------------------------

    def _check_state(self, state: OnlineState) -> None:
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

    def update(
        self,
        state: OnlineState,
        y: ArrayLike,
        dt: float,
        z: Covariates | None = None,
    ) -> tuple[OnlineState, Array]:
        """Absorb one observation `y (M,)` (`NaN` = unobserved) after a gap `dt`; `z` the lagged rows.

        Returns:
            `(state', filtered)` with `filtered (N,)` the filtered regime probabilities.

        Raises:
            NumericalError: a non-finite log-likelihood (every regime impossible).
        """
        self._check_state(state)
        ins = self._inputs(y, dt, z)
        alpha, log_norm, t, filtered = self._step(
            self.theta,
            self.space,
            state.log_alpha,
            state.t,
            *ins,
        )
        if not math.isfinite(float(log_norm)):
            raise NumericalError(f"non-finite running log-likelihood at step {int(t)}", path="state.log_norm")
        new = OnlineState(alpha, log_norm, t, None, state.last_time, state.buffer, state.spec_hash)
        return new, filtered

    def update_many(
        self,
        state: OnlineState,
        ys: ArrayLike,
        dts: ArrayLike,
        zs: Covariates | None = None,
    ) -> tuple[OnlineState, Array]:
        """`lax.scan` of `update` over `T'` observations: `ys (T', M)`, `dts (T',)`, `zs` blocks `(T', K)`.

        Returns:
            `(state', filtered)` with `filtered (T', N)`.
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

        alpha, log_norm, t, filtered = self._steps(
            self.theta,
            self.space,
            state.log_alpha,
            state.t,
            jnp.asarray(ys_arr),
            jnp.asarray(dts_arr),
            jnp.asarray(block("transition", self._w_pi)),
            jnp.asarray(block("duration", self._w_d)),
            jnp.asarray(block("emission", None)),
        )
        if n and not bool(jnp.isfinite(log_norm)):
            raise NumericalError("non-finite running log-likelihood", path="state.log_norm")
        new = OnlineState(
            alpha, log_norm if n else state.log_norm, t, None, state.last_time, state.buffer, state.spec_hash
        )
        return new, filtered

    def update_from_time(
        self,
        state: OnlineState,
        time: Any,  # noqa: ANN401
        y: ArrayLike,
        z: Covariates | None = None,
    ) -> tuple[OnlineState, Array]:
        """`update` with `dt` computed from `state.last_time` and `time` by the configured clock.

        The first observation of a stream without a warm start has no predecessor: its `dt` is `nominal_dt`
        (it enters the emission, e.g. a family with `scale_by_dt`). The returned `last_time` is `time`.
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
        new, filtered = self.update(state, y, dt, z)
        t = _normalize_time(time)
        return OnlineState(
            new.log_alpha, new.log_norm, new.t, new.carry, t, new.buffer, new.spec_hash
        ), filtered
