"""EM (Baum-Welch) cross-check for the covariate-free Gaussian / geometric-or-negbin model (E5-T06, v1.x).

Scope (anything else raises `ConfigError`): `GaussianFamily` (or an alias) without covariate links, a
single series, the expanded engine, `transition.order == 1` with `JumpTransition`
(`transition.pi_rows`, no covariates), a geometric or negative-binomial phase duration (`duration.rate`,
no covariates), `initial: uniform`, and a regular time grid. Parameters in `prior` mode are estimated by
maximum likelihood (EM ignores priors).

E-step: `forward_backward` gives the smoothed posterior; the expected transition counts `Xi = sum_t xi_t`,
`xi_t(s, s') = alpha_{t-1}(s) A_t(s, s') ell_t(s') beta_t(s') / L`, are summed in one `lax.scan` as an
`(S, S)` matrix (never `(T, S, S)`).

M-step (closed form):
* emission: `GaussianFamily.sufficient_stats` -> `m_i = w_y / w_dt`, `sigma_i^2 = (...) / w`,
  `mu_i = m_i + sigma_i^2 / 2`;
* advance probability per regime `a_i = (expected advance + exit counts) / (those + expected stay counts)`
  summed over the regime's phases, `rate_i = -log(1 - a_i) / dt`;
* jump rows: expected exit counts into each other regime, normalised.

The log-likelihood is non-decreasing (EM property); `debug=True` raises `NumericalError` on a decrease.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from regimekit.engine.expanded_hmm import ExpandedHMMEngine, _regime_log_emis
from regimekit.estimation.init import observed_rows, rollvol_kmeans
from regimekit.model.duration.geometric import GeometricDuration
from regimekit.model.duration.negbin import NegBinDuration
from regimekit.model.emission.gaussian import GaussianFamily
from regimekit.model.params.manager import ParamManager
from regimekit.model.transition import JumpTransition
from regimekit.statespace.builders import build_expanded
from regimekit.utils.clock import is_irregular
from regimekit.utils.errors import ConfigError, NumericalError
from regimekit.utils.rng import PURPOSE_EM_INIT, key_for
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.config.schema import EstimationSection
    from regimekit.data.contracts import PreparedDataset
    from regimekit.model.params import Params
    from regimekit.model.spec import ModelSpec
    from regimekit.statespace.index import StateSpace

_COVARIATE_PARAMS = frozenset(
    {"emission.gamma", "emission.delta", "duration.c", "transition.a", "transition.b"}
)
_REQUIRED = ("duration.rate", "transition.pi_rows")
MONOTONE_SLACK = 1e-8  # relative loglik decrease tolerated before `debug` raises (rounding)


@dataclass(frozen=True)
class EMResult:
    """Outcome of `em_fit`.

    Attributes:
        params: constrained parameters, shaped like `spec.param_specs`.
        uparams: unconstrained vector `(n_free,)` (`ParamManager.flatten`).
        loglik: final log-likelihood (Python float).
        logliks: log-likelihood at the start of each iteration plus the final value.
        n_iter: EM iterations run.
        converged: whether the relative change fell below `tol_f`.
    """

    params: Params
    uparams: Array
    loglik: float
    logliks: tuple[float, ...]
    n_iter: int
    converged: bool


def check_em_supported(spec: ModelSpec, data: PreparedDataset) -> None:
    """Raise `ConfigError` unless `spec` is the covariate-free Gaussian / geometric-or-negbin jump model."""
    if not isinstance(spec.emission, GaussianFamily):
        raise ConfigError(
            f"EM supports the gaussian emission family only, got {type(spec.emission).__name__}",
            path="model.emission.family",
        )
    names = {s.name for s in spec.param_specs}
    if spec.emission.mean_idx or spec.emission.scale_idx or (names & _COVARIATE_PARAMS):
        raise ConfigError("EM does not support covariates", path="model.emission.covariates")
    if getattr(spec.duration, "cov_idx", ()) or getattr(spec.transition, "cov_idx", ()):
        raise ConfigError("EM does not support covariates", path="model.duration.covariates")
    if spec.duration_conditioning != "current":
        raise ConfigError("EM supports duration_conditioning 'current' only", path="model.transition")
    if any(s.name == spec.emission.scale_name() and s.constraint == "increasing" for s in spec.param_specs):
        raise ConfigError(
            "EM does not keep an 'increasing' scale ordering; use a free scale (MAP applies the ordering)",
            path=f"model.emission.params.{spec.emission.scale_param}",
        )
    if spec.engine != "expanded" or spec.order != 1 or spec.initial != "uniform" or spec.n_series != 1:
        raise ConfigError(
            "EM supports the expanded engine, order 1, a uniform initial distribution and one series",
            path="model",
        )
    if not isinstance(spec.duration, GeometricDuration | NegBinDuration) or not isinstance(
        spec.transition, JumpTransition
    ):
        raise ConfigError(
            "EM supports geometric/negbin durations with a jump transition", path="model.duration"
        )
    required = (spec.emission.mean_name(), spec.emission.scale_name(), *_REQUIRED)
    missing = [n for n in required if n not in names]
    if missing:
        raise ConfigError(f"EM needs the parameters {missing}", path="model")
    for s in spec.param_specs:
        modes = (s.mode,) if isinstance(s.mode, str) else s.mode
        if s.name in required and any(m not in ("free", "prior") for m in modes):
            raise ConfigError(f"EM estimates free parameters only, {s.name} has mode {s.mode!r}", path=s.name)
    dt = np.asarray(data.dt)
    if is_irregular(dt) or not np.allclose(dt, dt[0], rtol=1e-9, atol=0.0):
        raise ConfigError(
            "EM needs a constant time step (closed-form advance-probability update)", path="data"
        )


@partial(jax.jit, static_argnames=("spec",))
def _expected_counts(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    data: PreparedDataset,
    log_alpha: Array,
    log_beta: Array,
    loglik: Array,
) -> Array:
    """`Xi = sum_t xi_t`, `(S, S)`, accumulated by a scan (no `(T, S, S)` array)."""
    ell = jnp.take(_regime_log_emis(theta, spec, data), space.index.regime_of, axis=1)  # (T, S)

    def step(acc: Array, xs: tuple[Array, Array, Array, Array]) -> tuple[Array, None]:
        la_prev, ell_t, lb_t, dt_t = xs
        _, log_a = space.transition_fn(theta, None, None, dt_t)
        log_xi = la_prev[:, None] + log_a + (ell_t + lb_t)[None, :] - loglik
        ok = jnp.isfinite(log_xi)
        return acc + jnp.where(ok, jnp.exp(jnp.where(ok, log_xi, 0.0)), 0.0), None

    s = log_alpha.shape[1]
    xs = (log_alpha[:-1], ell[1:], log_beta[1:], data.dt[1:])
    total, _ = jax.lax.scan(step, jnp.zeros((s, s)), xs)
    return total


def _m_step(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    data: PreparedDataset,
    smoothed: Array,
    xi: Array,
) -> Params:
    """Closed-form parameter update from the expected statistics."""
    n = spec.n_states
    idx = space.index
    shapes = {s.name: s.shape for s in spec.param_specs}
    new = dict(theta)

    # emission
    gamma = (smoothed @ idx.collapse) * jnp.all(data.mask, axis=-1)[:, None]  # (T, N), observed steps only
    emis = spec.emission
    assert isinstance(emis, GaussianFamily)
    st = emis.sufficient_stats(data.observations, data.dt, gamma)
    assert isinstance(st, dict)
    if not bool(jnp.all(st["w"] > 0)):
        raise NumericalError(
            "a regime has no posterior mass; reduce n_states or change the start", path="gamma"
        )
    m = st["w_y"] / st["w_dt"]
    var = (st["w_y2_dt"] - 2.0 * m * st["w_y"] + m**2 * st["w_dt"]) / st["w"]
    if not bool(jnp.all(var > 0)):
        raise NumericalError("degenerate (zero) variance in an EM M-step", path=emis.scale_name())
    mean_name, scale_name = emis.mean_name(), emis.scale_name()
    new[scale_name] = jnp.sqrt(var).reshape(shapes[scale_name])
    new[mean_name] = (m + 0.5 * var if emis.ito_correction else m).reshape(shapes[mean_name])

    # advance probability and jump rows from the expected transition counts
    xi_np = np.asarray(xi)
    regime = np.asarray(idx.regime_of)
    stay = np.diag(xi_np)
    leave = xi_np.sum(axis=1) - stay
    stay_i = np.bincount(regime, weights=stay, minlength=n)
    leave_i = np.bincount(regime, weights=leave, minlength=n)
    if not np.all(leave_i + stay_i > 0):
        raise NumericalError("a regime has no expected transition mass", path="duration.rate")
    a = np.clip(leave_i / (leave_i + stay_i), 1e-12, 1.0 - 1e-12)
    dt = float(np.asarray(data.dt)[0])
    rate = jnp.asarray(-np.log1p(-a) / dt)
    new["duration.rate"] = rate.reshape(shapes["duration.rate"])
    adv = np.asarray(spec.duration.advance_prob_t(new, dt, None))
    if not np.allclose(adv, a, rtol=1e-9, atol=1e-12):
        raise ConfigError(
            "EM expects advance_prob = 1 - exp(-rate * dt) for the duration family", path="duration"
        )

    exits = np.zeros((n, n))
    last = np.asarray(space.is_last_phase)
    targets = np.asarray(space.exit_targets)
    for s_idx in np.flatnonzero(last):
        for j in range(n):
            if targets[s_idx, j] >= 0:
                exits[regime[s_idx], j] += xi_np[s_idx, targets[s_idx, j]]
    rows = np.stack([np.array([exits[i, j] for j in range(n) if j != i]) for i in range(n)])
    prev = np.asarray(theta["transition.pi_rows"]).reshape(rows.shape)
    total = rows.sum(axis=1, keepdims=True)
    rows = np.where(
        total > 0, rows / np.where(total > 0, total, 1.0), prev
    )  # a regime never left keeps its row
    new["transition.pi_rows"] = jnp.asarray(rows).reshape(shapes["transition.pi_rows"])
    return new


def em_fit(
    spec: ModelSpec,
    data: PreparedDataset,
    cfg: EstimationSection,
    seed: int,
    *,
    theta0: Params | None = None,
    debug: bool = False,
) -> EMResult:
    """Fit by EM from `theta0` (default: the `rollvol_kmeans` start for `seed`).

    Args:
        spec: a model accepted by `check_em_supported`.
        data: prepared data (one regular grid).
        cfg: uses `max_iter` and `tol_f` (relative log-likelihood change).
        seed: only used for the default start.
        theta0: optional starting parameters.
        debug: raise `NumericalError` if the log-likelihood decreases (beyond rounding).

    Raises:
        ConfigError: unsupported model (covariates, non-Gaussian, explicit engine, ...).
        NumericalError: a non-finite log-likelihood, or a decrease under `debug`.
    """
    check_em_supported(spec, data)
    manager = ParamManager(spec.param_specs)
    space = build_expanded(spec)
    engine = ExpandedHMMEngine()
    theta = (
        dict(theta0)
        if theta0 is not None
        else dict(rollvol_kmeans(observed_rows(data), spec, key_for(seed, PURPOSE_EM_INIT)))
    )

    logliks: list[float] = []
    converged = False
    n_iter = 0
    for n_iter in range(1, cfg.max_iter + 1):
        post = engine.forward_backward(theta, spec, space, data)
        ll = float(post.loglik)
        if not np.isfinite(ll):
            raise NumericalError("non-finite log-likelihood during EM", path="loglik")
        if debug and logliks and ll < logliks[-1] - MONOTONE_SLACK * abs(logliks[-1]):
            raise NumericalError(
                f"EM log-likelihood decreased: {logliks[-1]!r} -> {ll!r} at iteration {n_iter}", path="loglik"
            )
        logliks.append(ll)
        if len(logliks) > 1 and abs(ll - logliks[-2]) <= cfg.tol_f * max(abs(ll), 1.0):
            converged = True
            break
        xi = _expected_counts(theta, spec, space, data, post.log_alpha, post.log_beta, post.loglik)
        theta = _m_step(theta, spec, space, data, post.smoothed, xi)

    final = float(engine.loglik(theta, spec, space, data))
    return EMResult(
        params=theta,
        uparams=manager.flatten(theta),
        loglik=final,
        logliks=(*logliks, final),
        n_iter=n_iter,
        converged=converged,
    )
