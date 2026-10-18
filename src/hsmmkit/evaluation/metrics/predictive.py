"""Predictive diagnostics (PIT, Ljung-Box; E5-T04) and forecast metrics (E7-T01).

Concept refs: §4.10 Diagnostics. `predictive_regime_probs` computes `P(regime_t | y_{1:t-1})` from an
already-computed `Posterior` (its `log_alpha` is the *filtered* joint, `P(y_{1:t}, s_t)`): step 0 is the
prior `softmax(log_rho)`; step `t >= 1` reuses the same per-step `A_t` builder the engines use
(`space.transition_fn`), applied to `softmax(log_alpha[t-1])`, following the same covariate/`dt` timing
as the forward recursion (CLAUDE.md §3.7: step `t`'s transition uses covariate/`dt` row `t`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np

from regimekit.estimation.results import FitResult
from regimekit.evaluation.metrics.registry import metric
from regimekit.inference.results import ForecastResult
from regimekit.statespace.builders import initial_log_rho
from regimekit.utils.errors import ConfigError, NumericalError

if TYPE_CHECKING:
    from regimekit.data.contracts import PreparedDataset
    from regimekit.engine.base import Posterior
    from regimekit.model.params.manager import Params
    from regimekit.model.spec import ModelSpec
    from regimekit.statespace.index import StateSpace
    from regimekit.utils.typing import Array


def _block(x: Array | None, t: int) -> Array:
    return jnp.zeros((t, 0)) if x is None else x


def predictive_regime_probs(
    posterior: Posterior, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> Array:
    """`(T, N)`: `P(regime_t = i | y_{1:t-1})`, the one-step-ahead predictive regime distribution."""
    t_len = data.observations.shape[0]
    log_rho = initial_log_rho(spec, space, theta, data.nominal_dt)
    pred_0 = jax.nn.softmax(log_rho)
    if t_len == 1:
        state_probs = pred_0[None, :]
    else:
        z_pi = _block(data.cov_transition, t_len)[1:]
        z_d = _block(data.cov_duration, t_len)[1:]
        dt = data.dt[1:]
        log_alpha_prev = posterior.log_alpha[:-1]

        def step(carry: None, xs: tuple[Array, Array, Array, Array]) -> tuple[None, Array]:
            del carry
            la_prev, z_pi_t, z_d_t, dt_t = xs
            a_t, _log_a = space.transition_fn(theta, z_pi_t, z_d_t, dt_t)
            return None, jax.nn.softmax(la_prev) @ a_t

        _, rest = jax.lax.scan(step, None, (log_alpha_prev, z_pi, z_d, dt))
        state_probs = jnp.concatenate([pred_0[None, :], rest], axis=0)
    out: Array = state_probs @ space.index.collapse
    return out


def pit_values(
    predictive_regime_probs_: Array, spec: ModelSpec, theta: Params, data: PreparedDataset, series: int = 0
) -> Array:
    """`(T,)`: PIT `F_t(y_t) = sum_i P(regime_t = i | y_{1:t-1}) F_i(y_t)` for one series.

    `F_i` is the emission family's own `cdf` (per-regime predictive CDF). Single-series only: PIT is a
    univariate calibration check, so `series` selects which of the `M` observed series to evaluate.
    `y` is sanitized (`NaN` at masked rows -> `0.0`, like `engine/expanded_hmm.py`'s `_regime_log_emis`)
    before `cdf`, and the corresponding PIT entries are set to `NaN` (never a fabricated real number)
    so callers can filter them out rather than silently treating an unobserved step as calibrated.
    """
    t_len = data.observations.shape[0]
    z_e = _block(data.cov_emission, t_len)
    observed = jnp.all(data.mask[:, series : series + 1], axis=-1)
    y_safe = jnp.where(data.mask, data.observations, 0.0)
    cdf_t = spec.emission.cdf(y_safe, data.dt, theta, z_e)[:, :, series]  # (T, N)
    pit = jnp.clip(jnp.sum(predictive_regime_probs_ * cdf_t, axis=-1), 0.0, 1.0)
    out: Array = jnp.where(observed, pit, jnp.nan)
    return out


def pit_histogram(pit: Array | np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, float, float]:
    """`(hist (n_bins,), ks_statistic, ks_p_value)`: PIT histogram and a KS test against `Uniform(0, 1)`.

    The p-value uses the standard asymptotic Kolmogorov distribution (no `scipy` dependency): `p =
    2 sum_{k=1}^inf (-1)^(k-1) exp(-2 k^2 n D^2)`, truncated at `k = 100` (converges within machine
    precision well before that for any `D` seen in practice). `NaN` entries (masked/unobserved steps,
    see `pit_values`) are dropped first.
    """
    values = np.asarray(pit)
    values = values[np.isfinite(values)]
    n = values.shape[0]
    hist, _ = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    sorted_v = np.sort(values)
    ecdf_upper = np.arange(1, n + 1) / n
    ecdf_lower = np.arange(0, n) / n
    d_stat = float(np.max(np.maximum(ecdf_upper - sorted_v, sorted_v - ecdf_lower)))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d_stat
    k = np.arange(1, 101)
    p = 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k**2 * lam**2))
    return hist, d_stat, float(np.clip(p, 0.0, 1.0))


def ljung_box(
    residuals: Array | np.ndarray, lags: tuple[int, ...] = (10, 20)
) -> dict[int, tuple[float, float]]:
    """`{lag: (Q_statistic, p_value)}`: Ljung-Box portmanteau test for residual autocorrelation.

    `Q = n(n+2) sum_{k=1}^{lag} rho_k^2 / (n-k)`, asymptotically `chi2(lag)` under the null of no
    autocorrelation; the p-value is the upper tail `1 - gammainc(lag / 2, Q / 2)` (regularized lower
    incomplete gamma, `jax.scipy.special.gammainc`, matching a chi-square CDF). `NaN` entries
    (masked/unobserved steps) are dropped first (a small gap in the lag structure, not a NaN leak).
    """
    r = np.asarray(residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.shape[0]
    r = r - r.mean()
    denom = np.sum(r**2)
    out: dict[int, tuple[float, float]] = {}
    for lag in lags:
        if lag >= n:
            out[lag] = (float("nan"), float("nan"))
            continue
        acf = np.array([np.sum(r[k:] * r[:-k]) / denom for k in range(1, lag + 1)])
        q_stat = float(n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lag + 1))))
        p_value = float(1.0 - jax.scipy.special.gammainc(lag / 2.0, q_stat / 2.0))
        out[lag] = (q_stat, p_value)
    return out


# --- forecast metrics (E7-T01) -----------------------------------------------------------------
#
# Context: `forecast` (`ForecastResult`), `realized (H, M)` realized future log-returns for steps
# 1..H and `horizon` (<= H). The predictive quantity at horizon h is the cumulative return
# `Y_h = sum_{k<=h} y_{t+k}`, compared with the forecast samples `cumsum(samples, axis=1)[:, h-1]`.
# With `M > 1` series every metric is evaluated per series and returned as `<name>_<m>` entries.


def forecast_of(ctx: dict[str, object]) -> ForecastResult:
    """`ctx['forecast']` as a `ForecastResult`."""
    forecast = ctx.get("forecast")
    if not isinstance(forecast, ForecastResult):
        raise ConfigError("this metric needs ctx['forecast'] (a ForecastResult)", path="evaluation.metrics")
    return forecast


def _forecast_ctx(ctx: dict[str, object]) -> tuple[np.ndarray, np.ndarray, int]:
    """`(cum_samples (n_sim, M), y_h (M,), horizon)` from the metric context."""
    forecast, realized, horizon = ctx.get("forecast"), ctx.get("realized"), ctx.get("horizon")
    if not isinstance(forecast, ForecastResult) or realized is None or not isinstance(horizon, int):
        raise ConfigError(
            "forecast metrics need ctx['forecast'], ctx['realized'] and ctx['horizon']",
            path="evaluation.metrics",
        )
    h = horizon
    samples = np.asarray(forecast.samples, dtype=np.float64)
    real = np.atleast_2d(np.asarray(realized, dtype=np.float64))
    if samples.shape[0] == 0:
        raise ConfigError("the forecast carries no samples (write it with samples)", path="forecast.samples")
    if not 1 <= h <= samples.shape[1] or h > real.shape[0]:
        raise ConfigError(
            f"horizon {h} is outside the forecast ({samples.shape[1]} steps) or realized ({real.shape[0]})",
            path="evaluation.backtest.horizons",
        )
    cum = np.cumsum(samples, axis=1)[:, h - 1, :]
    return cum, real[:h].sum(axis=0), h


def _series_result(name: str, values: np.ndarray) -> float | dict[str, float]:
    if not np.all(np.isfinite(values)):
        raise NumericalError(f"non-finite value in metric {name!r}", path=name)
    if values.shape[0] == 1:
        return float(values[0])
    return {f"{name}_{m}": float(v) for m, v in enumerate(values)}


def kde_log_density(samples: np.ndarray, y: float) -> float:
    """`log` of a Gaussian-KDE density estimate at `y` (Silverman's rule-of-thumb bandwidth).

    `bw = 0.9 min(sd, IQR / 1.34) n^(-1/5)`, floored at `1e-12 (1 + |y|)` for degenerate samples.
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.shape[0]
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    q75, q25 = np.quantile(x, [0.75, 0.25])
    iqr = float(q75 - q25)
    scale = min(sd, iqr / 1.34) if iqr > 0 else sd
    bw = max(0.9 * scale * n ** (-0.2), 1e-12 * (1.0 + abs(y)))
    z = -0.5 * ((y - x) / bw) ** 2
    zmax = float(z.max())
    return float(zmax + np.log(np.exp(z - zmax).sum()) - np.log(n) - np.log(bw) - 0.5 * np.log(2.0 * np.pi))


def crps_from_samples(samples: np.ndarray, y: float) -> float:
    """Sample CRPS `E|X - y| - 0.5 E|X - X'|` in `O(n log n)`.

    With the sorted samples `x_(1) <= ... <= x_(n)`, `E|X - X'| = (2 / n^2) sum_i (2i - n - 1) x_(i)`
    (the V-statistic over all `n^2` pairs, including `i = j`).
    """
    x = np.sort(np.asarray(samples, dtype=np.float64))
    n = x.shape[0]
    term_y = float(np.mean(np.abs(x - y)))
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    term_xx = float(2.0 * np.sum(weights * x) / n**2)
    return term_y - 0.5 * term_xx


@metric("log_score", {"forecast"})
def log_score(**ctx: object) -> float | dict[str, float]:
    """Log of a Gaussian-KDE density estimate of the forecast at the realized `Y_h` (higher is better).

    The density is estimated from the Monte-Carlo samples with Silverman's bandwidth (`kde_log_density`);
    `log_score_mixture` is the exact-component alternative for `h = 1`.
    """
    cum, y_h, _ = _forecast_ctx(ctx)
    return _series_result(
        "log_score", np.asarray([kde_log_density(cum[:, m], y_h[m]) for m in range(cum.shape[1])])
    )


@metric("crps", {"forecast"})
def crps(**ctx: object) -> float | dict[str, float]:
    """Sample-based continuous ranked probability score of `Y_h` (lower is better)."""
    cum, y_h, _ = _forecast_ctx(ctx)
    return _series_result(
        "crps", np.asarray([crps_from_samples(cum[:, m], y_h[m]) for m in range(cum.shape[1])])
    )


def one_step_mixture_weights(
    log_alpha_t: Array,
    theta: Params,
    space: StateSpace,
    dt_next: float,
    z_pi_next: Array | None = None,
    z_d_next: Array | None = None,
) -> Array:
    """Exact one-step regime weights `(softmax(alpha_T) A_{T+1}) C`, shape `(N,)`.

    `log_alpha_t` is the last row of the filtered `log_alpha`; `A_{T+1}` is built by the same per-step
    builder the engines use (`space.transition_fn`) from the covariate rows and `dt` of the first
    forecast step (absent covariates: pass `None`).
    """
    z_pi = jnp.zeros((0,)) if z_pi_next is None else z_pi_next
    z_d = jnp.zeros((0,)) if z_d_next is None else z_d_next
    a_t, _ = space.transition_fn(theta, z_pi, z_d, jnp.asarray(float(dt_next)))
    weights: Array = jax.nn.softmax(log_alpha_t) @ a_t @ space.index.collapse
    return weights


@metric("log_score_mixture", {"forecast", "fit"})
def log_score_mixture(**ctx: object) -> float:
    """`h = 1` log score from the predictive mixture `sum_i w_i f_i(y)` with exact regime densities.

    The components are the emission family's own `log_prob` at `ctx['dt_next']` (and the optional
    `ctx['z_emission_next']`), summed over the series, at the realized first-step return. The weights
    are `ctx['mixture_weights']` when given -- the exact `one_step_mixture_weights` -- and otherwise the
    forecast's one-step regime probabilities `regime_probs[0]` (a Monte-Carlo estimate of the same
    quantity, so a regime with tiny probability can be missing). Only defined for `horizon == 1`.
    """
    forecast, fit, realized = ctx.get("forecast"), ctx.get("fit"), ctx.get("realized")
    dt_next, horizon = ctx.get("dt_next"), ctx.get("horizon", 1)
    if (
        not isinstance(forecast, ForecastResult)
        or not isinstance(fit, FitResult)
        or realized is None
        or not isinstance(dt_next, int | float | np.floating)
    ):
        raise ConfigError(
            "log_score_mixture needs ctx['forecast'], ['fit'], ['realized'] and ['dt_next']",
            path="evaluation.metrics",
        )
    if horizon != 1:
        raise ConfigError(
            f"log_score_mixture is only defined for horizon 1, got {horizon}", path="evaluation.metrics"
        )
    y1 = jnp.atleast_2d(jnp.asarray(realized, dtype=jnp.float64))[:1]  # (1, M)
    z = ctx.get("z_emission_next")
    z_arr = None if z is None else jnp.atleast_2d(jnp.asarray(z, dtype=jnp.float64))
    log_f = fit.spec.emission.log_prob(y1, jnp.full((1,), float(dt_next)), fit.params, z_arr)[0]  # (N,)
    given = ctx.get("mixture_weights")
    weights = jnp.asarray(forecast.regime_probs)[0] if given is None else jnp.asarray(given)
    safe_w = jnp.where(weights > 0, weights, 1.0)
    log_w = jnp.where(weights > 0, jnp.log(safe_w), -jnp.inf)
    value = float(jax.scipy.special.logsumexp(log_w + log_f))
    if not np.isfinite(value):
        raise NumericalError("non-finite log_score_mixture", path="log_score_mixture")
    return value


@metric("pit_coverage", {"forecast"})
def pit_coverage(**ctx: object) -> dict[str, float]:
    """`pit` (= share of samples `<= Y_h`) and hit indicators of the central intervals.

    For every configured quantile pair `(q, 1 - q)` with `q < 0.5` present in
    `forecast.quantile_levels`, `coverage_<100 (1 - 2q)>` is `1.0` when `Y_h` lies inside the interval of
    the forecast samples, else `0.0`. A runner averages the indicators over origins (the interval hit
    rate) and pools the `pit` values for a KS statistic against uniformity.
    """
    cum, y_h, _ = _forecast_ctx(ctx)
    levels = tuple(float(q) for q in forecast_of(ctx).quantile_levels)
    out: dict[str, float] = {}
    for m in range(cum.shape[1]):
        suffix = "" if cum.shape[1] == 1 else f"_{m}"
        out[f"pit{suffix}"] = float(np.mean(cum[:, m] <= y_h[m]))
        for lo in levels:
            hi = 1.0 - lo
            if lo < 0.5 and any(abs(hi - q) < 1e-12 for q in levels):
                low, high = np.quantile(cum[:, m], [lo, hi])
                out[f"coverage_{round(100 * (1 - 2 * lo))}{suffix}"] = float(low <= y_h[m] <= high)
    return out


@metric("rmse", {"forecast"}, aggregate="rms")
def rmse(**ctx: object) -> float | dict[str, float]:
    """`|mean[h-1] - Y_h|` for one origin; aggregate with `aggregate == "rms"` (root mean square)."""
    cum, y_h, h = _forecast_ctx(ctx)
    mean = np.asarray(forecast_of(ctx).mean, dtype=np.float64)[h - 1]
    del cum
    return _series_result("rmse", np.abs(mean - y_h))


@metric("mae", {"forecast"})
def mae(**ctx: object) -> float | dict[str, float]:
    """`|mean[h-1] - Y_h|` for one origin (aggregated by the mean over origins)."""
    cum, y_h, h = _forecast_ctx(ctx)
    mean = np.asarray(forecast_of(ctx).mean, dtype=np.float64)[h - 1]
    del cum
    return _series_result("mae", np.abs(mean - y_h))
