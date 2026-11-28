"""Normal pseudo-residuals, Shapiro-Wilk test, QQ data (E14-T05; Zucchini, MacDonald & Langrock 2016).

One-step predictive distribution of series `m`:
`F_t(y) = sum_i P(regime_t = i | y_{1:t-1}) F_{i,m}(y | dt_t, z_t)` with the predictive regime
probabilities of
`evaluation.metrics.predictive.predictive_regime_probs`.

* continuous support (`real`): `u_t = F_t(y_t)`, `r_t = Phi^-1(u_t)`;
* discrete supports (`count`, `binary`, `categorical`): the mid-distribution
  `u_t = (F_t(y_t) + F_t(y_t - 1)) / 2` with `F_t(-1) = 0` (the category ids of a categorical model are an
  arbitrary order: reported, not meant for testing);
* a Gaussian with a `full` covariance (`M > 1`): the sequential Rosenblatt transform along the series order,
  `u_{t,m} = sum_i w_i^{(m)} Phi(z_{i,m})` with the regime weights updated by the earlier series of the
  same step (`w_i^{(m)} proportional to P_i prod_{k<m} phi(z_{i,k}) / L_{i,kk}`), `z_i = L_i^{-1} (y - mu_i)`.

Families without a usable `cdf` are skipped with a note. `shapiro_wilk` is an own implementation of Royston's
AS R94 algorithm (`n <= 5000`; longer samples use a seeded random subsample of 5000 values, so the p-value is
indicative only), validated against scipy in the dev tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from hsmmkit.evaluation.metrics.predictive import ljung_box, predictive_regime_probs
from hsmmkit.model.emission.base import Support
from hsmmkit.utils.errors import HsmmkitError

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.engine.base import InferenceEngine
    from hsmmkit.estimation.results import FitResult

__all__ = ["ResidualResult", "pseudo_residuals", "shapiro_wilk"]

_EPS = 1e-9
_SW_MAX = 5000
_DISCRETE: tuple[Support, ...] = ("count", "binary", "categorical")


def _poly(coefs: tuple[float, ...], x: float) -> float:
    """`coefs[0] + coefs[1] x + coefs[2] x^2 + ...`."""
    out = 0.0
    for c in reversed(coefs):
        out = out * x + c
    return out


def _norm_ppf(p: np.ndarray | float) -> np.ndarray:
    return np.asarray(jax.scipy.special.ndtri(jnp.asarray(p, dtype=jnp.float64)))


def _norm_sf(x: float) -> float:
    return float(0.5 * jax.scipy.special.erfc(x / np.sqrt(2.0)))


def shapiro_wilk(x: np.ndarray | list[float]) -> tuple[float, float]:
    """`(W, p)` of the Shapiro-Wilk normality test (Royston 1992, algorithm AS R94), `3 <= n <= 5000`.

    Samples with more than 5000 values use a seeded random subsample of 5000 values (the p-value is then
    indicative only). A constant sample gives
    `(nan, nan)`.

    Raises:
        ValueError: fewer than 3 finite values.
    """
    y = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = y[np.isfinite(y)]
    n = y.shape[0]
    if n < 3:
        raise ValueError("Shapiro-Wilk needs at least 3 values")
    if (
        n > _SW_MAX
    ):  # a seeded random subsample: an evenly spaced quantile grid would be too smooth for the null
        y = np.sort(np.random.default_rng(0).choice(y, size=_SW_MAX, replace=False))
        n = _SW_MAX
    if y[-1] - y[0] < 1e-12 * max(abs(y[-1]), 1.0):
        return float("nan"), float("nan")
    nn2 = n // 2
    if n == 3:
        a = np.zeros(1)
        a[0] = np.sqrt(0.5)
    else:
        m = _norm_ppf((np.arange(1, nn2 + 1) - 0.375) / (n + 0.25))  # lower-half Blom scores (negative)
        summ2 = 2.0 * float(np.sum(m**2))
        ssumm2 = np.sqrt(summ2)
        rsn = 1.0 / np.sqrt(n)
        a_n = -m[0] / ssumm2 + _poly((0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056), rsn)
        if n > 5:
            a_n1 = -m[1] / ssumm2 + _poly((0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633), rsn)
            fac = np.sqrt((summ2 - 2.0 * m[0] ** 2 - 2.0 * m[1] ** 2) / (1.0 - 2.0 * a_n**2 - 2.0 * a_n1**2))
            a = -m / fac
            a[0], a[1] = a_n, a_n1
        else:
            fac = np.sqrt((summ2 - 2.0 * m[0] ** 2) / (1.0 - 2.0 * a_n**2))
            a = -m / fac
            a[0] = a_n
    # W from the paired differences of the order statistics
    num = float(np.sum(a * (y[::-1][:nn2] - y[:nn2]))) ** 2
    den = float(np.sum((y - y.mean()) ** 2))
    w = min(num / den, 1.0)
    if n == 3:
        pi6, stqr = 1.90985931710274, 1.04719755119660
        p = max(pi6 * (np.arcsin(np.sqrt(w)) - stqr), 0.0)
        return w, float(p)
    if w >= 1.0:
        return 1.0, 1.0
    if n <= 11:
        gamma = _poly((-2.273, 0.459), n)
        arg = gamma - np.log1p(-w)
        if arg <= 0.0:
            return w, 0.0
        y_ = -np.log(arg)
        mu = _poly((0.544, -0.39978, 0.025054, -6.714e-4), n)
        sigma = np.exp(_poly((1.3822, -0.77857, 0.062767, -0.0020322), n))
    else:
        y_ = np.log1p(-w)
        xx = np.log(n)
        mu = _poly((-1.5861, -0.31082, -0.083751, 0.0038915), xx)
        sigma = np.exp(_poly((-0.4803, -0.082676, 0.0030302), xx))
    return w, _norm_sf((y_ - mu) / sigma)


@dataclass(frozen=True)
class ResidualResult:
    """Pseudo-residuals of a fitted model.

    Attributes:
        residuals: `(T, M)` normal pseudo-residuals, `NaN` at unobserved steps (and for skipped series).
        u: `(T, M)` uniform pseudo-residuals (PIT / mid-distribution values).
        table: one row per series: `series`, `n`, `shapiro_w`, `shapiro_p`, `lb_resid_{10,20}_p`,
            `lb_sq_{10,20}_p` (Ljung-Box of `r` and `r^2`).
        qq: `series`, `theoretical`, `sample` (sorted residuals against Blom normal scores).
        notes: why a series or the whole model was skipped.
        kind: `"continuous"`, `"mid-distribution"` or `"rosenblatt"`.
    """

    residuals: np.ndarray
    u: np.ndarray
    table: pd.DataFrame
    qq: pd.DataFrame
    notes: tuple[str, ...] = ()
    kind: str = "continuous"
    extra: dict[str, object] = field(default_factory=dict)


def _engine_for(fit: FitResult) -> InferenceEngine:
    from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
    from hsmmkit.engine.explicit_duration import ExplicitDurationEngine

    return ExplicitDurationEngine() if fit.spec.engine == "explicit" else ExpandedHMMEngine()


def _rosenblatt(
    fit: FitResult, data: PreparedDataset, pred: jnp.ndarray, z_e: jnp.ndarray | None
) -> np.ndarray:
    """`(T, M)` Rosenblatt `u` of a `full`-covariance Gaussian (regime weights updated along the series)."""
    emission = fit.spec.emission
    loc, chol, _ = emission._full_parts(data.dt, fit.params, z_e)  # type: ignore[attr-defined]
    y = jnp.where(data.mask, data.observations, 0.0)
    resid = y[:, None, :] - loc  # (T, N, M)
    z = jax.scipy.linalg.solve_triangular(chol, resid[..., None], lower=True)[..., 0]  # standardized
    diag = jnp.diagonal(chol, axis1=-2, axis2=-1)  # (T, N, M)
    log_phi = -0.5 * z**2 - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(diag)  # per-series conditional log-density
    prefix = jnp.cumsum(log_phi, axis=-1) - log_phi  # sum over k < m
    logw = jnp.log(jnp.maximum(pred, 1e-300))[:, :, None] + prefix  # (T, N, M)
    w = jax.nn.softmax(logw, axis=1)
    u = jnp.sum(w * jax.scipy.stats.norm.cdf(z), axis=1)
    return np.asarray(u)


def _discrete_u(
    emission: object,
    y: jnp.ndarray,
    fit: FitResult,
    data: PreparedDataset,
    pred: jnp.ndarray,
    z_e: jnp.ndarray | None,
) -> np.ndarray:
    cdf = emission.cdf  # type: ignore[attr-defined]
    upper = cdf(y, data.dt, fit.params, z_e)  # (T, N, M)
    lower_raw = cdf(jnp.maximum(y - 1.0, 0.0), data.dt, fit.params, z_e)
    lower = jnp.where((y <= 0.0)[:, None, :], 0.0, lower_raw)
    f_up = jnp.sum(pred[:, :, None] * upper, axis=1)
    f_lo = jnp.sum(pred[:, :, None] * lower, axis=1)
    return np.asarray(0.5 * (f_up + f_lo))


def pseudo_residuals(
    fit: FitResult, data: PreparedDataset, engine: InferenceEngine | None = None
) -> ResidualResult:
    """Normal pseudo-residuals of `fit` on `data`, their Shapiro-Wilk / Ljung-Box tests and QQ data.

    A `SequenceBatch` is not accepted here: call it per sequence.

    Raises:
        NumericalError: the engine's forward pass is not finite.
    """
    spec = fit.spec
    engine = engine or _engine_for(fit)
    from hsmmkit.statespace.builders import build_expanded, build_explicit

    space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
    posterior = engine.forward_backward(fit.params, spec, space, data)
    pred = predictive_regime_probs(posterior, fit.params, spec, space, data)  # (T, N)
    t_len, m_series = data.observations.shape
    emission = spec.emission
    z_e = None
    if data.cov_emission is not None and data.cov_emission.shape[-1] > 0:
        z_e = data.cov_emission
    y = jnp.where(data.mask, data.observations, 0.0)
    notes: list[str] = []
    u = np.full((t_len, m_series), np.nan)
    kind = "continuous"
    try:
        if getattr(emission, "joint", False) and m_series > 1 and hasattr(emission, "_full_parts"):
            u = _rosenblatt(fit, data, pred, z_e)
            kind = "rosenblatt"
        elif getattr(emission, "joint", False) and m_series > 1:
            notes.append(f"{type(emission).__name__}: joint emission without a Rosenblatt transform; skipped")
        elif emission.support in _DISCRETE:
            u = _discrete_u(emission, y, fit, data, pred, z_e)
            kind = "mid-distribution"
        else:
            cdf_t = emission.cdf(y, data.dt, fit.params, z_e)  # (T, N, M)
            u = np.asarray(jnp.sum(pred[:, :, None] * cdf_t, axis=1))
    except (NotImplementedError, AttributeError, TypeError, ValueError, HsmmkitError) as exc:
        notes.append(f"{type(emission).__name__}: no usable cdf ({type(exc).__name__}: {exc}); skipped")
        u = np.full((t_len, m_series), np.nan)
    observed = np.asarray(data.mask)
    u = np.where(observed, np.clip(u, 0.0, 1.0), np.nan)
    resid = np.where(np.isfinite(u), _norm_ppf(np.clip(np.nan_to_num(u, nan=0.5), _EPS, 1.0 - _EPS)), np.nan)
    rows, qq_rows = [], []
    for m in range(m_series):
        r = resid[:, m]
        r = r[np.isfinite(r)]
        row: dict[str, float | int] = {"series": m, "n": int(r.shape[0])}
        if r.shape[0] >= 3:
            w, p = shapiro_wilk(r)
            row.update(shapiro_w=w, shapiro_p=p)
            lb = ljung_box(r, lags=(10, 20))
            lb2 = ljung_box(r**2, lags=(10, 20))
            for lag in (10, 20):
                row[f"lb_resid_{lag}_p"] = lb[lag][1]
                row[f"lb_sq_{lag}_p"] = lb2[lag][1]
            srt = np.sort(r)
            theo = _norm_ppf((np.arange(1, srt.shape[0] + 1) - 0.375) / (srt.shape[0] + 0.25))
            qq_rows.append(pd.DataFrame({"series": m, "theoretical": theo, "sample": srt}))
        rows.append(row)
    table = pd.DataFrame(rows)
    for col in ("shapiro_w", "shapiro_p", "lb_resid_10_p", "lb_resid_20_p", "lb_sq_10_p", "lb_sq_20_p"):
        if col not in table:
            table[col] = np.nan
    qq = (
        pd.concat(qq_rows, ignore_index=True)
        if qq_rows
        else pd.DataFrame(columns=["series", "theoretical", "sample"])
    )
    return ResidualResult(resid, u, table, qq, tuple(notes), kind)
