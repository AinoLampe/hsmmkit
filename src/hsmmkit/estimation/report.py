"""Post-fit diagnostics and Hessian standard errors (E5-T04).

Concept refs: §4.8 Standard errors, §4.10 Diagnostics, §5.4 (`FitResult.diagnostics`). `map.py`'s `fit`
does not call these directly (both take the already-built `spec`/`space`/`engine`/`objective` the fit
step already has); the caller (E5-T03's `fit`, or a future `Project` facade) wires them in when
`cfg.std_errors` is set, or whenever a post-fit diagnostic report is wanted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.stats
import numpy as np

from hsmmkit.engine.viterbi import segments
from hsmmkit.evaluation.metrics.predictive import (
    ljung_box,
    pit_histogram,
    pit_values,
    predictive_regime_probs,
)
from hsmmkit.model.lattice import has_identifiability_guard

if TYPE_CHECKING:
    import pandas as pd

    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.engine.base import InferenceEngine
    from hsmmkit.estimation.objective import Objective
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.model.params.manager import ParamManager
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace
    from hsmmkit.utils.typing import Array

OCCUPANCY_WARN_THRESHOLD = 0.02
DISPERSION_WARN_THRESHOLD = 0.01
_PIT_EPS = 1e-6

_SCALE_NAMES = ("emission.scale", "emission.sigma", "emission.s")


def _entry(value: object, warning: str | None = None) -> dict[str, object]:
    return {"value": value, "warning": warning}


def _regime_occupancy(regime_smoothed: Array) -> dict[str, object]:
    occupancy = np.asarray(jnp.mean(regime_smoothed, axis=0))
    warning = (
        "misspecified fixed parameters or too many states"
        if bool(occupancy.min() < OCCUPANCY_WARN_THRESHOLD)
        else None
    )
    return _entry(occupancy, warning)


def _posterior_mean_duration(
    theta: dict[str, Array], spec: ModelSpec, space: StateSpace, data: PreparedDataset, viterbi_path: Array
) -> tuple[dict[str, object], dict[str, object]]:
    n = spec.n_states
    regime_path = np.asarray(space.index.regime_of[np.asarray(viterbi_path)])
    total = np.zeros(n)
    count = np.zeros(n)
    for start, end, regime in segments(regime_path):
        total[regime] += end - start + 1
        count[regime] += 1
    empirical = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)
    analytic = np.asarray(spec.duration.mean_sojourn(theta, data.nominal_dt)) / data.nominal_dt
    ratio = np.divide(empirical, analytic, out=np.full(n, np.nan), where=analytic > 0)
    return _entry(empirical), _entry(ratio)


def _ordering_check(theta: dict[str, Array], spec: ModelSpec) -> tuple[dict[str, object], dict[str, object]]:
    scale_name = next((name for name in _SCALE_NAMES if name in theta), None)
    if scale_name is None:
        return _entry(True), _entry(False)
    sigma = np.asarray(theta[scale_name]).reshape(spec.n_states, -1)[:, 0]
    sorted_ok = bool(np.all(np.diff(sigma) >= 0))
    risk = (not sorted_ok) and (not has_identifiability_guard(spec))
    return _entry(sorted_ok), _entry(risk)


def _restart_dispersion(restarts: pd.DataFrame) -> dict[str, object]:
    objectives = restarts["objective"].to_numpy(dtype=np.float64)
    finite = np.sort(objectives[np.isfinite(objectives)])
    if finite.size < 2:
        return _entry(0.0)
    scale = max(abs(float(finite[0])), 1e-12)
    gap = abs(float(finite[1]) - float(finite[0])) / scale
    warning = "likely multimodality" if gap > DISPERSION_WARN_THRESHOLD else None
    return _entry(gap, warning)


def diagnose(
    fit: FitResult, spec: ModelSpec, space: StateSpace, engine: InferenceEngine, data: PreparedDataset
) -> dict[str, object]:
    """The concept's mandatory post-fit diagnostics, each `{"value": ..., "warning": str | None}`."""
    theta = fit.params
    posterior = engine.forward_backward(theta, spec, space, data)
    viterbi_path = engine.viterbi(theta, spec, space, data)

    mean_duration, mean_duration_ratio = _posterior_mean_duration(theta, spec, space, data, viterbi_path)
    ordering_ok, label_switching_risk = _ordering_check(theta, spec)

    pred_probs = predictive_regime_probs(posterior, theta, spec, space, data)
    pit = pit_values(pred_probs, spec, theta, data)
    hist, ks_stat, ks_p = pit_histogram(np.asarray(pit))
    residuals = jax.scipy.stats.norm.ppf(jnp.clip(pit, _PIT_EPS, 1.0 - _PIT_EPS))
    lb = ljung_box(np.asarray(residuals), lags=(10, 20))

    return {
        "regime_occupancy": _regime_occupancy(posterior.regime_smoothed()),
        "posterior_mean_duration": mean_duration,
        "posterior_mean_duration_ratio": mean_duration_ratio,
        "pit_hist": _entry(hist),
        "pit_ks_statistic": _entry(ks_stat),
        "pit_ks_pvalue": _entry(ks_p),
        "ljung_box": _entry(lb),
        "ordering_check": ordering_ok,
        "label_switching_risk": label_switching_risk,
        "restart_dispersion": _restart_dispersion(fit.restarts),
        "timings": _entry(
            {
                "wall_s_total": float(fit.restarts["wall_s"].sum()),
                "wall_s_per_restart": fit.restarts["wall_s"].tolist(),
                "n_restarts": len(fit.restarts),
            }
        ),
    }


def std_errors(objective: Objective, u_hat: Array, manager: ParamManager) -> dict[str, Array] | None:
    """Hessian standard errors per named `theta` parameter, or `None` if `H` is not positive definite.

    `H = hessian(objective.value)(u_hat)`; `Cov_u = inv(H)`; `Cov_theta = J Cov_u J^T` (delta method,
    `J = jacfwd(manager.unflatten)`). `objective.value` is already the *negative* log-posterior (or
    log-likelihood, `use_priors=False`), so `H` is the observed information directly -- no sign flip.
    """
    hessian = jax.hessian(objective.value)(u_hat)
    eigvals = jnp.linalg.eigvalsh(hessian)
    if not bool(jnp.all(jnp.isfinite(eigvals))) or float(jnp.min(eigvals)) <= 0:
        return None
    cov_u = jnp.linalg.inv(hessian)

    jac = jax.jacfwd(manager.unflatten)(u_hat)
    out: dict[str, Array] = {}
    for name, j in jac.items():
        j_flat = jnp.reshape(j, (-1, u_hat.shape[0]))
        cov_theta = j_flat @ cov_u @ j_flat.T
        se_flat = jnp.sqrt(jnp.clip(jnp.diagonal(cov_theta), 0.0))
        out[name] = jnp.reshape(se_flat, j.shape[:-1])
    return out
