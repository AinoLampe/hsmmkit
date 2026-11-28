"""Parametric bootstrap: standard errors and nested-model tests (E14-T04; Part II §14.12).

`bootstrap_se` simulates `R` datasets from the fitted parameters `theta_hat` with the lengths, time steps and
covariate paths of the data, refits the model on each and reports the sample standard deviation and the
2.5 / 97.5 percentiles of the refitted (constrained) parameters. Refits that fail (an error, a non-finite
objective, no convergence when `require_converged`) are counted, listed and excluded. `bootstrap_lrt` tests a
nested pair: the observed `D = 2 (l_1 - l_0)` against `D_r` of datasets simulated under the null model,
with the p-value `(1 + #{D_r >= D}) / (R + 1)`. The chi-square approximation of `D` fails when the models
differ in the number of regimes (Hansen 1992; the extra regimes' parameters are unidentified under the
null), which is the main use.

* Replicate `r` draws its data with `key_for(seed, PURPOSE_BOOT, r)` and fits with
  `derive_seed(seed, PURPOSE_BOOT, r)`: results are deterministic per seed and independent of the scheduling
  (`workers > 1` runs the replicates in a `spawn` process pool).
* `refit_from="mle"` starts the refit at `theta_hat` only (one start); `"init"` uses the configured init
  strategies (`estimation.restarts` starts) as a fit of real data does.
* Refitted draws are relabelled towards `theta_hat` (the assignment of `estimation.relabel`), so a refit that
  lands in a permuted labelling does not inflate the spread.
* Supported: models without own-lag emission covariates (the lags of simulated data would have to be rebuilt),
  fitted by `map`, `em` or `gem` (estimators with a warm start).
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from hsmmkit.config.schema import EstimationSection
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.estimation import fit as run_fit
from hsmmkit.estimation.relabel import location_features, relabel_permutation
from hsmmkit.model.lattice import permute_states
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.selection.replicate import simulate_like as _simulate_like
from hsmmkit.statespace.builders import build_expanded, build_explicit
from hsmmkit.utils.errors import ConfigError, HsmmkitError
from hsmmkit.utils.rng import PURPOSE_BOOT, derive_seed, key_for

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec

__all__ = [
    "BootstrapResult",
    "LRTResult",
    "bootstrap_lrt",
    "bootstrap_se",
    "render_bootstrap_section",
    "replace_bootstrap_section",
]

_SUPPORTED = ("map", "em", "gem")


@dataclass(frozen=True)
class BootstrapResult:
    """`bootstrap_se` output.

    Attributes:
        table: one row per free parameter element: `parameter`, `index`, `estimate`, `se`, `lower`, `upper`
            and, when available, `hessian_se` and `ratio` (`se / hessian_se`).
        estimates: refitted constrained parameters `{name: (R_ok, *shape)}` (relabelled).
        n_requested: `R`.
        n_failed: refits excluded.
        failures: `(replicate, reason)` of every excluded refit.
        compare_hessian: the Hessian-SE comparison (`parameter`, `index`, `bootstrap_se`, `hessian_se`,
            `ratio`), `None` when the Hessian is not available (not positive definite, unsupported).
    """

    table: pd.DataFrame
    estimates: dict[str, np.ndarray]
    n_requested: int
    n_failed: int
    failures: tuple[tuple[int, str], ...]
    compare_hessian: pd.DataFrame | None = None

    @property
    def se(self) -> dict[str, np.ndarray]:
        """Bootstrap standard errors per parameter name (sample SD over the accepted refits)."""
        return {k: np.std(v, axis=0, ddof=1) for k, v in self.estimates.items()}


@dataclass(frozen=True)
class LRTResult:
    """`bootstrap_lrt` output: the observed statistic, the null replicates and the bootstrap p-value."""

    statistic: float
    null_statistics: np.ndarray
    p_value: float
    n_requested: int
    n_failed: int
    failures: tuple[tuple[int, str], ...] = field(default=())


def _space(spec: ModelSpec) -> Any:  # noqa: ANN401
    return build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)


def _check_model(fit: FitResult, *, name: str = "fit") -> None:
    if fit.estimator not in _SUPPORTED:
        raise ConfigError(
            f"{name}: the bootstrap refits with {', '.join(_SUPPORTED)}; the fit used {fit.estimator!r}",
            path="estimation.method",
        )
    if int(getattr(fit.spec.emission, "requires_own_lags", 0)):
        raise ConfigError(
            f"{name}: autoregressive emissions are not supported (the own lags of simulated data)",
            path="model.emission",
        )


def _relabel(theta: Params, pivot: Params, spec: ModelSpec) -> Params:
    """`theta` with its regimes permuted to the labelling closest to `pivot`."""
    n = spec.n_states
    f_pivot = location_features({k: np.asarray(v) for k, v in pivot.items()}, n)
    spread = np.std(f_pivot, axis=0)
    scale = np.where(spread > 1e-12, spread, np.maximum(np.abs(f_pivot).mean(axis=0), 1.0))
    perm = relabel_permutation(
        location_features({k: np.asarray(v) for k, v in theta.items()}, n), f_pivot, scale
    )
    if np.array_equal(perm, np.arange(n)):
        return theta
    return dict(permute_states(theta, spec, perm))


def _est_cfg(estimation: EstimationSection | None, fit: FitResult) -> EstimationSection:
    if estimation is not None:
        return estimation
    return EstimationSection.model_validate({"method": fit.estimator, "restarts": 1})


def _refit(
    spec: ModelSpec,
    data: PreparedDataset | SequenceBatch,
    est: EstimationSection,
    seed: int,
    warm: Any,  # noqa: ANN401
    refit_from: str,
) -> FitResult:
    """One refit: from `warm` only (`mle`; `restarts = 0` leaves just the warm start) or the inits."""
    if refit_from == "mle":
        single = est.model_copy(update={"restarts": 0})
        if single.em is not None and single.em.warm_start_from != "none":  # that would add a MAP start
            single = single.model_copy(
                update={"em": single.em.model_copy(update={"warm_start_from": "none"})}
            )
        return run_fit(spec, data, single, seed, warm)
    return run_fit(spec, data, est, seed)


def _replicate_se(
    r: int,
    fit: FitResult,
    data: PreparedDataset | SequenceBatch,
    est: EstimationSection,
    seed: int,
    refit_from: str,
    require_converged: bool,
) -> tuple[Params | None, str]:
    """`(relabelled theta_r, "")` or `(None, reason)` of replicate `r`."""
    try:
        sim = _simulate_like(fit.params, fit.spec, data, key_for(seed, PURPOSE_BOOT, r))
        res = _refit(fit.spec, sim, est, derive_seed(seed, PURPOSE_BOOT, r), fit.uparams, refit_from)
    except (HsmmkitError, NotImplementedError, FloatingPointError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not np.isfinite(res.loglik):
        return None, "non-finite log-likelihood"
    if require_converged and not res.converged:
        return None, "not converged"
    theta_r = {k: np.asarray(v) for k, v in _relabel(res.params, fit.params, fit.spec).items()}
    return theta_r, ""  # type: ignore[return-value]


def _pool_map(fn: Any, jobs: list[tuple[Any, ...]], workers: int) -> list[Any]:  # noqa: ANN401
    if workers <= 1:
        return [fn(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(fn, *job) for job in jobs]
        return [f.result() for f in futures]


def _free_mask(fit: FitResult) -> dict[str, np.ndarray]:
    """`{name: bool array}`: which elements of each parameter move with the free coordinates `u`."""
    import jax

    manager = ParamManager(fit.spec.param_specs)
    jac = jax.jacfwd(manager.unflatten)(fit.uparams)
    return {
        name: np.any(np.abs(np.asarray(j)).reshape(-1, manager.n_free) > 0.0, axis=1).reshape(j.shape[:-1])
        for name, j in jac.items()
    }


def _hessian_se(fit: FitResult, data: PreparedDataset | SequenceBatch) -> dict[str, np.ndarray] | None:
    """Hessian standard errors of `fit` on `data`, `None` when unavailable."""
    from hsmmkit.engine.batch import BatchedEngine
    from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
    from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
    from hsmmkit.estimation.objective import make_objective
    from hsmmkit.estimation.report import std_errors

    if fit.std_errors is not None:
        return {k: np.asarray(v) for k, v in fit.std_errors.items()}
    try:
        manager = ParamManager(fit.spec.param_specs)
        engine = ExplicitDurationEngine() if fit.spec.engine == "explicit" else ExpandedHMMEngine()
        space = _space(fit.spec)
        if isinstance(data, SequenceBatch):
            objective = make_objective(
                fit.spec, space, BatchedEngine(engine), manager, data.padded(), use_priors=False
            )
        else:
            objective = make_objective(fit.spec, space, engine, manager, data, use_priors=False)
        se = std_errors(objective, fit.uparams, manager)
    except (NotImplementedError, HsmmkitError, ValueError, TypeError):
        return None
    return None if se is None else {k: np.asarray(v) for k, v in se.items()}


def bootstrap_se(
    fit: FitResult,
    data: PreparedDataset | SequenceBatch,
    R: int,  # noqa: N803
    seed: int,
    *,
    refit_from: Literal["mle", "init"] = "mle",
    estimation: EstimationSection | None = None,
    require_converged: bool = False,
    workers: int = 1,
    hessian: bool = True,
) -> BootstrapResult:
    """Parametric-bootstrap standard errors and percentile intervals of the free parameters of `fit`.

    Args:
        fit: the fitted model (`map`, `em` or `gem`).
        data: the data it was fitted on (lengths, `Delta_t` and covariates are reused).
        R: number of bootstrap replicates (>= 2).
        seed: replicate `r` uses `key_for(seed, PURPOSE_BOOT, r)` (data), `derive_seed(...)` (fit).
        refit_from: `"mle"` (one refit from `theta_hat`) or `"init"` (the configured init methods).
        estimation: the refit configuration (default: the fit's estimator, one start).
        require_converged: also exclude refits that stopped at `max_iter`.
        workers: replicates in a `spawn` process pool when `> 1`.
        hessian: add the Hessian-SE comparison when the Hessian is available.

    Raises:
        ConfigError: unsupported estimator or emission, `R < 2`, or fewer than two accepted refits.
    """
    _check_model(fit)
    if R < 2:
        raise ConfigError("R must be >= 2", path="R")
    est = _est_cfg(estimation, fit)
    jobs = [(r, fit, data, est, seed, refit_from, require_converged) for r in range(R)]
    outcomes = _pool_map(_replicate_se, jobs, workers)
    ok = [(r, th) for r, (th, _) in enumerate(outcomes) if th is not None]
    failures = tuple((r, why) for r, (th, why) in enumerate(outcomes) if th is None)
    if len(ok) < 2:
        raise ConfigError(
            f"only {len(ok)} of {R} refits succeeded ({failures[0][1] if failures else ''})", path="R"
        )
    free = _free_mask(fit)
    names = [n for n in fit.params if bool(free[n].any())]
    estimates = {n: np.stack([th[n] for _, th in ok]) for n in names}
    hess = _hessian_se(fit, data) if hessian else None
    rows, cmp_rows = [], []
    for n in names:
        draws = estimates[n].reshape(len(ok), -1)
        hat = np.asarray(fit.params[n]).reshape(-1)
        sd = draws.std(axis=0, ddof=1)
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        h = None if hess is None or n not in hess else np.asarray(hess[n]).reshape(-1)
        for i in np.flatnonzero(free[n].reshape(-1)):
            idx = np.unravel_index(i, np.shape(fit.params[n]))
            row: dict[str, Any] = {
                "parameter": n,
                "index": ",".join(str(int(j)) for j in idx),
                "estimate": float(hat[i]),
                "se": float(sd[i]),
                "lower": float(lo[i]),
                "upper": float(hi[i]),
                "hessian_se": float("nan") if h is None else float(h[i]),
            }
            row["ratio"] = float(sd[i] / h[i]) if h is not None and h[i] > 0 else float("nan")
            rows.append(row)
            if h is not None:
                cmp_rows.append({k: row[k] for k in ("parameter", "index", "se", "hessian_se", "ratio")})
    compare = pd.DataFrame(cmp_rows).rename(columns={"se": "bootstrap_se"}) if cmp_rows else None
    return BootstrapResult(pd.DataFrame(rows), estimates, R, len(failures), failures, compare)


def _replicate_lrt(
    r: int,
    fit0: FitResult,
    fit1: FitResult,
    data: PreparedDataset | SequenceBatch,
    est: EstimationSection,
    seed: int,
) -> tuple[float | None, str]:
    """`(D_r, "")` or `(None, reason)`: both models fitted to one dataset simulated under the null."""
    try:
        sim = _simulate_like(fit0.params, fit0.spec, data, key_for(seed, PURPOSE_BOOT, r))
        s = derive_seed(seed, PURPOSE_BOOT, r)
        a = _refit(fit0.spec, sim, est, s, fit0.uparams, "mle")
        b = run_fit(fit1.spec, sim, est, s, fit1.uparams)
    except (HsmmkitError, NotImplementedError, FloatingPointError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not (np.isfinite(a.loglik) and np.isfinite(b.loglik)):
        return None, "non-finite log-likelihood"
    return max(2.0 * (b.loglik - a.loglik), 0.0), ""


def bootstrap_lrt(
    fit0: FitResult,
    fit1: FitResult,
    data: PreparedDataset | SequenceBatch,
    R: int,  # noqa: N803
    seed: int,
    *,
    estimation: EstimationSection | None = None,
    workers: int = 1,
) -> LRTResult:
    """Parametric-bootstrap likelihood-ratio test of the null model `fit0` against `fit1` (`fit0 < fit1`).

    The null model is refitted from its optimum on each simulated dataset; the alternative is refitted from
    its own optimum **and** the configured init methods (`estimation.restarts` starts, default 1 plus the
    warm start), so that `D_r >= 0` is not lost to a poor local optimum. `D_r` is clipped at 0. The two
    models are not refitted with equal effort (the alternative has more starts), which makes the p-value
    conservative; the observed `D` uses the log-likelihoods of the fits passed in.

    Raises:
        ConfigError: unsupported fits, `R < 2` or fewer than two accepted replicates.
    """
    _check_model(fit0, name="fit0")
    _check_model(fit1, name="fit1")
    if R < 2:
        raise ConfigError("R must be >= 2", path="R")
    est = _est_cfg(estimation, fit1)
    jobs = [(r, fit0, fit1, data, est, seed) for r in range(R)]
    outcomes = _pool_map(_replicate_lrt, jobs, workers)
    ok = np.array([d for d, _ in outcomes if d is not None], dtype=np.float64)
    failures = tuple((r, why) for r, (d, why) in enumerate(outcomes) if d is None)
    if ok.shape[0] < 2:
        raise ConfigError(f"only {ok.shape[0]} of {R} replicates succeeded", path="R")
    observed = max(2.0 * (fit1.loglik - fit0.loglik), 0.0)
    p = (1.0 + float(np.sum(ok >= observed))) / (ok.shape[0] + 1.0)
    return LRTResult(observed, ok, p, R, len(failures), failures)


def render_bootstrap_section(result: BootstrapResult) -> str:
    """The `## Bootstrap` section of `report.md`: the SE table and the failure count."""
    t = result.table
    lines = [
        "## Bootstrap",
        "",
        f"Parametric bootstrap, {result.n_requested - result.n_failed} of {result.n_requested} refits "
        "accepted"
        f" ({result.n_failed} failed).",
        "",
        "| parameter | index | estimate | bootstrap SE | 2.5 % | 97.5 % | Hessian SE |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['parameter']} | {r['index']} | {r['estimate']:.6g} | {r['se']:.4g} | {r['lower']:.6g} "
            f"| {r['upper']:.6g} | {_fmt(r['hessian_se'])} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(v: float) -> str:
    return "-" if not np.isfinite(v) else format(v, ".4g")


def replace_bootstrap_section(report: str, section: str) -> str:
    """`report` with its `## Bootstrap` section replaced by `section` (appended when absent)."""
    heading = "## Bootstrap"
    start = report.find(heading)
    if start < 0:
        return (report.rstrip("\n") + "\n\n" if report.strip() else "") + section
    end = report.find("\n## ", start + len(heading))
    tail = "" if end < 0 else report[end + 1 :]
    return report[:start] + section + ("\n" + tail if tail else "")
