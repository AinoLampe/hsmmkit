"""Posterior predictive checks (E14-T05; Part II §14.12).

`ppc` simulates `R` replicate datasets from the fitted model with the lengths, time steps and covariate paths
of the data (for an MCMC fit each replicate draws its own parameters from the posterior), evaluates registered
statistics `T(y)` on the observed and on every replicate, and reports the two-sided tail probability
`p = 2 min(P(T_rep >= T_obs), P(T_rep <= T_obs))` (clipped at 1) per statistic element.

Statistics are registry kind `ppc_statistic`: `fn(y, ctx) -> {element: value}` with `y (T, M)` the observed
rows and `ctx` holding `fit`, `data` (the dataset `y` came from) and `regimes()` (the Viterbi regime path of
that dataset under the *fitted* parameters, computed lazily). Defaults: `acf_abs` (autocorrelation of `|y|`
at lags 1-20), `sojourn_mean` (mean decoded segment length per regime) and `kurtosis` (excess kurtosis).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.selection.replicate import simulate_like
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.rng import PURPOSE_PPC, derive_seed, key_for

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.estimation.results import FitResult

__all__ = ["DEFAULT_STATISTICS", "PPCResult", "PPCStatistic", "ppc"]

PPCStatistic = Callable[[np.ndarray, dict[str, Any]], Mapping[str, float]]
DEFAULT_STATISTICS = ("acf_abs", "sojourn_mean", "kurtosis")
_ACF_LAGS = 20


def acf_abs(y: np.ndarray, ctx: dict[str, Any]) -> dict[str, float]:
    """Autocorrelation of `|y|` at lags 1-20, per series: `m{series}.lag{k}`."""
    del ctx
    out: dict[str, float] = {}
    for m in range(y.shape[1]):
        a = np.abs(y[:, m]) - np.abs(y[:, m]).mean()
        denom = float(np.sum(a**2))
        for k in range(1, _ACF_LAGS + 1):
            out[f"m{m}.lag{k}"] = (
                float(np.sum(a[k:] * a[:-k]) / denom) if denom > 0 and a.shape[0] > k else np.nan
            )
    return out


def kurtosis(y: np.ndarray, ctx: dict[str, Any]) -> dict[str, float]:
    """Excess kurtosis per series: `m{series}`."""
    del ctx
    out: dict[str, float] = {}
    for m in range(y.shape[1]):
        c = y[:, m] - y[:, m].mean()
        var = float(np.mean(c**2))
        out[f"m{m}"] = float(np.mean(c**4) / var**2 - 3.0) if var > 0 else np.nan
    return out


def sojourn_mean(y: np.ndarray, ctx: dict[str, Any]) -> dict[str, float]:
    """Mean length of the decoded segments of each regime (`NaN` when a regime is never decoded)."""
    del y
    path = np.asarray(ctx["regimes"]())
    n = int(ctx["fit"].spec.n_states)
    change = np.flatnonzero(np.diff(path) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [path.shape[0]]])
    out: dict[str, float] = {}
    for i in range(n):
        lengths = (ends - starts)[path[starts] == i]
        out[f"regime{i}"] = float(lengths.mean()) if lengths.size else np.nan
    return out


for _name, _fn in (("acf_abs", acf_abs), ("kurtosis", kurtosis), ("sojourn_mean", sojourn_mean)):
    REGISTRY.register("ppc_statistic", _name, _fn, provenance=provenance_of(__name__))


@dataclass(frozen=True)
class PPCResult:
    """`ppc` output.

    Attributes:
        table: `statistic`, `element`, `observed`, `rep_mean`, `rep_lower`, `rep_upper` (2.5 / 97.5 %), `p`.
        replicates: `{statistic: (R, n_elements)}` raw replicate values; `elements` names the columns.
        elements: `{statistic: tuple of element names}`.
    """

    table: pd.DataFrame
    replicates: dict[str, np.ndarray]
    elements: dict[str, tuple[str, ...]]


def _draw_params(fit: FitResult, rng: np.random.Generator) -> dict[str, Any]:
    """One posterior draw of an MCMC fit (the point estimate for any other fit)."""
    samples = fit.posterior_samples
    if samples is None:
        return dict(fit.params)
    post = samples["posterior"]
    out: dict[str, Any] = dict(fit.params)
    chain_draw: tuple[int, int] | None = None
    for name in fit.params:
        if name not in post:
            continue
        arr = np.asarray(post[name])
        if chain_draw is None:
            chain_draw = (int(rng.integers(arr.shape[0])), int(rng.integers(arr.shape[1])))
        out[name] = np.asarray(arr[chain_draw[0], chain_draw[1]])
    return out


def _regimes_of(fit: FitResult, data: PreparedDataset) -> np.ndarray:
    from hsmmkit.inference.decode import decode

    return np.asarray(decode(fit, data).viterbi)


def _evaluate(name: str, fit: FitResult, data: PreparedDataset) -> Mapping[str, float]:
    fn = REGISTRY.get("ppc_statistic", name)
    rows = np.asarray(data.mask).all(axis=-1)
    y = np.asarray(data.observations)[rows]
    cache: dict[str, np.ndarray] = {}

    def regimes() -> np.ndarray:
        if "path" not in cache:
            cache["path"] = _regimes_of(fit, data)
        return cache["path"]

    result: Mapping[str, float] = fn(y, {"fit": fit, "data": data, "regimes": regimes})
    return result


def ppc(
    fit: FitResult,
    data: PreparedDataset | SequenceBatch,
    R: int,  # noqa: N803
    seed: int,
    statistics: Sequence[str] = DEFAULT_STATISTICS,
) -> PPCResult:
    """Posterior predictive check of `fit` on `data` with `R` replicates.

    Args:
        fit: the fitted model (an MCMC fit draws its parameters from the posterior per replicate).
        data: the (single-sequence) data the model was fitted on.
        R: number of replicates (>= 2).
        seed: replicate `r` uses `key_for(seed, PURPOSE_PPC, r)`; deterministic per seed.
        statistics: registered `ppc_statistic` names.

    Raises:
        ConfigError: unknown statistic names, `R < 2`, a `SequenceBatch` or an own-lag emission.
    """
    if isinstance(data, SequenceBatch):
        raise ConfigError("ppc takes a single sequence; call it per sequence", path="data")
    if int(getattr(fit.spec.emission, "requires_own_lags", 0)):
        raise ConfigError("autoregressive emissions are not supported by ppc", path="model.emission")
    if R < 2:
        raise ConfigError("R must be >= 2", path="R")
    for name in statistics:
        REGISTRY.get("ppc_statistic", name)  # unknown name: the registry lists the registered ones
    rng = np.random.default_rng(derive_seed(seed, PURPOSE_PPC))
    observed = {name: dict(_evaluate(name, fit, data)) for name in statistics}
    elements = {name: tuple(obs) for name, obs in observed.items()}
    reps: dict[str, list[list[float]]] = {name: [] for name in statistics}
    for r in range(R):
        theta = _draw_params(fit, rng)
        sim = simulate_like(theta, fit.spec, data, key_for(seed, PURPOSE_PPC, r))
        sim = dataclasses.replace(sim, mask=data.mask)
        for name in statistics:
            values = _evaluate(name, fit, sim)
            reps[name].append([float(values.get(e, np.nan)) for e in elements[name]])
    rows = []
    replicates = {name: np.asarray(v, dtype=np.float64) for name, v in reps.items()}
    for name in statistics:
        for k, element in enumerate(elements[name]):
            obs = float(observed[name][element])
            vals = replicates[name][:, k]
            vals = vals[np.isfinite(vals)]
            if vals.size and np.isfinite(obs):
                p = min(1.0, 2.0 * min(float(np.mean(vals >= obs)), float(np.mean(vals <= obs))))
                lo, hi = np.percentile(vals, [2.5, 97.5])
                rows.append((name, element, obs, float(vals.mean()), float(lo), float(hi), p))
            else:
                rows.append((name, element, obs, np.nan, np.nan, np.nan, np.nan))
    table = pd.DataFrame(
        rows, columns=["statistic", "element", "observed", "rep_mean", "rep_lower", "rep_upper", "p"]
    )
    return PPCResult(table, replicates, elements)
