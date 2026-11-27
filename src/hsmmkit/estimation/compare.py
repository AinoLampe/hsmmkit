"""Compare estimators on the same model, data and seed (E13-T07: `compare CONFIG --estimators ...`)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.estimation import fit as dispatch_fit
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.plugins import REGISTRY
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.optional import optional_import

if TYPE_CHECKING:
    from hsmmkit.config.schema import EstimationSection
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.model.spec import ModelSpec

__all__ = ["COLUMNS", "compare_estimators", "render_estimator_table"]

COLUMNS = (
    "estimator",
    "loglik",
    "objective",
    "elbo",
    "n_params",
    "bic",
    "icl",
    "wall_s",
    "n_iter",
    "converged",
    "max_abs_dtheta_vs_map",
)


def _max_abs_diff(a: FitResult, b: FitResult) -> float:
    """`max |theta_a - theta_b|` over the parameters of both fits (constrained scale)."""
    return float(
        max(
            (
                float(np.max(np.abs(np.asarray(a.params[k]) - np.asarray(b.params[k]))))
                for k in a.params
                if k in b.params and np.size(a.params[k])
            ),
            default=float("nan"),
        )
    )


def compare_estimators(
    spec: ModelSpec,
    data: PreparedDataset | SequenceBatch,
    cfg: EstimationSection,
    seed: int,
    estimators: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, FitResult]]:
    """Fit `spec` to `data` with every estimator of `estimators` and tabulate the fits.

    Columns (`COLUMNS`): `loglik` at the fitted parameters, `objective` (the quantity the estimator optimizes:
    `-loglik` for EM / GEM, `-log_joint` for MAP, `-ELBO` for VB, `-log_joint` at the posterior mean for
    MCMC),
    `elbo` (VB only), `n_params`, `bic` / `icl` (single sequences; BIC only for a batch), `wall_s`, `n_iter`,
    `converged` and the largest absolute parameter difference to the MAP fit (`NaN` without `map`).

    Returns:
        `(table, fits)`, the fits by estimator name.

    Raises:
        ConfigError: an unknown estimator name, no estimator, or an estimator that cannot fit the model.
    """
    names = list(dict.fromkeys(estimators))
    if not names:
        raise ConfigError("no estimator to compare", path="compare.estimators")
    known = {e.name for e in REGISTRY.entries("estimator")}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ConfigError(
            f"unknown estimator(s) {unknown}; registered: {sorted(known)}", path="compare.estimators"
        )
    if "gem" in names and spec.engine != "explicit":
        raise ConfigError("estimator 'gem' needs model.engine: explicit", path="compare.estimators")
    if "mcmc" in names:
        try:
            optional_import("arviz", extra="bayes")
            optional_import("numpyro", extra="bayes")
        except ImportError as exc:
            raise ConfigError(str(exc), path="compare.estimators") from exc
    n_free = ParamManager(spec.param_specs).n_free
    fits: dict[str, FitResult] = {}
    rows: list[dict[str, object]] = []
    for name in names:
        started = time.perf_counter()
        fit = dispatch_fit(spec, data, cfg.model_copy(update={"method": name}), seed)
        wall = time.perf_counter() - started
        fits[name] = fit
        if isinstance(data, SequenceBatch):
            n_obs = int(sum(int(np.asarray(s.mask).all(axis=-1).sum()) for s in data.sequences))
            bic, icl = -2.0 * fit.loglik + n_free * float(np.log(max(n_obs, 1))), float("nan")
        else:
            from hsmmkit.inference.decode import decode

            post = decode(fit, data)
            bic, icl = post.bic, post.icl
        rows.append(
            {
                "estimator": name,
                "loglik": fit.loglik,
                "objective": fit.objective,
                "elbo": fit.elbo if fit.elbo is not None else float("nan"),
                "n_params": n_free,
                "bic": bic,
                "icl": icl,
                "wall_s": wall,
                "n_iter": fit.n_iter,
                "converged": fit.converged,
                "max_abs_dtheta_vs_map": float("nan"),
            }
        )
    if "map" in fits:
        for row in rows:
            row["max_abs_dtheta_vs_map"] = _max_abs_diff(fits[str(row["estimator"])], fits["map"])
    return pd.DataFrame(rows, columns=list(COLUMNS)), fits


def render_estimator_table(table: pd.DataFrame) -> str:
    """`compare.md` of an estimator comparison."""
    lines = [
        "# Estimator comparison",
        "",
        "Same model, data and seed. `objective` is what each estimator optimizes (MAP: `-log_joint`; "
        "EM / GEM: `-loglik`, without the prior; VB: `-ELBO`; MCMC: `-log_joint` at the posterior mean), "
        "so it is comparable only between estimators of the same kind; "
        "`max_abs_dtheta_vs_map` is on the "
        "constrained scale.",
        "",
        "| " + " | ".join(table.columns) + " |",
        "|" + "---|" * len(table.columns),
    ]
    for _, row in table.iterrows():
        cells = []
        for col in table.columns:
            v = row[col]
            if isinstance(v, float | np.floating):
                cells.append("-" if not np.isfinite(v) else f"{v:.6g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
