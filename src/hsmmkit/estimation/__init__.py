"""Estimators (MAP, EM), initialization and fit reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hsmmkit.estimation import em as _em  # noqa: F401  (registers the `em` estimator)
from hsmmkit.estimation import gem as _gem  # noqa: F401  (registers the `gem` estimator)
from hsmmkit.estimation.map import fit as fit_map
from hsmmkit.plugins import REGISTRY

if TYPE_CHECKING:
    from hsmmkit.config.schema import EstimationSection
    from hsmmkit.data.batch import SequenceBatch
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.utils.typing import Array


def fit(
    spec: ModelSpec,
    data: PreparedDataset | SequenceBatch,
    cfg: EstimationSection,
    seed: int,
    warm_start: Array | None = None,
) -> FitResult:
    """Fit with the estimator `cfg.method` names (`map` by default, `em`): `fit(spec, data, cfg, seed)`."""
    estimator = REGISTRY.get("estimator", cfg.method)
    result: FitResult = estimator.fit(spec, data, cfg, seed, warm_start)
    return result


__all__ = ["fit", "fit_map"]
