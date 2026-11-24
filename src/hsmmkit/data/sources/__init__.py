"""Domain-neutral data sources (E10-T04a): `numpy`, `pandas`, `csv`, `parquet`.

Importing the package registers the sources under the kind `source`. `load_data` is the config-driven
entry point: it resolves `data.source`, loads, and checks the result against `data.clock`.
"""

from __future__ import annotations

from typing import Any

from regimekit.config.schema import DataSection
from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.data.sources import csv as csv  # registers `csv`
from regimekit.data.sources import numpy as numpy  # registers `numpy`
from regimekit.data.sources import pandas as pandas  # registers `pandas`
from regimekit.data.sources import parquet as parquet  # registers `parquet`
from regimekit.data.sources.base import Source, TabularOpts, get_source, validate_options
from regimekit.utils.clock import get_clock
from regimekit.utils.errors import ConfigError

__all__ = ["Source", "TabularOpts", "get_source", "load_data", "validate_options"]


def load_data(data: DataSection, *, obj: Any | None = None) -> tuple[ObservationSet, list[CovariateSeries]]:  # noqa: ANN401
    """Load the observations and the declared covariates of a `data:` section.

    Covariates without their own `source` are taken from the primary source's covariate columns
    (`column`, else `name`); a covariate with its own `source` takes the first observation series of that
    source.

    Args:
        data: the `data:` config section.
        obj: the in-memory object of the primary source (a `DataFrame`, a dict of arrays), if any.

    Returns:
        The observations and one `CovariateSeries` per `data.covariates[i]`, in declared order.

    Raises:
        ConfigError: unknown source, invalid options, time-less data under a clock that needs times
            (`path="data.clock"`), an unknown covariate column.
        DataError: the data does not satisfy the source contract.
    """
    kind = data.source.kind
    source = get_source(kind)
    opts = validate_options(kind, data.source.options(), "data.source")
    assert opts is not None
    obs, available = source.load(opts, obj=obj)
    if obs.times is None and get_clock(data.clock).unit != "step":
        raise ConfigError(
            f"data without a time column needs a step clock (data.clock: steps), not '{data.clock}'",
            path="data.clock",
        )
    by_name = {c.name: c for c in available}
    covs: list[CovariateSeries] = []
    for i, cov in enumerate(data.covariates):
        if cov.source is not None:
            other = get_source(cov.source.kind)
            other_opts = validate_options(
                cov.source.kind, cov.source.options(), f"data.covariates[{i}].source"
            )
            assert other_opts is not None
            other_obs, _ = other.load(other_opts)
            covs.append(
                CovariateSeries(
                    name=cov.name,
                    times=other_obs.times,
                    values=other_obs.values[:, 0].astype("float64"),
                    source=other_obs.source,
                    series_id=other_obs.series_names[0],
                    snapshot_id=other_obs.snapshot_id,
                    frequency="grid",
                )
            )
            continue
        column = cov.column or cov.name
        if column not in by_name:
            raise ConfigError(
                f"covariate column '{column}' not found; available: {', '.join(by_name) or '<none>'}",
                path=f"data.covariates[{i}].column",
            )
        covs.append(by_name[column])
    return obs, covs
