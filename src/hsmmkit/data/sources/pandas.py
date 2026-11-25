"""The `pandas` source: a `DataFrame` passed through the Python API (`obj`)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel

from hsmmkit.data.sources.base import Loaded, TabularOpts, frame_to_loaded
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.utils.errors import ConfigError


class PandasSource:
    """Observations and covariates from an in-memory `pandas.DataFrame` (Python API only)."""

    name = "pandas"
    options_model: type[BaseModel] = TabularOpts

    def load(self, opts: BaseModel, *, obj: Any | None = None) -> Loaded:  # noqa: ANN401
        """Load from `obj`; a YAML `kind: pandas` without an object is a `ConfigError`."""
        if not isinstance(obj, pd.DataFrame):
            raise ConfigError(
                "the `pandas` source needs a DataFrame passed through the Python API; "
                "in YAML use `kind: csv` or `kind: parquet`",
                path="data.source.kind",
            )
        assert isinstance(opts, TabularOpts)
        return frame_to_loaded(obj, opts, source=self.name, snapshot_id=None)


REGISTRY.register("source", PandasSource.name, PandasSource(), provenance=provenance_of(__name__))
