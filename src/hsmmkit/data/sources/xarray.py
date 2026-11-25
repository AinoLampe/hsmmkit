"""The `xarray` source: an `xarray.Dataset` (`obj`) or a `.nc` / `.zarr` store (`path`)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel

from hsmmkit.data.sources.base import Loaded, TabularOpts, frame_to_loaded, sha256_file
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.utils.errors import ConfigError, DataError
from hsmmkit.utils.optional import optional_import


class XarraySource:
    """Observations from data variables of a Dataset.

    `observation_columns` are data variables, `time_column` a coordinate (or `None`), `sequence_column` a
    coordinate or variable, `covariate_columns` further variables. The Dataset is flattened to a frame
    (`to_dataframe().reset_index()`) and goes through the tabular logic.
    """

    name = "xarray"
    options_model: type[BaseModel] = TabularOpts

    def load(self, opts: BaseModel, *, obj: Any | None = None) -> Loaded:  # noqa: ANN401
        """Load from `obj` (a Dataset) or `opts.path` (`.nc` via `open_dataset`, `.zarr` via `open_zarr`)."""
        assert isinstance(opts, TabularOpts)
        xr = optional_import("xarray", "xarray")
        digest: str | None = None
        if obj is not None:
            if not isinstance(obj, xr.Dataset):
                raise ConfigError(
                    "the `xarray` source takes an xarray.Dataset as `obj`", path="data.source.kind"
                )
            dataset = obj
        else:
            if opts.path is None:
                raise ConfigError("the `xarray` source needs a path or an object", path="data.source.path")
            path = opts.path.expanduser()
            if not path.exists():
                raise DataError(f"file not found: {path}", path="data.source.path")
            try:
                dataset = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
            except (OSError, ValueError, ImportError) as exc:
                raise DataError(f"cannot read {path}: {exc}", path="data.source.path") from exc
            if path.is_file():
                digest = sha256_file(path)
        missing = [
            c for c in (*opts.observation_columns, *opts.covariate_columns) if c not in dataset.variables
        ]
        if missing:
            raise DataError(
                f"variable {missing[0]!r} not found; available: {', '.join(map(str, dataset.variables))}",
                path="data.source.observation_columns",
            )
        frame: pd.DataFrame = dataset.to_dataframe().reset_index()
        return frame_to_loaded(frame, opts, source=self.name, snapshot_id=digest)


REGISTRY.register("source", XarraySource.name, XarraySource(), provenance=provenance_of(__name__))
