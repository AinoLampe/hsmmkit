"""The `parquet` source: a parquet file read with pyarrow."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel

from regimekit.data.sources.base import Loaded, TabularOpts, frame_to_loaded, sha256_file
from regimekit.plugins import REGISTRY, provenance_of
from regimekit.utils.errors import ConfigError, DataError


class ParquetSource:
    """Observations and covariates from a parquet file (`path`)."""

    name = "parquet"
    options_model: type[BaseModel] = TabularOpts

    def load(self, opts: BaseModel, *, obj: Any | None = None) -> Loaded:  # noqa: ANN401
        """Read `opts.path` and delegate to the tabular logic; the snapshot id hashes the file bytes."""
        assert isinstance(opts, TabularOpts)
        if opts.path is None:
            raise ConfigError("the `parquet` source needs a path", path="data.source.path")
        path = opts.path.expanduser()
        if not path.is_file():
            raise DataError(f"file not found: {path}", path="data.source.path")
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, pa.ArrowException) as exc:
            raise DataError(f"cannot read {path}: {exc}", path="data.source.path") from exc
        return frame_to_loaded(frame, opts, source=self.name, snapshot_id=sha256_file(path))


REGISTRY.register("source", ParquetSource.name, ParquetSource(), provenance=provenance_of(__name__))
