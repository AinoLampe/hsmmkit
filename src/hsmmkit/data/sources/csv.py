"""The `csv` source: a CSV file read with pandas."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel

from regimekit.data.sources.base import Loaded, TabularOpts, frame_to_loaded, sha256_file
from regimekit.plugins import REGISTRY, provenance_of
from regimekit.utils.errors import ConfigError, DataError


class CsvSource:
    """Observations and covariates from a CSV file (`path`)."""

    name = "csv"
    options_model: type[BaseModel] = TabularOpts

    def load(self, opts: BaseModel, *, obj: Any | None = None) -> Loaded:  # noqa: ANN401
        """Read `opts.path` and delegate to the tabular logic; the snapshot id hashes the file bytes."""
        assert isinstance(opts, TabularOpts)
        if opts.path is None:
            raise ConfigError("the `csv` source needs a path", path="data.source.path")
        path = opts.path.expanduser()
        if not path.is_file():
            raise DataError(f"file not found: {path}", path="data.source.path")
        try:
            frame = pd.read_csv(path)
        except (ValueError, pd.errors.ParserError) as exc:
            raise DataError(f"cannot read {path}: {exc}", path="data.source.path") from exc
        return frame_to_loaded(frame, opts, source=self.name, snapshot_id=sha256_file(path))


REGISTRY.register("source", CsvSource.name, CsvSource(), provenance=provenance_of(__name__))
