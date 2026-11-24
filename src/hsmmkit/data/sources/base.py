"""The `Source` protocol, the tabular options and the shared loading logic (E10-T04a, FR-G04, FR-PL02).

A source turns a file or an in-memory object into an `ObservationSet` plus covariate series, without any
domain knowledge. Sources are registered under the kind `source`; their `options_model` validates the extra
keys of `data.source:` (error path `data.source.<key>`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pydantic
from pydantic import BaseModel, ConfigDict, Field

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.plugins import REGISTRY
from regimekit.utils.errors import ConfigError, DataError

Loaded = tuple[ObservationSet, list[CovariateSeries]]


@runtime_checkable
class Source(Protocol):
    """A registered data source."""

    name: str
    options_model: type[BaseModel]

    def load(self, opts: BaseModel, *, obj: Any | None = None) -> Loaded:  # noqa: ANN401
        """Load the observations and the covariate series described by `opts` (or held by `obj`)."""
        ...


class TabularOpts(BaseModel):
    """Options shared by the tabular sources (`pandas`, `csv`, `parquet`).

    Attributes:
        path: file to read (`csv`, `parquet`).
        observation_columns: the observed series, in model order; `M = len(observation_columns)`.
        time_column: timestamp column (parsed as UTC); `None` = time-less data (valid under a `steps` clock).
        sequence_column: column of sequence ids (consumed from E12-T01a on).
        covariate_columns: columns offered as covariate series.
        dtype: dtype of the observation values.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path | None = None
    observation_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None
    sequence_column: str | None = None
    covariate_columns: list[str] = Field(default_factory=list)
    dtype: Literal["float64", "int64"] = "float64"


def get_source(name: str) -> Source:
    """The registered source `name` (the core sources are registered on first use).

    Raises:
        ConfigError: unknown name or not a source object (`path="data.source.kind"`).
    """
    import regimekit.data.sources  # noqa: F401  (registers the core sources)

    entry = REGISTRY.get("source", name)
    if not isinstance(entry, Source):
        raise ConfigError(
            f"source '{name}' is a configuration section, not a loadable source", path="data.source.kind"
        )
    return entry


def validate_options(kind: str, raw: Mapping[str, Any], path: str) -> BaseModel | None:
    """`raw` validated by the options model of the source `kind`; `None` if `kind` has no options model.

    Raises:
        ConfigError: unknown kind or an invalid option (`path="<path>.<key>"`).
    """
    import regimekit.data.sources  # noqa: F401

    options_model = getattr(REGISTRY.get("source", kind), "options_model", None)
    if options_model is None:
        return None
    try:
        parsed: BaseModel = options_model.model_validate(dict(raw))
    except pydantic.ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(part) for part in first["loc"])
        raise ConfigError(first["msg"], path=f"{path}.{loc}" if loc else path) from exc
    return parsed


def sha256_file(path: Path) -> str:
    """Hex sha256 of the file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(names: Sequence[str], arrays: Sequence[np.ndarray]) -> str:
    """Hex sha256 over the column names and the contiguous array buffers (in-memory data)."""
    digest = hashlib.sha256()
    for name, array in zip(names, arrays, strict=True):
        digest.update(name.encode("utf-8") + b"\0")
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.dtype).encode() + str(arr.shape).encode() + b"\0")
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], path: str) -> None:
    for column in columns:
        if column not in frame.columns:
            raise DataError(
                f"column {column!r} not found; available: {', '.join(map(str, frame.columns))}", path=path
            )


def parse_times(raw: Any, path: str = "data.source.time_column") -> np.ndarray:  # noqa: ANN401
    """Timestamps as strictly increasing naive-UTC `datetime64[ns]`.

    Raises:
        DataError: unparseable or not strictly increasing times (`path`).
    """
    try:
        stamps = pd.to_datetime(pd.Series(np.asarray(raw)), utc=True)
    except (ValueError, TypeError) as exc:
        raise DataError(f"cannot parse the time column: {exc}", path=path) from exc
    if stamps.isna().any():
        raise DataError("the time column holds missing values", path=path)
    times = stamps.dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    if times.shape[0] > 1 and not bool(np.all(np.diff(times) > np.timedelta64(0, "ns"))):
        raise DataError("times must be strictly increasing", path=path)
    return times


def frame_to_loaded(
    frame: pd.DataFrame, opts: TabularOpts, *, source: str, snapshot_id: str | None
) -> Loaded:
    """The pandas logic every tabular source delegates to.

    Args:
        frame: the data.
        opts: the tabular options.
        source: name recorded in `ObservationSet.source`.
        snapshot_id: the sha256 of the file bytes, or `None` to hash the arrays (in-memory data).

    Raises:
        ConfigError: no `observation_columns`.
        DataError: a missing column (the message names it), bad times, values not convertible to `dtype`.
    """
    if not opts.observation_columns:
        raise ConfigError(
            "observation_columns must name at least one column", path="data.source.observation_columns"
        )
    _require_columns(frame, opts.observation_columns, "data.source.observation_columns")
    if opts.time_column is not None:
        _require_columns(frame, [opts.time_column], "data.source.time_column")
    if opts.sequence_column is not None:
        _require_columns(frame, [opts.sequence_column], "data.source.sequence_column")
    _require_columns(frame, opts.covariate_columns, "data.source.covariate_columns")

    try:
        values = frame[opts.observation_columns].to_numpy(dtype=np.dtype(opts.dtype))
    except (ValueError, TypeError) as exc:
        raise DataError(
            f"observation columns cannot be converted to {opts.dtype}: {exc}", path="data.source.dtype"
        ) from exc
    if opts.dtype == "int64" and not np.array_equal(
        values, frame[opts.observation_columns].to_numpy(dtype="float64")
    ):
        raise DataError("observation values are not integral (dtype: int64)", path="data.source.dtype")
    times = parse_times(frame[opts.time_column]) if opts.time_column is not None else None
    sequence_ids = frame[opts.sequence_column].to_numpy() if opts.sequence_column is not None else None
    cov_values = {c: frame[c].to_numpy(dtype=np.float64) for c in opts.covariate_columns}

    if snapshot_id is None:
        names = ["<observations>:" + ",".join(opts.observation_columns), *opts.covariate_columns]
        arrays = [values, *cov_values.values()]
        if times is not None:
            names.append("<time>")
            arrays.append(times.astype("int64"))
        if sequence_ids is not None:
            names.append("<sequence>")
            arrays.append(np.asarray(sequence_ids))
        snapshot_id = sha256_arrays(names, arrays)

    obs = ObservationSet(
        series_names=tuple(opts.observation_columns),
        times=times,
        values=values,
        source=source,
        snapshot_id=snapshot_id,
        sequence_ids=sequence_ids,
    )
    obs.validate()
    cov_times = times if times is not None else np.arange(values.shape[0], dtype=np.int64)
    covs = [
        CovariateSeries(
            name=column,
            times=cov_times,
            values=series,
            source=source,
            series_id=column,
            snapshot_id=snapshot_id,
            frequency="grid",
        )
        for column, series in cov_values.items()
    ]
    return obs, covs
