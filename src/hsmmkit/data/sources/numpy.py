"""The `numpy` source: a `.npy` / `.npz` file, or a dict of arrays passed through the Python API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hsmmkit.data.contracts import CovariateSeries, ObservationSet
from hsmmkit.data.sources.base import Loaded, parse_times, sha256_arrays, sha256_file
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.utils.errors import ConfigError, DataError

_KEYS = ("observations", "times", "covariates", "covariate_names", "sequence_ids")


class NumpyOpts(BaseModel):
    """Options of the `numpy` source.

    Attributes:
        path: a `.npy` file (observations only) or a `.npz` file with the keys `observations` and the
            optional `times`, `covariates` `(T0, K)`, `covariate_names`, `sequence_ids`.
        observation_columns: optional names of the observed series (default `x0, x1, ...`).
        dtype: dtype of the observation values.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path | None = None
    observation_columns: list[str] = Field(default_factory=list)
    dtype: Literal["float64", "int64"] = "float64"


def _arrays_of(opts: NumpyOpts, obj: Any) -> tuple[dict[str, np.ndarray], str | None]:  # noqa: ANN401
    """The named arrays and the file digest (`None` for in-memory data)."""
    if obj is not None:
        if not isinstance(obj, Mapping):
            raise ConfigError("the `numpy` source takes a dict of arrays as `obj`", path="data.source.kind")
        arrays = {str(k): np.asarray(v) for k, v in obj.items()}
        digest = None
    else:
        if opts.path is None:
            raise ConfigError("the `numpy` source needs a path or an object", path="data.source.path")
        path = opts.path.expanduser()
        if not path.is_file():
            raise DataError(f"file not found: {path}", path="data.source.path")
        try:
            loaded = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise DataError(f"cannot read {path}: {exc}", path="data.source.path") from exc
        if isinstance(loaded, np.ndarray):
            arrays = {"observations": loaded}
        else:
            with loaded:
                arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        digest = sha256_file(path)
    unknown = sorted(set(arrays) - set(_KEYS))
    if unknown:
        raise DataError(f"unknown array key(s) {unknown}; expected {list(_KEYS)}", path="data.source.path")
    if "observations" not in arrays:
        raise DataError("the key 'observations' is missing", path="data.source.observation_columns")
    return arrays, digest


class NumpySource:
    """Observations (and optional times, covariates, sequence ids) from numpy arrays."""

    name = "numpy"
    options_model: type[BaseModel] = NumpyOpts

    def load(self, opts: BaseModel, *, obj: Any | None = None) -> Loaded:  # noqa: ANN401
        """Load `.npy`/`.npz` (or a dict of arrays); the snapshot id hashes the file or the array buffers."""
        assert isinstance(opts, NumpyOpts)
        arrays, digest = _arrays_of(opts, obj)
        raw = arrays["observations"]
        values = raw.reshape(-1, 1) if raw.ndim == 1 else raw
        try:
            values = values.astype(np.dtype(opts.dtype))
        except (ValueError, TypeError) as exc:
            raise DataError(
                f"observations cannot be converted to {opts.dtype}: {exc}", path="data.source.dtype"
            ) from exc
        if values.ndim != 2:
            raise DataError("observations must be 1-D or 2-D", path="data.source.observation_columns")
        names = tuple(opts.observation_columns) or tuple(f"x{j}" for j in range(values.shape[1]))
        if len(names) != values.shape[1]:
            raise DataError(
                f"observation_columns has {len(names)} names, the observations {values.shape[1]} columns",
                path="data.source.observation_columns",
            )
        times = parse_times(arrays["times"], "data.source.path") if "times" in arrays else None
        sequence_ids = arrays.get("sequence_ids")
        if digest is None:
            labels = ["<names>:" + ",".join(names), *(f"<{k}>" for k in arrays if k != "observations")]
            bufs = [
                values,
                *(
                    times.astype("int64") if k == "times" and times is not None else arrays[k]
                    for k in arrays
                    if k != "observations"
                ),
            ]
            digest = sha256_arrays(labels[:1] + labels[1:], bufs)
        obs = ObservationSet(
            series_names=names,
            times=times,
            values=values,
            source=self.name,
            snapshot_id=digest,
            sequence_ids=sequence_ids,
        )
        obs.validate()
        covs: list[CovariateSeries] = []
        if "covariates" in arrays:
            block = np.asarray(arrays["covariates"], dtype=np.float64)
            block = block.reshape(-1, 1) if block.ndim == 1 else block
            cov_names = (
                [str(n) for n in arrays["covariate_names"]]
                if "covariate_names" in arrays
                else [f"c{k}" for k in range(block.shape[1])]
            )
            if block.shape[0] != values.shape[0] or len(cov_names) != block.shape[1]:
                raise DataError("covariates must have shape (T0, K) with K names", path="data.source.path")
            cov_times = times if times is not None else np.arange(values.shape[0], dtype=np.int64)
            covs = [
                CovariateSeries(
                    name=cov_names[k],
                    times=cov_times,
                    values=block[:, k],
                    source=self.name,
                    series_id=cov_names[k],
                    snapshot_id=digest,
                    frequency="grid",
                )
                for k in range(block.shape[1])
            ]
        return obs, covs


REGISTRY.register("source", NumpySource.name, NumpySource(), provenance=provenance_of(__name__))
