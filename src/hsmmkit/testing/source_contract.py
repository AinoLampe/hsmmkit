"""Source contract suite (E10-T04a): the checks every registered source must pass.

A source is exercised through a `SourceCase`: factories that build its options (and in-memory object) for the
synthetic `gauss` fixture (`tests/fixtures/sources`, `T0 = 40`, observed `y1, y2`, covariate `x`).
`check_source` raises `AssertionError` with a message naming the source when a check fails; the test modules
parametrize it over `REGISTRY.names("source")` that have a case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regimekit.data.contracts import ObservationSet
from regimekit.data.sources.base import get_source, validate_options
from regimekit.utils.errors import DataError

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "sources"
OBSERVATION_COLUMNS = ["y1", "y2"]
COVARIATE_COLUMNS = ["x"]


@dataclass(frozen=True)
class SourceCase:
    """How to drive one source in the contract suite.

    Attributes:
        options: `options(time)` gives the `data.source` options (without `kind`); `time=False` omits the
            time column / array.
        obj: the in-memory object for sources that need one (`None` for file sources); `obj(time)`.
        missing: `missing()` gives `(options, obj, needle)` of a load that must raise `DataError` whose
            message contains `needle` (the missing column or key).
        supports_timeless: `time=False` loads must work.
    """

    options: Callable[[bool], dict[str, Any]]
    obj: Callable[[bool], Any | None]
    missing: Callable[[], tuple[dict[str, Any], Any | None, str]]
    supports_timeless: bool = True


def _frame(time: bool) -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / "gauss.csv")
    return frame if time else frame.drop(columns=["t"])


def _tabular_options(time: bool) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "observation_columns": OBSERVATION_COLUMNS,
        "covariate_columns": COVARIATE_COLUMNS,
        "time_column": "t" if time else None,
    }
    return opts


def _npz_arrays(time: bool) -> dict[str, np.ndarray]:
    frame = _frame(True)
    arrays: dict[str, np.ndarray] = {
        "observations": frame[OBSERVATION_COLUMNS].to_numpy(),
        "covariates": frame[COVARIATE_COLUMNS].to_numpy(),
        "covariate_names": np.array(COVARIATE_COLUMNS),
    }
    if time:
        arrays["times"] = pd.to_datetime(frame["t"]).to_numpy().astype("datetime64[ns]")
    return arrays


def core_cases() -> dict[str, SourceCase]:
    """Contract cases of the four core sources, keyed by registered name."""

    def with_path(name: str, **extra: Any) -> Callable[[bool], dict[str, Any]]:  # noqa: ANN401
        return lambda time: {"path": FIXTURES / name, **extra}

    def tabular_file(name: str) -> SourceCase:
        def missing() -> tuple[dict[str, Any], Any | None, str]:
            return (
                {**_tabular_options(True), "path": FIXTURES / name, "observation_columns": ["y1", "nope"]},
                None,
                "nope",
            )

        return SourceCase(
            options=lambda time: {**_tabular_options(time), "path": FIXTURES / name},
            obj=lambda time: None,
            missing=missing,
        )

    def numpy_missing() -> tuple[dict[str, Any], Any | None, str]:
        return {}, {"times": np.arange(3)}, "observations"

    return {
        "csv": tabular_file("gauss.csv"),
        "parquet": tabular_file("gauss.parquet"),
        "pandas": SourceCase(
            options=_tabular_options,
            obj=_frame,
            missing=lambda: (
                {**_tabular_options(True), "observation_columns": ["y1", "nope"]},
                _frame(True),
                "nope",
            ),
        ),
        "numpy": SourceCase(
            options=lambda time: {"observation_columns": OBSERVATION_COLUMNS},
            obj=_npz_arrays,
            missing=numpy_missing,
        ),
    }


def _load(name: str, options: dict[str, Any], obj: Any | None) -> tuple[ObservationSet, list[Any]]:  # noqa: ANN401
    source = get_source(name)
    opts = validate_options(name, options, "data.source")
    assert opts is not None, f"{name}: the source has no options model"
    return source.load(opts, obj=obj)


def check_source(name: str, case: SourceCase) -> None:
    """Run the contract on the registered source `name` with its `case`.

    Raises:
        AssertionError: a check failed.
    """
    source = get_source(name)
    assert source.name == name, f"{name}: the source names itself {source.name!r}"

    obs, covs = _load(name, case.options(True), case.obj(True))
    obs.validate()
    assert obs.times is not None, f"{name}: a load with a time column must return times"
    assert len(obs.series_names) == len(OBSERVATION_COLUMNS) == obs.values.shape[1], f"{name}: M mismatch"
    assert obs.series_names == tuple(OBSERVATION_COLUMNS), f"{name}: series_names {obs.series_names}"
    assert obs.values.dtype == np.float64, f"{name}: values dtype {obs.values.dtype}"
    assert obs.source == name, f"{name}: ObservationSet.source is {obs.source!r}"
    assert obs.snapshot_id, f"{name}: the snapshot id must be set"
    assert [c.name for c in covs] == COVARIATE_COLUMNS, f"{name}: covariates {[c.name for c in covs]}"
    for cov in covs:
        assert cov.times is not None, f"{name}: covariate without times"
        assert np.array_equal(cov.times, obs.times), f"{name}: covariate not aligned"
        assert cov.values.shape == (obs.values.shape[0],), f"{name}: covariate length"

    again, _ = _load(name, case.options(True), case.obj(True))
    assert again.snapshot_id == obs.snapshot_id, f"{name}: the snapshot id differs across two loads"
    assert np.array_equal(again.values, obs.values), f"{name}: values differ across two loads"

    if case.supports_timeless:
        bare, bare_covs = _load(name, case.options(False), case.obj(False))
        bare.validate()
        assert bare.times is None, f"{name}: a load without a time column must return times=None"
        assert np.array_equal(bare.values, obs.values), f"{name}: time-less values differ"
        assert [c.name for c in bare_covs] == COVARIATE_COLUMNS, f"{name}: time-less covariates"

    bad_options, bad_obj, needle = case.missing()
    message = None
    try:
        _load(name, bad_options, bad_obj)
    except DataError as exc:
        message = str(exc)
    assert message is not None, f"{name}: a missing {needle!r} did not raise DataError"
    assert needle in message, f"{name}: the error does not name {needle!r}: {message}"
