"""E10-T04a: the `Source` protocol, the numpy/pandas/csv/parquet sources and `source_contract`."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import DataSection
from hsmmkit.data.sources import load_data
from hsmmkit.data.sources.base import Source, get_source
from hsmmkit.plugins import REGISTRY
from hsmmkit.testing.source_contract import FIXTURES, SourceCase, check_source, core_cases
from hsmmkit.utils.errors import ConfigError, DataError

CASES = core_cases()


def _section(kind: str, clock: str = "steps", **opts: Any) -> DataSection:  # noqa: ANN401
    base: dict[str, Any] = {"observation_columns": ["y1", "y2"], "covariate_columns": ["x"]}
    return DataSection.model_validate({"source": {"kind": kind, **base, **opts}, "clock": clock})


def test_e10_t04a_core_sources_are_registered_with_provenance() -> None:
    """The four core sources are registered as kind `source` by the core."""
    get_source("csv")  # registers on first use
    prov = {e.name: e.provenance for e in REGISTRY.entries("source")}
    for name in ("numpy", "pandas", "csv", "parquet"):
        assert prov[name] == "hsmmkit.core"
        assert isinstance(get_source(name), Source)


@pytest.mark.parametrize("name", sorted(CASES))
def test_e10_t04a_source_passes_the_contract_suite(name: str) -> None:
    """AC1: numpy, pandas, csv and parquet pass `source_contract`."""
    check_source(name, CASES[name])


def test_e10_t04a_contract_suite_flags_a_broken_source() -> None:
    """The suite fails (and names the source) when a check does not hold."""
    case = CASES["csv"]
    bad = SourceCase(
        options=lambda time: {**case.options(time), "dtype": "int64"},  # int64 breaks the float64 expectation
        obj=case.obj,
        missing=case.missing,
    )
    with pytest.raises((AssertionError, DataError), match=r"csv|int64|converted"):
        check_source("csv", bad)


def _frame(time: bool) -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / "gauss.csv")
    return frame if time else frame.drop(columns=["t"])


@pytest.mark.parametrize(
    ("clock", "time", "outcome"),
    [
        ("steps", True, "times"),
        ("steps", False, "no times"),
        ("calendar", True, "times"),
        ("calendar", False, "ConfigError"),
    ],
)
def test_e10_t04a_dataframe_loads_by_clock_and_time_column(clock: str, time: bool, outcome: str) -> None:
    """AC2: steps with/without times, calendar with times; calendar without times → data.clock error."""
    section = _section("pandas", clock, time_column="t" if time else None)
    if outcome == "ConfigError":
        with pytest.raises(ConfigError) as info:
            load_data(section, obj=_frame(time))
        assert info.value.path == "data.clock"
        return
    obs, _ = load_data(section, obj=_frame(time))
    assert (obs.times is not None) == (outcome == "times")
    assert obs.values.shape == (40, 2)


def test_e10_t04a_pandas_in_yaml_without_an_object_suggests_csv_or_parquet() -> None:
    """`kind: pandas` without an object is a ConfigError that points at csv/parquet."""
    with pytest.raises(ConfigError, match=r"csv.*parquet") as info:
        load_data(_section("pandas"))
    assert info.value.path == "data.source.kind"


def test_e10_t04a_unknown_option_keys_fail_with_the_key_path() -> None:
    """Each source's options model validates the extra keys: `ConfigError(path="data.source.<key>")`."""
    with pytest.raises(ConfigError) as info:
        _section("csv", path=FIXTURES / "gauss.csv", observation_colums=["y1"])
    assert info.value.path == "data.source.observation_colums"
    with pytest.raises(ConfigError) as info:
        DataSection.model_validate({"source": {"kind": "numpy", "time_column": "t"}})  # no such numpy option
    assert info.value.path == "data.source.time_column"
    with pytest.raises(ConfigError, match="unknown source") as info:
        _section("nope")
    assert info.value.path == "data.source.kind"


def test_e10_t04a_covariate_source_options_are_validated_with_their_index() -> None:
    """A covariate's `source` is validated like `data.source`."""
    with pytest.raises(ConfigError) as info:
        DataSection.model_validate(
            {
                "source": {"kind": "csv", "path": str(FIXTURES / "gauss.csv"), "observation_columns": ["y1"]},
                "covariates": [
                    {
                        "name": "x",
                        "source": {"kind": "csv", "paht": "x.csv"},
                        "lag": 1,
                        "use_in": ["emission"],
                    }
                ],
            }
        )
    assert info.value.path == "data.covariates[0].source.paht"


def test_e10_t04a_file_snapshot_is_the_sha256_of_the_bytes() -> None:
    """Files: `snapshot_id` = sha256 over the file bytes (csv, parquet, npz)."""
    for kind, name, extra in (
        ("csv", "gauss.csv", {"time_column": "t"}),
        ("parquet", "gauss.parquet", {"time_column": "t"}),
        ("numpy", "gauss.npz", {"observation_columns": ["y1", "y2"], "covariate_columns": None}),
    ):
        opts = {k: v for k, v in extra.items() if v is not None}
        section = DataSection.model_validate(
            {
                "source": {
                    "kind": kind,
                    "path": str(FIXTURES / name),
                    "observation_columns": ["y1", "y2"],
                    **({} if kind == "numpy" else {"covariate_columns": ["x"]}),
                    **opts,
                }
            }
        )
        obs, _ = load_data(section)
        assert obs.snapshot_id == hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()


def test_e10_t04a_in_memory_snapshot_is_deterministic_and_content_sensitive() -> None:
    """In-memory data: same content → same id; a changed value or column name → another id."""
    section = _section("pandas", time_column="t")
    a, _ = load_data(section, obj=_frame(True))
    b, _ = load_data(section, obj=_frame(True))
    assert a.snapshot_id == b.snapshot_id
    changed = _frame(True)
    changed.loc[3, "y1"] += 1e-9
    assert load_data(section, obj=changed)[0].snapshot_id != a.snapshot_id
    renamed = _frame(True).rename(columns={"x": "z"})
    other = _section("pandas", time_column="t", covariate_columns=["z"])
    assert load_data(other, obj=renamed)[0].snapshot_id != a.snapshot_id


def test_e10_t04a_non_monotone_times_are_a_data_error_on_the_time_column() -> None:
    """Non-increasing timestamps: `DataError(path="data.source.time_column")`."""
    frame = _frame(True).iloc[::-1]
    with pytest.raises(DataError) as info:
        load_data(_section("pandas", time_column="t"), obj=frame)
    assert info.value.path == "data.source.time_column"


def test_e10_t04a_times_are_utc_nanoseconds() -> None:
    """Time columns are parsed as UTC; offsets are converted."""
    frame = pd.DataFrame({"t": ["2024-01-01T01:00:00+01:00", "2024-01-02T00:00:00Z"], "y": [1.0, 2.0]})
    section = DataSection.model_validate(
        {"source": {"kind": "pandas", "observation_columns": ["y"], "time_column": "t"}}
    )
    obs, _ = load_data(section, obj=frame)
    assert obs.times is not None
    assert obs.times.dtype == np.dtype("datetime64[ns]")
    assert obs.times.tolist()[0] == np.datetime64("2024-01-01T00:00:00", "ns").tolist()


def test_e10_t04a_sequence_column_fills_sequence_ids_and_stays_out_of_the_values() -> None:
    """`sequence_column` gives `(T0,)` ids; the observation matrix has only the observation columns."""
    obs, _ = load_data(_section("pandas", time_column="t", sequence_column="seq"), obj=_frame(True))
    assert obs.sequence_ids is not None
    assert obs.sequence_ids.shape == (40,)
    assert sorted(set(obs.sequence_ids.tolist())) == [0, 1]
    assert obs.values.shape == (40, 2)


def test_e10_t04a_int64_observations_for_count_data() -> None:
    """`dtype: int64` yields an integer matrix; non-integral values are a DataError on `dtype`."""
    frame = pd.DataFrame({"n": [0, 1, 4, 2]})
    section = DataSection.model_validate(
        {"source": {"kind": "pandas", "observation_columns": ["n"], "dtype": "int64"}}
    )
    obs, _ = load_data(section, obj=frame)
    assert obs.values.dtype == np.int64
    with pytest.raises(DataError) as info:
        load_data(section, obj=pd.DataFrame({"n": [0.5, 1.0]}))
    assert info.value.path == "data.source.dtype"


def test_e10_t04a_covariates_follow_the_declared_order_and_unknown_columns_fail() -> None:
    """`data.covariates[i]` picks `column` (else `name`) of the primary source; unknown → ConfigError."""
    base = {"source": {"kind": "pandas", "observation_columns": ["y1"], "covariate_columns": ["x", "y2"]}}
    cov = {"lag": 1, "use_in": ["emission"]}
    ok = DataSection.model_validate(
        {**base, "covariates": [{"name": "b", "column": "y2", **cov}, {"name": "x", **cov}]}
    )
    _, covs = load_data(ok, obj=_frame(True))
    assert [c.series_id for c in covs] == ["y2", "x"]
    bad = DataSection.model_validate({**base, "covariates": [{"name": "nope", **cov}]})
    with pytest.raises(ConfigError) as info:
        load_data(bad, obj=_frame(True))
    assert info.value.path == "data.covariates[0].column"


def test_e10_t04a_numpy_npy_file_and_one_dimensional_arrays(tmp_path: Path) -> None:
    """`.npy` holds observations only; a 1-D array is one series named `x0`."""
    path = tmp_path / "obs.npy"
    np.save(path, np.arange(5.0))
    obs, covs = load_data(DataSection.model_validate({"source": {"kind": "numpy", "path": str(path)}}))
    assert obs.series_names == ("x0",)
    assert obs.values.shape == (5, 1)
    assert obs.times is None
    assert covs == []
    with pytest.raises(DataError) as info:
        load_data(
            DataSection.model_validate(
                {"source": {"kind": "numpy", "path": str(path), "observation_columns": ["a", "b"]}}
            )
        )
    assert info.value.path == "data.source.observation_columns"
    with pytest.raises(DataError) as info:
        load_data(
            DataSection.model_validate({"source": {"kind": "numpy", "path": str(tmp_path / "gone.npy")}})
        )
    assert info.value.path == "data.source.path"


def test_e10_t04a_csv_requires_a_path_and_reports_missing_files() -> None:
    """No `path`: ConfigError; missing file: DataError (both `data.source.path`)."""
    with pytest.raises(ConfigError) as info:
        load_data(_section("csv"))
    assert info.value.path == "data.source.path"
    with pytest.raises(DataError) as info:
        load_data(_section("parquet", path="/nonexistent/x.parquet"))
    assert info.value.path == "data.source.path"
