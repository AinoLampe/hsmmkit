"""E10-T04a: the generic `ObservationSet`, time-less alignment and the finance `price_view`."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hsmmkit.data.calendar import align_covariate, build_grid
from hsmmkit.data.contracts import CovariateSeries, ObservationSet
from hsmmkit.utils.errors import ConfigError, DataError

DAYS = np.arange("2024-01-01", "2024-01-13", dtype="datetime64[D]").astype(
    "datetime64[ns]"
)  # 12 daily stamps


def _obs(times: np.ndarray | None, n: int = 12, **kw: object) -> ObservationSet:
    return ObservationSet(
        series_names=("a",),
        times=times,
        values=np.linspace(1.0, 2.0, n)[:, None],
        source="test",
        snapshot_id="s",
        **kw,  # type: ignore[arg-type]
    )


def _cov(times: np.ndarray | None, values: np.ndarray) -> CovariateSeries:
    return CovariateSeries("c", times, values, "test", "c", "s", "grid")


def test_e10_t04a_observation_set_has_neutral_field_names_only() -> None:
    """AC4: no `prices`, `symbols`, `provider` (nor the finance `interval`/`adjusted`) attribute."""
    fields = {f.name for f in dataclasses.fields(ObservationSet)}
    assert fields == {"series_names", "times", "values", "source", "snapshot_id", "meta", "sequence_ids"}
    obs = _obs(DAYS)
    for gone in ("prices", "symbols", "provider", "interval", "adjusted"):
        assert not hasattr(obs, gone)


def test_e10_t04a_validate_accepts_a_valid_set_with_and_without_times() -> None:
    """`validate()` passes for timestamps, for time-less data and with sequence ids."""
    _obs(DAYS).validate()
    _obs(None).validate()
    _obs(DAYS, sequence_ids=np.zeros(12, dtype=np.int64)).validate()
    ints = ObservationSet(("k",), None, np.arange(6).reshape(3, 2).astype(np.int64)[:, :1], "t", "")
    ints.validate()


@pytest.mark.parametrize(
    ("obs", "path"),
    [
        (ObservationSet(("a",), DAYS, np.zeros(12), "t", ""), "values"),
        (ObservationSet(("a", "b"), DAYS, np.zeros((12, 1)), "t", ""), "series_names"),
        (ObservationSet(("a",), DAYS[::-1], np.zeros((12, 1)), "t", ""), "times"),
        (ObservationSet(("a",), DAYS[:5], np.zeros((12, 1)), "t", ""), "times"),
        (ObservationSet(("a",), DAYS, np.zeros((12, 1)), "t", "", sequence_ids=np.zeros(3)), "sequence_ids"),
        (ObservationSet(("a",), DAYS, np.zeros((12, 1), dtype=bool), "t", ""), "values"),
    ],
)
def test_e10_t04a_validate_rejects_violations_with_the_field_path(obs: ObservationSet, path: str) -> None:
    """Each contract violation raises `DataError` carrying the field name."""
    with pytest.raises(DataError) as info:
        obs.validate()
    assert info.value.path == path


def test_e10_t04a_time_less_grid_is_a_step_index_and_needs_a_step_clock() -> None:
    """Time-less data: an integer grid under `steps`, else `ConfigError(path="data.clock")`."""
    grid = build_grid(_obs(None), "drop", clock="steps")
    assert grid.times.dtype == np.int64
    assert grid.times.tolist() == list(range(12))
    assert grid.dt.tolist() == [1.0] * 11
    for clock in ("calendar", "trading"):
        with pytest.raises(ConfigError) as info:
            build_grid(_obs(None), "drop", clock=clock)
        assert info.value.path == "data.clock"


@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("max_staleness", [0, 1, 5])
def test_e10_t04a_row_alignment_equals_timestamp_alignment_on_a_regular_grid(
    lag: int, max_staleness: int
) -> None:
    """Lag and staleness by row index give what the timestamp path gives for one print per grid step."""
    rng = np.random.default_rng(3)
    values = rng.normal(size=12)
    values[[4, 5, 9]] = np.nan  # gaps exercise the staleness counter
    dated = build_grid(_obs(DAYS), "drop", clock="steps")
    bare = build_grid(_obs(None), "drop", clock="steps")
    v_dated, s_dated = align_covariate(dated, _cov(DAYS, values), lag, max_staleness)
    v_bare, s_bare = align_covariate(bare, _cov(None, values), lag, max_staleness)
    np.testing.assert_array_equal(v_bare, v_dated)
    np.testing.assert_array_equal(s_bare, s_dated)


def test_e10_t04a_lag_by_row_hand_checked() -> None:
    """Row `t` holds the print of row `t + 1 - lag` (known at the end of the previous step for lag 1)."""
    grid = build_grid(_obs(None, 6), "drop", clock="steps")
    values, staleness = align_covariate(grid, _cov(None, np.arange(10.0, 16.0)), 2, 3)
    assert np.isnan(values[0])  # row 0 would need row -1
    assert values[1:].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert staleness.tolist() == [-1, 0, 0, 0, 0]


def test_e10_t04a_time_less_covariate_needs_the_grid_length_and_matching_stamps() -> None:
    """A covariate without timestamps must have one value per grid row; a step index needs a step grid."""
    grid = build_grid(_obs(None), "drop", clock="steps")
    with pytest.raises(DataError):
        align_covariate(grid, _cov(None, np.zeros(5)), 1, 1)
    dated = build_grid(_obs(DAYS), "drop", clock="steps")
    with pytest.raises(DataError):
        align_covariate(dated, _cov(np.arange(12), np.zeros(12)), 1, 1)
    values, _ = align_covariate(dated, _cov(None, np.arange(12.0)), 1, 1)  # row index also on a dated grid
    assert values.shape == (11,)
