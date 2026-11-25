"""E10-T05: the generic `Preparer` (transforms, missing policies, standardization, time-less data)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import DataSection
from hsmmkit.data.contracts import PrepStats
from hsmmkit.data.prepare import Preparer, prep_stats_from_json, prep_stats_to_json
from hsmmkit.data.sources import load_data
from hsmmkit.data.transforms import get_observation_transform
from hsmmkit.plugins import REGISTRY
from hsmmkit.testing.source_contract import FIXTURES
from hsmmkit.utils.errors import ConfigError, DataError

N = 40


def _frame(time: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / "gauss.csv")
    return frame if time else frame.drop(columns=["t"])


def _section(
    *,
    clock: str = "steps",
    time: bool = True,
    prep: dict[str, Any] | None = None,
    covariates: bool = False,
    lag: int = 1,
) -> DataSection:
    source: dict[str, Any] = {
        "kind": "pandas",
        "observation_columns": ["y1", "y2"],
        "time_column": "t" if time else None,
        "covariate_columns": ["x"],
    }
    cov = [{"name": "x", "lag": lag, "use_in": ["emission"], "max_staleness": 2}] if covariates else []
    return DataSection.model_validate(
        {"source": source, "clock": clock, "preparation": prep or {}, "covariates": cov}
    )


def _prepare(
    section: DataSection,
    frame: pd.DataFrame | None = None,
    train_end: Any = "2024-02-09",  # noqa: ANN401
) -> Any:  # noqa: ANN401
    obs, covs = load_data(
        section, obj=_frame(section.source.option("time_column") is not None) if frame is None else frame
    )
    return Preparer(section).fit(obs, covs, train_end).transform(obs, covs), obs


def test_e10_t05_core_transforms_are_registered() -> None:
    """`identity` and `diff` are core transforms; unknown names list the registered ones."""
    prov = {e.name: e.provenance for e in REGISTRY.entries("transform")}
    assert prov["identity"] == prov["diff"] == "hsmmkit.core"
    assert get_observation_transform(None).name == "identity"
    with pytest.raises(ConfigError, match="registered") as info:
        get_observation_transform("nope")
    assert info.value.path == "data.preparation.observation_transform"


def test_e10_t05_identity_keeps_the_values_and_every_row() -> None:
    """AC3: `identity` on the csv fixture: `observations == values` (float64), `T == T0`."""
    data, obs = _prepare(_section())
    assert data.observations.dtype == np.float64
    np.testing.assert_array_equal(np.asarray(data.observations), obs.values)
    assert data.observations.shape == (N, 2)
    assert data.dt.shape == (N,)
    assert bool(np.all(np.asarray(data.mask)))
    assert data.support == "real"
    assert data.clock == "steps"


def test_e10_t05_identity_under_calendar_has_a_spacing_for_the_first_row() -> None:
    """Under `calendar` the first observation gets the median spacing (one virtual start time)."""
    data, _ = _prepare(_section(clock="calendar"))
    np.testing.assert_allclose(np.asarray(data.dt), 1 / 365.25)
    assert data.times.shape == (N,)


def test_e10_t05_diff_transform_drops_the_first_row() -> None:
    """`diff`: `y_t = x_t - x_{t-1}` and `T = T0 - 1` with the shortened times."""
    data, obs = _prepare(_section(prep={"observation_transform": "diff"}))
    assert data.observations.shape == (N - 1, 2)
    np.testing.assert_allclose(np.asarray(data.observations), np.diff(obs.values, axis=0), rtol=0, atol=1e-15)
    assert obs.times is not None
    np.testing.assert_array_equal(data.times, obs.times[1:])


def test_e10_t05_mask_policy_keeps_rows_and_masks_missing_values() -> None:
    """`missing_observations: mask` (the default): non-finite values give `mask=False` and value 0."""
    frame = _frame()
    frame.loc[5, "y1"] = np.nan
    frame.loc[7, "y2"] = np.nan
    data, _ = _prepare(_section(), frame)
    mask, y = np.asarray(data.mask), np.asarray(data.observations)
    assert data.observations.shape[0] == N
    assert not mask[5, 0]
    assert not mask[7, 1]
    assert mask.sum() == 2 * N - 2
    assert y[5, 0] == 0.0
    assert y[7, 1] == 0.0


def test_e10_t05_drop_policy_removes_rows_of_the_primary_series() -> None:
    """`drop` removes the rows where the first series is missing, jointly with the covariates."""
    frame = _frame()
    frame.loc[5, "y1"] = np.nan
    data, _ = _prepare(_section(prep={"missing_observations": "drop"}, covariates=True), frame)
    assert data.observations.shape[0] == N - 1 - 1  # the dropped row, and the covariate lag warm-up
    assert data.cov_emission is not None
    assert data.cov_emission.shape[0] == data.observations.shape[0]


def test_e10_t05_error_policy_names_the_first_missing_value() -> None:
    """`error` raises a DataError naming the first row and series."""
    frame = _frame()
    frame.loc[9, "y2"] = np.nan
    section = _section(prep={"missing_observations": "error"})
    obs, covs = load_data(section, obj=frame)
    with pytest.raises(DataError, match="y2") as info:
        Preparer(section).fit(obs, covs, "2024-02-09")
    assert info.value.path == "data.preparation.missing_observations"


def test_e10_t05_standardize_observations_uses_the_train_window_only() -> None:
    """Per-series train-window mean / sd (ddof 0) are stored in `PrepStats` and applied to every row."""
    section = _section(prep={"standardize_observations": True})
    obs, covs = load_data(section, obj=_frame())
    prep = Preparer(section).fit(obs, covs, "2024-01-20")
    stats = prep.prep_stats
    train = obs.values[:20]  # 2024-01-01 .. 2024-01-20 inclusive
    np.testing.assert_allclose(stats.obs_mean, train.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(stats.obs_std, train.std(axis=0), rtol=1e-14)
    data = prep.transform(obs, covs)
    np.testing.assert_allclose(
        np.asarray(data.observations), (obs.values - stats.obs_mean) / stats.obs_std, rtol=1e-14
    )


def test_e10_t05_prep_stats_json_round_trip_and_old_files_load() -> None:
    """`obs_mean` / `obs_std` round-trip; a 0.1.x / 0.2.x file without them loads with `None`."""
    stats = PrepStats(
        np.array([1.0]),
        np.array([2.0]),
        ("a", "b"),
        ("x",),
        {"x": []},
        {"x": 1},
        np.array([0.5, 1.5]),
        np.array([2.0, 3.0]),
    )
    assert prep_stats_from_json(prep_stats_to_json(stats)) == stats
    old = PrepStats(np.array([1.0]), np.array([2.0]), ("a", "b"), ("x",), {"x": []}, {"x": 1})
    text = prep_stats_to_json(old)
    assert "obs_mean" not in json.loads(text)  # the file format is unchanged without observation statistics
    loaded = prep_stats_from_json(text)
    assert loaded.obs_mean is None
    assert loaded.obs_std is None
    assert loaded == old


def test_e10_t05_time_less_data_uses_the_step_index_and_row_aligned_covariates() -> None:
    """Without times: an `int64` step index, `dt == 1`, covariates aligned by row with lag and staleness."""
    section = _section(time=False, covariates=True, lag=2)
    data, obs = _prepare(section, _frame(False), train_end=19)
    assert obs.times is None
    assert data.times.dtype == np.int64
    assert data.clock == "steps"
    assert bool(np.all(np.asarray(data.dt) == 1.0))
    # row t holds the covariate of row t + 1 - lag (lag 2 -> the previous row); the warm-up drops row 0
    x = _frame(False)["x"].to_numpy()
    assert data.cov_emission is not None
    raw = np.asarray(data.cov_emission)[:, 0] * data.prep_stats.cov_std[0] + data.prep_stats.cov_mean[0]
    np.testing.assert_allclose(raw, x[: len(raw)], rtol=1e-12)


def test_e10_t05_covariate_pipeline_on_timestamps_matches_the_time_less_one() -> None:
    """The same data with daily timestamps (steps clock) and without give the same covariate block."""
    dated, _ = _prepare(_section(covariates=True, lag=2), train_end="2024-01-20")
    bare, _ = _prepare(_section(time=False, covariates=True, lag=2), _frame(False), train_end=19)
    np.testing.assert_array_equal(np.asarray(dated.cov_emission), np.asarray(bare.cov_emission))
    np.testing.assert_array_equal(np.asarray(dated.observations), np.asarray(bare.observations))


def test_e10_t05_standardize_needs_real_support() -> None:
    """`standardize_observations` with a non-real support is a ConfigError."""
    with pytest.raises(ConfigError) as info:
        Preparer(_section(prep={"standardize_observations": True}), support="count")
    assert info.value.path == "data.preparation.standardize_observations"
