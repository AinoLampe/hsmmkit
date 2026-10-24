"""E9-T01: multi-series master grid (primary decides the grid; other series mask) and `per_series` config."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ruamel.yaml import YAML

from regimekit.config.schema import DataSection, EmissionCfg, ProjectConfig
from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config
from regimekit.data.calendar import build_grid
from regimekit.data.contracts import ObservationSet
from regimekit.data.prepare import Preparer
from regimekit.utils.errors import DataError
from tests.helpers.two_series import APPENDIX_A

N_DAYS = 60
TIMES = np.asarray(pd.bdate_range("2021-01-04", periods=N_DAYS).as_unit("ns").values, dtype="datetime64[ns]")
_YAML = YAML(typ="safe")


def _obs(prices: np.ndarray) -> ObservationSet:
    return ObservationSet(
        symbols=("A", "B")[: prices.shape[1]],
        times=TIMES,
        prices=prices,
        interval="1d",
        adjusted=True,
        provider="csv",
        snapshot_id="snap",
    )


def _prices() -> np.ndarray:
    rng = np.random.default_rng(1)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (N_DAYS, 2)), axis=0))


def _data_cfg(missing: str = "drop") -> DataSection:
    raw = _YAML.load(APPENDIX_A.read_text(encoding="utf-8"))
    raw["data"]["primary"]["symbols"] = ["A", "B"]
    raw["data"]["covariates"] = []
    raw["data"]["preparation"]["missing_returns"] = missing
    return DataSection.model_validate(raw["data"])


# --- master grid ----------------------------------------------------------------------------------


def test_e9_t01_secondary_gap_keeps_the_grid_row_and_masks_that_series() -> None:
    """A missing price in series 2 does not remove the primary's row; the series is masked instead."""
    prices = _prices()
    prices[20, 1] = np.nan
    grid = build_grid(_obs(prices), "drop", clock="trading")
    assert grid.times.shape[0] == N_DAYS  # no row dropped
    assert np.isnan(grid.prices[20, 1])
    assert np.isfinite(grid.prices[:, 0]).all()


def test_e9_t01_primary_gap_drops_the_row_for_every_series() -> None:
    """The first series is the primary: its missing price removes the grid row (as for `M = 1`)."""
    prices = _prices()
    prices[20, 0] = np.nan
    grid = build_grid(_obs(prices), "drop", clock="trading")
    assert grid.times.shape[0] == N_DAYS - 1
    assert TIMES[20] not in grid.times
    assert grid.prices.shape == (N_DAYS - 1, 2)


def test_e9_t01_error_policy_rejects_a_gap_in_any_series() -> None:
    """`missing_returns: error` stays strict for every series and names the series."""
    prices = _prices()
    prices[20, 1] = np.nan
    with pytest.raises(DataError, match="'B'") as exc:
        build_grid(_obs(prices), "error", clock="trading")
    assert exc.value.path == "data.preparation.missing_returns"


def test_e9_t01_single_series_grid_is_unchanged() -> None:
    """`M = 1` keeps dropping every row with a missing price (the primary is the only series)."""
    prices = _prices()[:, :1]
    prices[20, 0] = np.nan
    grid = build_grid(_obs(prices), "drop", clock="trading")
    assert grid.times.shape[0] == N_DAYS - 1


def test_e9_t01_preparer_masks_the_return_a_gap_touches() -> None:
    """A price hole at row 20 masks series 2's returns at rows 19 and 20 (both need `P_20`) and no others."""
    prices = _prices()
    prices[20, 1] = np.nan
    prep = Preparer(_data_cfg()).fit(_obs(prices), [], train_end=TIMES[-1])
    ds = prep.transform(_obs(prices), [])
    mask = np.asarray(ds.mask)
    assert mask.shape == (N_DAYS - 1, 2)
    assert mask[:, 0].all()
    assert np.flatnonzero(~mask[:, 1]).tolist() == [19, 20]  # returns r_t = log(P_{t+1} / P_t) (row t)
    assert np.isfinite(np.asarray(ds.observations)).all()
    assert (np.asarray(ds.observations)[~mask] == 0.0).all()
    ds.validate()


# --- config: model.emission.per_series ----------------------------------------------------------------


def test_e9_t01_per_series_defaults_to_true_and_keeps_appendix_a_valid() -> None:
    """Default `True` (separate parameters per series, the pre-E9-T01 behaviour)."""
    assert EmissionCfg(family="gbm", params={}).per_series is True
    assert EmissionCfg(family="gbm", params={}, per_series=False).per_series is False
    assert isinstance(load_config(APPENDIX_A), ProjectConfig)


def test_e9_t01_per_series_in_json_schema_and_resolved_dump(tmp_path: Path) -> None:
    """The JSON Schema lists `per_series` and the resolved dump writes the defaulted value."""
    schema_path = tmp_path / "schema.json"
    export_json_schema(schema_path)
    props = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["EmissionCfg"]["properties"]
    assert props["per_series"]["type"] == "boolean"
    assert props["per_series"]["default"] is True

    resolved = tmp_path / "config.resolved.yaml"
    dump_resolved(load_config(APPENDIX_A), resolved)
    assert _YAML.load(resolved)["model"]["emission"]["per_series"] is True
