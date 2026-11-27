"""E15-T01: state persistence, `spec_hash`, `update_from_time`, covariate preparer (AC2)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hsmmkit.data.calendar import Grid, align_covariate
from hsmmkit.data.contracts import CovariateSeries, PrepStats
from hsmmkit.data.transforms import apply_transforms
from hsmmkit.online import ExpandedOnlineFilter, OnlineCovariatePreparer, load_state, save_state, spec_hash
from hsmmkit.utils.errors import ConfigError, PersistenceError
from tests.helpers.engine_fixtures import fit_of, make_scenario

T = 400


@pytest.fixture(scope="module")
def case() -> dict[str, Any]:
    """A small scenario with covariates and its `FitResult`."""
    sc = make_scenario(3, [3, 3, 3], 1, T, n_cov_pi=2, n_cov_d=2, seed=2)
    d = sc.data
    return {
        "sc": sc,
        "fit": fit_of(sc),
        "y": np.asarray(d.observations),
        "dt": np.asarray(d.dt),
        "z": {"transition": np.asarray(d.cov_transition), "duration": np.asarray(d.cov_duration)},
    }


def test_e15_t01_save_restore_continue_is_bitwise_identical(case: dict[str, Any], tmp_path: Path) -> None:
    """E15-T01: save restore continue is bitwise identical."""
    filt = ExpandedOnlineFilter(case["fit"])
    y, dt, z = case["y"], case["dt"], case["z"]
    sl = slice(0, T // 2)
    full, f_full = filt.update_many(filt.start(), y, dt, z)
    half, _ = filt.update_many(filt.start(), y[sl], dt[sl], {k: v[sl] for k, v in z.items()})
    save_state(half, tmp_path / "s")
    assert (tmp_path / "s" / "state.safetensors").exists()
    meta = json.loads((tmp_path / "s" / "state.json").read_text())
    assert meta["t"] == T // 2
    assert meta["spec_hash"] == spec_hash(case["fit"])
    assert "library_version" in meta
    restored = load_state(tmp_path / "s", case["fit"])
    assert restored.step == T // 2
    rest = slice(T // 2, None)
    cont, f2 = filt.update_many(restored, y[rest], dt[rest], {k: v[rest] for k, v in z.items()})
    np.testing.assert_array_equal(np.asarray(f2), np.asarray(f_full)[rest])
    np.testing.assert_array_equal(np.asarray(cont.log_alpha), np.asarray(full.log_alpha))
    np.testing.assert_array_equal(np.asarray(cont.log_norm), np.asarray(full.log_norm))


def test_e15_t01_a_state_of_another_fit_is_rejected_by_spec_hash(
    case: dict[str, Any], tmp_path: Path
) -> None:
    """E15-T01: a state of another fit is rejected by spec hash."""
    filt = ExpandedOnlineFilter(case["fit"])
    state, _ = filt.update_many(
        filt.start(), case["y"][:10], case["dt"][:10], {k: v[:10] for k, v in case["z"].items()}
    )
    save_state(state, tmp_path / "s")
    other = fit_of(make_scenario(4, [3, 3, 3, 3], 1, 30, n_cov_pi=2, n_cov_d=2, seed=2))
    assert spec_hash(other) != spec_hash(case["fit"])
    with pytest.raises(PersistenceError, match=spec_hash(other)) as exc:
        load_state(tmp_path / "s", other)
    assert spec_hash(case["fit"]) in str(exc.value)
    with pytest.raises(ConfigError, match="another model"):
        ExpandedOnlineFilter(other).update(state, case["y"][0], 1.0)


def test_e15_t01_missing_state_files_raise_persistence_error(case: dict[str, Any], tmp_path: Path) -> None:
    """E15-T01: missing state files raise persistence error."""
    with pytest.raises(PersistenceError):
        load_state(tmp_path / "nothing", case["fit"])


def test_e15_t01_update_from_time_uses_the_step_clock(case: dict[str, Any]) -> None:
    """E15-T01: update from time uses the step clock."""
    filt = ExpandedOnlineFilter(case["fit"], nominal_dt=1.0)
    y, z = case["y"], case["z"]
    state = filt.start()
    ref = filt.start()
    for t in range(5):
        zt = {k: v[t] for k, v in z.items()}
        state, a = filt.update_from_time(state, t, y[t], zt)
        ref, b = filt.update(ref, y[t], 1.0, zt)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert state.last_time == 4


def test_e15_t01_covariate_row_and_input_validation(case: dict[str, Any]) -> None:
    """E15-T01: covariate row and input validation."""
    filt = ExpandedOnlineFilter(case["fit"])
    with pytest.raises(ConfigError, match="covariate row"):
        filt.update(filt.start(), case["y"][0], 1.0, None)
    with pytest.raises(ConfigError, match="dt must be positive"):
        filt.update(filt.start(), case["y"][0], -1.0, {k: v[0] for k, v in case["z"].items()})


def test_e15_t01_update_is_compiled_once(case: dict[str, Any]) -> None:
    """Further updates reuse the compiled step (no recompilation)."""
    filt = ExpandedOnlineFilter(case["fit"])
    state = filt.start()
    for t in range(3):
        state, _ = filt.update(state, case["y"][t], case["dt"][t], {k: v[t] for k, v in case["z"].items()})
    size = filt._step._cache_size()
    for t in range(3, 10):
        state, _ = filt.update(state, case["y"][t], case["dt"][t], {k: v[t] for k, v in case["z"].items()})
    assert filt._step._cache_size() == size


@pytest.mark.parametrize(
    ("transforms", "lag", "stale"),
    [([], 1, None), (["log"], 2, None), (["diff"], 1, 3), (["log", "pct"], 3, 2)],
)
def test_e15_t01_covariate_preparer_matches_the_offline_alignment(
    transforms: list[str], lag: int, stale: int | None
) -> None:
    """Transforms, lag, stale limit and standardization equal the offline alignment path."""
    rng = np.random.default_rng(4)
    n = 60
    raw = np.exp(rng.normal(0.0, 0.3, size=n + 1))
    raw[rng.random(n + 1) < 0.2] = np.nan
    mean, std = 0.3, 1.7
    stats = PrepStats(
        cov_mean=np.array([mean]),
        cov_std=np.array([std]),
        train_window=("0", "1"),
        cov_names=("x",),
        transforms={"x": transforms},
        lags={"x": lag},
    )
    grid = Grid(
        times=np.arange(n + 1, dtype=np.int64),
        values=np.zeros((n + 1, 1)),
        dt=np.ones(n),
        nominal_dt=1.0,
        irregular=False,
        clock="steps",
    )
    series = CovariateSeries("x", None, apply_transforms(raw, transforms), "t", "x", "s", "d")
    big = 10**6 if stale is None else stale
    offline, _ = align_covariate(grid, series, lag, big)
    offline = (offline - mean) / std

    prep = OnlineCovariatePreparer(stats, max_staleness=stale, entry_points={"transition": [0]})
    for j in range(n + 1):
        prep.push([raw[j]])
        t = j - 1 + lag  # the newest row whose source print is j
        if 0 <= t < n:
            np.testing.assert_allclose(prep.row(t), [offline[t]], rtol=1e-12, atol=1e-12, equal_nan=True)
            assert set(prep.z(t)) == {"transition"}
    with pytest.raises(ConfigError):
        prep.row(n + 5)


def test_e15_t01_the_online_gate_is_removed_but_alerts_stay_gated(tmp_path: Path) -> None:
    """The `online` key builds; `online.alerts` stays gated until E15-T03."""
    from hsmmkit.config import gates
    from hsmmkit.config.schema import parse_project_config
    from tests.helpers.core_config import core_raw

    raw = core_raw(tmp_path)
    raw["online"] = {}
    gates.check(parse_project_config(raw))  # builds now
    raw["online"] = {"alerts": [{"regime": 1, "p_on": 0.8, "p_off": 0.5}]}
    with pytest.raises(ConfigError, match="not available before E15-T03"):
        gates.check(parse_project_config(raw))


def test_e15_t01_review_fixes_nominal_dt_param_values_and_time_types(case: dict[str, Any]) -> None:
    """Review: cold `update_from_time` needs `nominal_dt`; the hash covers values; times are normalized."""
    filt = ExpandedOnlineFilter(case["fit"])
    with pytest.raises(ConfigError, match="nominal_dt"):
        filt.update_from_time(filt.start(), 0, case["y"][0], {k: v[0] for k, v in case["z"].items()})
    shifted = dataclasses.replace(case["fit"], params={k: v + 1e-9 for k, v in case["fit"].params.items()})
    assert spec_hash(shifted) != spec_hash(case["fit"])
    from hsmmkit.online.filter import _normalize_time

    assert _normalize_time("2020-01-02") == np.datetime64("2020-01-02", "ns")
    assert _normalize_time(np.datetime64("2020-01-02", "D")).dtype == np.dtype("datetime64[ns]")
    assert _normalize_time(3) == 3
