"""E15-T02: fixed-lag smoothing equals the offline smoother on the same prefix (AC1, AC2, AC4)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.online import ExpandedOnlineFilter, load_state, save_state
from hsmmkit.online.buffer import smoothed_series
from hsmmkit.utils.errors import ConfigError
from tests.helpers.engine_fixtures import fit_of, make_scenario

T = 2000
LAG = 12
CHECK_AT = (12, 13, 40, 333, 777, 1500, 1999)


@pytest.fixture(scope="module")
def case() -> dict[str, Any]:
    """Scenario with covariates, masked steps and irregular gaps, plus the online replay."""
    sc = make_scenario(3, [3, 3, 3], 1, T, n_cov_pi=2, n_cov_d=2, seed=21)
    rng = np.random.default_rng(8)
    mask = np.ones((T, 1), dtype=bool)
    mask[rng.random(T) < 0.03] = False
    mask[0] = True
    dt = np.asarray(sc.data.dt).copy()
    dt[rng.random(T) < 0.1] *= 2
    data = dataclasses.replace(sc.data, mask=jnp.asarray(mask), dt=jnp.asarray(dt))
    y = np.asarray(data.observations).copy()
    y[~mask] = np.nan
    z = {"transition": np.asarray(data.cov_transition), "duration": np.asarray(data.cov_duration)}
    fit = fit_of(sc)
    filt = ExpandedOnlineFilter(fit, lag=LAG)
    state, filtered, smoothed = filt.update_many(filt.start(), y, dt, z)
    return {
        "sc": sc, "data": data, "fit": fit, "filt": filt, "y": y, "dt": dt, "z": z,
        "state": state, "filtered": filtered, "smoothed": np.asarray(smoothed),
    }  # fmt: skip


def _prefix_smoothed(case: dict[str, Any], t: int) -> np.ndarray:
    sc = case["sc"]
    prefix = case["data"].window(0, t + 1)
    post = ExpandedHMMEngine().forward_backward(sc.theta, sc.spec, sc.space, prefix)
    return np.asarray(post.regime_smoothed())


def test_e15_t02_streamed_smoothed_equals_the_offline_smoother_on_every_prefix(case: dict[str, Any]) -> None:
    """AC1: row t of the replay is the offline smoothed marginal at `t - L` of `window(0, t + 1)`."""
    assert np.isnan(case["smoothed"][:LAG]).all()
    for t in CHECK_AT:
        ref = _prefix_smoothed(case, t)[t - LAG]
        np.testing.assert_allclose(case["smoothed"][t], ref, rtol=0, atol=1e-12)


def test_e15_t02_smoothed_series_helper_matches_the_stream(case: dict[str, Any]) -> None:
    """The vectorized replay helper agrees with the streaming results at every step."""
    series = np.asarray(smoothed_series(case["fit"], case["data"], LAG))
    assert np.isnan(series[:LAG]).all()
    np.testing.assert_allclose(series[LAG:], case["smoothed"][LAG:], rtol=0, atol=1e-12)


def test_e15_t02_at_the_end_it_equals_the_full_offline_smoother(case: dict[str, Any]) -> None:
    """AC2: at `t = T` `smoothed_at` is the full-data smoother at `T - L`."""
    sc = case["sc"]
    full = ExpandedHMMEngine().forward_backward(sc.theta, sc.spec, sc.space, case["data"])
    got = np.asarray(case["filt"].smoothed_at(case["state"]))
    np.testing.assert_allclose(got, np.asarray(full.regime_smoothed())[T - 1 - LAG], rtol=0, atol=1e-12)
    # a shorter lag from the same buffer
    got5 = np.asarray(case["filt"].smoothed_at(case["state"], lag=5))
    np.testing.assert_allclose(got5, np.asarray(full.regime_smoothed())[T - 1 - 5], rtol=0, atol=1e-12)


def test_e15_t02_stepwise_update_and_filtered_unchanged_by_the_lag(case: dict[str, Any]) -> None:
    """Per-step `update` returns the same triple as the scan; the filtered output ignores the lag."""
    filt = case["filt"]
    plain = ExpandedOnlineFilter(case["fit"])
    _, plain_f = plain.update_many(plain.start(), case["y"], case["dt"], case["z"])
    np.testing.assert_array_equal(np.asarray(case["filtered"]), np.asarray(plain_f))
    state = filt.start()
    for t in range(LAG + 5):
        z = {k: v[t] for k, v in case["z"].items()}
        state, f, s = filt.update(state, case["y"][t], case["dt"][t], z)
        np.testing.assert_array_equal(np.asarray(s), case["smoothed"][t])
        np.testing.assert_array_equal(np.asarray(f), np.asarray(case["filtered"])[t])
    assert np.isfinite(np.asarray(s)).all()


def test_e15_t02_warm_start_fills_the_buffer_from_the_prefix(case: dict[str, Any]) -> None:
    """`start(initial)` with a lag reproduces the smoothed outputs of the uninterrupted stream."""
    filt = case["filt"]
    k = 500
    state = filt.start(case["data"].window(0, k))
    sl = slice(k, None)
    _, _, smoothed = filt.update_many(
        state, case["y"][sl], case["dt"][sl], {a: v[sl] for a, v in case["z"].items()}
    )
    np.testing.assert_allclose(np.asarray(smoothed), case["smoothed"][sl], rtol=0, atol=1e-12)
    short = filt.start(case["data"].window(0, 5))  # fewer than L + 1 steps: still filling
    assert np.isnan(np.asarray(filt.smoothed_at(short))).all()


def test_e15_t02_buffer_round_trips_through_save_and_load(case: dict[str, Any], tmp_path: Path) -> None:
    """AC4: the buffer is part of the saved state; continuing from it is bitwise identical."""
    filt = case["filt"]
    k = 900
    sl, rest = slice(0, k), slice(k, None)
    half, *_ = filt.update_many(
        filt.start(), case["y"][sl], case["dt"][sl], {a: v[sl] for a, v in case["z"].items()}
    )
    save_state(half, tmp_path / "s")
    restored = load_state(tmp_path / "s", case["fit"])
    assert restored.buffer is not None
    for name in ("ell", "z_pi", "z_d", "dt", "log_alpha", "head", "count"):
        np.testing.assert_array_equal(
            np.asarray(getattr(restored.buffer, name)), np.asarray(getattr(half.buffer, name))
        )
    _, f2, s2 = filt.update_many(
        restored, case["y"][rest], case["dt"][rest], {a: v[rest] for a, v in case["z"].items()}
    )
    np.testing.assert_array_equal(np.asarray(s2), case["smoothed"][rest])
    np.testing.assert_array_equal(np.asarray(f2), np.asarray(case["filtered"])[rest])


def test_e15_t02_lag_mismatch_and_bad_arguments(case: dict[str, Any]) -> None:
    """A state needs a filter of the same lag; `smoothed_at` rejects a lag beyond the buffer."""
    plain = ExpandedOnlineFilter(case["fit"])
    with pytest.raises(ConfigError, match="fixed-lag buffer"):
        plain.update(case["state"], case["y"][0], 1.0, {k: v[0] for k, v in case["z"].items()})
    with pytest.raises(ConfigError, match="lag"):
        case["filt"].smoothed_at(case["state"], lag=LAG + 1)
    with pytest.raises(ConfigError):
        ExpandedOnlineFilter(case["fit"], lag=-1)


def test_e15_t02_the_online_lag_key_builds(tmp_path: Path) -> None:
    """The `online.lag` gate row is gone."""
    from hsmmkit.config import gates
    from hsmmkit.config.schema import parse_project_config
    from tests.helpers.core_config import core_raw

    raw = core_raw(tmp_path)
    raw["online"] = {"lag": 5}
    gates.check(parse_project_config(raw))
    assert "online.lag" not in gates.GATES
