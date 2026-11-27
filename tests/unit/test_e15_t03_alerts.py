"""E15-T03: alert rules, events (AC1, AC2, AC4) and the config fields they use."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.config.schema import parse_project_config
from hsmmkit.online import AlertEvent, ExpandedOnlineFilter, load_state, save_state
from hsmmkit.online.alerts import HYSTERESIS, AlertRule, evaluate
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw
from tests.helpers.engine_fixtures import fit_of, make_scenario

HYST = AlertRule(name="turbulent", regime=1, p_on=0.8, p_off=0.5)


def _run(rule: AlertRule, path: list[float]) -> list[AlertEvent]:
    active = np.zeros(1, dtype=bool)
    events: list[AlertEvent] = []
    for t, p in enumerate(path, start=1):
        active, ev = evaluate([rule], active, t, t, np.array([1 - p, p]), None, None)
        events.extend(ev)
    return events


def test_e15_t03_one_switch_gives_one_on_and_one_off() -> None:
    """AC1: calm -> turbulent -> calm gives exactly one `on` and one `off`."""
    path = [0.05] * 20 + list(np.linspace(0.05, 0.99, 15)) + [0.99] * 20 + list(np.linspace(0.99, 0.02, 15))
    events = _run(HYST, path)
    assert [e.kind for e in events] == ["on", "off"]
    assert events[0].value >= 0.8
    assert events[1].value < 0.5
    assert events[0].t < events[1].t


def test_e15_t03_no_flapping_on_a_noisy_path_that_stays_above_p_off() -> None:
    """AC1: crossing `p_on` again and again without reaching `p_off` gives a single `on`."""
    rng = np.random.default_rng(1)
    path = [0.1] * 10 + [float(0.8 + 0.1 * np.sign(rng.normal()) * rng.random()) for _ in range(200)]
    path = [min(max(p, 0.55), 0.99) if i >= 10 else p for i, p in enumerate(path)]
    assert sum(a >= 0.8 > b for a, b in zip(path[1:], path[:-1], strict=True)) > 5  # crosses p_on repeatedly
    assert [e.kind for e in _run(HYST, path)] == ["on"]


def test_e15_t03_events_round_trip_through_json() -> None:
    """AC4: `to_dict` is JSON-serializable and `from_dict` restores the event."""
    for time in (np.datetime64("2020-03-04T05:06:07", "ns"), 17, 2.5, None):
        ev = AlertEvent(time, 12, "turbulent", "on", 1, 0.93, 0.8)
        blob = json.loads(json.dumps(ev.to_dict()))
        back = AlertEvent.from_dict(blob)
        assert back == ev
    assert AlertEvent.from_dict(AlertEvent(np.int64(3), 1, "a", "off", 0, 0.1, 0.5).to_dict()).time == 3


def test_e15_t03_rule_defaults_and_validation() -> None:
    """Names default to `alert<i>`; the schema accepts `name`, `use`, `p_min` and checks them."""
    rule = AlertRule.from_config({"regime": 1, "p_on": 0.8, "p_off": 0.5}, 3)
    assert rule.name == "alert3"
    assert rule.kind == "probability"
    dur = AlertRule.from_config({"regime": 0, "remaining_duration_below": 2.0, "p_min": 0.7}, 0)
    assert dur.kind == "duration"
    assert dur.p_min == 0.7
    assert HYSTERESIS == 1.2


def _cfg(tmp_path: Path, online: dict[str, Any]) -> Any:  # noqa: ANN401
    raw = core_raw(tmp_path)
    raw["online"] = online
    return parse_project_config(raw)


def test_e15_t03_config_fields_are_validated(tmp_path: Path) -> None:
    """`name` unique, `use: smoothed` needs a lag, `p_min` in (0, 1]; the `online.alerts` key builds."""
    from hsmmkit.config import gates

    ok = _cfg(
        tmp_path,
        {
            "lag": 3,
            "alerts": [
                {"name": "a", "regime": 1, "p_on": 0.8, "p_off": 0.5, "use": "smoothed"},
                {"name": "b", "regime": 0, "remaining_duration_below": 0.1, "p_min": 0.6},
            ],
        },
    )
    gates.check(ok)
    assert ok.online.alerts[1].p_min == 0.6
    assert "online.alerts" not in gates.GATES
    with pytest.raises(ConfigError, match="unique"):
        _cfg(tmp_path, {"alerts": [{"name": "a", "regime": 0, "p_on": 0.8, "p_off": 0.5}] * 2})
    with pytest.raises(ConfigError, match=r"needs online\.lag"):
        _cfg(tmp_path, {"alerts": [{"regime": 0, "p_on": 0.8, "p_off": 0.5, "use": "smoothed"}]})
    with pytest.raises(ConfigError):
        _cfg(tmp_path, {"alerts": [{"regime": 0, "remaining_duration_below": 1.0, "p_min": 0.0}]})


# --- filter-level behaviour ---------------------------------------------------------------


def _two_regime(rate: float, r: int, t_len: int) -> Any:  # noqa: ANN401
    sc = make_scenario(2, [r, r], 1, t_len, seed=1)
    theta = dict(sc.theta)
    theta["emission.mean"] = jnp.asarray([0.0, 100.0])
    theta["emission.scale"] = jnp.asarray([0.05, 0.05])
    theta["duration.rate"] = jnp.asarray([rate, rate])
    return dataclasses.replace(sc, theta=theta)


def _fit(sc: Any) -> Any:  # noqa: ANN401
    return fit_of(sc)


def test_e15_t03_probability_alert_through_the_filter_and_flags_survive_save_load(tmp_path: Path) -> None:
    """A calm -> turbulent -> calm stream raises one `on` and one `off`; the flags persist (AC1, AC4)."""
    dt = 1 / 252
    sc = _two_regime(rate=2.0, r=1, t_len=10)  # sticky regimes: mean sojourn ~ 1/(rate dt) ... see below
    theta = dict(sc.theta)
    theta["duration.rate"] = jnp.asarray([0.5, 0.5])  # mean ~ 500 steps: regimes persist
    sc = dataclasses.replace(sc, theta=theta)
    rule = {"name": "turb", "regime": 1, "p_on": 0.8, "p_off": 0.5}
    filt = ExpandedOnlineFilter(_fit(sc), alerts=[rule])
    rng = np.random.default_rng(3)
    sd = 0.05 * np.sqrt(dt)
    mean = np.r_[np.zeros(60), 100 * dt * np.ones(40), np.zeros(60)]
    y = (mean + rng.normal(0, sd, size=mean.shape))[:, None]
    dts = np.full(y.shape[0], dt)

    state, _, events = filt.update_many(filt.start(), y, dts)
    assert [e.kind for e in events] == ["on", "off"]
    assert 60 < events[0].t <= 70
    assert 100 < events[1].t <= 110
    assert bool(np.asarray(state.alerts)[0]) is False

    # the flags persist: save while the alert is active, restore, and the `off` event still comes
    cut = 80
    mid, _, first = filt.update_many(filt.start(), y[:cut], dts[:cut])
    assert [e.kind for e in first] == ["on"]
    save_state(mid, tmp_path / "s")
    restored = load_state(tmp_path / "s", _fit(sc))
    assert np.asarray(restored.alerts).tolist() == [True]
    _, _, rest = filt.update_many(restored, y[cut:], dts[cut:])
    assert [e.kind for e in rest] == ["off"]
    assert rest[0].t == events[1].t  # same step number as the uninterrupted run

    # stepwise `update` returns the events of its step as the last element
    state = filt.start()
    seen: list[str] = []
    for i in range(y.shape[0]):
        state, _, ev = filt.update(state, y[i], dt)
        seen += [e.kind for e in ev]
    assert seen == ["on", "off"]


def test_e15_t03_remaining_duration_fires_at_the_analytic_crossing() -> None:
    """AC2: deterministic NB chain (advance prob ~ 1): regime 1 lasts exactly r steps, remaining = r - k."""
    dt = 1 / 252
    r = 10
    sc = _two_regime(rate=1e4, r=r, t_len=10)
    # regimes alternate every `r` steps; regime 1 occupies code indices 10..19, 30..39, ...
    n = 60
    regime = (np.arange(n) // r) % 2
    y = (np.where(regime == 1, 100.0 * dt, 0.0))[:, None]
    d_steps = 4
    rule = {"name": "ending", "regime": 1, "remaining_duration_below": d_steps * dt, "p_min": 0.5}
    filt = ExpandedOnlineFilter(_fit(sc), alerts=[rule], nominal_dt=dt)
    _, _, events = filt.update_many(filt.start(), y, np.full(n, dt))
    ons = [e.t for e in events if e.kind == "on"]
    # entering regime 1 at code index e = 10 (+20 k); remaining after k steps in it is r - k steps,
    # `< d` from k = r - d + 1 = 7: code index e + 7, i.e. step number (1-based) e + 8
    expected = [10 + (r - d_steps + 1) + 1 + 20 * k for k in range(len(ons))]
    assert len(ons) >= 2
    for got, want in zip(ons, expected, strict=True):
        assert abs(got - want) <= 3
    offs = [e.t for e in events if e.kind == "off"]
    assert offs
    assert all(
        e.regime == 1 and e.threshold in (d_steps * dt, HYSTERESIS * d_steps * dt, 0.5) for e in events
    )


def test_e15_t03_state_without_matching_rules_is_rejected() -> None:
    """A state made by a filter with other rules cannot be used (flag count mismatch)."""
    sc = _two_regime(rate=0.5, r=1, t_len=10)
    plain = ExpandedOnlineFilter(_fit(sc))
    ruled = ExpandedOnlineFilter(_fit(sc), alerts=[{"regime": 1, "p_on": 0.8, "p_off": 0.5}])
    with pytest.raises(ConfigError, match="alert flags"):
        ruled.update(plain.start(), [0.0], 1.0)
    with pytest.raises(ConfigError, match="regime must be"):
        ExpandedOnlineFilter(_fit(sc), alerts=[{"regime": 5, "p_on": 0.8, "p_off": 0.5}])
    with pytest.raises(ConfigError, match="smoothed needs"):
        ExpandedOnlineFilter(_fit(sc), alerts=[{"regime": 1, "p_on": 0.8, "p_off": 0.5, "use": "smoothed"}])


def test_e15_t03_remaining_steps_equal_the_offline_expectation_on_the_filtered_distribution() -> None:
    """The in-step formula is `expected_remaining_steps` evaluated on `softmax(log_alpha)`."""
    import jax

    from hsmmkit.engine.augmented import expected_remaining_steps
    from hsmmkit.engine.base import Posterior
    from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
    from hsmmkit.online.filter import _remaining_steps

    sc = make_scenario(3, [3, 4, 2], 1, 80, n_cov_pi=2, n_cov_d=2, seed=9)
    d = sc.data
    log_alpha, _ = ExpandedHMMEngine().forward(sc.theta, sc.spec, sc.space, d)
    probs = jax.nn.softmax(log_alpha, axis=-1)
    post = Posterior(log_alpha, jnp.zeros_like(probs), probs, jnp.zeros(()), sc.space.index)
    ref = expected_remaining_steps(post, sc.spec, sc.theta, d.dt, d.cov_duration)
    for t in (0, 5, 40, 79):
        got = _remaining_steps(sc.theta, sc.spec, sc.space, log_alpha[t], d.dt[t], d.cov_duration[t])
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref[t]), rtol=1e-12, atol=1e-12)
