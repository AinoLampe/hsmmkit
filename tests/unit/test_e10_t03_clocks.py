"""E10-T03: the clock registry, the core clocks and the contract suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from regimekit.config.schema import parse_project_config
from regimekit.data.calendar import build_grid
from regimekit.data.contracts import ObservationSet
from regimekit.plugins import REGISTRY
from regimekit.testing.clock_contract import check_clock, fixture_times
from regimekit.utils.clock import days_to_steps, future_grid, get_clock, parse_clock_options
from regimekit.utils.errors import ConfigError

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

ROOT = Path(__file__).resolve().parents[2]
FRIDAY = np.datetime64("2024-01-05T00:00:00", "ns")


@pytest.mark.parametrize("name", ["steps", "calendar", "trading"])
def test_e10_t03_every_clock_passes_the_contract_suite(name: str) -> None:
    """AC5: `steps`, `calendar` and `trading` pass the clock contract."""
    check_clock(name)


def test_e10_t03_registry_holds_the_three_clocks_with_provenance() -> None:
    """The core registers `steps` and `calendar`; the finance package registers `trading`."""
    prov = {e.name: e.provenance for e in REGISTRY.entries("clock")}
    assert prov == {"calendar": "regimekit.core", "steps": "regimekit.core", "trading": "regimekit.finance"}


def test_e10_t03_steps_and_calendar_dt() -> None:
    """Steps and calendar dt."""
    times = fixture_times()
    assert get_clock("steps").dt(times, 4, parse_clock_options("steps", None)).tolist() == [1.0] * 4
    cal = get_clock("calendar").dt(times, times.shape[0] - 1, parse_clock_options("calendar", None))
    assert cal[0] == pytest.approx(1 / 365.25)
    with pytest.raises(ConfigError):
        get_clock("calendar").dt(None, 3, parse_clock_options("calendar", None))


def test_e10_t03_trading_dt_uses_252_and_bars() -> None:
    """Trading dt uses 252 and bars."""
    clock = get_clock("trading")
    dt = clock.dt(None, 3, parse_clock_options("trading", {"bars_per_day": 4}))
    assert dt.tolist() == [1.0 / 1008] * 3


def test_e10_t03_future_grid_per_clock() -> None:
    """`horizon_times` per clock: weekend skipping for the weekday clocks, none for `steps`."""
    t_tr, dt_tr = future_grid(FRIDAY, 3, "trading")
    assert [str(t)[:10] for t in t_tr] == ["2024-01-08", "2024-01-09", "2024-01-10"]
    assert dt_tr.tolist() == [1 / 252] * 3
    t_cal, dt_cal = future_grid(FRIDAY, 2, "calendar")
    assert [str(t)[:10] for t in t_cal] == ["2024-01-08", "2024-01-09"]
    assert dt_cal[0] == pytest.approx(3 / 365.25)
    t_st, dt_st = future_grid(FRIDAY, 2, "steps", None, fixture_times())
    assert [str(t)[:10] for t in t_st] == ["2024-01-06", "2024-01-07"]
    assert dt_st.tolist() == [1.0, 1.0]
    ints, dt_int = future_grid(10, 3, "steps")
    assert ints.tolist() == [11, 12, 13]
    assert dt_int.tolist() == [1.0] * 3


def test_e10_t03_future_grid_rejects_bad_arguments() -> None:
    """Future grid rejects bad arguments."""
    with pytest.raises(ConfigError):
        future_grid(FRIDAY, 0, "trading")
    with pytest.raises(ConfigError) as err:
        future_grid(FRIDAY, 2, "trading", {"bars_per_day": 0})
    assert err.value.path == "data.clock_options.bars_per_day"


def test_e10_t03_days_to_steps_dispatches_through_the_clock() -> None:
    """Days to steps dispatches through the clock."""
    assert days_to_steps(60.0, "steps", 1.0) == 60.0
    assert days_to_steps(252.0, "trading", 1 / 252) == pytest.approx(252.0)
    assert days_to_steps(365.25, "calendar", 1 / 365.25) == pytest.approx(365.25)


def test_e10_t03_unknown_clock_lists_the_registered_names() -> None:
    """Unknown clock lists the registered names."""
    with pytest.raises(ConfigError) as err:
        get_clock("lunar")
    assert err.value.path == "data.clock"
    assert "calendar, steps, trading" in str(err.value)


def test_e10_t03_build_grid_is_clock_agnostic() -> None:
    """Build grid is clock agnostic."""
    times = np.array(["2024-01-01", "2024-01-02", "2024-01-05"], dtype="datetime64[ns]")
    obs = ObservationSet(("X",), times, np.array([[1.0], [2.0], [3.0]]), "t", "s")
    assert build_grid(obs, "drop", clock="steps").dt.tolist() == [1.0, 1.0]
    assert build_grid(obs, "drop", clock="calendar").irregular is True
    with pytest.raises(ConfigError) as err:
        build_grid(obs, "drop", clock="trading", clock_options={"bars_per_day": 0})
    assert err.value.path == "data.clock_options.bars_per_day"


def test_e10_t03_core_registry_without_the_finance_package_rejects_the_trading_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3: with `trading` unregistered, `data.clock: trading` is `ConfigError(data.clock)`.

    The `HSMM_CORE_ONLY` finder cannot show this before E10-T07b (`regimekit` still imports the finance
    package at import time), so the finance entry is removed from the registry here.
    """
    monkeypatch.delitem(REGISTRY._entries["clock"], "trading")
    with pytest.raises(ConfigError) as err:
        parse_project_config(_core_with_clock("trading"))
    assert err.value.path == "data.clock"
    assert "registered: calendar, steps" in str(err.value)
    assert parse_project_config(_core_with_clock("calendar")).data.clock == "calendar"


def _core_with_clock(clock: str) -> dict[str, object]:
    """The v2 Appendix A without finance-only keys, on the given clock."""
    from tests.unit.test_e10_t02b_schema_core import _core_raw

    raw = _core_raw()
    raw["data"]["clock"] = clock  # type: ignore[index]
    return raw


def test_e10_t03_clock_options_bars_per_day_zero_is_rejected_in_a_config() -> None:
    """AC4: `clock_options: {bars_per_day: 0}` -> ConfigError(path="data.clock_options.bars_per_day")."""
    from regimekit.finance import FinanceConfig
    from tests.helpers.finance_config import load_finance

    raw = load_finance(ROOT / "tests" / "fixtures" / "configs" / "appendix_a.yaml").model_dump(mode="json")
    raw["data"]["clock_options"] = {"bars_per_day": 0}
    with pytest.raises(ConfigError) as err:
        parse_project_config(raw, FinanceConfig)
    assert err.value.path == "data.clock_options.bars_per_day"
