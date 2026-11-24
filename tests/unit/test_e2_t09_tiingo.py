"""E2-T09: tiingo adapter against a recorded (synthetic) JSON cassette; no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finance"))
from provider_contract import check_price_contract, check_window_inclusive

from regimekit.finance.providers import TiingoProvider, get_provider
from regimekit.finance.providers.tiingo import TiingoQuota
from regimekit.utils.errors import ProviderError

CASSETTE = Path(__file__).resolve().parents[1] / "fixtures" / "cassettes" / "tiingo" / "SPY.json"
KEY = "TIINGO-SECRET-KEY-7"
T0 = 1_719_792_000.0  # 2024-07-01T00:00:00Z


pytestmark = pytest.mark.finance


class WallClock:
    """Settable wall clock (epoch seconds)."""

    def __init__(self, now: float = T0) -> None:
        """Start at `now`."""
        self.now = now

    def __call__(self) -> float:
        """Current epoch seconds."""
        return self.now


class CassetteDownloader:
    """Replays `SPY.json` for `SPY`; other tickers are unknown (Tiingo's 404)."""

    def __init__(self, fail_first: int = 0) -> None:
        """Optionally fail the first `fail_first` calls with a retryable `ProviderError`."""
        self.calls: list[tuple[str, str, str | None, str]] = []
        self._fail_first = fail_first

    def __call__(self, ticker: str, start: str, end: str | None, api_key: str) -> list[dict[str, Any]] | None:
        """Record the call and return the recorded rows, or `None` for unknown tickers."""
        self.calls.append((ticker, start, end, api_key))
        if len(self.calls) <= self._fail_first:
            raise ProviderError("temporary failure")
        if ticker != "SPY":
            return None
        rows: list[dict[str, Any]] = json.loads(CASSETTE.read_text(encoding="utf-8"))
        return rows


def _provider(
    tmp_path: Path, downloader: CassetteDownloader | None = None, **kwargs: object
) -> TiingoProvider:
    options: dict[str, object] = {
        "api_key": KEY,
        "quota_dir": tmp_path,
        "clock": WallClock(),
        "max_per_hour": 10_000,
        "max_per_day": 10_000,
        "sleep": lambda s: None,
    }
    options.update(kwargs)
    return TiingoProvider(downloader=downloader or CassetteDownloader(), **options)  # type: ignore[arg-type]


def _rows() -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(CASSETTE.read_text(encoding="utf-8")))
    frame.index = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    return frame


# --- AC1: contract suite ------------------------------------------------------------------------------


def test_e2_t09_passes_contract_suite_daily_and_resampled(tmp_path: Path) -> None:
    """AC1: recorded responses pass the E2-T01 contract suite (`1d`, and resampled `1wk`/`1mo`)."""
    p = _provider(tmp_path)
    check_price_contract(p, ["SPY"], "2024-02-01", "2024-04-30")
    check_price_contract(p, ["SPY"], "2024-01-02", "2024-06-28", interval="1wk")
    check_price_contract(p, ["SPY"], "2024-01-02", "2024-06-28", interval="1mo")
    check_window_inclusive(p, "SPY", "2024-02-05", "2024-04-26")


def test_e2_t09_adjusted_selects_adjclose_else_close(tmp_path: Path) -> None:
    """`adjusted=True` -> `adjClose`, `adjusted=False` -> `close`, at the recorded dates (UTC midnight)."""
    rows = _rows().loc["2024-02-01":"2024-04-30"]
    p = _provider(tmp_path)
    adj = p.fetch_prices(["SPY"], "2024-02-01", "2024-04-30", "1d", True)
    raw = p.fetch_prices(["SPY"], "2024-02-01", "2024-04-30", "1d", False)
    np.testing.assert_array_equal(adj.prices[:, 0], rows["adjClose"].to_numpy(dtype="float64"))
    np.testing.assert_array_equal(raw.prices[:, 0], rows["close"].to_numpy(dtype="float64"))
    np.testing.assert_array_equal(adj.times, rows.index.to_numpy(dtype="datetime64[ns]"))


def test_e2_t09_weekly_is_last_close_of_each_week(tmp_path: Path) -> None:
    """Weekly bars are the last daily close of each week, labelled with that trading date."""
    obs = _provider(tmp_path).fetch_prices(["SPY"], "2024-01-02", "2024-06-28", "1wk", True)
    daily = _rows()["adjClose"]
    expected = daily.groupby(daily.index.to_period("W-SUN")).tail(1)
    np.testing.assert_array_equal(obs.times, expected.index.to_numpy(dtype="datetime64[ns]"))
    np.testing.assert_array_equal(obs.prices[:, 0], expected.to_numpy(dtype="float64"))
    assert obs.meta["resampled_from"].startswith("1d")


def test_e2_t09_monthly_is_last_close_of_each_month(tmp_path: Path) -> None:
    """Monthly bars are the last daily close of each calendar month (six bars for Jan..Jun)."""
    obs = _provider(tmp_path).fetch_prices(["SPY"], "2024-01-02", "2024-06-28", "1mo", False)
    daily = _rows()["close"]
    expected = daily.groupby(daily.index.to_period("M")).tail(1)
    assert len(expected) == 6
    np.testing.assert_array_equal(obs.times, expected.index.to_numpy(dtype="datetime64[ns]"))
    np.testing.assert_array_equal(obs.prices[:, 0], expected.to_numpy(dtype="float64"))


def test_e2_t09_intraday_and_series_are_refused(tmp_path: Path) -> None:
    """`1h` and `fetch_series` raise `ProviderError`; the adapter declares no intraday support."""
    p = _provider(tmp_path)
    assert p.supports_intraday is False
    with pytest.raises(ProviderError, match="1h"):
        p.fetch_prices(["SPY"], "2024-02-01", "2024-04-30", "1h", True)
    with pytest.raises(ProviderError, match="prices only"):
        p.fetch_series("VIXCLS", "2024-02-01", "2024-04-30")


# --- AC2: limits and key ------------------------------------------------------------------------------


def test_e2_t09_hourly_limit_enforced_with_fake_clock(tmp_path: Path) -> None:
    """AC2: 50 requests within an hour pass, the 51st is refused, and one hour later it passes again."""
    clock = WallClock()
    dl = CassetteDownloader()
    p = _provider(tmp_path, dl, clock=clock, max_per_hour=50)
    for _ in range(50):
        p.fetch_prices(["SPY"], "2024-02-01", "2024-02-29", "1d", True)
        clock.now += 60.0  # 50 requests over 50 minutes
    with pytest.raises(ProviderError, match="this hour"):
        p.fetch_prices(["SPY"], "2024-02-01", "2024-02-29", "1d", True)
    assert len(dl.calls) == 50  # refused before any network call
    clock.now = T0 + 3600.0  # the first request has left the rolling hour
    p.fetch_prices(["SPY"], "2024-02-01", "2024-02-29", "1d", True)
    assert len(dl.calls) == 51


def test_e2_t09_quota_is_persisted_across_instances(tmp_path: Path) -> None:
    """The hourly count lives in the cache dir, so a new adapter instance sees earlier requests."""
    clock = WallClock()
    for _ in range(3):
        _provider(tmp_path, clock=clock, max_per_hour=3).fetch_prices(
            ["SPY"], "2024-02-01", "2024-02-29", "1d", True
        )
    with pytest.raises(ProviderError, match="this hour"):
        _provider(tmp_path, clock=clock, max_per_hour=3).fetch_prices(
            ["SPY"], "2024-02-01", "2024-02-29", "1d", True
        )
    assert (tmp_path / "tiingo_quota.json").exists()


def test_e2_t09_daily_limit_is_rolling_24h(tmp_path: Path) -> None:
    """The daily limit counts requests of the last 24 h, independent of the hourly window."""
    clock = WallClock()
    quota = TiingoQuota(tmp_path / "q.json", clock, max_per_hour=1000, max_per_day=5)
    for _ in range(5):
        quota.charge(["SPY"])
        clock.now += 2 * 3600.0
    with pytest.raises(ProviderError, match="today"):
        quota.charge(["SPY"])
    clock.now = T0 + 86400.0  # first request is 24 h old
    quota.charge(["SPY"])


def test_e2_t09_unique_symbols_per_month(tmp_path: Path) -> None:
    """Distinct symbols are limited per UTC month; repeating a known symbol is free; a new month resets."""
    clock = WallClock()
    quota = TiingoQuota(
        tmp_path / "q.json", clock, max_per_hour=1000, max_per_day=1000, max_symbols_per_month=2
    )
    quota.charge(["SPY", "QQQ"])
    quota.charge(["SPY"])
    with pytest.raises(ProviderError, match="distinct symbols"):
        quota.charge(["IWM"])
    clock.now = T0 + 31 * 86400.0  # 2024-08-01
    quota.charge(["IWM"])


def test_e2_t09_refused_charge_charges_nothing(tmp_path: Path) -> None:
    """A request batch that would exceed a limit is refused whole and leaves the counters unchanged."""
    clock = WallClock()
    quota = TiingoQuota(tmp_path / "q.json", clock, max_per_hour=3)
    quota.charge(["A", "B"])
    with pytest.raises(ProviderError):
        quota.charge(["C", "D"])
    state = json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))
    assert len(state["requests"]) == 2
    assert state["symbols"] == {"2024-07": ["A", "B"]}
    quota.charge(["C"])


def test_e2_t09_corrupt_quota_file_starts_fresh(tmp_path: Path) -> None:
    """An unreadable quota file is treated as empty rather than crashing."""
    (tmp_path / "q.json").write_text("{not json", encoding="utf-8")
    TiingoQuota(tmp_path / "q.json", WallClock(), max_per_hour=1).charge(["SPY"])


def test_e2_t09_missing_key_raises_without_network(tmp_path: Path) -> None:
    """AC2: no `TIINGO_API_KEY` -> `ProviderError` with instructions; no download, no quota file."""
    dl = CassetteDownloader()
    p = TiingoProvider(downloader=dl, env={}, quota_dir=tmp_path, sleep=lambda s: None)
    with pytest.raises(ProviderError, match="TIINGO_API_KEY"):
        p.fetch_prices(["SPY"], "2024-02-01", "2024-04-30", "1d", True)
    assert dl.calls == []
    assert not (tmp_path / "tiingo_quota.json").exists()


def test_e2_t09_key_from_env_and_never_logged(tmp_path: Path) -> None:
    """The key is read from the environment, passed to the downloader, and appears in no log or error."""
    dl = CassetteDownloader()
    p = TiingoProvider(
        downloader=dl,
        env={"TIINGO_API_KEY": KEY},
        quota_dir=tmp_path,
        clock=WallClock(),
        sleep=lambda s: None,
    )
    with capture_logs() as logs:
        p.fetch_prices(["SPY"], "2024-02-01", "2024-04-30", "1d", True)
        with pytest.raises(ProviderError) as info:
            p.fetch_prices(["NOPE"], "2024-02-01", "2024-04-30", "1d", True)
    assert dl.calls[0][3] == KEY
    assert KEY not in repr(logs)
    assert KEY not in str(info.value)
    assert KEY not in (tmp_path / "tiingo_quota.json").read_text(encoding="utf-8")
    assert any(e.get("provider") == "tiingo" and e.get("tos") for e in logs)


# --- other behaviour -------------------------------------------------------------------------------


def test_e2_t09_unknown_ticker_is_not_retried(tmp_path: Path) -> None:
    """Tiingo's 404 (`None`) fails once with a clear error."""
    dl = CassetteDownloader()
    with pytest.raises(ProviderError, match="does not know"):
        _provider(tmp_path, dl).fetch_prices(["NOPE"], "2024-02-01", "2024-04-30", "1d", True)
    assert len(dl.calls) == 1


def test_e2_t09_transient_failures_are_retried(tmp_path: Path) -> None:
    """Download failures are retried (shared `with_retries`)."""
    dl = CassetteDownloader(fail_first=2)
    obs = _provider(tmp_path, dl).fetch_prices(["SPY"], "2024-02-01", "2024-04-30", "1d", True)
    assert len(dl.calls) == 3
    assert obs.prices.shape[0] > 0


def test_e2_t09_empty_symbols_and_window_raise(tmp_path: Path) -> None:
    """No symbols, or a window without rows, is a `ProviderError`."""
    p = _provider(tmp_path)
    with pytest.raises(ProviderError, match="no symbols"):
        p.fetch_prices([], "2024-02-01", "2024-04-30", "1d", True)
    with pytest.raises(ProviderError, match="no rows"):
        p.fetch_prices(["SPY"], "2030-01-01", "2030-02-01", "1d", True)


def test_e2_t09_registered(tmp_path: Path) -> None:
    """The adapter is registered as `tiingo`."""
    assert isinstance(get_provider("tiingo", api_key=KEY, quota_dir=tmp_path), TiingoProvider)
