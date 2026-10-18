"""E2-T04: fred and cboe covariate adapters against recorded (synthetic) responses; no network."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finance"))
from provider_contract import check_series_contract

from regimekit.finance.providers import CboeProvider, FredProvider, get_provider
from regimekit.finance.providers.base import RateLimiter
from regimekit.finance.providers.cboe import DEFAULT_URLS, parse_cboe_csv
from regimekit.finance.providers.fred import MAX_REQUESTS_PER_MINUTE, parse_values
from regimekit.utils.errors import ProviderError

CASSETTES = Path(__file__).resolve().parents[1] / "fixtures" / "cassettes"
KEY = "FRED-SECRET-KEY-42"


class FakeClock:
    """Monotonic clock whose `sleep` advances time and records every wait."""

    def __init__(self) -> None:
        """Start at t = 0."""
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        """Current time in seconds."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record the wait and advance time by it."""
        self.sleeps.append(seconds)
        self.now += seconds


class CassetteFredClient:
    """Replays `fred/<id>.{series,observations}.json`; unknown ids do not exist, as on FRED."""

    def __init__(self, fail_first: int = 0, frequency_short: str | None = None) -> None:
        """Optionally fail the first `fail_first` requests with a retryable `ProviderError`."""
        self.requests: list[tuple[str, str]] = []
        self._fail_first = fail_first
        self._frequency_short = frequency_short

    def _tick(self, kind: str, series_id: str) -> None:
        self.requests.append((kind, series_id))
        if len(self.requests) <= self._fail_first:
            raise ProviderError("temporary failure")

    def series_info(self, series_id: str, api_key: str) -> Mapping[str, object] | None:
        """`seriess[0]` of the recorded `fred/series` response."""
        self._tick("info", series_id)
        assert api_key == KEY
        path = CASSETTES / "fred" / f"{series_id}.series.json"
        if not path.exists():
            return None
        info = dict(json.loads(path.read_text(encoding="utf-8"))["seriess"][0])
        if self._frequency_short is not None:
            info["frequency_short"] = self._frequency_short
        return info

    def observations(self, series_id: str, start: str, end: str | None, api_key: str) -> pd.DataFrame:
        """Recorded observations filtered to `[start, end]` like FRED's `observation_start/end`."""
        self._tick("observations", series_id)
        path = CASSETTES / "fred" / f"{series_id}.observations.json"
        obs = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["observations"])
        keep = (obs["date"] >= start) & (obs["date"] <= (end or "9999-12-31"))
        return obs.loc[keep, ["date", "value"]]


class NoNetworkClient:
    """Fails the test if any request is attempted."""

    def series_info(self, series_id: str, api_key: str) -> Mapping[str, object] | None:
        """Never allowed."""
        raise AssertionError("network used")

    def observations(self, series_id: str, start: str, end: str | None, api_key: str) -> pd.DataFrame:
        """Never allowed."""
        raise AssertionError("network used")


class CassetteCboeDownloader:
    """Maps URLs to the recorded CSV files by file name."""

    def __init__(self, fail_first: int = 0) -> None:
        """Optionally fail the first `fail_first` calls with a retryable `ProviderError`."""
        self.urls: list[str] = []
        self._fail_first = fail_first

    def __call__(self, url: str) -> str:
        """Return the CSV text of the cassette named like the URL's last path segment."""
        self.urls.append(url)
        if len(self.urls) <= self._fail_first:
            raise ProviderError("temporary failure")
        return (CASSETTES / "cboe" / url.rsplit("/", 1)[-1]).read_text(encoding="utf-8")


def _fred(client: object | None = None, clock: FakeClock | None = None, **kwargs: object) -> FredProvider:
    clock = clock or FakeClock()
    options: dict[str, object] = {
        "api_key": KEY,
        "limiter": RateLimiter(MAX_REQUESTS_PER_MINUTE, 60.0, clock=clock, sleep=clock.sleep),
        "sleep": lambda s: None,
    }
    options.update(kwargs)
    return FredProvider(client=client or CassetteFredClient(), **options)  # type: ignore[arg-type]


def _cboe(downloader: CassetteCboeDownloader | None = None, **kwargs: object) -> CboeProvider:
    return CboeProvider(downloader=downloader or CassetteCboeDownloader(), sleep=lambda s: None, **kwargs)  # type: ignore[arg-type]


# --- AC1: contract suite ------------------------------------------------------------------------------


@pytest.mark.parametrize("series_id", ["VIXCLS", "T10Y2Y"])
def test_e2_t04_fred_passes_series_contract(series_id: str) -> None:
    """AC1: recorded `VIXCLS` / `T10Y2Y` responses pass the E2-T01 `fetch_series` contract."""
    check_series_contract(_fred(), series_id, "2024-01-02", "2024-02-29", unknown="NOPE")


@pytest.mark.parametrize("series_id", ["VIX_History.csv", "VVIX_History.csv"])
def test_e2_t04_cboe_passes_series_contract(series_id: str) -> None:
    """AC1: the recorded `VIX_History.csv` (and the single-column VVIX layout) pass the contract."""
    check_series_contract(_cboe(), series_id, "2024-01-02", "2024-02-29", unknown="NOPE.csv")


def test_e2_t04_fred_values_dates_and_frequency() -> None:
    """Values equal the recorded strings, dates are UTC midnight, frequency comes from metadata."""
    ser = _fred().fetch_series("VIXCLS", "2024-01-02", "2024-02-29")
    recorded = json.loads((CASSETTES / "fred" / "VIXCLS.observations.json").read_text(encoding="utf-8"))
    rows = [o for o in recorded["observations"] if o["value"] != "."]
    finite = ~np.isnan(ser.values)
    np.testing.assert_array_equal(ser.values[finite], [float(o["value"]) for o in rows])
    np.testing.assert_array_equal(
        ser.times[finite], np.array([o["date"] for o in rows], dtype="datetime64[ns]")
    )
    assert np.all(ser.times == ser.times.astype("datetime64[D]").astype("datetime64[ns]"))
    assert ser.frequency == "d"
    assert ser.name == "VIXCLS"


def test_e2_t04_fred_window_is_inclusive_and_clipped() -> None:
    """`[start, end]` keeps both boundary dates and nothing outside."""
    ser = _fred().fetch_series("T10Y2Y", "2024-01-10", "2024-01-31")
    assert ser.times[0] == np.datetime64("2024-01-10", "ns")
    assert ser.times[-1] == np.datetime64("2024-01-31", "ns")


def test_e2_t04_cboe_value_is_close_and_dates_parse() -> None:
    """`value = CLOSE`; `MM/DD/YYYY` dates become UTC midnight; frequency is `"d"`."""
    ser = _cboe().fetch_series("VIX_History.csv", "2024-01-02", "2024-02-29")
    raw = pd.read_csv(CASSETTES / "cboe" / "VIX_History.csv")
    np.testing.assert_array_equal(ser.values, raw["CLOSE"].to_numpy(dtype="float64"))
    expected_times = pd.to_datetime(raw["DATE"], format="%m/%d/%Y").to_numpy(dtype="datetime64[ns]")
    np.testing.assert_array_equal(ser.times, expected_times)
    assert ser.frequency == "d"
    assert ser.values.dtype == np.float64


# --- AC2: "." -> NaN ------------------------------------------------------------------------------------


def test_e2_t04_fred_dot_values_become_nan() -> None:
    """AC2: the fixture's two `"."` holiday rows become NaN, at exactly those dates."""
    ser = _fred().fetch_series("VIXCLS", "2024-01-02", "2024-02-29")
    assert int(np.isnan(ser.values).sum()) == 2
    np.testing.assert_array_equal(
        ser.times[np.isnan(ser.values)], np.array(["2024-01-15", "2024-02-19"], dtype="datetime64[ns]")
    )


def test_e2_t04_parse_values_rules() -> None:
    """`"."`, empty and missing -> NaN; numeric strings and floats parse; garbage raises."""
    out = parse_values(pd.Series([".", "1.5", " . ", "", None, 2.0, float("nan")]), "X")
    np.testing.assert_array_equal(np.isnan(out), [True, False, True, True, True, False, True])
    assert out[1] == 1.5
    assert out[5] == 2.0
    with pytest.raises(ProviderError, match="non-numeric"):
        parse_values(pd.Series(["1.0", "n/a"]), "X")


# --- AC3: 120 requests per rolling minute ------------------------------------------------------------


def test_e2_t04_limiter_blocks_the_121st_call_within_a_minute() -> None:
    """AC3: 120 calls within a minute pass; the 121st waits until the first leaves the window."""
    clock = FakeClock()
    limiter = RateLimiter(120, 60.0, clock=clock, sleep=clock.sleep)
    for _ in range(120):
        limiter.acquire()
        clock.now += 0.25  # 120 calls spread over the first 30 s
    assert clock.sleeps == []
    limiter.acquire()  # t = 30 s: 121st call inside the minute
    assert clock.sleeps == [pytest.approx(30.0)]
    assert clock.now == pytest.approx(60.0)


def test_e2_t04_limiter_never_exceeds_limit_in_any_window() -> None:
    """Rolling window: no 60-s window ever contains more than 120 calls, over a long burst."""
    clock = FakeClock()
    limiter = RateLimiter(120, 60.0, clock=clock, sleep=clock.sleep)
    stamps = []
    for _ in range(500):
        limiter.acquire()
        stamps.append(clock.now)
        clock.now += 0.1
    t = np.asarray(stamps)
    counts = np.searchsorted(t, t + 60.0, side="left") - np.arange(len(t))
    assert counts.max() == 120


def test_e2_t04_limiter_frees_capacity_after_the_window() -> None:
    """Calls after the window has passed go through without waiting."""
    clock = FakeClock()
    limiter = RateLimiter(120, 60.0, clock=clock, sleep=clock.sleep)
    for _ in range(120):
        limiter.acquire()
    clock.now = 60.0
    for _ in range(120):
        limiter.acquire()
    assert clock.sleeps == []


def test_e2_t04_limiter_rejects_bad_settings() -> None:
    """Non-positive settings are configuration errors."""
    with pytest.raises(ProviderError):
        RateLimiter(0, 60.0)
    with pytest.raises(ProviderError):
        RateLimiter(10, 0.0)


def test_e2_t04_fred_requests_go_through_the_limiter() -> None:
    """AC3 end to end: each fetch is two requests, so the 61st fetch at t = 0 is the one that waits."""
    clock = FakeClock()
    client = CassetteFredClient()
    provider = _fred(client, clock=clock)
    for _ in range(60):
        provider.fetch_series("VIXCLS", "2024-01-02", "2024-01-31")
    assert len(client.requests) == 120
    assert clock.sleeps == []
    provider.fetch_series("VIXCLS", "2024-01-02", "2024-01-31")
    assert clock.sleeps == [pytest.approx(60.0)]


def test_e2_t04_fred_default_limiter_is_shared_and_120_per_minute() -> None:
    """Instances without an explicit limiter share one process-wide 120/60 s window."""
    a = FredProvider(client=CassetteFredClient(), api_key=KEY)
    b = FredProvider(client=CassetteFredClient(), api_key=KEY)
    assert a._limiter is b._limiter
    assert (a._limiter.max_calls, a._limiter.period_s) == (120, 60.0)


def test_e2_t04_fred_retries_count_against_the_limiter() -> None:
    """A transient failure is retried, and the retry is a limited request too."""
    clock = FakeClock()
    client = CassetteFredClient(fail_first=1)
    provider = _fred(client, clock=clock)
    limiter = provider._limiter
    ser = provider.fetch_series("VIXCLS", "2024-01-02", "2024-01-31")
    assert len(ser.values) > 0
    assert len(client.requests) == 3
    assert len(limiter._expiries) == 3


# --- AC4: missing key ------------------------------------------------------------------------------


def test_e2_t04_fred_missing_key_raises_without_network() -> None:
    """AC4: no `FRED_API_KEY` -> `ProviderError` with instructions, before any request."""
    provider = FredProvider(client=NoNetworkClient(), env={}, sleep=lambda s: None)
    with pytest.raises(ProviderError, match="FRED_API_KEY") as info:
        provider.fetch_series("VIXCLS", "2024-01-02", "2024-02-29")
    assert "export FRED_API_KEY" in str(info.value)


def test_e2_t04_fred_key_from_env_and_never_logged() -> None:
    """The key is read from the environment and appears in no log event or error."""
    client = CassetteFredClient()
    provider = FredProvider(client=client, env={"FRED_API_KEY": KEY}, limiter=RateLimiter(120, 60.0))
    with capture_logs() as logs:
        provider.fetch_series("VIXCLS", "2024-01-02", "2024-01-31")
        with pytest.raises(ProviderError) as info:
            provider.fetch_series("NOPE", "2024-01-02", "2024-01-31")
    assert KEY not in repr(logs)
    assert KEY not in str(info.value)
    assert any(e.get("provider") == "fred" and e.get("tos") for e in logs)


def test_e2_t04_fred_default_client_needs_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without `fredapi` the default client fails fast (no retries) with install instructions."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    sleeps: list[float] = []
    provider = FredProvider(env={"FRED_API_KEY": KEY}, sleep=sleeps.append)
    with pytest.raises(ProviderError, match="extra providers"):
        provider.fetch_series("VIXCLS", "2024-01-02", "2024-02-29")
    assert sleeps == []


# --- other behaviour -------------------------------------------------------------------------------


def test_e2_t04_fred_unknown_series_is_not_retried() -> None:
    """A series that does not exist fails once with a clear error, no retry storm."""
    client = CassetteFredClient()
    with pytest.raises(ProviderError, match="does not exist"):
        _fred(client).fetch_series("NOPE", "2024-01-02", "2024-02-29")
    assert client.requests == [("info", "NOPE")]


@pytest.mark.parametrize(("short", "expected"), [("D", "d"), ("W", "w"), ("M", "m"), ("Q", "q")])
def test_e2_t04_fred_frequency_mapping(short: str, expected: str) -> None:
    """FRED `frequency_short` maps to `d/w/m/q`."""
    ser = _fred(CassetteFredClient(frequency_short=short)).fetch_series("T10Y2Y", "2024-01-02", "2024-02-29")
    assert ser.frequency == expected


@pytest.mark.parametrize("short", ["A", "SA", "BW", ""])
def test_e2_t04_fred_unsupported_frequency_raises(short: str) -> None:
    """Annual, semiannual, biweekly or missing metadata frequency are refused before downloading data."""
    client = CassetteFredClient(frequency_short=short)
    with pytest.raises(ProviderError, match="frequency"):
        _fred(client).fetch_series("T10Y2Y", "2024-01-02", "2024-02-29")
    assert [kind for kind, _ in client.requests] == ["info"]


def test_e2_t04_fred_empty_window_raises() -> None:
    """A window with no observations is a `ProviderError`."""
    with pytest.raises(ProviderError):
        _fred().fetch_series("VIXCLS", "2030-01-01", "2030-02-01")


def test_e2_t04_neither_adapter_serves_prices() -> None:
    """`fetch_prices` raises `ProviderError` for both covariate providers."""
    with pytest.raises(ProviderError, match="not prices"):
        _fred().fetch_prices(["^GDAXI"], "2024-01-02", "2024-02-29", "1d", True)
    with pytest.raises(ProviderError, match="not prices"):
        _cboe().fetch_prices(["^GDAXI"], "2024-01-02", "2024-02-29", "1d", True)


def test_e2_t04_registry_builds_both_adapters() -> None:
    """Both adapters are registered under their config names."""
    assert isinstance(get_provider("fred", api_key=KEY), FredProvider)
    assert isinstance(get_provider("cboe"), CboeProvider)


def test_e2_t04_cboe_urls_are_configurable() -> None:
    """`urls` replaces the defaults; the downloader receives the configured URL."""
    dl = CassetteCboeDownloader()
    provider = _cboe(dl, urls={"VIX": "https://mirror.example/data/VIX_History.csv"})
    ser = provider.fetch_series("VIX", "2024-01-02", "2024-01-31")
    assert dl.urls == ["https://mirror.example/data/VIX_History.csv"]
    assert ser.series_id == "VIX"
    with pytest.raises(ProviderError, match="unknown cboe series"):
        provider.fetch_series("VIX_History.csv", "2024-01-02", "2024-01-31")
    assert len(dl.urls) == 1
    assert set(DEFAULT_URLS) == {"VIX_History.csv", "VVIX_History.csv"}
    assert all(url.startswith("https://") for url in DEFAULT_URLS.values())


def test_e2_t04_cboe_retries_transient_failures() -> None:
    """Download failures are retried (shared `with_retries`)."""
    dl = CassetteCboeDownloader(fail_first=2)
    ser = _cboe(dl).fetch_series("VIX_History.csv", "2024-01-02", "2024-01-31")
    assert len(dl.urls) == 3
    assert len(ser.values) > 0


def test_e2_t04_cboe_parser_rejects_malformed_files() -> None:
    """Missing DATE, ambiguous value columns, non-numeric closes and empty files raise `ProviderError`."""
    for text in (
        "OPEN,CLOSE\n1,2\n",
        "DATE,A,B\n01/02/2024,1,2\n",
        "DATE,CLOSE\n01/02/2024,abc\n",
        "",
    ):
        with pytest.raises(ProviderError):
            parse_cboe_csv(text, "X")


def test_e2_t04_cboe_parser_accepts_iso_dates_and_sorts() -> None:
    """ISO dates are accepted too; rows come out sorted and de-duplicated."""
    frame = parse_cboe_csv("DATE,CLOSE\n2024-01-03,2\n2024-01-02,1\n2024-01-03,3\n", "X")
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert list(frame["value"]) == [1.0, 3.0]
