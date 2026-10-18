"""E2-T02: yfinance adapter against recorded (synthetic) `^GDAXI` cassettes; no network."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finance"))
from provider_contract import check_price_contract, check_window_inclusive

from regimekit.finance.providers import YFinanceProvider, get_provider
from regimekit.finance.providers.yfinance import INTRADAY_CAP_DAYS, extract_close
from regimekit.utils.errors import ProviderError

CASSETTES = Path(__file__).resolve().parents[1] / "fixtures" / "cassettes" / "yfinance"
SYMBOL = "^GDAXI"


class CassetteDownloader:
    """Replays the recorded frames; unknown symbols come back empty, as yfinance does."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[tuple[list[str], str, str | None, str, bool]] = []

    def __call__(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> pd.DataFrame:
        """Record the call and return the cassette frame for `(interval, adjusted)`."""
        self.calls.append((symbols, start, end, interval, adjusted))
        kind = "adjusted" if adjusted else "raw"
        path = CASSETTES / f"GDAXI_{interval}_{kind}.parquet"
        if not path.is_file():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        return frame.loc[:, [s for s in symbols if s in frame.columns]]


def _provider(downloader: CassetteDownloader | None = None, **kwargs: object) -> YFinanceProvider:
    return YFinanceProvider(
        downloader=downloader or CassetteDownloader(),
        min_interval_s=0.0,
        sleep=lambda s: None,
        today=lambda: pd.Timestamp("2024-07-01"),
        **kwargs,  # type: ignore[arg-type]
    )


def test_e2_t02_daily_passes_contract_suite() -> None:
    """AC1: recorded `^GDAXI` 1d passes the E2-T01 contract suite (incl. inclusive window)."""
    check_price_contract(_provider(), [SYMBOL], "2024-06-03", "2024-06-28", interval="1d")
    check_window_inclusive(_provider(), SYMBOL, "2024-06-05", "2024-06-20", interval="1d")


def test_e2_t02_hourly_passes_contract_suite_and_converts_to_utc() -> None:
    """AC1: recorded `^GDAXI` 1h passes the suite; Berlin-local bars (CEST, +2h) are stored as UTC."""
    check_price_contract(_provider(), [SYMBOL], "2024-06-03", "2024-06-07", interval="1h")
    obs = _provider().fetch_prices([SYMBOL], "2024-06-03", "2024-06-03", "1h", False)
    assert obs.times[0] == np.datetime64("2024-06-03T07:00:00", "ns")  # 09:00 Berlin
    assert len(obs.times) == 9


def test_e2_t02_adjusted_flag_selects_recorded_frame_and_reaches_downloader() -> None:
    """`adjusted=True` maps to auto_adjust=True; `adjusted=False` to raw Close."""
    dl = CassetteDownloader()
    raw = _provider(dl).fetch_prices([SYMBOL], "2024-06-03", "2024-06-28", "1d", False)
    adj = _provider(dl).fetch_prices([SYMBOL], "2024-06-03", "2024-06-28", "1d", True)
    np.testing.assert_allclose(adj.prices, raw.prices * 0.97, rtol=1e-12)
    assert [c[4] for c in dl.calls] == [False, True]


def test_e2_t02_end_is_passed_exclusive_but_result_is_inclusive() -> None:
    """Yfinance `end` is exclusive: the adapter asks for end+1 day and returns the `end` bar."""
    dl = CassetteDownloader()
    obs = _provider(dl).fetch_prices([SYMBOL], "2024-06-03", "2024-06-07", "1d", False)
    assert dl.calls[0][2] == "2024-06-08 00:00:00"
    assert obs.times[-1] == np.datetime64("2024-06-07", "ns")


def test_e2_t02_three_years_of_hourly_warns_and_clips_to_cap() -> None:
    """AC2: 3y of `1h` warns and the downloader is asked only for the last 730 days."""
    dl = CassetteDownloader()
    with capture_logs() as logs, pytest.warns(UserWarning, match="clipping start"):
        _provider(dl).fetch_prices([SYMBOL], "2021-07-01", "2024-06-07", "1h", False)
    earliest = pd.Timestamp("2024-07-01") - pd.Timedelta(days=INTRADAY_CAP_DAYS["1h"])
    assert dl.calls[0][1] == str(earliest.date())
    assert any(e["event"] == "intraday_window_clipped" for e in logs)


def test_e2_t02_daily_and_in_cap_hourly_requests_do_not_warn() -> None:
    """No clipping (and no warning) for daily data or an intraday request within the cap."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _provider().fetch_prices([SYMBOL], "2000-01-01", "2024-06-28", "1d", False)
        _provider().fetch_prices([SYMBOL], "2024-06-03", "2024-06-07", "1h", False)


def test_e2_t02_tos_note_logged_on_each_fetch() -> None:
    """AC4: every fetch emits an INFO log carrying `provider="yfinance"` and the ToS note."""
    p = _provider()
    with capture_logs() as logs:
        p.fetch_prices([SYMBOL], "2024-06-03", "2024-06-28", "1d", False)
        p.fetch_prices([SYMBOL], "2024-06-03", "2024-06-28", "1d", True)
    fetches = [e for e in logs if e["event"] == "fetch"]
    assert len(fetches) == 2
    assert all(e["provider"] == "yfinance" and e["tos"] == p.tos_note for e in fetches)
    assert all(e["log_level"] == "info" for e in fetches)


def test_e2_t02_multi_symbol_union_alignment_pads_nan() -> None:
    """Symbols with different bar times are aligned on the union, NaN where missing."""
    daily = pd.read_parquet(CASSETTES / "GDAXI_1d_raw.parquet")

    def dl(symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool) -> pd.DataFrame:
        return pd.DataFrame({"A": daily[SYMBOL].iloc[:5], "B": daily[SYMBOL].iloc[2:8]})

    p = YFinanceProvider(downloader=dl, min_interval_s=0.0, sleep=lambda s: None)
    obs = p.fetch_prices(["A", "B"], "2024-06-01", "2024-06-30", "1d", False)
    assert obs.prices.shape == (8, 2)
    assert np.isnan(obs.prices[:2, 1]).all()
    assert np.isnan(obs.prices[5:, 0]).all()


def test_e2_t02_unknown_symbol_and_empty_results_raise_provider_error() -> None:
    """Missing/empty symbol data and empty windows are `ProviderError`s (not retried)."""
    dl = CassetteDownloader()
    with pytest.raises(ProviderError, match="no data"):
        _provider(dl).fetch_prices(["NOPE"], "2024-06-03", "2024-06-28", "1d", False)
    assert len(dl.calls) == 1
    with pytest.raises(ProviderError, match="no rows"):
        _provider().fetch_prices([SYMBOL], "2030-01-01", "2030-02-01", "1d", False)
    with pytest.raises(ProviderError, match="no symbols"):
        _provider().fetch_prices([], "2024-06-03", None, "1d", False)
    with pytest.raises(ProviderError):
        _provider().fetch_prices([SYMBOL], "2024-06-03", None, "5m", False)


def test_e2_t02_downloader_failures_are_retried_then_reraised() -> None:
    """A failing download is attempted `max_attempts` times before the error propagates."""
    calls: list[int] = []

    def flaky(*args: object) -> pd.DataFrame:
        calls.append(1)
        raise ProviderError("connection reset")

    p = YFinanceProvider(downloader=flaky, min_interval_s=0.0, max_attempts=3, sleep=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(ProviderError, match="connection reset"):
        p.fetch_prices([SYMBOL], "2024-06-03", None, "1d", False)
    assert len(calls) == 3


def test_e2_t02_downloads_are_throttled() -> None:
    """Consecutive downloads are spaced by at least `min_interval_s` (injected clock)."""
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    p = YFinanceProvider(
        downloader=CassetteDownloader(), min_interval_s=0.5, clock=lambda: now[0], sleep=sleep
    )
    p.fetch_prices([SYMBOL], "2024-06-03", "2024-06-28", "1d", False)
    p.fetch_prices([SYMBOL], "2024-06-03", "2024-06-28", "1d", False)
    assert slept == [0.5]


def test_e2_t02_fetch_series_is_unsupported_and_registry_resolves() -> None:
    """Price-only provider: `fetch_series` raises; `get_provider("yfinance")` resolves offline."""
    p = get_provider("yfinance", downloader=CassetteDownloader())
    assert isinstance(p, YFinanceProvider)
    with pytest.raises(ProviderError, match="prices only"):
        p.fetch_series("VIXCLS", "2024-01-01", None)


def test_e2_t02_extract_close_handles_multiindex_and_flat_layouts() -> None:
    """`extract_close` reduces raw yfinance frames (both layouts) to one Close column per symbol."""
    idx = pd.date_range("2024-01-01", periods=3)
    multi = pd.DataFrame(
        np.arange(12.0).reshape(3, 4),
        index=idx,
        columns=pd.MultiIndex.from_product([["Close", "Open"], ["A", "B"]]),
    )
    out = extract_close(multi, ["A", "B", "C"])
    assert list(out.columns) == ["A", "B"]
    assert out["A"].tolist() == [0.0, 4.0, 8.0]
    flat = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Open": [0.0, 0.0, 0.0]}, index=idx)
    assert extract_close(flat, ["A"])["A"].tolist() == [1.0, 2.0, 3.0]
    assert extract_close(flat, ["A", "B"]).shape[1] == 0
    assert extract_close(pd.DataFrame(index=idx), ["A"]).shape[1] == 0
    no_close = pd.DataFrame(
        np.zeros((3, 2)), index=idx, columns=pd.MultiIndex.from_product([["Open"], ["A", "B"]])
    )
    assert extract_close(no_close, ["A"]).shape[1] == 0


def test_e2_t02_cap_boundary_and_window_entirely_outside_cap() -> None:
    """Start exactly at the cap edge is untouched; one day earlier clips; a window wholly beyond it errors."""
    earliest = pd.Timestamp("2024-07-01") - pd.Timedelta(days=INTRADAY_CAP_DAYS["1h"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _provider().fetch_prices([SYMBOL], str(earliest.date()), "2024-06-07", "1h", False)
    with pytest.warns(UserWarning, match="clipping start"):
        _provider().fetch_prices(
            [SYMBOL], str((earliest - pd.Timedelta(days=1)).date()), "2024-06-07", "1h", False
        )
    dl = CassetteDownloader()
    with pytest.warns(UserWarning, match="clipping start"), pytest.raises(ProviderError, match="limit"):
        _provider(dl).fetch_prices([SYMBOL], "2021-01-01", "2022-01-01", "1h", False)
    assert dl.calls == []


def test_e2_t02_missing_extra_fails_fast_without_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the default downloader and no `yfinance`, the error surfaces at once (no backoff sleeps)."""
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    slept: list[float] = []
    provider = YFinanceProvider(min_interval_s=0.0, sleep=slept.append)
    with pytest.raises(ProviderError, match="providers"):
        provider.fetch_prices([SYMBOL], "2024-06-03", None, "1d", False)
    assert slept == []
