"""E2-T03: stooq adapter against a recorded (synthetic) daily cassette; no network."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finance"))
from provider_contract import check_price_contract, check_window_inclusive

from regimekit.finance.errors import ProviderError
from regimekit.finance.providers import StooqProvider, get_provider
from regimekit.finance.providers.stooq import to_stooq_symbol

CASSETTE = Path(__file__).resolve().parents[1] / "fixtures" / "cassettes" / "stooq" / "dax_1d.parquet"
KEY = "SECRET-KEY-123"


pytestmark = pytest.mark.finance


class CassetteDownloader:
    """Replays the recorded frame for `^dax`; other symbols come back empty, as stooq does."""

    def __init__(self, fail_first: int = 0) -> None:
        """Optionally fail the first `fail_first` calls with a retryable `ProviderError`."""
        self.calls: list[tuple[str, str, str | None, str]] = []
        self._fail_first = fail_first

    def __call__(self, symbol: str, start: str, end: str | None, api_key: str) -> pd.DataFrame:
        """Record the call and return the cassette frame (or an empty frame)."""
        self.calls.append((symbol, start, end, api_key))
        if len(self.calls) <= self._fail_first:
            raise ProviderError("temporary failure")
        return pd.read_parquet(CASSETTE) if symbol == "^dax" else pd.DataFrame()


def _provider(
    tmp_path: Path, downloader: CassetteDownloader | None = None, **kwargs: object
) -> StooqProvider:
    options: dict[str, object] = {
        "api_key": KEY,
        "quota_dir": tmp_path,
        "max_calls_per_day": 1000,
        "today": lambda: date(2024, 7, 1),
        "sleep": lambda s: None,
    }
    options.update(kwargs)
    return StooqProvider(downloader=downloader or CassetteDownloader(), **options)  # type: ignore[arg-type]


def _cassette_close() -> pd.Series:
    return pd.read_parquet(CASSETTE)["Close"]


def test_e2_t03_passes_contract_suite_daily_and_weekly(tmp_path: Path) -> None:
    """AC1: recorded responses pass the E2-T01 contract suite (`1d`, and resampled `1wk`)."""
    p = _provider(tmp_path)
    check_price_contract(
        p, ["^GDAXI"], "2024-06-03", "2024-06-28", adjusted_changes_prices=False, unknown="NOPE.XX"
    )
    check_price_contract(
        p, ["^GDAXI"], "2024-01-02", "2024-06-28", interval="1wk", adjusted_changes_prices=False
    )
    check_window_inclusive(p, "^GDAXI", "2024-06-05", "2024-06-20")


def test_e2_t03_symbol_mapping_and_passthrough(tmp_path: Path) -> None:
    """`^GDAXI -> ^dax`, `^GSPC -> ^spx`; other symbols pass through; results keep the requested names."""
    assert to_stooq_symbol("^GDAXI") == "^dax"
    assert to_stooq_symbol("^GSPC") == "^spx"
    assert to_stooq_symbol("sap.de") == "sap.de"
    dl = CassetteDownloader()
    obs = _provider(tmp_path, dl).fetch_prices(["^GDAXI"], "2024-06-03", "2024-06-28", "1d", True)
    assert dl.calls[0][0] == "^dax"
    assert obs.series_names == ("^GDAXI",)
    assert obs.values.shape == (20, 1)


def test_e2_t03_weekly_and_monthly_are_last_close_of_period(tmp_path: Path) -> None:
    """`1wk`/`1mo` equal the last daily close of each period, labelled with its real trading date."""
    close = _cassette_close()
    p = _provider(tmp_path)
    for interval, freq in (("1wk", "W-SUN"), ("1mo", "M")):
        obs = p.fetch_prices(["^GDAXI"], "2024-01-01", "2024-06-30", interval, False)
        expected = close.groupby(close.index.to_period(freq)).tail(1)
        assert obs.times.tolist() == expected.index.to_numpy(dtype="datetime64[ns]").tolist()
        np.testing.assert_array_equal(obs.values[:, 0], expected.to_numpy())
        assert "resampled_from" in obs.meta
    daily = p.fetch_prices(["^GDAXI"], "2024-01-01", "2024-06-30", "1d", False)
    assert daily.meta == {"interval": "1d", "adjusted": False}
    assert len(daily.times) == len(close)


def test_e2_t03_intraday_is_rejected_without_network(tmp_path: Path) -> None:
    """`1h` raises `ProviderError`; `supports_intraday` is False; the downloader is never called."""
    dl = CassetteDownloader()
    p = _provider(tmp_path, dl)
    assert p.supports_intraday is False
    with pytest.raises(ProviderError, match="daily"):
        p.fetch_prices(["^GDAXI"], "2024-06-03", None, "1h", False)
    assert dl.calls == []


def test_e2_t03_missing_key_raises_with_instructions_and_no_network(tmp_path: Path) -> None:
    """AC2: no key -> `ProviderError` naming `STOOQ_API_KEY`; downloader untouched; quota untouched."""
    dl = CassetteDownloader()
    p = StooqProvider(downloader=dl, quota_dir=tmp_path, env={}, sleep=lambda s: None)
    with pytest.raises(ProviderError, match="STOOQ_API_KEY"):
        p.fetch_prices(["^GDAXI"], "2024-06-03", None, "1d", False)
    assert dl.calls == []
    assert not (tmp_path / "stooq_quota.json").exists()


def test_e2_t03_key_comes_from_environment_and_is_never_logged(tmp_path: Path) -> None:
    """The env key reaches the downloader; it appears in no log record."""
    dl = CassetteDownloader()
    p = StooqProvider(
        downloader=dl,
        quota_dir=tmp_path,
        env={"STOOQ_API_KEY": KEY},
        today=lambda: date(2024, 7, 1),
    )
    with capture_logs() as logs:
        p.fetch_prices(["^GDAXI"], "2024-06-03", "2024-06-28", "1d", False)
    assert dl.calls[0][3] == KEY
    assert KEY not in repr(logs)
    fetch = next(e for e in logs if e["event"] == "fetch")
    assert fetch["provider"] == "stooq"
    assert fetch["tos"] == p.tos_note


def test_e2_t03_quota_exhaustion_raises_and_persists_across_instances(tmp_path: Path) -> None:
    """AC3: the daily budget survives new adapter instances; over budget -> `ProviderError("stooq quota")`."""
    args = (["^GDAXI"], "2024-06-03", "2024-06-28", "1d", False)
    _provider(tmp_path, max_calls_per_day=3).fetch_prices(*args)
    _provider(tmp_path, max_calls_per_day=3).fetch_prices(*args)
    dl = CassetteDownloader()
    third = _provider(tmp_path, dl, max_calls_per_day=3)
    third.fetch_prices(*args)
    assert json.loads((tmp_path / "stooq_quota.json").read_text())["calls"] == 3
    calls_before = len(dl.calls)
    with pytest.raises(ProviderError, match="stooq quota"):
        _provider(tmp_path, dl, max_calls_per_day=3).fetch_prices(*args)
    assert len(dl.calls) == calls_before


def test_e2_t03_quota_refuses_oversized_requests_without_charging_and_resets_daily(tmp_path: Path) -> None:
    """A 2-symbol request with 1 call left is refused whole; the counter resets on a new UTC day."""
    args = (["^GDAXI"], "2024-06-03", "2024-06-28", "1d", False)
    _provider(tmp_path, max_calls_per_day=2).fetch_prices(*args)
    dl = CassetteDownloader()
    with pytest.raises(ProviderError, match="stooq quota"):
        _provider(tmp_path, dl, max_calls_per_day=2).fetch_prices(
            ["^GDAXI", "sap.de"], "2024-06-03", "2024-06-28", "1d", False
        )
    assert dl.calls == []
    assert json.loads((tmp_path / "stooq_quota.json").read_text())["calls"] == 1
    tomorrow = date(2024, 7, 1) + timedelta(days=1)
    _provider(tmp_path, dl, max_calls_per_day=1, today=lambda: tomorrow).fetch_prices(*args)
    assert json.loads((tmp_path / "stooq_quota.json").read_text()) == {"date": "2024-07-02", "calls": 1}


def test_e2_t03_corrupt_quota_file_counts_as_unused(tmp_path: Path) -> None:
    """An unreadable quota file does not block fetching."""
    (tmp_path / "stooq_quota.json").write_text("{not json")
    _provider(tmp_path, max_calls_per_day=1).fetch_prices(["^GDAXI"], "2024-06-03", "2024-06-28", "1d", False)
    assert json.loads((tmp_path / "stooq_quota.json").read_text())["calls"] == 1


def test_e2_t03_transient_failures_are_retried_without_extra_quota(tmp_path: Path) -> None:
    """Two failures then success: 3 downloader calls, but a single quota charge."""
    dl = CassetteDownloader(fail_first=2)
    _provider(tmp_path, dl).fetch_prices(["^GDAXI"], "2024-06-03", "2024-06-28", "1d", False)
    assert len(dl.calls) == 3
    assert json.loads((tmp_path / "stooq_quota.json").read_text())["calls"] == 1


def test_e2_t03_empty_results_and_unsupported_calls_raise(tmp_path: Path) -> None:
    """Unknown symbol, empty window, no symbols, `fetch_series`: all `ProviderError`; registry resolves."""
    p = _provider(tmp_path)
    with pytest.raises(ProviderError, match="no data"):
        p.fetch_prices(["NOPE.XX"], "2024-06-03", None, "1d", False)
    with pytest.raises(ProviderError, match="no rows"):
        p.fetch_prices(["^GDAXI"], "2030-01-01", "2030-02-01", "1d", False)
    with pytest.raises(ProviderError, match="no symbols"):
        p.fetch_prices([], "2024-06-03", None, "1d", False)
    with pytest.raises(ProviderError, match="prices only"):
        p.fetch_series("VIXCLS", "2024-01-01", None)
    assert isinstance(get_provider("stooq", downloader=CassetteDownloader(), api_key=KEY), StooqProvider)


def test_e2_t03_multi_symbol_union_alignment_pads_nan_and_charges_per_symbol(tmp_path: Path) -> None:
    """Two symbols with different histories are outer-joined (NaN padded); each costs one quota call."""
    base = pd.read_parquet(CASSETTE)

    def downloader(symbol: str, start: str, end: str | None, api_key: str) -> pd.DataFrame:
        return base if symbol == "^dax" else base.iloc[3:]

    p = StooqProvider(downloader=downloader, api_key=KEY, quota_dir=tmp_path, today=lambda: date(2024, 7, 1))
    obs = p.fetch_prices(["^GDAXI", "sap.de"], "2024-01-01", "2024-06-30", "1d", False)
    assert obs.values.shape == (len(base), 2)
    assert np.isnan(obs.values[:3, 1]).all()
    assert not np.isnan(obs.values[:, 0]).any()
    assert json.loads((tmp_path / "stooq_quota.json").read_text())["calls"] == 2
