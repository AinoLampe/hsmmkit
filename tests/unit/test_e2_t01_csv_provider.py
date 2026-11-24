"""E2-T01: CsvProvider contract suite, registry and retry helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finance"))
from provider_contract import (
    check_price_contract,
    check_series_contract,
    check_window_inclusive,
)

from regimekit.finance.providers import (
    CsvProvider,
    get_provider,
    register,
    registered_providers,
)
from regimekit.finance.providers.base import throttle, with_retries
from regimekit.utils.errors import ProviderError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "csv"


pytestmark = pytest.mark.finance


@pytest.fixture
def provider() -> CsvProvider:
    """Provider rooted at the synthetic CSV fixtures."""
    return CsvProvider(FIXTURES)


def test_e2_t01_csv_passes_price_contract_suite(provider: CsvProvider) -> None:
    """AC1: the shared price contract suite passes, incl. inclusive window bounds."""
    check_price_contract(provider, ["DEMO"], "2020-01-02", "2020-01-10")
    check_window_inclusive(provider, "DEMO", "2020-01-03", "2020-01-09")
    check_price_contract(provider, ["DEMO"], "2020-01-02", "2020-01-10", interval="1wk")


def test_e2_t01_csv_passes_series_contract_suite(provider: CsvProvider) -> None:
    """AC1: the shared series contract suite passes."""
    check_series_contract(provider, "VIXCLS", "2020-01-02", "2020-01-10")


def test_e2_t01_csv_sorts_dedups_keep_last_and_uses_requested_column(provider: CsvProvider) -> None:
    """Unsorted rows are sorted, duplicate timestamps keep the last row, `adjusted` picks the column."""
    raw = provider.fetch_prices(["DEMO"], "2020-01-01", None, "1d", False)
    adj = provider.fetch_prices(["DEMO"], "2020-01-01", None, "1d", True)
    assert len(raw.times) == 7  # 8 rows, one duplicate timestamp dropped
    assert raw.prices[:, 0].tolist() == [100.5, 101.5, 102.0, 102.5, 103.75, 104.0, 105.0]
    assert adj.prices[-1, 0] == 52.5


def test_e2_t01_csv_series_dot_and_empty_become_nan(provider: CsvProvider) -> None:
    """`"."` and empty cells become NaN; frequency is inferred as daily."""
    ser = provider.fetch_series("VIXCLS", "2020-01-01", None)
    assert np.isnan(ser.values).tolist() == [False, True, False, True, False, False, False]
    assert ser.frequency == "d"


def test_e2_t01_csv_multi_symbol_outer_join_pads_nan(tmp_path: Path) -> None:
    """Symbols with different timestamps are outer-joined; tz-aware times are converted to UTC."""
    (tmp_path / "A.csv").write_text("time,close,adj_close\n2020-01-01,1,1\n2020-01-02,2,2\n")
    (tmp_path / "B.csv").write_text("time,close,adj_close\n2020-01-02T00:00:00+01:00,5,5\n2020-01-03,6,6\n")
    obs = CsvProvider(tmp_path).fetch_prices(["A", "B"], "2019-12-01", "2020-01-31", "1d", False)
    assert obs.prices.shape == (4, 2)  # tz-aware B timestamp is converted to UTC (2020-01-01T23:00)
    assert np.isnan(obs.prices).sum() == 4


def test_e2_t01_csv_errors(provider: CsvProvider, tmp_path: Path) -> None:
    """Bad interval, no symbols and a missing `adj_close` column raise `ProviderError`."""
    with pytest.raises(ProviderError):
        provider.fetch_prices(["DEMO"], "2020-01-01", None, "5m", False)
    with pytest.raises(ProviderError):
        provider.fetch_prices([], "2020-01-01", None, "1d", False)
    (tmp_path / "X.csv").write_text("time,close\n2020-01-01,1\n")
    with pytest.raises(ProviderError, match="adj_close"):
        CsvProvider(tmp_path).fetch_prices(["X"], "2020-01-01", None, "1d", True)
    assert CsvProvider(tmp_path).fetch_prices(["X"], "2020-01-01", None, "1d", False).prices.shape == (1, 1)


def test_e2_t01_registry_resolves_csv() -> None:
    """AC2: `get_provider("csv", root=...)` resolves; unknown names raise."""
    p = get_provider("csv", root=FIXTURES)
    assert isinstance(p, CsvProvider)
    assert "csv" in registered_providers()
    with pytest.raises(ProviderError, match="unknown provider"):
        get_provider("nope")


def test_e2_t01_duplicate_registration_raises() -> None:
    """AC2: registering a second `"csv"` (or a nameless class) raises `ProviderError`."""

    class Dup:
        name = "csv"

    with pytest.raises(ProviderError, match="already registered"):
        register(Dup)

    class Nameless:
        pass

    with pytest.raises(ProviderError):
        register(Nameless)


def test_e2_t01_with_retries_exhausts_and_reraises() -> None:
    """AC3: exactly `max_attempts` calls, exponential backoff, then the last error is re-raised."""
    calls: list[int] = []
    sleeps: list[float] = []

    def boom() -> int:
        calls.append(1)
        raise ProviderError("down")

    with pytest.raises(ProviderError, match="down"):
        with_retries(boom, max_attempts=4, sleep=sleeps.append)
    assert len(calls) == 4
    assert sleeps == [0.5, 1.0, 2.0]  # exponential backoff, no sleep after the final attempt


def test_e2_t01_with_retries_succeeds_and_ignores_other_errors() -> None:
    """Retries stop on success; non-`ProviderError` exceptions propagate immediately."""
    state = {"n": 0}

    def flaky() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise ProviderError("flaky")
        return "ok"

    def bug() -> int:
        raise ValueError("bug")

    assert with_retries(flaky, max_attempts=5, sleep=lambda s: None) == "ok"
    with pytest.raises(ValueError, match="bug"):
        with_retries(bug, sleep=lambda s: None)
    with pytest.raises(ProviderError):
        with_retries(lambda: 1, max_attempts=0)


def test_e2_t01_throttle_enforces_min_interval() -> None:
    """`throttle` sleeps only for the remaining part of the minimum interval."""
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    @throttle(2.0, clock=lambda: now[0], sleep=sleep)
    def f() -> int:
        return 1

    f()
    now[0] += 0.5
    f()
    now[0] += 5.0
    f()
    assert slept == [1.5]


def test_e2_t01_csv_bad_timestamp_and_missing_file_raise(tmp_path: Path) -> None:
    """Unparseable timestamps and missing files raise `ProviderError`."""
    (tmp_path / "BAD.csv").write_text("time,close\nnot-a-date,1\n")
    with pytest.raises(ProviderError, match="unparseable"):
        CsvProvider(tmp_path).fetch_prices(["BAD"], "2020-01-01", None, "1d", False)
    (tmp_path / "BIN.csv").write_bytes(b"time,close\n\xff\xfe,1\n")
    with pytest.raises(ProviderError, match="cannot parse"):
        CsvProvider(tmp_path).fetch_prices(["BIN"], "2020-01-01", None, "1d", False)
    with pytest.raises(ProviderError, match="no CSV file"):
        CsvProvider(tmp_path).fetch_series("MISSING", "2020-01-01", None)
