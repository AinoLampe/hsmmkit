"""E2-T05: parquet cache, snapshot ids, generations, offline replay and `max_age_days`."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regimekit.data.cache import Cache, CacheableProvider, CachedProvider, SnapshotPolicy
from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.utils.errors import ConfigError, DataError

START, END = "2024-01-02", "2024-06-28"
NOW = datetime(2024, 7, 1, 12, 0, tzinfo=UTC)


class Clock:
    """Settable UTC clock for `fetched_at` and staleness."""

    def __init__(self) -> None:
        """Start at `NOW`."""
        self.now = NOW

    def __call__(self) -> datetime:
        """Current time."""
        return self.now


class FakeInner:
    """Deterministic provider; `version` changes the prices (like a new split/dividend adjustment)."""

    name = "fake"
    supports_intraday = False
    tos_note = "synthetic test data"

    def __init__(self, version: int = 0) -> None:
        """Count calls; `version` shifts every price."""
        self.version = version
        self.price_calls = 0
        self.series_calls = 0

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Business-day closes on `[start, end]`, one column per symbol."""
        self.price_calls += 1
        times = pd.bdate_range(start, end or END).to_numpy(dtype="datetime64[ns]")
        i = np.arange(len(times), dtype="float64")[:, None]
        prices = (
            100.0 + i + np.arange(len(symbols))[None, :] + 0.5 * self.version + (0.25 if adjusted else 0.0)
        )
        return ObservationSet(
            symbols=tuple(symbols),
            times=times,
            prices=prices,
            interval=interval,  # type: ignore[arg-type]
            adjusted=adjusted,
            provider=self.name,
            snapshot_id="",
            meta={"note": "fake", "version": self.version},
        )

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """Daily series with one NaN."""
        self.series_calls += 1
        times = pd.bdate_range(start, end or END).to_numpy(dtype="datetime64[ns]")
        values = 15.0 + np.sin(np.arange(len(times), dtype="float64")) + self.version
        values[3] = np.nan
        return CovariateSeries(
            name=series_id,
            times=times,
            values=values,
            provider=self.name,
            series_id=series_id,
            snapshot_id="",
            frequency="d",
        )


class NoNetwork(FakeInner):
    """Fails the test if the inner provider is ever called."""

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Never allowed."""
        raise AssertionError("inner provider called")

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """Never allowed."""
        raise AssertionError("inner provider called")


def _cached(
    root: Path, inner: FakeInner | None = None, clock: Clock | None = None, **policy: object
) -> tuple[CachedProvider, FakeInner]:
    inner = inner if inner is not None else FakeInner()
    cache = Cache(root, now=clock or Clock())
    return CachedProvider(inner, cache, SnapshotPolicy(**policy)), inner  # type: ignore[arg-type]


def _bitwise_equal(a: ObservationSet, b: ObservationSet) -> bool:
    # Field-wise, not `a == b`: the contracts' `__eq__` uses `np.array_equal` without `equal_nan`.
    return (
        (a.symbols, a.interval, a.adjusted, a.provider, a.snapshot_id, a.meta)
        == (b.symbols, b.interval, b.adjusted, b.provider, b.snapshot_id, b.meta)
        and a.times.tobytes() == b.times.tobytes()
        and a.prices.tobytes() == b.prices.tobytes()
        and a.prices.dtype == b.prices.dtype
        and a.times.dtype == b.times.dtype
    )


def _series_bitwise_equal(a: CovariateSeries, b: CovariateSeries) -> bool:
    # `CovariateSeries.__eq__` compares values without `equal_nan`, so it is False whenever a value is NaN.
    return (
        (a.name, a.provider, a.series_id, a.snapshot_id, a.frequency)
        == (b.name, b.provider, b.series_id, b.snapshot_id, b.frequency)
        and a.times.tobytes() == b.times.tobytes()
        and a.values.tobytes() == b.values.tobytes()
        and a.values.dtype == b.values.dtype == np.float64
        and a.times.dtype == b.times.dtype
    )


# --- AC1: cache hit ------------------------------------------------------------------------------------


def test_e2_t05_second_identical_fetch_hits_cache(tmp_path: Path) -> None:
    """AC1: the second identical `fetch_prices` is served from the cache (`inner` called once)."""
    provider, inner = _cached(tmp_path)
    first = provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    second = provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert inner.price_calls == 1
    assert _bitwise_equal(first, second)
    assert len(first.snapshot_id) == 64
    int(first.snapshot_id, 16)


def test_e2_t05_series_second_fetch_hits_cache(tmp_path: Path) -> None:
    """Covariate series are cached the same way; NaN survives the round trip."""
    provider, inner = _cached(tmp_path)
    first = provider.fetch_series("VIXCLS", START, END)
    second = provider.fetch_series("VIXCLS", START, END)
    assert inner.series_calls == 1
    assert _series_bitwise_equal(first, second)
    assert np.isnan(second.values[3])
    assert len(first.snapshot_id) == 64
    assert (second.name, second.frequency, second.provider) == ("VIXCLS", "d", "fake")


def test_e2_t05_distinct_requests_are_distinct_entries(tmp_path: Path) -> None:
    """Different interval, adjustment or symbols are separate cache entries."""
    provider, inner = _cached(tmp_path)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", False)
    provider.fetch_prices(["^GDAXI"], START, END, "1wk", True)
    provider.fetch_prices(["^GDAXI", "^GSPC"], START, END, "1d", True)
    assert inner.price_calls == 4
    provider.fetch_prices(["^GDAXI"], START, END, "1d", False)
    assert inner.price_calls == 4


def test_e2_t05_narrower_window_is_served_and_clipped(tmp_path: Path) -> None:
    """A window inside the cached request is a hit, clipped to `[start, end]` inclusive."""
    provider, inner = _cached(tmp_path)
    full = provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    part = provider.fetch_prices(["^GDAXI"], "2024-02-01", "2024-02-29", "1d", True)
    assert inner.price_calls == 1
    assert part.times[0] == np.datetime64("2024-02-01", "ns")
    assert part.times[-1] == np.datetime64("2024-02-29", "ns")
    mask = (full.times >= part.times[0]) & (full.times <= part.times[-1])
    np.testing.assert_array_equal(part.prices, full.prices[mask])
    assert part.snapshot_id == full.snapshot_id


def test_e2_t05_wider_window_refetches(tmp_path: Path) -> None:
    """A window the cached request does not cover is refetched."""
    provider, inner = _cached(tmp_path)
    provider.fetch_prices(["^GDAXI"], "2024-02-01", "2024-02-29", "1d", True)
    wide = provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert inner.price_calls == 2
    assert wide.times[0] == np.datetime64(START, "ns")


def test_e2_t05_layout_and_meta(tmp_path: Path) -> None:
    """`<dir>/<provider>/<symbol>_<interval>_<adj|raw>.parquet` + `.meta.json`; series as `series_<id>`."""
    provider, _ = _cached(tmp_path)
    obs = provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    provider.fetch_series("VIXCLS", START, END)
    assert (tmp_path / "fake" / "^GDAXI_1d_adj.parquet").is_file()
    assert (tmp_path / "fake" / "series_VIXCLS.parquet").is_file()
    meta = json.loads((tmp_path / "fake" / "^GDAXI_1d_adj.meta.json").read_text(encoding="utf-8"))
    assert meta["snapshot_id"] == obs.snapshot_id
    assert meta["fetched_at"] == NOW.isoformat()
    assert meta["request"] == {
        "provider": "fake",
        "symbols": ["^GDAXI"],
        "interval": "1d",
        "adjusted": True,
        "start": START,
        "end": END,
    }
    assert meta["provider_version"]
    assert (tmp_path / "fake" / "series_VIXCLS.meta.json").is_file()


def test_e2_t05_unsafe_file_name_characters_are_replaced(tmp_path: Path) -> None:
    """Symbols with path separators or reserved characters cannot escape the provider directory."""
    provider, inner = _cached(tmp_path)
    provider.fetch_prices(["EUR/USD"], START, END, "1d", False)
    provider.fetch_prices(["EUR/USD"], START, END, "1d", False)
    assert inner.price_calls == 1
    assert (tmp_path / "fake" / "EUR_USD_1d_raw.parquet").is_file()


def test_e2_t05_cache_dir_expands_user() -> None:
    """`~` in the cache directory is expanded; nothing is written on construction."""
    assert Cache("~/regimekit-cache-test").root == Path.home() / "regimekit-cache-test"
    assert not (Path.home() / "regimekit-cache-test").exists()


def test_e2_t05_cached_provider_is_structural_provider(tmp_path: Path) -> None:
    """`CachedProvider` delegates the provider attributes and satisfies `CacheableProvider`."""
    provider, _ = _cached(tmp_path)
    assert isinstance(FakeInner(), CacheableProvider)
    assert isinstance(provider, CacheableProvider)
    assert (provider.name, provider.supports_intraday, provider.tos_note) == (
        "fake",
        False,
        "synthetic test data",
    )


# --- AC2: stable snapshot ids -------------------------------------------------------------------------


def test_e2_t05_snapshot_id_identical_across_fresh_caches(tmp_path: Path) -> None:
    """AC2: identical input data gives the same snapshot id (and parquet bytes) in two fresh caches."""
    a, _ = _cached(tmp_path / "a")
    b, _ = _cached(tmp_path / "b", clock=Clock())
    b_clock_later = Clock()
    b_clock_later.now = NOW + timedelta(days=30)
    c, _ = _cached(tmp_path / "c", clock=b_clock_later)
    ids = {p.fetch_prices(["^GDAXI"], START, END, "1d", True).snapshot_id for p in (a, b, c)}
    assert len(ids) == 1
    files = [(tmp_path / d / "fake" / "^GDAXI_1d_adj.parquet").read_bytes() for d in "abc"]
    assert files[0] == files[1] == files[2]
    series_ids = {p.fetch_series("VIXCLS", START, END).snapshot_id for p in (a, b, c)}
    assert len(series_ids) == 1


def test_e2_t05_snapshot_id_changes_with_data_and_request(tmp_path: Path) -> None:
    """Different prices, adjustment flag or provider give different snapshot ids."""
    base, _ = _cached(tmp_path / "a")
    changed, _ = _cached(tmp_path / "b", inner=FakeInner(version=1))
    id0 = base.fetch_prices(["^GDAXI"], START, END, "1d", True).snapshot_id
    assert changed.fetch_prices(["^GDAXI"], START, END, "1d", True).snapshot_id != id0
    assert base.fetch_prices(["^GDAXI"], START, END, "1d", False).snapshot_id != id0


# --- AC3: offline replay --------------------------------------------------------------------------------


def test_e2_t05_offline_replay_is_bitwise_identical(tmp_path: Path) -> None:
    """AC3: `offline=True` never calls `inner` and returns a bitwise-identical `ObservationSet`."""
    online, _ = _cached(tmp_path)
    original = online.fetch_prices(["^GDAXI", "^GSPC"], START, END, "1d", True)
    stale = Clock()
    stale.now = NOW + timedelta(days=365)
    offline, _ = _cached(tmp_path, inner=NoNetwork(), clock=stale, offline=True)
    replay = offline.fetch_prices(["^GDAXI", "^GSPC"], START, END, "1d", True)
    assert _bitwise_equal(original, replay)
    assert replay.meta == {"note": "fake", "version": 0}


def test_e2_t05_offline_series_replay(tmp_path: Path) -> None:
    """Offline replay works for covariate series too."""
    online, _ = _cached(tmp_path)
    original = online.fetch_series("T10Y2Y", START, END)
    offline, _ = _cached(tmp_path, inner=NoNetwork(), offline=True)
    replay = offline.fetch_series("T10Y2Y", START, END)
    assert _series_bitwise_equal(replay, original)


def test_e2_t05_offline_without_cache_raises(tmp_path: Path) -> None:
    """AC3: offline with an empty cache raises `DataError` telling the user to run `regimekit fetch`."""
    offline, _ = _cached(tmp_path, inner=NoNetwork(), offline=True)
    with pytest.raises(DataError, match="regimekit fetch"):
        offline.fetch_prices(["^GDAXI"], START, END, "1d", True)
    with pytest.raises(DataError, match="regimekit fetch"):
        offline.fetch_series("VIXCLS", START, END)


def test_e2_t05_offline_uncovered_window_raises(tmp_path: Path) -> None:
    """Offline, a window the cache does not cover is a `DataError`, not a silent partial answer."""
    online, _ = _cached(tmp_path)
    online.fetch_prices(["^GDAXI"], "2024-02-01", "2024-02-29", "1d", True)
    offline, _ = _cached(tmp_path, inner=NoNetwork(), offline=True)
    with pytest.raises(DataError):
        offline.fetch_prices(["^GDAXI"], START, END, "1d", True)


def test_e2_t05_corrupt_meta_is_a_miss(tmp_path: Path) -> None:
    """An unreadable `.meta.json` is treated as a missing entry (refetch online, `DataError` offline)."""
    online, inner = _cached(tmp_path)
    online.fetch_prices(["^GDAXI"], START, END, "1d", True)
    (tmp_path / "fake" / "^GDAXI_1d_adj.meta.json").write_text("{broken", encoding="utf-8")
    offline, _ = _cached(tmp_path, inner=NoNetwork(), offline=True)
    with pytest.raises(DataError):
        offline.fetch_prices(["^GDAXI"], START, END, "1d", True)
    online.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert inner.price_calls == 2


# --- AC4: max_age and pinned generations ------------------------------------------------------------


def test_e2_t05_max_age_zero_forces_refetch(tmp_path: Path) -> None:
    """AC4: `max_age_days=0` refetches on every call."""
    provider, inner = _cached(tmp_path, max_age_days=0)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert inner.price_calls == 2


def test_e2_t05_max_age_uses_fetched_at(tmp_path: Path) -> None:
    """An entry younger than `max_age_days` is served; an older one is refetched."""
    clock = Clock()
    provider, inner = _cached(tmp_path, clock=clock, max_age_days=1)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    clock.now = NOW + timedelta(hours=23)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert inner.price_calls == 1
    clock.now = NOW + timedelta(days=1, seconds=1)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert inner.price_calls == 2


def test_e2_t05_pinned_snapshot_returns_old_generation_after_refetch(tmp_path: Path) -> None:
    """AC4: after a refetch with changed data, `snapshot="<old id>"` still serves the old generation."""
    inner = FakeInner(version=0)
    latest, _ = _cached(tmp_path, inner=inner, max_age_days=0)
    old = latest.fetch_prices(["^GDAXI"], START, END, "1d", True)
    inner.version = 1
    new = latest.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert new.snapshot_id != old.snapshot_id
    assert (tmp_path / "fake" / f"^GDAXI_1d_adj.{old.snapshot_id}.parquet").is_file()

    pinned_old, _ = _cached(tmp_path, inner=NoNetwork(), snapshot=old.snapshot_id)
    assert _bitwise_equal(pinned_old.fetch_prices(["^GDAXI"], START, END, "1d", True), old)
    pinned_new, _ = _cached(tmp_path, inner=NoNetwork(), snapshot=new.snapshot_id)
    assert _bitwise_equal(pinned_new.fetch_prices(["^GDAXI"], START, END, "1d", True), new)
    latest_again, _ = _cached(tmp_path, inner=NoNetwork())
    assert _bitwise_equal(latest_again.fetch_prices(["^GDAXI"], START, END, "1d", True), new)


def test_e2_t05_pinned_unknown_snapshot_raises(tmp_path: Path) -> None:
    """A pinned id that is not in the cache raises `DataError` and never calls `inner`."""
    online, _ = _cached(tmp_path)
    online.fetch_prices(["^GDAXI"], START, END, "1d", True)
    pinned, _ = _cached(tmp_path, inner=NoNetwork(), snapshot="0" * 64)
    with pytest.raises(DataError, match="0000"):
        pinned.fetch_prices(["^GDAXI"], START, END, "1d", True)


def test_e2_t05_unchanged_refetch_keeps_one_generation(tmp_path: Path) -> None:
    """Refetching identical data keeps the id, refreshes `fetched_at`, and writes no extra generation."""
    clock = Clock()
    provider, inner = _cached(tmp_path, clock=clock, max_age_days=0)
    first = provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    clock.now = NOW + timedelta(days=2)
    second = provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert inner.price_calls == 2
    assert first.snapshot_id == second.snapshot_id
    assert sorted(p.name for p in (tmp_path / "fake").glob("*.parquet")) == ["^GDAXI_1d_adj.parquet"]
    meta = json.loads((tmp_path / "fake" / "^GDAXI_1d_adj.meta.json").read_text(encoding="utf-8"))
    assert meta["fetched_at"] == clock.now.isoformat()


def test_e2_t05_generations_newest_first(tmp_path: Path) -> None:
    """`Cache.generations` lists every stored generation, newest first (for E2-T08's discontinuity check)."""
    clock = Clock()
    inner = FakeInner()
    provider, _ = _cached(tmp_path, inner=inner, clock=clock, max_age_days=0)
    ids = []
    for version in range(3):
        inner.version = version
        clock.now = NOW + timedelta(days=version)
        ids.append(provider.fetch_prices(["^GDAXI"], START, END, "1d", True).snapshot_id)
    key = provider.cache.price_key("fake", ["^GDAXI"], "1d", True)
    gens = provider.cache.generations(key)
    assert [g.snapshot_id for g in gens] == ids[::-1]
    assert gens[0].fetched_at > gens[1].fetched_at > gens[2].fetched_at
    older = provider.cache.load_prices(key, ids[0])
    assert older is not None
    assert older.meta["version"] == 0


def test_e2_t05_policy_validation() -> None:
    """Negative `max_age_days` or an empty snapshot are configuration errors with the config path."""
    with pytest.raises(ConfigError) as info:
        SnapshotPolicy(max_age_days=-1)
    assert info.value.path == "data.cache.max_age_days"
    with pytest.raises(ConfigError) as info:
        SnapshotPolicy(snapshot="")
    assert info.value.path == "data.cache.snapshot"


def test_e2_t05_mislabelled_provider_output_raises(tmp_path: Path) -> None:
    """Data whose `provider` differs from the adapter's `name` is refused, not cached under a wrong key."""

    class Mislabelled(FakeInner):
        def fetch_prices(
            self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
        ) -> ObservationSet:
            return dataclasses.replace(
                super().fetch_prices(symbols, start, end, interval, adjusted), provider="x"
            )

        def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
            return dataclasses.replace(super().fetch_series(series_id, start, end), provider="x")

    provider, _ = _cached(tmp_path, inner=Mislabelled())
    with pytest.raises(DataError, match="provider='x'"):
        provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    with pytest.raises(DataError, match="provider='x'"):
        provider.fetch_series("VIXCLS", START, END)
    assert not (tmp_path / "fake").exists() or not any((tmp_path / "fake").iterdir())


def test_e2_t05_open_ended_requests(tmp_path: Path) -> None:
    """A cached `end=None` request covers any end; a cached fixed end does not cover `end=None`."""
    provider, inner = _cached(tmp_path)
    provider.fetch_prices(["^GDAXI"], START, None, "1d", True)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    provider.fetch_prices(["^GDAXI"], START, None, "1d", True)
    assert inner.price_calls == 1
    other, inner2 = _cached(tmp_path / "b")
    other.fetch_prices(["^GDAXI"], START, END, "1d", True)
    other.fetch_prices(["^GDAXI"], START, None, "1d", True)
    assert inner2.price_calls == 2


def test_e2_t05_timestamp_and_tz_aware_bounds(tmp_path: Path) -> None:
    """A timestamped `end` is inclusive at that instant; tz-aware bounds are compared in UTC."""
    provider, _ = _cached(tmp_path)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    obs = provider.fetch_prices(["^GDAXI"], "2024-02-01T00:00:00+00:00", "2024-02-05 00:00:00", "1d", True)
    assert obs.times[0] == np.datetime64("2024-02-01", "ns")
    assert obs.times[-1] == np.datetime64("2024-02-05", "ns")  # 2024-02-05 00:00 itself is included
    shifted = provider.fetch_prices(["^GDAXI"], "2024-02-01T01:00:00+01:00", "2024-02-02", "1d", True)
    assert shifted.times[0] == np.datetime64("2024-02-01", "ns")  # 01:00+01:00 is midnight UTC


def test_e2_t05_pinned_snapshot_must_cover_window(tmp_path: Path) -> None:
    """A pinned generation fetched for a narrower window raises instead of returning partial data."""
    online, _ = _cached(tmp_path)
    obs = online.fetch_prices(["^GDAXI"], "2024-02-01", "2024-02-29", "1d", True)
    pinned, _ = _cached(tmp_path, inner=NoNetwork(), snapshot=obs.snapshot_id)
    with pytest.raises(DataError, match="does not cover"):
        pinned.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert pinned.fetch_prices(["^GDAXI"], "2024-02-05", "2024-02-09", "1d", True).times.shape == (5,)


class NaNInner(FakeInner):
    """Two symbols, the second with a NaN price (a non-trading day for that market)."""

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Like `FakeInner`, with one NaN and distinct per-column scales."""
        obs = super().fetch_prices(symbols, start, end, interval, adjusted)
        prices = obs.prices * np.array([1.0, 1000.0 / 3.0])[: len(symbols)]
        prices[5, -1] = np.nan
        return dataclasses.replace(obs, prices=prices)


def test_e2_t05_cached_result_equals_provider_output(tmp_path: Path) -> None:
    """The cached result equals the provider's output: column order, NaN and every bit of every price."""
    direct = NaNInner().fetch_prices(["^GDAXI", "^GSPC"], START, END, "1d", True)
    provider, _ = _cached(tmp_path, inner=NaNInner())
    cached = provider.fetch_prices(["^GDAXI", "^GSPC"], START, END, "1d", True)
    replay = provider.fetch_prices(["^GDAXI", "^GSPC"], START, END, "1d", True)
    assert cached.symbols == direct.symbols
    assert cached.times.tobytes() == direct.times.tobytes()
    assert cached.prices.tobytes() == direct.prices.tobytes()
    assert np.isnan(cached.prices[5, 1])
    assert cached.meta == direct.meta
    assert _bitwise_equal(cached, replay)
    ser_direct = FakeInner().fetch_series("VIXCLS", START, END)
    ser = provider.fetch_series("VIXCLS", START, END)
    assert ser.values.tobytes() == ser_direct.values.tobytes()
    assert ser.times.tobytes() == ser_direct.times.tobytes()


def test_e2_t05_data_reverting_to_an_older_generation(tmp_path: Path) -> None:
    """A -> B -> A: A becomes current again, its stale copy is removed, B is kept; generations are [A, B]."""
    clock = Clock()
    inner = FakeInner(version=0)
    provider, _ = _cached(tmp_path, inner=inner, clock=clock, max_age_days=0)
    a = provider.fetch_prices(["^GDAXI"], START, END, "1d", True).snapshot_id
    inner.version, clock.now = 1, NOW + timedelta(days=1)
    b = provider.fetch_prices(["^GDAXI"], START, END, "1d", True).snapshot_id
    inner.version, clock.now = 0, NOW + timedelta(days=2)
    again = provider.fetch_prices(["^GDAXI"], START, END, "1d", True).snapshot_id
    assert again == a
    files = sorted(p.name for p in (tmp_path / "fake").glob("*.parquet"))
    assert files == sorted(["^GDAXI_1d_adj.parquet", f"^GDAXI_1d_adj.{b}.parquet"])
    key = provider.cache.price_key("fake", ["^GDAXI"], "1d", True)
    assert [g.snapshot_id for g in provider.cache.generations(key)] == [a, b]


def test_e2_t05_colliding_file_names_do_not_share_data(tmp_path: Path) -> None:
    """`a/b` and `a_b` map to the same file name; each request still gets its own data."""
    provider, inner = _cached(tmp_path)
    first = provider.fetch_series("a/b", START, END)
    second = provider.fetch_series("a_b", START, END)
    assert inner.series_calls == 2
    assert (first.series_id, second.series_id) == ("a/b", "a_b")
    prices = [provider.fetch_prices(s, START, END, "1d", True) for s in (["A+B"], ["A", "B"])]
    assert inner.price_calls == 2
    assert [p.symbols for p in prices] == [("A+B",), ("A", "B")]
    again = provider.fetch_series("a_b", START, END)
    assert again.series_id == "a_b"
    assert inner.series_calls == 2


def test_e2_t05_relabelled_symbols_or_interval_raise(tmp_path: Path) -> None:
    """An adapter answering with other symbols or another interval is refused with a clear `DataError`."""

    class Relabel(FakeInner):
        def fetch_prices(
            self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
        ) -> ObservationSet:
            obs = super().fetch_prices(symbols, start, end, interval, adjusted)
            return dataclasses.replace(obs, symbols=tuple(s + ".DE" for s in symbols))

    provider, _ = _cached(tmp_path, inner=Relabel())
    with pytest.raises(DataError, match="symbols="):
        provider.fetch_prices(["SAP"], START, END, "1d", True)

    class WrongInterval(FakeInner):
        def fetch_prices(
            self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
        ) -> ObservationSet:
            return super().fetch_prices(symbols, start, end, "1d", adjusted)

    provider, _ = _cached(tmp_path / "b", inner=WrongInterval())
    with pytest.raises(DataError, match="interval="):
        provider.fetch_prices(["SAP"], START, END, "1wk", True)


def test_e2_t05_resampled_intervals_need_the_exact_window(tmp_path: Path) -> None:
    """For `1wk`/`1mo` the last bar depends on the window end, so only the identical window is a hit."""
    provider, inner = _cached(tmp_path)
    provider.fetch_prices(["^GDAXI"], START, END, "1wk", True)
    provider.fetch_prices(["^GDAXI"], START, END, "1wk", True)
    assert inner.price_calls == 1
    provider.fetch_prices(["^GDAXI"], START, "2024-02-14", "1wk", True)
    assert inner.price_calls == 2
    daily, inner_d = _cached(tmp_path / "d")
    daily.fetch_prices(["^GDAXI"], START, END, "1d", True)
    daily.fetch_prices(["^GDAXI"], START, "2024-02-14", "1d", True)
    assert inner_d.price_calls == 1


def test_e2_t05_intraday_end_does_not_cover_the_whole_day(tmp_path: Path) -> None:
    """A cached end at 12:00 does not cover a date-only request for that day (which means the whole day)."""
    provider, inner = _cached(tmp_path)
    provider.fetch_prices(["^GDAXI"], START, "2024-02-01T12:00:00", "1h", True)
    provider.fetch_prices(["^GDAXI"], START, "2024-02-01", "1h", True)
    assert inner.price_calls == 2
    provider.fetch_prices(["^GDAXI"], START, "2024-02-01T12:00:00", "1h", True)
    assert inner.price_calls == 2


def test_e2_t05_bad_dates_are_data_errors(tmp_path: Path) -> None:
    """Unparseable window bounds raise `DataError` with the config path, even on a cache hit."""
    provider, _ = _cached(tmp_path)
    provider.fetch_prices(["^GDAXI"], START, None, "1d", True)
    with pytest.raises(DataError) as info:
        provider.fetch_prices(["^GDAXI"], START, "not-a-date", "1d", True)
    assert info.value.path == "data.end"
    with pytest.raises(DataError) as info:
        provider.fetch_series("VIXCLS", "yesterday-ish", END)
    assert info.value.path == "data.start"


def test_e2_t05_unreadable_parquet_is_a_data_error(tmp_path: Path) -> None:
    """A truncated cache file raises `DataError` naming the cache, not a pyarrow exception."""
    provider, _ = _cached(tmp_path)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    (tmp_path / "fake" / "^GDAXI_1d_adj.parquet").write_bytes(b"PAR1 truncated")
    with pytest.raises(DataError, match="unreadable"):
        provider.fetch_prices(["^GDAXI"], START, END, "1d", True)


def test_e2_t05_malformed_generation_meta_is_skipped(tmp_path: Path) -> None:
    """`generations()` ignores entries whose `fetched_at` cannot be parsed; such entries count as stale."""
    provider, _ = _cached(tmp_path)
    provider.fetch_prices(["^GDAXI"], START, END, "1d", True)
    meta_path = tmp_path / "fake" / "^GDAXI_1d_adj.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = "sometime"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    key = provider.cache.price_key("fake", ["^GDAXI"], "1d", True)
    assert provider.cache.generations(key) == []
    stale, inner = _cached(tmp_path)
    stale.fetch_prices(["^GDAXI"], START, END, "1d", True)
    assert inner.price_calls == 1


def test_e2_t05_policy_rejects_malformed_pins_and_nan() -> None:
    """A pinned snapshot must be a 64-hex id (typos like `Latest` fail early); NaN max age is rejected."""
    with pytest.raises(ConfigError) as info:
        SnapshotPolicy(snapshot="Latest")
    assert info.value.path == "data.cache.snapshot"
    with pytest.raises(ConfigError):
        SnapshotPolicy(max_age_days=float("nan"))
    assert SnapshotPolicy(snapshot="a" * 64).pinned == "a" * 64
    assert SnapshotPolicy().pinned is None
