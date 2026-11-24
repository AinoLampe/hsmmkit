"""Local parquet cache with snapshot ids, generations, `max_age_days` and offline replay (FR-D03, E2-T05).

Layout under `cache_dir` (default `~/.cache/regimekit`):

- prices: `<provider>/<symbols>_<interval>_<adj|raw>.parquet` (`symbols` joined by `+`),
- covariates: `<provider>/series_<series_id>.parquet`,

each with a `.meta.json` (`snapshot_id`, `fetched_at`, versions, request parameters, payload fields). When
a refetch yields different data, the previous generation is kept as `<name>.<snapshot_id>.parquet` (+ meta),
so a pinned snapshot stays replayable and E2-T08 can compare the two latest generations.

`snapshot_id = sha256(provider | symbols | interval | adjusted | parquet bytes)` for prices and
`sha256(provider | series_id | parquet bytes)` for covariates. Parquet is written with fixed pyarrow options
(no pandas metadata, no statistics, fixed compression), so identical data gives identical bytes and ids for a
given pyarrow version (recorded in the meta file).

The data layer must not import `regimekit.finance` (profile rule), so the wrapped provider is typed by the
structural `CacheableProvider` protocol declared here; the finance adapters satisfy it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.utils.errors import ConfigError, DataError

DEFAULT_CACHE_DIR = "~/.cache/regimekit"
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_HEX64 = re.compile(r"[0-9a-f]{64}")


@runtime_checkable
class CacheableProvider(Protocol):
    """Anything with a provider `name` and the two fetch methods (e.g. every finance adapter)."""

    name: str

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Fetch prices for `symbols` on `[start, end]`."""
        ...

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """Fetch one covariate series on `[start, end]`."""
        ...


@dataclass(frozen=True)
class SnapshotPolicy:
    """How `CachedProvider` chooses between cache and provider (config `data.cache`, CLI `--offline`).

    Attributes:
        snapshot: `"latest"` (serve the newest generation if fresh, else refetch) or a snapshot id to pin.
        max_age_days: A `latest` entry older than this is refetched; `0` always refetches.
        offline: Never call the provider; serve from the cache or raise `DataError`.
    """

    snapshot: str = "latest"
    max_age_days: float = 1.0
    offline: bool = False

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not self.snapshot:
            raise ConfigError("must be 'latest' or a snapshot id", path="data.cache.snapshot")
        if self.snapshot != "latest" and not _HEX64.fullmatch(self.snapshot):
            raise ConfigError(
                f"must be 'latest' or a 64-character hex snapshot id, got {self.snapshot!r}",
                path="data.cache.snapshot",
            )
        if not self.max_age_days >= 0:  # also rejects NaN
            raise ConfigError(f"must be >= 0, got {self.max_age_days}", path="data.cache.max_age_days")

    @property
    def pinned(self) -> str | None:
        """The pinned snapshot id, or `None` for `latest`."""
        return None if self.snapshot == "latest" else self.snapshot


@dataclass(frozen=True)
class CacheKey:
    """One cache entry: provider directory, file stem, and the snapshot-id prefix of the request."""

    provider: str
    stem: str
    id_prefix: str


@dataclass(frozen=True)
class Generation:
    """A stored generation of a cache entry."""

    snapshot_id: str
    fetched_at: datetime
    path: Path


def _safe(name: str) -> str:
    return _UNSAFE.sub("_", name)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover  (always installed in practice)
        return "unknown"


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        table, sink, compression="zstd", compression_level=3, use_dictionary=False, write_statistics=False
    )
    return bytes(sink.getvalue().to_pybytes())


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _naive_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def _upper(end: str) -> pd.Timestamp:
    """Exclusive upper bound of an inclusive `end`; a date-only `end` covers that whole day."""
    hi = _naive_utc(end)
    if len(end.strip()) <= 10 and hi == hi.normalize():
        return hi + pd.Timedelta(days=1)
    return hi + pd.Timedelta(1, unit="ns")


def _parse_window(start: str, end: str | None) -> None:
    """Raise `DataError` for unparseable window bounds (before any cache or provider access)."""
    for value, path in ((start, "data.start"), (end, "data.end")):
        if value is None:
            continue
        try:
            _naive_utc(value)
        except (ValueError, TypeError) as exc:
            raise DataError(f"unparseable date {value!r}: {exc}", path=path) from None


def _window_mask(times: np.ndarray, start: str, end: str | None) -> np.ndarray:
    """Inclusive `[start, end]` (see `_upper`)."""
    mask = times >= _naive_utc(start).to_datetime64()
    if end is not None:
        mask &= times < _upper(end).to_datetime64()
    return np.asarray(mask)


def _covers(request: dict[str, Any], start: str, end: str | None, exact: bool = False) -> bool:
    """Whether a cached request window contains `[start, end]` (an open cached end covers any end).

    With `exact=True` (resampled `1wk`/`1mo` bars, whose last bar depends on where the window ends) only the
    identical window counts.
    """
    try:
        cached_start, cached_end = request["start"], request["end"]
        if exact:
            same_end = (cached_end is None and end is None) or (
                cached_end is not None and end is not None and _upper(cached_end) == _upper(end)
            )
            return same_end and _naive_utc(cached_start) == _naive_utc(start)
        if _naive_utc(cached_start) > _naive_utc(start):
            return False
        if cached_end is None:
            return True
        return end is not None and _upper(cached_end) >= _upper(end)
    except (KeyError, TypeError, ValueError):
        return False


def _fetched_at(meta: dict[str, Any] | None) -> datetime | None:
    try:
        return datetime.fromisoformat(meta["fetched_at"]) if meta is not None else None
    except (KeyError, TypeError, ValueError):
        return None


class Cache:
    """Parquet store for `ObservationSet`/`CovariateSeries` generations.

    Args:
        root: Cache directory (`~` is expanded); nothing is created until the first write.
        now: Injectable UTC clock for `fetched_at` and staleness.
    """

    def __init__(
        self, root: Path | str = DEFAULT_CACHE_DIR, now: Callable[[], datetime] | None = None
    ) -> None:
        """Configure the cache; no file access happens here."""
        self.root = Path(root).expanduser()
        self.now: Callable[[], datetime] = now if now is not None else (lambda: datetime.now(UTC))

    # --- keys and paths ---------------------------------------------------------------------------------

    def price_key(
        self, provider: str, symbols: list[str] | tuple[str, ...], interval: str, adjusted: bool
    ) -> CacheKey:
        """Key of a price request."""
        stem = f"{_safe('+'.join(symbols))}_{_safe(interval)}_{'adj' if adjusted else 'raw'}"
        return CacheKey(provider, stem, f"{provider}|{','.join(symbols)}|{interval}|{adjusted}|")

    def series_key(self, provider: str, series_id: str) -> CacheKey:
        """Key of a covariate-series request."""
        return CacheKey(provider, f"series_{_safe(series_id)}", f"{provider}|{series_id}|")

    def _dir(self, key: CacheKey) -> Path:
        return self.root / _safe(key.provider)

    def _paths(self, key: CacheKey, snapshot_id: str | None = None) -> tuple[Path, Path]:
        name = key.stem if snapshot_id is None else f"{key.stem}.{snapshot_id}"
        return self._dir(key) / f"{name}.parquet", self._dir(key) / f"{name}.meta.json"

    @staticmethod
    def _read_meta(path: Path) -> dict[str, Any] | None:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return meta if isinstance(meta, dict) and isinstance(meta.get("snapshot_id"), str) else None

    def _read_own_meta(self, key: CacheKey, path: Path) -> dict[str, Any] | None:
        """Meta at `path` if it belongs to `key` (file names can collide, e.g. `a/b` vs `a_b`, or by case)."""
        meta = self._read_meta(path)
        return meta if meta is not None and meta.get("key") == key.id_prefix else None

    def meta(self, key: CacheKey, snapshot_id: str | None = None) -> dict[str, Any] | None:
        """Meta of the current generation (`snapshot_id=None`) or of a specific one; `None` if absent."""
        current = self._read_own_meta(key, self._paths(key)[1])
        if snapshot_id is None or (current is not None and current["snapshot_id"] == snapshot_id):
            if current is not None and self._paths(key)[0].is_file():
                return current
            return None
        parquet, meta_path = self._paths(key, snapshot_id)
        meta = self._read_own_meta(key, meta_path)
        return meta if meta is not None and meta["snapshot_id"] == snapshot_id and parquet.is_file() else None

    def _parquet_path(self, key: CacheKey, snapshot_id: str) -> Path:
        current = self._read_own_meta(key, self._paths(key)[1])
        if current is not None and current["snapshot_id"] == snapshot_id:
            return self._paths(key)[0]
        return self._paths(key, snapshot_id)[0]

    def generations(self, key: CacheKey) -> list[Generation]:
        """All stored generations of `key`, newest `fetched_at` first."""
        found: list[Generation] = []
        current = self.meta(key)
        fetched = _fetched_at(current)
        if current is not None and fetched is not None:
            found.append(Generation(current["snapshot_id"], fetched, self._paths(key)[0]))
        directory = self._dir(key)
        if directory.is_dir():
            for meta_path in directory.glob(f"{key.stem}.*.meta.json"):
                sid = meta_path.name[len(key.stem) + 1 : -len(".meta.json")]
                if not _HEX64.fullmatch(sid) or (current is not None and sid == current["snapshot_id"]):
                    continue
                fetched = _fetched_at(self.meta(key, sid))
                if fetched is not None:
                    found.append(Generation(sid, fetched, self._paths(key, sid)[0]))
        return sorted(found, key=lambda g: g.fetched_at, reverse=True)

    # --- writing ------------------------------------------------------------------------------------------

    def _store(self, key: CacheKey, table: pa.Table, request: dict[str, Any], payload: dict[str, Any]) -> str:
        data = _parquet_bytes(table)
        snapshot_id = hashlib.sha256(key.id_prefix.encode("utf-8") + data).hexdigest()
        parquet, meta_path = self._paths(key)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        current = self.meta(key)
        if current is not None and current["snapshot_id"] != snapshot_id:
            old_parquet, old_meta = self._paths(key, current["snapshot_id"])
            parquet.replace(old_parquet)
            meta_path.replace(old_meta)
        if current is None or current["snapshot_id"] != snapshot_id:
            _atomic_write(parquet, data)
        meta = {
            "snapshot_id": snapshot_id,
            "key": key.id_prefix,
            "fetched_at": self.now().isoformat(),
            "provider_version": f"regimekit {_package_version('regimekit')}",
            "pyarrow_version": pa.__version__,
            "request": request,
            "payload": payload,
        }
        _atomic_write(meta_path, json.dumps(meta, indent=1, default=str).encode("utf-8"))
        for stale in self._paths(key, snapshot_id):  # the new current may equal an older generation
            stale.unlink(missing_ok=True)
        return snapshot_id

    def store_prices(self, obs: ObservationSet, request: dict[str, Any]) -> str:
        """Write `obs` as the current generation of its key; return its snapshot id."""
        if obs.times is None:
            raise DataError("only observations with timestamps can be cached", path="times")
        if "interval" not in obs.meta or "adjusted" not in obs.meta:
            raise DataError("only provider observations (meta interval/adjusted) can be cached", path="meta")
        interval, adjusted = str(obs.meta["interval"]), bool(obs.meta["adjusted"])
        key = self.price_key(obs.source, obs.series_names, interval, adjusted)
        arrays = [pa.array(obs.times, type=pa.timestamp("ns"))]
        arrays += [
            pa.array(np.ascontiguousarray(obs.values[:, j]), type=pa.float64())
            for j in range(len(obs.series_names))
        ]
        table = pa.Table.from_arrays(arrays, names=["time", *obs.series_names])
        payload = {
            "symbols": list(obs.series_names),
            "interval": interval,
            "adjusted": adjusted,
            "provider": obs.source,
            "meta": {k: v for k, v in obs.meta.items() if k not in ("interval", "adjusted")},
        }
        return self._store(key, table, request, payload)

    def store_series(self, ser: CovariateSeries, request: dict[str, Any]) -> str:
        """Write `ser` as the current generation of its key; return its snapshot id."""
        key = self.series_key(ser.source, ser.series_id)
        if ser.times is None:
            raise DataError("only series with timestamps can be cached", path="times")
        table = pa.Table.from_arrays(
            [pa.array(ser.times, type=pa.timestamp("ns")), pa.array(ser.values, type=pa.float64())],
            names=["time", "value"],
        )
        payload = {
            "name": ser.name,
            "series_id": ser.series_id,
            "frequency": ser.frequency,
            "provider": ser.source,
        }
        return self._store(key, table, request, payload)

    # --- reading ------------------------------------------------------------------------------------------

    def _table(self, key: CacheKey, snapshot_id: str | None) -> tuple[pa.Table, dict[str, Any]] | None:
        meta = self.meta(key, snapshot_id)
        if meta is None:
            return None
        try:
            table = pq.read_table(self._parquet_path(key, meta["snapshot_id"]))
        except (OSError, pa.ArrowException) as exc:
            raise DataError(
                f"cache file for {key.stem!r} is unreadable: {exc}", path="data.cache.dir"
            ) from exc
        return table, meta

    def load_prices(self, key: CacheKey, snapshot_id: str | None = None) -> ObservationSet | None:
        """The current (or a pinned) generation as an `ObservationSet`, or `None` if not cached."""
        loaded = self._table(key, snapshot_id)
        if loaded is None:
            return None
        table, meta = loaded
        payload = meta["payload"]
        values = (
            np.column_stack(
                [table.column(j + 1).to_numpy().astype("float64") for j in range(table.num_columns - 1)]
            )
            if table.num_columns > 1
            else np.empty((table.num_rows, 0))
        )
        return ObservationSet(
            series_names=tuple(payload["symbols"]),
            times=table.column(0).to_numpy().astype("datetime64[ns]"),
            values=values,
            source=payload["provider"],
            snapshot_id=meta["snapshot_id"],
            meta={
                **dict(payload.get("meta") or {}),
                "interval": payload["interval"],
                "adjusted": bool(payload["adjusted"]),
            },
        )

    def load_series(self, key: CacheKey, snapshot_id: str | None = None) -> CovariateSeries | None:
        """The current (or a pinned) generation as a `CovariateSeries`, or `None` if not cached."""
        loaded = self._table(key, snapshot_id)
        if loaded is None:
            return None
        table, meta = loaded
        payload = meta["payload"]
        return CovariateSeries(
            name=payload["name"],
            times=table.column(0).to_numpy().astype("datetime64[ns]"),
            values=table.column(1).to_numpy().astype("float64"),
            source=payload["provider"],
            series_id=payload["series_id"],
            snapshot_id=meta["snapshot_id"],
            frequency=payload["frequency"],
        )

    def is_fresh(self, meta: dict[str, Any], max_age_days: float) -> bool:
        """Whether an entry fetched at `meta["fetched_at"]` is younger than `max_age_days`."""
        fetched = _fetched_at(meta)
        try:
            return fetched is not None and self.now() - fetched < timedelta(days=max_age_days)
        except TypeError:  # naive vs aware clock: treat as stale rather than crash
            return False


def _clip_obs(obs: ObservationSet, start: str, end: str | None) -> ObservationSet:
    assert obs.times is not None
    mask = _window_mask(obs.times, start, end)
    return ObservationSet(
        series_names=obs.series_names,
        times=obs.times[mask],
        values=obs.values[mask],
        source=obs.source,
        snapshot_id=obs.snapshot_id,
        meta=obs.meta,
    )


def _clip_series(ser: CovariateSeries, start: str, end: str | None) -> CovariateSeries:
    assert ser.times is not None
    mask = _window_mask(ser.times, start, end)
    return CovariateSeries(
        name=ser.name,
        times=ser.times[mask],
        values=ser.values[mask],
        source=ser.source,
        series_id=ser.series_id,
        snapshot_id=ser.snapshot_id,
        frequency=ser.frequency,
    )


def _check_labels(returned: dict[str, Any], requested: dict[str, Any]) -> None:
    """The adapter must label its result exactly as requested, or it would be cached under another key."""
    for field, value in requested.items():
        if returned[field] != value:
            raise DataError(
                f"provider {requested['provider']!r} returned data labelled {field}={returned[field]!r} for "
                f"a request with {field}={value!r}; cannot cache it",
                path="data.provider",
            )


class CachedProvider:
    """Wraps any `CacheableProvider`: serves from the cache per `policy`, fills `snapshot_id`.

    Results are always read back from the cache, so an online fetch and a later offline replay of the same
    request are bitwise identical.

    Args:
        inner: The provider adapter to wrap.
        cache: Where to store and read generations.
        policy: `latest`/pinned snapshot, `max_age_days`, `offline`.
    """

    def __init__(self, inner: CacheableProvider, cache: Cache, policy: SnapshotPolicy | None = None) -> None:
        """Wrap `inner`; no file or network access happens here."""
        self.inner = inner
        self.cache = cache
        self.policy = policy if policy is not None else SnapshotPolicy()

    @property
    def name(self) -> str:
        """The wrapped provider's name."""
        return self.inner.name

    @property
    def supports_intraday(self) -> bool:
        """Delegated to the wrapped provider."""
        return bool(getattr(self.inner, "supports_intraday", False))

    @property
    def tos_note(self) -> str:
        """Delegated to the wrapped provider."""
        return str(getattr(self.inner, "tos_note", ""))

    def _resolve[T](
        self,
        key: CacheKey,
        what: str,
        start: str,
        end: str | None,
        load: Callable[[CacheKey, str | None], T | None],
        fetch_and_store: Callable[[], str],
        exact: bool = False,
    ) -> T:
        def loaded(snapshot_id: str | None) -> T:
            result = load(key, snapshot_id)
            if result is None:  # pragma: no cover  (meta was just checked or written; a concurrent delete)
                raise DataError(f"{what} vanished from the cache {self.cache.root}", path="data.cache.dir")
            return result

        _parse_window(start, end)
        pinned = self.policy.pinned
        if pinned is not None:
            meta = self.cache.meta(key, pinned)
            if meta is None:
                raise DataError(
                    f"snapshot {pinned} of {what} is not in the cache {self.cache.root}",
                    path="data.cache.snapshot",
                )
            if not _covers(meta["request"], start, end, exact):
                raise DataError(
                    f"snapshot {pinned} of {what} does not cover [{start}, {end}]", path="data.cache.snapshot"
                )
            return loaded(pinned)
        meta = self.cache.meta(key)
        usable = meta is not None and _covers(meta["request"], start, end, exact)
        if usable and (
            self.policy.offline or self.cache.is_fresh(cast(dict[str, Any], meta), self.policy.max_age_days)
        ):
            return loaded(None)
        if self.policy.offline:
            reason = "is not cached" if meta is None else f"is not cached for [{start}, {end}]"
            raise DataError(f"{what} {reason} (offline); run `regimekit fetch` first", path="data.cache")
        fetch_and_store()
        return loaded(None)

    def fetch_prices(
        self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool
    ) -> ObservationSet:
        """Prices from the cache or the wrapped provider, clipped to `[start, end]`, `snapshot_id` set."""
        key = self.cache.price_key(self.inner.name, symbols, interval, adjusted)
        request = {
            "provider": self.inner.name,
            "symbols": list(symbols),
            "interval": interval,
            "adjusted": adjusted,
            "start": start,
            "end": end,
        }

        def fetch_and_store() -> str:
            obs = self.inner.fetch_prices(symbols, start, end, interval, adjusted)
            _check_labels(
                {
                    "provider": obs.source,
                    "symbols": obs.series_names,
                    "interval": obs.meta.get("interval"),
                    "adjusted": obs.meta.get("adjusted"),
                },
                {
                    "provider": self.inner.name,
                    "symbols": tuple(symbols),
                    "interval": interval,
                    "adjusted": adjusted,
                },
            )
            return self.cache.store_prices(obs, request)

        what = f"{self.inner.name} prices {symbols} {interval} {'adjusted' if adjusted else 'raw'}"
        exact = interval in ("1wk", "1mo")
        obs = self._resolve(key, what, start, end, self.cache.load_prices, fetch_and_store, exact)
        return _clip_obs(obs, start, end)

    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries:
        """A covariate series from the cache or the wrapped provider, clipped, with `snapshot_id` set."""
        key = self.cache.series_key(self.inner.name, series_id)
        request = {"provider": self.inner.name, "series_id": series_id, "start": start, "end": end}

        def fetch_and_store() -> str:
            ser = self.inner.fetch_series(series_id, start, end)
            _check_labels(
                {"provider": ser.source, "series_id": ser.series_id},
                {"provider": self.inner.name, "series_id": series_id},
            )
            return self.cache.store_series(ser, request)

        what = f"{self.inner.name} series {series_id!r}"
        ser = self._resolve(key, what, start, end, self.cache.load_series, fetch_and_store)
        return _clip_series(ser, start, end)
