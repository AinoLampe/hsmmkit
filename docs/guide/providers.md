# Data providers & terms of service

regimekit fetches data from third-party providers (yfinance, stooq, FRED, CBOE, tiingo). Provider data is
never redistributed or committed to this repository (ADR-0002); you are responsible for complying with each
provider's own terms of service when you fetch and cache data locally.

## yfinance (default price provider)

`regimekit.finance.providers.yfinance.YFinanceProvider` downloads indices, ETFs and FX from Yahoo! Finance.
It needs the optional extra: `uv sync --extra providers` (`yfinance >= 1.7` and `curl_cffi`, used to
impersonate a browser session).

- Intervals `1h`, `1d`, `1wk`, `1mo` map one-to-one. `adjusted=True` requests split/dividend-adjusted
  prices (`auto_adjust=True`); `adjusted=False` requests the raw `Close`.
- `start`/`end` are both inclusive (yfinance's own `end` is exclusive; the adapter compensates).
- Intraday history is capped (`1h`: 730 days, table `INTRADAY_CAP_DAYS`). Older `start` dates are clipped with
  a `UserWarning` and a log line.
- Downloads are retried (`with_retries`, 5 attempts) and spaced at least 0.5 s apart.
- Multiple symbols are aligned on the union of bar times (NaN where missing); alignment to the master grid
  happens later, in the data-preparation step.
- Every fetch logs `provider="yfinance"` together with the terms-of-service note at INFO level.

!!! warning "Terms of service"
    Yahoo! data is licensed for personal, non-commercial use. regimekit never commits or redistributes it
    (ADR-0002); keep your local cache private and check Yahoo's current terms before any other use.

## stooq (daily fallback, EU coverage)

`regimekit.finance.providers.stooq.StooqProvider` is an end-of-day fallback for when Yahoo is unavailable.

- **API key.** Set the environment variable `STOOQ_API_KEY`. Without it the adapter raises a `ProviderError`
  with setup instructions *before* any network call. The key is never logged and never appears in errors.
- **Intervals.** Daily only (`supports_intraday = False`; `1h` raises). `1wk` and `1mo` are resampled from
  daily data as the last close of each period, labelled with the last actual trading date; `meta` records
  the resampling. The `[start, end]` window is applied to the daily data first, so a period cut by the
  window edge yields a partial first or last bar.
- **Adjustment.** Stooq closes are split-adjusted only, so `adjusted` is echoed but does not change prices.
- **Quota.** A client-side counter (default 100 calls per UTC day, `max_calls_per_day`) is persisted in
  `~/.cache/regimekit/stooq_quota.json` and shared by every adapter instance. One call is charged per
  symbol; retries are free. Exceeding it raises `ProviderError("stooq quota ...")`. Cache aggressively.
- **Symbols.** Yahoo-style names are mapped (`^GDAXI` -> `^dax`, `^GSPC` -> `^spx`, see `STOOQ_SYMBOLS`);
  anything else is passed through unchanged. Results keep the names you asked for.

!!! warning "Terms of service"
    Stooq data is for personal use under stooq.com's terms; do not redistribute it.

## fred (macro and volatility covariates)

`regimekit.finance.providers.fred.FredProvider` fetches covariate series such as `VIXCLS` (VIX close) and
`T10Y2Y` (10y-2y Treasury spread) from FRED. The default client uses `fredapi` (`uv sync --extra providers`).

- **API key.** Set the environment variable `FRED_API_KEY` (free at fredaccount.stlouisfed.org). Without it
  the adapter raises a `ProviderError` with setup instructions *before* any network call. The key is never
  logged and never appears in errors.
- **Values.** FRED marks missing observations (e.g. market holidays) with `"."`; these become `NaN`. How NaN
  is filled is decided later by the covariate's `missing`/`max_staleness` settings.
- **Times and frequency.** Observation dates are returned as UTC midnight. `frequency` is taken from FRED's
  series metadata (`D`, `W`, `M`, `Q` -> `d`, `w`, `m`, `q`); other frequencies (annual, semiannual,
  biweekly) raise `ProviderError`. Alignment and publication lag are applied by the data preparation.
- **Rate limit.** At most 120 requests in any rolling 60-second window, shared by all adapter instances in
  the process; each fetch makes two requests (metadata, observations) and retries count too. Excess calls
  wait rather than fail.
- `fetch_prices` raises `ProviderError`: FRED is not a price provider.

!!! warning "Terms of service"
    FRED data is provided by the Federal Reserve Bank of St. Louis under its terms of use, and some series
    carry third-party copyrights. Do not redistribute it.

## cboe (VIX / VVIX history files)

`regimekit.finance.providers.cboe.CboeProvider` reads CBOE's daily index history CSV files. No API key and no
extra dependency are needed.

- **Series.** A `series_id` is a key of the adapter's `urls` mapping. The defaults (`DEFAULT_URLS`) are
  `VIX_History.csv` and `VVIX_History.csv` on `cdn.cboe.com`; pass `urls={...}` to point at other files or
  mirrors (this replaces the defaults). Unknown ids raise `ProviderError` without a network call.
- **Values.** Files with `DATE, OPEN, HIGH, LOW, CLOSE` use `CLOSE`; single-value files such as
  `DATE, VVIX` use that column. Dates (`MM/DD/YYYY`) become UTC midnight; `frequency` is `"d"`.
- Downloads are retried (5 attempts) and spaced at least 0.5 s apart. `fetch_prices` raises `ProviderError`.

!!! warning "Terms of service"
    CBOE publishes index history for informational use under cboe.com's terms; do not redistribute it.

## tiingo (optional, clean US end-of-day)

`regimekit.finance.providers.tiingo.TiingoProvider` fetches US end-of-day prices from Tiingo's REST API. It
uses only the Python standard library, so no extra dependency is needed and the adapter is always available.

- **API key.** Set the environment variable `TIINGO_API_KEY`. Without it the adapter raises a
  `ProviderError` with setup instructions *before* any network call. The key is sent in the `Authorization`
  header (never in a URL) and is never logged or written to the quota file.
- **Prices.** `adjusted=True` uses `adjClose`, `adjusted=False` uses `close`. Daily only
  (`supports_intraday = False`); `1wk`/`1mo` are resampled from daily data like stooq (last close of each
  period).
- **Free-tier limits** are enforced client-side and persisted in `~/.cache/regimekit/tiingo_quota.json`
  (shared by every adapter instance and process): 50 requests in any rolling hour (`max_per_hour`), 1000 in
  any rolling 24 hours (`max_per_day`), and 500 distinct symbols per UTC calendar month
  (`max_symbols_per_month`). One request is charged per symbol before downloading; a batch that would exceed
  a limit is refused whole with a `ProviderError`. Retries are not charged.

!!! warning "Terms of service"
    Tiingo data is licensed for the account holder's own use under tiingo.com's terms; do not redistribute it.

## Local cache, snapshots and offline replay

`regimekit.data.cache.CachedProvider` wraps a provider (the `fetch` command wires it up) and stores what was
fetched as parquet under `data.cache.dir` (default `~/.cache/regimekit`, `~` expanded):

```text
<cache_dir>/<provider>/<symbols>_<interval>_<adj|raw>.parquet      # prices (symbols joined by "+")
<cache_dir>/<provider>/series_<series_id>.parquet                  # covariates
<name>.meta.json                                                   # snapshot_id, fetched_at, versions, request
<name>.<snapshot_id>.parquet (+ .meta.json)                        # older generations
```

- **Snapshot ids.** `snapshot_id = sha256(provider | symbols | interval | adjusted | parquet bytes)` (covariates:
  `sha256(provider | series_id | parquet bytes)`). Parquet is written with fixed options, so identical data
  gives identical ids on any machine with the same pyarrow version (recorded in the meta file). Adapters
  return an empty `snapshot_id`; the cache fills it, and prepared datasets and result bundles record it.
- **`snapshot: latest`** serves the cached entry while it is younger than `max_age_days` (default 1 day;
  `0` always refetches) and its request window covers the one asked for; otherwise it refetches. When the
  refetched data differs, the previous generation is kept as `<name>.<old_id>.parquet`.
- **`snapshot: <id>`** serves exactly that generation, never calls the provider, and raises `DataError` if the
  generation is missing or does not cover the requested window.
- **Offline** (`--offline`): the provider is never called; a missing entry raises
  `DataError("... run `regimekit fetch` first")`.
- Results are always read back from the cache, so an online run and a later offline replay of the same
  request are bitwise identical. For `1h`/`1d`, a narrower window than the cached one is served by
  clipping; for resampled `1wk`/`1mo` bars (whose last bar depends on where the window ends) only the
  identical window is a hit.
- Each meta file records the exact request key, so two requests whose file names collide (unsafe
  characters become `_`; case-insensitive file systems) never serve each other's data.
- Older generations are never deleted automatically; remove `<name>.<id>.parquet` files you no longer need.

## `regimekit fetch`

`regimekit [--out DIR] [--offline] fetch CONFIG` runs the data pipeline end to end:

1. Each provider named in `data.primary` and `data.covariates` is wrapped in the cache (see above),
   using `data.cache.snapshot`, `data.cache.max_age_days` and `--offline`. Each provider's
   terms-of-service note is logged at INFO level. `csv` files are looked up relative to the config
   file's directory (`symbols: [DEMO]` reads `DEMO.csv`, `series: VIXCLS` reads `VIXCLS.csv`).
   stooq/tiingo keep their client-side quota files in `data.cache.dir`.
2. The `Preparer` is fitted on the initial training window: the first
   `evaluation.backtest.initial_train_years` years, capped at the full span. The data is then
   transformed (see [Data contracts](contracts.md)).
3. Outputs go to `--out`, or to `project.output_dir` when `--out` is not given:
   - `prepared.parquet`: the `PreparedDataset`, bit-exact (`regimekit.persistence.results.read_prepared`
     reads it back);
   - `prep_stats.json`: the standardization statistics;
   - `snapshots.json`: the snapshot id of the primary and each covariate. `--offline` re-runs from the
     *current* cache generations (bitwise identical as long as the cache has not been refreshed since);
     it does not check the ids in `snapshots.json`. A pinned `data.cache.snapshot` id only works for runs
     without covariates, because one id names one cache entry.
   - A failing run removes `prepared.parquet`, `prep_stats.json` and `snapshots.json` left in the output
     directory by an earlier run, so the new report never sits next to a stale dataset.
   - `quality.json` / `quality.md`: the data-quality report.

The report lists, each with its timestamps:
- gaps against a plain weekday calendar (daily data only; exchange holidays show up by design);
- duplicated timestamps;
- zero or negative prices;
- log-returns beyond 8 sigma. Sigma is the robust MAD scale, and these are flagged only.
- adjustment discontinuities: the maximum relative price difference between the two newest cache
  generations of the primary request, which exist after a refetch that changed the data. Anything
  above 1e-6 is flagged.
- each covariate's staleness histogram, in grid steps since the print became visible, with
  `no_value` for none yet;
- the rows removed for missing prices, for warm-up and by each covariate's missing policy. When
  preparation fails, only the missing-price count is reported.

The report is also written when preparation fails.

Exit codes: 0 success, 1 output cannot be written (`PersistenceError`), 2 invalid config, 3 data
error (for example `missing: error` with a stale covariate), 5 provider error.
