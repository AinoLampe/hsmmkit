# E2-T01 — `DataProvider` protocol, registry, `csv` adapter, shared provider contract test-suite

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-D01 | M1 | §5.5 (`DataProvider`), §5.6 Providers; Part II §15.7 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendment 3: providers live in `regimekit.finance.providers`). No behavioural change for the finance profile.

## Goal
Establish the provider abstraction and the one test-suite every adapter must pass, using the local CSV adapter as reference implementation.

## Context (self-contained)
```python
class DataProvider(Protocol):
    name: str; supports_intraday: bool; tos_note: str
    def fetch_prices(self, symbols: list[str], start: str, end: str | None, interval: str, adjusted: bool) -> ObservationSet: ...
    def fetch_series(self, series_id: str, start: str, end: str | None) -> CovariateSeries: ...   # macro providers; price-only providers raise ProviderError
```
- Location: `src/regimekit/finance/providers/` — providers are part of the finance *profile*, not of the domain-neutral `regimekit.data` (ADR-0012). `regimekit.data` never imports them; `cli/` and `project.py` resolve providers by name through the registry.
- Registry: `finance/providers/base.py` with `register(provider_cls)` decorator and `get_provider(name, **kwargs)`; duplicate names raise `ProviderError`. Adding a provider must only touch `regimekit.finance.providers`.
- Common conversion helpers in `base.py`: to UTC `datetime64[ns]`, float64, sorted, de-duplicated timestamps (keep last), `snapshot_id = sha256(provider, symbols, interval, adjusted, raw parquet bytes)` computed by E2-T05's cache — here adapters return `snapshot_id=""` and the cache fills it (document this).
- Retry helper `with_retries(fn, max_attempts=5, backoff=exponential)` for network adapters; `throttle(min_interval_s)` decorator.
- `csv` adapter: `CsvProvider(root: Path)`; `fetch_prices` reads `<root>/<symbol>.csv` with columns `time, open, high, low, close, adj_close, volume` (only `close`/`adj_close` required), `fetch_series` reads `<root>/<series_id>.csv` with `time, value`; `"."` and empty → NaN.
- Contract suite (`tests/finance/provider_contract.py`): parametrized over adapter fixtures; asserts contract validity (`validate()`), UTC, monotonic times, float64, `interval` echoed, `adjusted` respected, `start/end` filtering inclusive, `tos_note` non-empty, unknown symbol → `ProviderError`.

## Creates
- `src/regimekit/finance/providers/{__init__,base,csv}.py`
- `tests/finance/provider_contract.py`, `tests/unit/test_e2_t01_csv_provider.py`, `tests/fixtures/csv/{DEMO.csv,VIXCLS.csv}` (synthetic, not real provider data)

## Acceptance criteria
1. `CsvProvider` passes the full contract suite.
2. `get_provider("csv", root=...)` resolves; registering a second `"csv"` raises `ProviderError`.
3. `with_retries` retries exactly `max_attempts` times on a raised `ProviderError` and re-raises.

## Tests
Contract suite + registry + retry unit tests.

## Out of scope
Network adapters (E2-T02/T03/T04/T09); caching (E2-T05).

## Connections
- Depends on: E1-T01
- Blocks: E2-T02, E2-T03, E2-T04, E2-T05, E2-T09
- Relates to: NFR-08 (`tos_note`)
