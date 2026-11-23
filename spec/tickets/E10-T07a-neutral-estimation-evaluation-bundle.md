# E10-T07a — Neutral init naming, walk-forward origin hooks, strategies to finance, bundle v2

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | M | FR-G01, FR-G08, D21 | M7a | Part II §15.2 (`BacktestResult.extras`), §15.3 (`OriginHook`), §15.8; Part I §4.8 init, §5.8, E7-T02b, E7-T03, E8-T01 |

## Goal
Remove the remaining finance coupling from estimation, evaluation and persistence by renaming, by an origin-hook extension point for walk-forward backtests, and by a bundle format that carries extension data separately.

## Context (self-contained)
**Init.** `rollvol_kmeans` → `rolling_kmeans` (same algorithm: 21-step rolling scale `sqrt(mean(y²)/nominal_dt)`, k-means++ with the given key, clusters sorted by scale). Finance registers `rollvol_kmeans` as an alias entry (same object, provenance finance) and `migrate_1_to_2` rewrites it. Docstrings describe "rolling scale", not volatility of returns.
**Walk-forward origin hooks.** `evaluation/backtest.py::run_backtest(cfg, data_loader, seed, hooks: Sequence[OriginHook] = ())`:
```python
class OriginHook(Protocol):
    name: str                                    # key in BacktestResult.extras
    def on_origin(self, fold: int, origin: int, posterior: PosteriorResult,
                  forecast: ForecastResult | None, data: PreparedDataset) -> dict[str, Any]: ...
    def finalize(self, rows: list[dict[str, Any]]) -> pd.DataFrame: ...
```
`BacktestResult.strategy` → `BacktestResult.extras: dict[str, pd.DataFrame]`; finance's `strategy` becomes `extras["strategy"]` and a finance helper `strategy_frame(bt)` returns it. Fold scheduling uses `initial_train` / `rolling_window` in clock units (or the `_steps` variants, E10-T02b) — the elapsed-time rule of E7-T02a with `cumsum(dt)`; results unchanged for finance (years under `trading`). `run_backtest` receives a `data_loader` callable (returns `(ObservationSet, covs)`) instead of provider objects.
**Strategies to finance.** `evaluation/strategies.py` → `finance/strategies.py` (`regime_long_flat`, `regime_target_vol`, costs) wrapped as an `OriginHook` named `strategy`; `evaluation/metrics/strategy.py` → `finance/metrics.py` (`sharpe`, `max_drawdown`, `turnover`, `hit_rate`; registered kind `metric`, provenance finance). Metrics with `needs={"strategy"}` read `ctx["extras"]["strategy"]`.
**Report.** `evaluation/report.py` builds core sections (header, model, fit, diagnostics, backtest summary, timings) and appends `extra_sections: list[Callable[[ctx], str]]` contributed by hooks; finance contributes the strategy table. Header shows "data provenance" (snapshot IDs) instead of provider names.
**Bundle v2** (`persistence/store.py`): `bundle_version = 2`; files `manifest.json` (+ `package`, `package_version`, `extensions: {name: version}`; `profile` removed), `config.resolved.yaml`, `spec.json`, `params.safetensors`, `prep_stats.json`, `provenance.json` (snapshot IDs — was `snapshots.json` without provider metadata), `fit_report.json`, `extensions/<name>.json` (opaque, written by extension save hooks; finance writes provider metadata to `extensions/regimekit.json`). `ModelStore(suffix=".rkmodel")` parameter; loading checks that every extension listed in the manifest is importable (else `PersistenceError` naming it). **v1 read compatibility:** `bundle_version 1` → map `snapshots.json` to `provenance.json` + `extensions/regimekit.json`, `profile` ignored; write always v2.

## Creates
- edits: `estimation/init.py`, `evaluation/{backtest,report}.py`, `evaluation/results.py`, `persistence/store.py`
- `src/regimekit/finance/{strategies,metrics}.py` (moved), `finance/report.py` (strategy section), `finance/persistence.py` (extension save/load hooks)
- `tests/integration/test_e10_t07a_origin_hooks.py`, `tests/integration/test_e10_t07a_bundle_v2.py`, `tests/finance/test_e10_t07a_strategy_hook.py`, fixture `tests/fixtures/bundles/v0_2_0.rkmodel/` (bundle written by tag `v0.2.0`; `bundle_version 1`, same as 0.1.0)

## Acceptance criteria
1. Finance backtest (Appendix A v2, offline, 2 folds) yields `metrics` and `extras["strategy"]` equal bitwise to the v0.2.0 golden `backtest_metrics.parquet` / strategy frame; the E7-T02b look-ahead test still passes.
2. A core backtest with no hooks runs with `HSMM_CORE_ONLY=1` on the gaussian csv fixture.
3. The v0.2.0 bundle fixture loads (v1 path) and reproduces the golden loglik bitwise; saving it again writes v2 with `extensions/regimekit.json`.
4. Missing extension package for a bundle that lists it → `PersistenceError` naming it.
5. Allow-list entries tagged `E10-T07a` gone; v0.2.0 golden values bitwise.

## Tests
Hook protocol tests, bundle v1→v2 compatibility, finance strategy hook parity, golden test.

## Out of scope
CLI/`Project` split (E10-T07b); `Splitter` interface (E14-T01).

## Connections
- Depends on: E10-T02b, E10-T03
- Blocks: E10-T07b
- Relates to: E5-T03, E7-T02b, E7-T03, E8-T01, E10-T09
