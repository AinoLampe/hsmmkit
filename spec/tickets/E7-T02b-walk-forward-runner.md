# E7-T02b — Walk-forward runner: per-fold `Preparer.fit` on train only, refit, per-origin forecasts, metric table, look-ahead test

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-E02, NFR-03 | M5 | §3.1 FR-E02, §5.4 (`BacktestResult`), §8 risks (leakage, recompilation) |

## Goal
Execute the walk-forward scheme end-to-end and produce `BacktestResult.folds/metrics/summary`.

## Context (self-contained)
```python
def run_backtest(cfg: ProjectConfig, obs: ObservationSet, covs: list[CovariateSeries], seed: int, strategy=None) -> BacktestResult
```
Per fold: `Preparer.fit(obs, covs, train_end=times[fold.train_end−1])` → `transform` full → `window(train_start, train_end)` for fitting (E5-T03 `fit`; warm-start `u` from the previous fold's optimum as an *additional* restart), `window(0, test_end + max_h)` for decoding/forecasting; for every origin in the test range and each `h`: `MonteCarloForecaster.forecast(fit, data.window(0, origin+1), h, n_sim, seed=key_for(seed, fold, origin))`; realized `Y_h`; evaluate resolved metrics (E7-T01) → rows `(fold, origin, horizon, metric, value)`; aggregate `metrics` per `(fold, horizon, metric)` (mean) and `summary` per `(horizon, metric)` (mean, sd across folds, n). PIT KS statistic is computed on the pooled PIT values per horizon in `summary`.
Recompilation control: pad `T` of each window to the next multiple of 256 with masked rows (`mask=False`, `dt` positive dummy) so the jitted engine reuses compiled code; the padded rows contribute zero log-density. `n_sim` per origin may be reduced via `evaluation.backtest.n_sim` (default `forecast.n_sim`).
Sequential folds by default (§9 #8); `fit_time` recorded per fold. Strategy hook: if `strategy` given, positions per origin are computed by E7-T03 and appended to `BacktestResult.strategy`.

## Creates
- `evaluation/backtest.py` (`run_backtest`), `evaluation/padding.py`
- `tests/integration/test_e7_t02b_backtest.py`, `tests/integration/test_e7_t02b_lookahead.py`

## Acceptance criteria
1. Look-ahead test: build a `CovariateSeries` whose value stamped at `τ_t` equals the *next* return `y_{t+1}` (a perfect future signal if leaked). Run it through the official pipeline with `lag: 1`: the value first becomes usable at step `t+1`, by which time `y_{t+1}` is already realized, so it carries no information about future returns. Assert the out-of-sample log score with this covariate is not better than the no-covariate model beyond MC noise (paired comparison over folds, one-sided 5 % test). A second variant bypasses the preparer (`lag=0` injected directly into arrays) and asserts a large gain, proving the test is sensitive.
2. Fold table equals `schedule_folds` output; `metrics` has one row per `(fold, horizon, metric)`; `summary` per `(horizon, metric)`.
3. Standardization per fold uses only the fold's train window (assert `PrepStats.train_window`).
4. Padding: results with padding equal results without padding ≤ 1e-8 on a short synthetic run; compile count across folds is 1 for the engine functions.
5. Determinism: two runs with the same seed give identical `metrics`.

## Out of scope
Strategies (E7-T03); CLI (E7-T04); parallel folds (ADR needed).

## Connections
- Depends on: E5-T03, E6-T04b, E7-T01, E7-T02a
- Blocks: E7-T03, E7-T04
- Relates to: E4-T03b (masking), E5-T01 (data as dynamic arg)
