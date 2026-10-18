# E7-T02a — Walk-forward fold scheduler (pure): expanding/rolling windows, refit cadence, embargo

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-E02 | M5 | §3.1 FR-E02, §5.7 `BacktestCfg` |

## Goal
Compute the fold table deterministically from the grid and backtest config, with no model code involved.

## Context (self-contained)
```python
@dataclass(frozen=True)
class Fold: fold: int; train_start: int; train_end: int; test_start: int; test_end: int   # grid indices, ends exclusive
def schedule_folds(times: np.ndarray, dt: np.ndarray, cfg: BacktestCfg, max_horizon: int) -> list[Fold]
```
Rules: initial training window = first `initial_train_years` of elapsed time (`cumsum(dt) ≥ initial_train_years`); `expanding`: `train_start = 0`; `rolling`: `train_start` such that the window spans `rolling_years`; refit every `refit_every_steps`: fold `k` has `train_end = initial_end + k·refit_every_steps`, `test_start = train_end + embargo_steps`, `test_end = min(test_start + refit_every_steps, T − max_horizon)` (so every test origin has `max_horizon` realized future steps); drop folds with empty test ranges. Per fold, forecast origins are every step in `[test_start, test_end)`; for each origin and horizon `h ∈ cfg.horizons`, realized `Y_h` uses steps `origin+1..origin+h`.
Also `fold_table(folds, times) -> pd.DataFrame` with dates.

## Creates
- `src/regimekit/evaluation/backtest.py` (`schedule_folds`, `fold_table`)
- `tests/unit/test_e7_t02a_folds.py`

## Acceptance criteria
1. Hand-checked fold boundaries for `T=1000, initial 2 years (504 steps), refit 63, embargo 5, horizons [1,5,20]`: first fold `train (0,504)`, `test (509, 572)`, last fold's `test_end ≤ 980`.
2. No fold's test range overlaps its training range; consecutive test ranges are contiguous and disjoint.
3. `rolling` keeps train length constant (± 1 step).
4. Empty-test folds are dropped; a config with `initial_train_years` > span raises `ConfigError`.

## Out of scope
Running fits (E7-T02b).

## Connections
- Depends on: E2-T07 (uses `PreparedDataset.window` semantics), E1-T02a
- Blocks: E7-T02b
- Relates to: —
