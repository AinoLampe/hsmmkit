# E7-T03 — Strategy protocol, `regime_long_flat`, `regime_target_vol`, transaction costs, strategy metrics

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-E03 | M5 | §3.1 FR-E03, §5.5 (`Strategy`), Appendix A `strategy` |

## Goal
Map posteriors/forecasts to positions, apply proportional costs, and compute Sharpe / max drawdown / turnover / hit rate.

## Context (self-contained)
```python
class Strategy(Protocol):
    def positions(self, posterior: PosteriorResult, forecast: ForecastResult | None) -> Array: ...  # (T, M) in [-1, 1]
```
- `regime_long_flat(flat_if={regime, prob_above, use: filtered|smoothed})`: position 1 unless `P(regime) > prob_above` (using the chosen probabilities at the decision time, i.e. filtered probs at `t` decide the position for `t+1`) → 0. Only *filtered* probabilities are admissible in backtests; `smoothed` is allowed for in-sample illustration with a warning.
- `regime_target_vol(target_vol, cap=1.0)`: position `= min(cap, target_vol / σ_pred)` where `σ_pred` is the one-step predictive annualized volatility `sqrt(var[0] / Δ)` from `ForecastResult` (`h ≥ 1`), sign from the predictive mean (long-only when `mean ≥ 0`, else 0 or −cap if `allow_short`).
- Costs: `cost_bps` proportional to `|Δposition|`: strategy return `r_t = pos_{t−1} · y_t − cost · |pos_{t−1} − pos_{t−2}|` (log-return approximation; document).
- Equity curve `exp(cumsum(r))`; metrics registered in E7-T01's registry: `sharpe` (annualized with `dt`), `max_drawdown`, `turnover` (mean `|Δpos|` per year), `hit_rate` (fraction of periods with `pos·y > 0` among non-zero positions).
- Backtest integration: `run_backtest(..., strategy=)` collects per-origin positions (decision at each test origin using that origin's filtered posterior/forecast) into `BacktestResult.strategy` (`time, position, return, equity, turnover`).

## Creates
- `src/regimekit/evaluation/strategies.py`, `evaluation/metrics/strategy.py`
- `tests/unit/test_e7_t03_strategies.py`

## Acceptance criteria
1. On a synthetic path with known positions, costs reduce cumulative return by exactly `cost_bps·1e-4·Σ|Δpos|` (≤ 1e-12).
2. `regime_long_flat` positions equal a hand-computed rule on a fixture; using `smoothed` in a backtest warns.
3. `regime_target_vol` positions scale inversely with predictive σ and are capped.
4. Sharpe/MDD/turnover/hit-rate verified on closed-form fixtures (e.g. constant return → MDD 0, Sharpe ∞ guarded to `inf`).
5. Equity curve is reproducible bitwise for the same inputs.

## Out of scope
Portfolio optimization, multi-asset allocation.

## Connections
- Depends on: E7-T02b
- Blocks: E7-T04
- Relates to: E7-T01 (registry)
