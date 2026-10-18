# E7-T01 — Metric registry and metrics (log score, CRPS, PIT coverage, RMSE/MAE, IC wrappers, occupancy/persistence, balanced accuracy)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-E01, D9 | M5 | §3.1 FR-E01, §5.5 (`Metric`), Gneiting & Raftery 2007 |

## Goal
Provide a configurable, name-resolved set of forecast and model metrics computed from a common context.

## Context (self-contained)
```python
class Metric(Protocol):
    name: str; needs: set[Literal["forecast", "posterior", "fit", "labels", "strategy"]]
    def __call__(self, **ctx) -> float | dict[str, float]: ...
# ctx keys: forecast: ForecastResult; realized: Array (h, M) realized future log-returns; posterior: PosteriorResult;
#           fit: FitResult; labels: np.ndarray (T,) int | None; strategy: pd.DataFrame | None; horizon: int
```
Registry `evaluation/metrics/registry.py`: `register(metric)`, `resolve(names: list[str]) -> list[Metric]`; unknown name → `ConfigError(path="evaluation.metrics")`.
Predictive metrics at horizon `h` use the cumulative return `Y_h = Σ_{k≤h} y_{t+k}` and the forecast samples `cum[:, h−1]`:
- `log_score`: log of a Gaussian-KDE density estimate at `Y_h` (bandwidth Silverman) — document; also `log_score_mixture` for `h=1` using the exact predictive mixture (from E5-T04 helpers).
- `crps` (sample-based): `E|X − Y_h| − ½ E|X − X'|` with `X` the samples (use the sorted-sample formula, `O(n log n)`).
- `pit_coverage`: PIT `= mean(samples ≤ Y_h)`; report coverage of central intervals at the configured quantile pairs (e.g. 90 % interval hit rate) and a KS statistic across folds (aggregation in E7-T02b).
- `rmse`, `mae`: on `mean[h−1]` vs `Y_h`.
- `aic`, `bic`, `icl`: wrappers reading `posterior`.
- `regime_occupancy (dict per regime)`, `regime_persistence`: mean Viterbi segment length per regime.
- `balanced_accuracy`: Viterbi regimes vs `labels` (requires labels; Hungarian matching of label ids via `scipy.optimize.linear_sum_assignment`? — scipy is dev-only: implement a brute-force permutation match for `N ≤ 6`, else raise `ConfigError`).
Strategy metrics (`sharpe`, `max_drawdown`, `turnover`, `hit_rate`) are registered by E7-T03.

## Creates
- `src/regimekit/evaluation/metrics/{registry,predictive,model,labels}.py`
- `tests/unit/test_e7_t01_metrics.py`

## Acceptance criteria
1. CRPS on Gaussian samples vs the closed-form Gaussian CRPS `σ[z(2Φ(z)−1) + 2φ(z) − 1/√π]` within 1 % for `n=20000`.
2. `log_score_mixture` equals `log` of the analytic `h=1` mixture density ≤ 1e-8.
3. PIT on samples drawn from the true distribution is uniform (KS p > 0.05 over 500 folds simulated in-test).
4. `resolve(["log_score", "nope"])` → `ConfigError`; all names in Appendix A's `metrics` list resolve.
5. `balanced_accuracy` is invariant to label permutation.

## Out of scope
Walk-forward orchestration (E7-T02b); strategy metrics (E7-T03).

## Connections
- Depends on: E6-T04b, E5-T04 (predictive helpers)
- Blocks: E7-T02b
- Relates to: E7-T03
