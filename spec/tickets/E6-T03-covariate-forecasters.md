# E6-T03 — Covariate forecasters: `freeze`, `scenario`, `bootstrap`, `ar`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-I05, D7 | M4 | §4.9 Covariate forecasters, §5.5 (`CovariateForecaster`) |

## Goal
Generate future covariate paths `(n_sim, h, K)` for multi-step forecasts.

## Context (self-contained)
```python
class CovariateForecaster(Protocol):
    name: str
    def fit(self, data: PreparedDataset) -> "CovariateForecaster": ...        # uses training window only
    def sample_paths(self, key, h: int, n_sim: int, last: Array) -> Array: ... # (n_sim, h, K), standardized units
```
`K` = union of all covariate columns (transition ∪ duration ∪ emission, de-duplicated by name; a `ColumnMap` splits paths back into the three blocks). Because covariates are lagged by ≥ 1 step, `z_{t+1}` is already known at the origin `t` (passed in as `last`); `h = 1` therefore never calls a forecaster. `sample_paths` returns `(n_sim, h, K)` where row 0 equals `last` and rows `1..h−1` are generated.
| Type | Rule |
|---|---|
| `freeze` | all rows `= last` |
| `scenario` | user path `(h, K')` for a subset of columns (by name, from YAML/CSV); missing columns → `freeze`; length ≠ h → `ConfigError` |
| `bootstrap` | stationary block bootstrap (Politis–Romano, geometric block lengths with mean `mean_block_length`) of historical *first differences* of all covariates jointly, cumulated from `last`; NumPy RNG `np.random.default_rng(seed)` (data-layer exception to the JAX-RNG rule) |
| `ar` | VAR(1) `z_{k+1} = c + Φ z_k + ε`, OLS on the training window, Gaussian innovations with the residual covariance; `diagonal: true` → per-column AR(1) |
All outputs are in standardized units (the dataset is already standardized), so no re-standardization is needed; note the concept's "re-standardized with training stats" is satisfied by operating on standardized data.

## Creates
- `src/regimekit/inference/covariate_forecasters.py`
- `tests/unit/test_e6_t03_forecasters.py`

## Acceptance criteria
1. Shapes `(n_sim, h, K)` for all types; row 0 equals `last`.
2. `bootstrap`: sample cross-correlation of differences within 0.1 of the historical one on a synthetic bivariate series; mean block length within 10 % of configured.
3. `ar`: recovers synthetic VAR(1) coefficients within 0.05 on `T=5000`; `diagonal: true` yields zero off-diagonals.
4. `scenario` with a missing column freezes that column; wrong length → `ConfigError`.
5. Deterministic per seed/key.

## Out of scope
Forecast sampling (E6-T04a).

## Connections
- Depends on: E1-T01
- Blocks: E6-T04a
- Relates to: E1-T02a (`ForecastSection.covariate_forecaster`)
