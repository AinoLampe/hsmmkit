# E16-T02 — `to_pandas()`/`to_xarray()` for all result objects; `hmmlearn` parameter import

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-IO02, FR-IO04 | M12 | Part II §13.1 FR-IO02/IO04, §15.5 (`xarray` extra); Part I §5.4 (result contracts) |

## Goal
Make every result object easy to analyse with pandas and xarray, and let `hmmlearn` users bring existing models over.

## Context (self-contained)
**Conversions** (methods on each dataclass, implemented in `interop/xarray.py` / `interop/pandas.py`):
| Object | `to_pandas()` (long format) | `to_xarray()` dims |
|---|---|---|
| `PosteriorResult` | `time, sequence, state, filtered, smoothed` + `viterbi` column | `filtered/smoothed (sequence, time, state)`, `viterbi (sequence, time)` |
| `ForecastResult` | `horizon, time, series, quantile, value` and `sim, horizon, series, value` (samples, optional) | `samples (sim, horizon, series)`, `regime_probs (horizon, state)`, `cum_quantiles (horizon, quantile, series)` |
| `FitResult` | parameters long: `name, index, value, se` | `Dataset` with one variable per parameter, dims from `ParamSpec` (`state, series, covariate, …`) |
| `BacktestResult` / `CVResult` | as stored (already long) | `metrics (fold, horizon, metric)` |
| batched results | `sequence` column / dim with ids | |
Coordinates carry names from the config (state labels `s0..s{N−1}` or `model.state_names` if set, series names from `observation_columns`, covariate names). `from_xarray(ds)` classmethods reconstruct `PosteriorResult` and `ForecastResult` (round trip). xarray via `optional_import("xarray", extra="xarray")`; pandas needs no extra.
**hmmlearn import** (`interop/hmmlearn.py`, one-way): `from_hmmlearn(model) -> tuple[ProjectConfig, Params]` for `GaussianHMM` (`covariance_type` `diag`/`full`/`spherical`/`tied` → `gaussian` diag/full with tied mapped to `tie` groups) and `CategoricalHMM`: emission parameters copied; `transmat_` → geometric durations with per-step exit `a_i = 1 − A_ii` and jump matrix `π_ij = A_ij / (1 − A_ii)` (`clock: steps`); `startprob_` → `initial_state: free`. Self-transition probability 1 → absorbing state (topology). Fitted `hmmlearn` model required (`check_is_fitted`). The result builds a `ModelSpec` and can be used for decode/score.

## Creates
- `src/hsmmkit/interop/{pandas,xarray,hmmlearn}.py`, methods on result classes
- `tests/integration/test_e16_t02_conversions.py` (marker `xarray` for xarray parts), `tests/dev/test_e16_t02_hmmlearn.py` (hmmlearn dev-only)

## Acceptance criteria
1. `from_xarray(to_xarray(x))` equals `x` bitwise for posterior and forecast results (single and batched).
2. Long-format pandas schemas documented and asserted.
3. Imported `GaussianHMM` (diag and full) and `CategoricalHMM` reproduce `hmmlearn`'s `score` log-likelihood ≤ 1e-8 relative on the same data.
4. Spherical/tied covariance types map correctly (log-likelihood check).

## Tests
Round trips, schema tests, dev-only hmmlearn comparison.

## Out of scope
Export to `hmmlearn` (one-way only).

## Connections
- Depends on: E11-T02, E11-T04, E10-T11
- Blocks: E16-T03
- Relates to: E16-T01, E12-T03 (absorbing)
