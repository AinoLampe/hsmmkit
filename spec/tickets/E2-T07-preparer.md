# E2-T07 — `Preparer` fit/transform: log-returns, transforms, train-only standardization, missing policies, `PrepStats`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-D05 | M1 | §5.5 (`Preparer`), §5.6 Preparation, §4.4 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendment 1: `observations`; new `PreparedDataset` fields filled with defaults). No behavioural change for the finance profile.

## Goal
Turn an `ObservationSet` plus covariates into a `PreparedDataset` with statistics fitted only on the training window, sklearn-style.

## Context (self-contained)
```python
class Preparer:
    def __init__(self, cfg: DataSection): ...
    def fit(self, obs: ObservationSet, covs: list[CovariateSeries], train_end: str) -> "Preparer": ...
    def transform(self, obs: ObservationSet, covs: list[CovariateSeries]) -> PreparedDataset: ...
    prep_stats: PrepStats
```
- Observations (finance profile): `y_t = log(P_t / P_{t−1})` per series, implemented as one pure function `transforms.log_returns(prices) -> Array` so Part II can move it into the profile without touching the `Preparer`; `mask = isfinite(y)`; masked values set to 0. The output field is `PreparedDataset.observations`.
- Part II fields are filled with their 0.1.0 constants: `support="real"`, `n_categories=None`, `labels=None`, `boundaries=None`, `group=None`; `clock` and `nominal_dt` are passed through from the grid (E2-T06).
- Per covariate, in declared order: transforms `log`, `diff`, `pct` (`x_t/x_{t−1} − 1`), `zscore` (**deprecated alias of standardization**: apply standardization here instead; keep for config compat). Then alignment + lag via E2-T06. Then missing policy: `ffill` (already done up to `max_staleness`; remaining NaN → apply next), `drop` (drop grid rows with NaN in that covariate — applied jointly, returns and all covariates re-sliced, logged count), `error` (→ `DataError` with first offending timestamp).
- Standardization (if `standardize_covariates`): `mean`/`std` computed on rows with `times ≤ train_end` only; applied to all rows; `std < 1e-12` → `DataError`. Stats stored in `PrepStats(cov_mean, cov_std, train_window, cov_names, transforms, lags)`.
- Entry-point split: `cov_transition/cov_duration/cov_emission` are column subsets by `use_in`; `cov_names` per entry point; a covariate may appear in several.
- `snapshot_ids = (obs.snapshot_id, *cov.snapshot_id)`. `PrepStats` JSON serialization helpers `to_json/from_json`.

## Creates
- `src/regimekit/data/prepare.py`, `src/regimekit/data/transforms.py`
- `tests/unit/test_e2_t07_preparer.py`

## Acceptance criteria
1. `transform` on a test window uses train-window stats: fit on the first 60 % of a fixture, transform full series, assert train-part mean ≈ 0/std ≈ 1 and full-series mean ≠ 0.
2. Policies `ffill/drop/error` each tested; `drop` keeps `observations`, `dt`, `mask`, covariates aligned (same `T`).
3. Transform order `log → diff` verified numerically against pandas.
4. `PrepStats` JSON round-trip equality; `PreparedDataset.validate()` passes on the output, including under `clock: steps` (all `dt == 1`).
5. A covariate in `use_in: [transition, duration]` appears in both blocks with identical columns.

## Out of scope
Fetching/caching; quality report (E2-T08).

## Connections
- Depends on: E2-T06
- Blocks: E2-T08, E7-T02a (per-fold refit), E7-T02b
- Relates to: E8-T01 (persists `PrepStats`)
