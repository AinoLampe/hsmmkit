# E10-T05 — Generic `Preparer` with observation-transform hook, observation supports and validation, quality-report split

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | M | FR-G03, FR-G01, FR-D05 | M7a | Part II §15.3 (`ObservationTransform`), §15.4 (`data.preparation`), §14.2 (supports); Part I §5.6, E2-T07, E2-T08 |

## Goal
Make preparation domain-neutral (observations pass through unchanged unless a registered transform is configured), validate observation supports against the emission family, and split the quality report into generic and finance checks.

## Context (self-contained)
**Observation transforms** (registry kind `transform`):
```python
class ObservationTransform(Protocol):
    name: str
    def __call__(self, values: np.ndarray, times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]: ...
```
Core registers `identity` (default when `observation_transform: null`) and `diff` (`y_t = x_t − x_{t−1}`, drops the first row). Finance registers `log_returns` (moved from `data/transforms.py::log_returns`: `y_t = log(P_t / P_{t−1})`, drops the first row) in `finance/transforms.py`. Length change is handled by the transform returning shortened `times`.
**Generic `Preparer`** (`data/prepare.py`): `fit(obs, covs, train_end)`, `transform(obs, covs) -> PreparedDataset` as in Part I, but: (1) grid from `obs.times` (or step index) via the clock (E10-T03); (2) observations = `transform(values)`; (3) `missing_observations`: `mask` (non-finite → `mask=False`, value 0), `drop` (rows removed jointly with covariates), `error` (→ `DataError` with first index); (4) `standardize_observations` (train-window mean/sd per series, stored in `PrepStats.obs_mean/obs_std`, real support only); (5) covariate pipeline unchanged (transforms `log, diff, pct, zscore`, alignment, lag ≥ 1, missing policies, train-only standardization). `PrepStats` gains `obs_mean`, `obs_std` (`None` when not standardized) — JSON round-trip.
**Supports** (FR-G03): `PrepCfg` does not carry the support; it is derived from the emission family (`family.support`) at build time and checked by the preparer:
| support | dtype after transform | rule |
|---|---|---|
| `real` | float64 | finite where `mask` |
| `count` | int64 (float input accepted if integral) | `≥ 0` |
| `binary` | int64 | in `{0, 1}` |
| `categorical` | int64, `M = 1` | in `[0, n_categories)`; `n_categories` from family params or `max+1` with warning |
Violation → `DataError(path="data.source.observation_columns", "<support> support violated at row r, series m")`. `PreparedDataset.support/n_categories` are filled accordingly (Part I always `real`). Non-real supports do not allow `standardize_observations` (`ConfigError`).
**Quality report split** (`data/quality.py` core): duplicates, non-monotone or gapped index (gap = spacing > 1.5 × median), outliers `|y − median| > 8·MAD` (flag only), covariate staleness histogram. `finance/quality.py`: gaps vs exchange weekdays, zero/negative prices, adjustment discontinuities between cache generations. `fetch` composes both; JSON + Markdown output unchanged in structure for finance (extra keys allowed).

## Creates
- `src/regimekit/data/{prepare,transforms,quality}.py` (edited), `src/regimekit/finance/{transforms,quality}.py`
- `tests/unit/test_e10_t05_preparer.py`, `tests/unit/test_e10_t05_supports.py`, `tests/finance/test_e10_t05_quality_finance.py`
- `docs/guide/preparation.md`

## Acceptance criteria
1. Each support has an accepting and a rejecting fixture with the documented `DataError` message.
2. Finance (`observation_transform: log_returns`, `missing_observations: drop`) produces a `PreparedDataset` bitwise equal to the v0.2.0 golden; `PrepStats` JSON written by 0.1.x/0.2.x still loads (missing `obs_*` → `None`).
3. A core config on the csv fixture with `identity` keeps `observations == values` (float64) and `T == T0`.
4. Quality report on the E2-T08 defect fixture lists the same finance findings as v0.2.0 plus the generic section.
5. Allow-list entries tagged `E10-T05` gone; v0.2.0 golden values bitwise.

## Tests
Transform registry tests, preparer policies, support validation, quality split, golden test.

## Out of scope
Lag generation for MS-AR (E11-T05); sequence splitting (E12-T01a).

## Connections
- Depends on: E10-T03, E10-T04a
- Blocks: E10-T07b, E11-T05
- Relates to: E2-T07, E2-T08, E11-T04, E11-T05
