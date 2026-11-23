# E10-T04a — Generic `ObservationSet`, `Source` protocol, numpy/pandas/csv/parquet sources, `source_contract`

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | M | FR-G04, FR-PL02, FR-G01 | M7a | Part II §15.2 (`ObservationSet`), §15.3 (`Source`), §15.4 (`data.source`); Part I §5.4, E2-T01 |

## Goal
Give the core a domain-neutral raw-data contract and in-memory/file sources, so that data can enter the model without any provider.

## Context (self-contained)
**Contract change** (`data/contracts.py`):
```python
@dataclass(frozen=True)
class ObservationSet:
    series_names: tuple[str, ...]        # M        (was: symbols)
    times: np.ndarray | None             # (T0,) datetime64[ns] UTC, strictly increasing; None only under unit="step" clocks
    values: np.ndarray                   # (T0, M) float64 or int64   (was: prices)
    source: str                          # (was: provider)
    snapshot_id: str                     # sha256 of the source bytes/array buffer; "" if not computed
    meta: dict[str, Any]                 # finance puts interval, adjusted, field here
    sequence_ids: np.ndarray | None = None   # (T0,) group ids; consumed from E12-T01a, ignored before
```
`validate()`: `values.ndim == 2`, `len(series_names) == M`, `times` monotone when present, `sequence_ids` length `T0` when present. Finance adapters (E2-T02…T09) are updated mechanically (`symbols→series_names`, `prices→values`, `provider→source`, `interval/adjusted` → `meta`); a finance-side helper `price_view(obs)` gives the old attribute names for finance code only. `PreparedDataset.times` may be an `int64` step index when `times is None` (validated: only with a `unit="step"` clock).
**Source protocol and options** (`data/sources/base.py`, registry kind `source`):
```python
class Source(Protocol):
    name: str
    options_model: type[BaseModel]         # validates the extra keys of data.source / covariates[*].source
    def load(self, opts: BaseModel, *, obj: Any | None = None) -> tuple[ObservationSet, list[CovariateSeries]]: ...
```
Common options `TabularOpts`: `path: Path | None`, `observation_columns: list[str]`, `time_column: str | None`, `sequence_column: str | None`, `covariate_columns: list[str] = []`, `dtype: Literal["float64","int64"] = "float64"`. Sources:
- `numpy`: `path` to `.npy` (observations only) or `.npz` (keys `observations`, optional `times`, `covariates`, `covariate_names`, `sequence_ids`); `obj` may be a dict of arrays (Python API).
- `pandas`: Python API only (`obj: pd.DataFrame`); YAML with `kind: pandas` and no object → `ConfigError` suggesting `csv`/`parquet`.
- `csv`, `parquet`: read via pandas/pyarrow then delegate to the pandas logic.
Covariate columns become `CovariateSeries(name=col, times=times or step index, values, source=…, series_id=col, snapshot_id=…, frequency="grid")`. `snapshot_id` = sha256 over the file bytes (files) or the array buffers + column names (in-memory). Time column parsing: `pd.to_datetime(utc=True)`; non-monotone → `DataError(path="data.source.time_column")`.
**Alignment without times:** `data/calendar.py::align_covariate` accepts `times=None` series only if the series has exactly the grid's length; lag and staleness are then applied by row index (same code path, step index as "time").
**`source_contract`** (`testing/source_contract.py`): parametrized over registered sources with fixture factories: returns valid `ObservationSet`, `validate()` passes, `M` equals `len(observation_columns)`, covariates aligned, time-less load works for every source that supports `time_column: null`, snapshot id deterministic across two loads, missing column → `DataError` with the column name.
Config: core `SourceCfg.kind` validated against registry; each source's `options_model` validates the extra keys (error path `data.source.<key>`).

## Creates
- `src/regimekit/data/sources/{__init__,base,numpy,pandas,csv,parquet}.py`, edits in `data/contracts.py`, `data/calendar.py`, finance adapters
- `src/regimekit/testing/source_contract.py`
- `tests/unit/test_e10_t04a_sources.py`, `tests/unit/test_e10_t04a_contracts.py`, fixtures `tests/fixtures/sources/{gauss.csv,gauss.parquet,gauss.npz}`
- `docs/guide/data_sources.md`

## Acceptance criteria
1. numpy, pandas, csv and parquet sources pass `source_contract`.
2. A `DataFrame` with and without a time column loads under `steps`; with a time column under `calendar`; without one under `calendar` → `ConfigError(path="data.clock")`.
3. Finance provider tests (cassettes) pass unchanged in behaviour after the field renames; v0.2.0 golden values bitwise.
4. `ObservationSet` has no attribute named `prices`, `symbols` or `provider` (scanner entries tagged `E10-T04a` gone).

## Tests
Contract suite, time-less alignment and lag-by-row test, snapshot determinism, golden test.

## Out of scope
xarray source, provider-as-source and cache move (E10-T04b); multi-sequence splitting (E12-T01a uses `sequence_ids`).

## Connections
- Depends on: E10-T02b
- Blocks: E10-T04b, E10-T05
- Relates to: E2-T01, E12-T01a, E16-T04
