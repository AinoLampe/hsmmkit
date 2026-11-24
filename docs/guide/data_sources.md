# Data sources

A *source* turns a file or an in-memory object into observations, without any domain knowledge. The core ships
four sources, registered under the kind `source`: `numpy`, `pandas`, `csv` and `parquet`. Applications register
more (the finance application registers `provider`, which fetches prices from market-data providers).

```yaml
data:
  clock: steps                 # or calendar; time-less data needs steps
  source:
    kind: csv                  # numpy | pandas | csv | parquet
    path: data/gauss.csv
    observation_columns: [y1, y2]
    time_column: t             # null = no timestamps (steps clock only)
    sequence_column: seq       # optional group ids; used from E12-T01a on
    covariate_columns: [x]
    dtype: float64             # or int64 for count data
```

The keys next to `kind` are validated by the options model of the source. A wrong key fails with
`ConfigError(path="data.source.<key>")`.

## What a source returns

`Source.load(opts, obj=None)` returns an `ObservationSet` and a list of `CovariateSeries`:

| Field | Meaning |
|---|---|
| `series_names` | the observed series, `M = len(observation_columns)` |
| `times` | `(T0,)` `datetime64[ns]`, UTC, strictly increasing; `None` without a time column |
| `values` | `(T0, M)` `float64` (`int64` with `dtype: int64`) |
| `source` | name of the source |
| `snapshot_id` | sha256 of the file bytes, or of the column names and array buffers for in-memory data |
| `meta` | free-form extras of the source |
| `sequence_ids` | `(T0,)` group ids when `sequence_column` is set |

Covariate columns become `CovariateSeries` on the same time axis (`frequency="grid"`).

## The four sources

- **`csv`, `parquet`**: read with pandas / pyarrow, then the pandas logic below. The snapshot id hashes the file.
- **`pandas`**: Python API only. Pass the frame as `obj`; YAML with `kind: pandas` and no object is a
  `ConfigError` that points at `csv` / `parquet`.
- **`numpy`**: a `.npy` file (observations only) or a `.npz` file with the keys `observations` and the optional
  `times`, `covariates`, `covariate_names`, `sequence_ids`. `obj` may be a dict with the same keys.

Time columns are parsed as UTC (`pd.to_datetime(utc=True)`); timestamps that do not strictly increase raise
`DataError(path="data.source.time_column")`. A missing column raises `DataError` naming the column.

## Loading from the configuration

```python
from regimekit.data.sources import load_data

obs, covariates = load_data(cfg.data, obj=my_dataframe)
```

`load_data` checks the result against the clock: data without a time column needs `data.clock: steps`, otherwise
`ConfigError(path="data.clock")`. `data.covariates[i]` picks the covariate column `column` (else `name`) of the
primary source, in declared order.

## Time-less data

Without timestamps the step index is the time axis. `data.calendar.build_grid` builds an integer grid
(`Δ_t ≡ 1`), and covariates without timestamps need exactly one value per grid row; lag and staleness then count
rows. The result equals the timestamp path for one print per grid step. (The preparer still needs timestamps; it
is generalized in E10-T05.)

## Writing a source

Implement the `Source` protocol (`name`, `options_model`, `load`) and register an instance:

```python
REGISTRY.register("source", "mine", MySource(), provenance=provenance_of(__name__))
```

Then add a `SourceCase` to `regimekit.testing.source_contract` and run `check_source` on it: it checks a valid
`ObservationSet`, `M`, aligned covariates, the time-less load, a deterministic snapshot id and the error for a
missing column.
