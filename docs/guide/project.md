# Python API and CLI parity

`regimekit.Project` is the single Python entry point; every CLI command is a thin wrapper over one of its
methods (the CLI passes `write=True`). Methods return their result objects and **write nothing unless
`write=True`**. Importing `regimekit` or constructing a `Project` triggers no network access and no file
writes.

```python
from regimekit import Project

p = Project.from_yaml("dax_3state.yaml", overrides={"project.seed": 7})
data = p.fetch()  # PreparedDataset (quality report on p.last_quality)
fit = p.fit(data)  # FitResult
post = p.decode(fit, data)  # PosteriorResult
fc = p.forecast(fit, data, horizon=20)
bt = p.backtest()  # BacktestResult (fetches the raw inputs itself)
sim = p.simulate(params="config", steps=500, n_paths=1, data=data)
p.save(fit, "runs/dax_3state.rkmodel")
fit2 = Project.load("runs/dax_3state.rkmodel")
table = Project.compare(["a.yaml", "b.yaml"], allow_mixed=False)
```

| CLI | `Project` | Writes with `write=True` |
|---|---|---|
| `validate-config` | `validate()` | `config.resolved.yaml` |
| `fetch` | `fetch()` | `prepared.parquet`, `prep_stats.json`, `snapshots.json`, `quality.json/md` |
| `fit` | `fit(data)` | `<name>.rkmodel/`, `fit_report.json`, `config.resolved.yaml` |
| `decode --model` | `decode(fit, data)` | `posterior.parquet`, `segments.json`, `criteria.json` |
| `forecast --model` | `forecast(fit, data)` | `forecast.parquet`, `forecast.json` |
| `simulate` | `simulate(...)` | `simulation.parquet`, `simulation_segments.json` |
| `backtest` | `backtest()` | metrics, folds, strategy, `backtest.json`, `report.md` |
| `compare` | `Project.compare([...])` | `compare.parquet`, `compare.md` |
| `selftest` | `Project.selftest()` | nothing |

Where the CLI and `Project` both run, the artifacts are equal: parquet frames exactly, and forecasts
bitwise for the same seed (`tests/integration/test_e8_t02_parity.py`).

## Differences worth knowing

- `Project.backtest(data=...)` takes the *raw* inputs (`Project.fetch_inputs()`), not a `PreparedDataset`:
  the walk-forward runner refits the preparation in every fold.
- `Project.save(fit, path)` also needs the dataset the model was fitted on (for the bundle's data-derived
  facts); it defaults to the dataset of the last `fit`/`fetch`.
- `overrides` are dotted config keys validated like the YAML (`ConfigError` with the key as path). The
  CLI's global `--seed`, `--device` and `--log-level` are overrides of `project.seed`,
  `estimation.device` and `project.log_level` and show up in the written `config.resolved.yaml`.

## Global options and exit codes

`--out`, `--offline`, `--device`, `--seed` and `--log-level` precede the command and apply to all of
them. Exit codes: 0 ok, 2 config, 3 data, 4 numerical, 5 provider, 1 for a persistence error or any
unexpected exception (the traceback is logged at `DEBUG`).
