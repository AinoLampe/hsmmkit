# Command line

The CLI is built by `hsmmkit.cli.app.build_app(name, config_cls, project_cls, extra_commands)`; `hsmmkit` is the
app of the core.

| Command | |
|---|---|
| `validate-config`, `config-schema` | validate a configuration, export its JSON Schema |
| `fit`, `decode`, `forecast`, `simulate` | data from `data.source` (`decode --samples R` also draws `R` exact posterior regime paths: `path_samples.parquet`, `boundary_prob`) |
| `backtest`, `compare`, `selftest` | walk-forward evaluation, comparison, smoke test |
| `stream` | CSV rows from stdin or a (growing) file -> JSONL regime probabilities and alert events (`--model`, `--input`, `--follow`, `--state`, `--emit`, `--every`, `--strict`) |

Global options: `--out`, `--device`, `--seed`, `--log-level` and `--offline` (ignored unless a source honours it).

## Exit codes

`0` ok, `1` unexpected (and `PersistenceError`), `2` configuration, `3` data, `4` numerical. The table is
`hsmmkit.cli.app.EXIT_CODES`; an application adds its own with `register_exit_code(exc, code)`; the most specific
registered class wins.

## Python

`hsmmkit.project.Project` is the facade (`load_data`, `fit`, `decode`, `forecast`, `simulate`, `backtest`,
`compare`, `save`/`load`). An application subclasses it (`config_cls`, `_load_raw`, `_hooks`, `_report_sections`,
`_extensions`, `needs_network`) and builds its own app:

```python
from hsmmkit.cli.app import build_app, register_exit_code

app = build_app(name="mytool", config_cls=MyConfig, project_cls=MyProject, extra_commands=[add_my_command])
```

`regimekit` (market-regime detection) is an example extension: it adds `fetch`, market-data providers, a
`trading` clock and trading strategies on top of this core.
