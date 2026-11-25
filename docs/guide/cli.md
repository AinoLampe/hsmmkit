# Command line: core and finance

The CLI is built by `regimekit.cli.app.build_app(name, config_cls, project_cls, extra_commands)` (E10-T07b). The
core ships a complete domain-neutral application, `regimekit-core`; the finance application builds its own app on
top, `regimekit` (`regimekit.finance.cli`).

| Command | `regimekit-core` | `regimekit` |
|---|---|---|
| `validate-config`, `config-schema` | yes | yes |
| `fit`, `decode`, `forecast`, `simulate` | yes (data from `data.source`) | yes |
| `backtest`, `compare`, `selftest` | yes | yes (strategy hook, strategy metrics) |
| `fetch` | no | yes (providers through the cache, quality report) |

Global options: `--out`, `--device`, `--seed`, `--log-level` and `--offline` (the core ignores `--offline`
unless a source honours it).

## Exit codes

`0` ok, `1` unexpected (and `PersistenceError`), `2` configuration, `3` data, `4` numerical. The table is
`regimekit.cli.app.EXIT_CODES`; an application adds its own with `register_exit_code(exc, code)`. The most specific
registered class wins. The finance app registers `ProviderError -> 5` (`ProviderError` subclasses `DataError`).

## Python

`regimekit.project.Project` is the domain-neutral facade (`load_data`, `fit`, `decode`, `forecast`, `simulate`,
`backtest`, `compare`, `save`/`load`). `regimekit.finance.project.FinanceProject` adds `fetch`, the strategy hook
and the finance bundle extension; `regimekit.Project` is an alias of it (the 0.2.x API).

```python
from regimekit import Project  # the finance facade
from regimekit.project import Project as CoreProject
from regimekit.config.schema import ProjectConfig

p = CoreProject.from_yaml("config.yaml", config_cls=ProjectConfig)
data = p.load_data()  # or load_data(obj=my_dataframe) for pandas / numpy / xarray sources
fit = p.fit(data)
```

An extension builds its own app:

```python
from regimekit.cli.app import build_app, register_exit_code

app = build_app(name="mytool", config_cls=MyConfig, project_cls=MyProject, extra_commands=[add_my_command])
```
