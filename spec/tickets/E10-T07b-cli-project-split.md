# E10-T07b — CLI app factory and `Project`/`FinanceProject` split; neutral-name allow-list empty

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | M | FR-G01, FR-G08, FR-C02, FR-C03 | M7a | Part II §15.1 (`cli/app.py`, `project.py`), §15.8; Part I §5.9, E8-T02 |

## Goal
Make CLI and Python facade extensible so the core ships a complete domain-neutral application and finance adds `fetch`, providers and strategies on top — and close the decouple phase.

## Context (self-contained)
**App factory** (`cli/app.py`):
```python
def build_app(*, name: str, config_cls: type[ProjectConfig], project_cls: type[Project],
              extra_commands: Sequence[Callable[[typer.Typer], None]] = ()) -> typer.Typer: ...
EXIT_CODES: dict[type[BaseException], int]      # core: ConfigError 2, DataError 3, NumericalError 4, PersistenceError 2; default 1
def register_exit_code(exc: type[BaseException], code: int) -> None: ...
```
Core commands: `validate-config`, `config-schema`, `fit`, `decode`, `forecast`, `simulate`, `backtest` (hooks from config are resolved by name via `REGISTRY` kind `metric`/hook names only if registered), `compare`, `selftest` — all taking data from `data.source`. Global options `--out --device --seed --log-level` (and `--offline`, which the core ignores with a debug log unless a source honours it). Core console script `regimekit-core` (temporary, removed by E10-T08 where it becomes `hsmmkit`).
Finance (`finance/cli.py`): `app = build_app(name="regimekit", config_cls=FinanceConfig, project_cls=FinanceProject, extra_commands=[add_fetch])`; registers `ProviderError → 5`; console script `regimekit` points here.
**Project** (`project.py`): `Project(config)`, `from_yaml(path, overrides, config_cls=ProjectConfig)`, `load_data(obj=None) -> PreparedDataset` (via the configured source; `obj` for numpy/pandas/xarray), `fit`, `decode`, `forecast`, `backtest(data=None, hooks=())`, `simulate`, `save/load`, `compare`. `FinanceProject(Project)` (`finance/project.py`) adds `fetch(offline=False)` (= `load_data` through the provider source + quality report), default hooks from `evaluation.strategy`, and keeps `regimekit.Project` as an alias of `FinanceProject` for 0.1/0.2 API compatibility.
**Close the phase.** Remaining allow-list entries tagged `E10-T07b` (CLI help texts, `project.py`, `__init__.py`) are fixed; both allow-list files are deleted and the scanner asserts that no allow-list file exists; the `core-only` CI job becomes required.

## Creates
- `src/regimekit/cli/app.py`, `src/regimekit/project.py` (rewritten), `src/regimekit/finance/{cli,project}.py`
- `pyproject.toml` console scripts (`regimekit` → finance app, `regimekit-core` → core app)
- `tests/integration/test_e10_t07b_core_cli.py`, `tests/finance/test_e10_t07b_finance_cli_parity.py`
- `docs/guide/cli.md` (core vs finance commands)

## Acceptance criteria
1. E8-T02 parity tests pass for the finance app (`fetch, fit, decode, forecast, backtest, simulate, compare, selftest, validate-config`) with artifacts equal to the v0.2.0 golden values.
2. `regimekit-core fit/decode/forecast/backtest` run end-to-end on the gaussian csv fixture with `HSMM_CORE_ONLY=1`; `regimekit-core fetch` does not exist.
3. Neutral-name scanner passes with **no** allow-list; finance-import test passes with no allow-list; `core-only` job required and green.
4. `from regimekit import Project` still returns the finance facade; `Project.fetch` works as in 0.2.0.

## Tests
Core CLI integration, finance parity, exit codes (`ProviderError → 5` only in the finance app), scanner without allow-list, golden test.

## Out of scope
Extraction (E10-T08); `stream`/`presets` commands (E15-T04, E10-T10).

## Connections
- Depends on: E10-T07a, E10-T04b, E10-T05, E10-T06
- Blocks: E10-T08
- Relates to: E8-T02, E10-T01
