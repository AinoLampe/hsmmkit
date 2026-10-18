# E8-T02 — `Project` facade and CLI completion (`fit`, global options, exit codes, logging, parity)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-C02, FR-C03 | M6 | §5.9 |

## Goal
Provide one Python entry point mirroring every CLI command, and finish the CLI (`fit`, global options everywhere).

## Context (self-contained)
```python
from regimekit import Project
p = Project.from_yaml("dax_3state.yaml", overrides={"project.seed": 7})   # or Project(config: ProjectConfig)
data = p.fetch(offline=False)          # PreparedDataset (+ quality report on p.last_quality)
fit  = p.fit(data)                     # FitResult
post = p.decode(fit, data)             # PosteriorResult
fc   = p.forecast(fit, data, horizon=20, n_sim=None)
bt   = p.backtest(data=None)           # BacktestResult (fetches if data None)
sim  = p.simulate(params="config", steps=500, n_paths=1)
p.save(fit, "runs/dax_3state"); fit2 = p.load("runs/dax_3state")
Project.compare([cfg1, cfg2], allow_mixed=False)
```
- `Project` holds the config, resolved engine/families/space (built once), logger and cache; every method returns the result object and writes nothing unless `write=True` (CLI passes `write=True`).
- CLI `fit CONFIG [--out] [--offline] [--device] [--seed] [--log-level]`: fetch → fit → save bundle to `<out>/<name>.rkmodel` → write `fit_report.json`, `config.resolved.yaml`. Global options apply to *all* commands (implemented as a typer callback); `--device` calls `jax_setup.setup` before any JAX use; `--log-level` configures structlog.
- Exit-code mapping in `main()` for every `RegimekitError`; unexpected exceptions → 1 with traceback at DEBUG level.
- `regimekit/__init__.py` re-exports `Project`.

## Creates
- `src/regimekit/project.py`, `cli/app.py` (final), `__init__.py` re-export
- `tests/integration/test_e8_t02_parity.py`

## Acceptance criteria
1. Parity: for each of `fetch, fit, decode, forecast, backtest, simulate, compare, selftest, validate-config` the CLI on a synthetic csv project and the `Project` method produce equal artifacts/results (parquet frames equal; forecasts bitwise with the same seed).
2. `--seed`/`--device`/`--log-level` reflected in `config.resolved.yaml` and logs.
3. A `ConfigError` raised inside `Project.fit` yields exit code 2 through the CLI; `ProviderError` → 5.
4. `Project` is importable without triggering network or file writes.

## Out of scope
New functionality.

## Connections
- Depends on: E8-T01, E7-T04, E6-T05, E6-T02, E2-T08, E5-T05b
- Blocks: E8-T03
- Relates to: E1-T02b
