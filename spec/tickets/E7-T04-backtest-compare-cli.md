# E7-T04 — `backtest` and `compare` CLI, comparison table across configs, `report.md`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-E04, FR-C02, FR-P02 | M5 | §5.8 Results, §5.9, §4.10 |

## Goal
Expose backtesting and multi-config comparison on the CLI with parquet outputs and a Markdown report.

## Context (self-contained)
- `regimekit backtest CONFIG`: fetch (respecting `--offline`) → `run_backtest` → write `backtest_metrics.parquet` (long), `backtest_folds.parquet`, `strategy.parquet` (if configured), `report.md`.
- `regimekit compare CONFIG... [--allow-mixed] [--metric M]`: runs `backtest` (or loads existing results from each config's `output_dir` if present and `--reuse`) for each config; refuses if snapshot IDs differ across configs unless `--allow-mixed`; writes `compare.parquet` and `compare.md`: one row per config, columns = summary metrics per horizon (mean ± sd), plus in-sample `bic`, `icl`, `n_free`, fit time; best value per column marked.
- `report.md` (`evaluation/report.py`): header (config name, snapshot IDs, seed, versions, backend), model summary (families, N, order, covariates), fit summary (best objective, restarts dispersion, convergence), diagnostics with warnings (E5-T04), backtest summary table, strategy table, timings.

## Creates
- `backtest`, `compare` commands; `src/regimekit/evaluation/compare.py`, `evaluation/report.py`; writers in `persistence/results.py`
- `tests/integration/test_e7_t04_cli.py`

## Acceptance criteria
1. `backtest` on a synthetic csv project writes all four artifacts; `backtest_metrics.parquet` schema asserted.
2. `compare` with two configs on the same snapshot produces a two-row table; with different snapshots exits 3 unless `--allow-mixed`.
3. `report.md` contains every section listed above (assert headings).

## Out of scope
`Project` facade (E8-T02); plotting.

## Connections
- Depends on: E7-T03, E1-T02b, E2-T05
- Blocks: E8-T02
- Relates to: E5-T04
