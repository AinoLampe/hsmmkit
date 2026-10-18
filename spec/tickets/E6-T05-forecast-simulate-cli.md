# E6-T05 — `forecast` and `simulate` CLI commands + `forecast.parquet` / `simulation.parquet` writers

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-C02, FR-P02 | M4 | §5.8 Results, §5.9 |

## Goal
Expose forecasting and simulation on the command line with documented long-format parquet outputs.

## Context (self-contained)
- `regimekit forecast CONFIG --model DIR [--horizon H] [--n-sim N] [--scenario PATH]`: loads model + prepared data, runs `MonteCarloForecaster`, writes `forecast.parquet` (long: `origin_time, horizon_time, step, kind ∈ {regime_prob, quantile, mean, var}, key (regime index / quantile level / series), value`), `forecast_samples.parquet` optional (`--samples`), and `forecast.json` summary (origin, horizon, n_sim, forecaster, seed, snapshot IDs).
- `regimekit simulate CONFIG [--params model|config] [--model DIR] [--steps T] [--n-paths P]`: `config` builds `θ` from `init`/`values` in the config (via `lattice_jitter` with zero jitter); writes `simulation.parquet` (long: `path, step, time, regime, phase, return`) and `simulation_segments.json`.
- `--out` overrides `project.output_dir`; `--seed` overrides `project.seed`.

## Creates
- `forecast`, `simulate` commands; `persistence/results.py::{write_forecast, write_simulation, read_forecast}`
- `tests/integration/test_e6_t05_cli.py`; `docs/guide/outputs.md` (schemas)

## Acceptance criteria
1. Round trip on synthetic data: `forecast` then `read_forecast` reconstructs `regime_probs`, `cum_quantiles`, `mean`, `var` bitwise.
2. `simulate --params config --steps 500 --n-paths 3` produces 1500 rows with valid regimes/phases.
3. Parquet schemas documented and asserted in tests (column names/dtypes).

## Out of scope
`Project` facade (E8-T02).

## Connections
- Depends on: E6-T04b, E6-T01, E1-T02b
- Blocks: E8-T02
- Relates to: E6-T02
