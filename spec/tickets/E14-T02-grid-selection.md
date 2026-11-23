# E14-T02 — Grid model selection (`compare --grid`) over `N`, `r`, families and estimators

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EV04 | M11 | Part II §14.12 (Grid selection), §15.4 (`evaluation.grid`); Part I §4.10, E7-T04 |

## Goal
Select model structure systematically: one command evaluates a Cartesian grid of configurations by information criteria and cross-validated scores and returns one table.

## Context (self-contained)
- Config `evaluation.grid: {n_states: [2, 3, 4], r: [[1], [2], [3]] | "same_for_all": [1, 2, 3], emission: [gaussian, student_t], duration: [negbin, gamma], estimator: [map, em]}`; every axis optional (missing axis = base config value). Cells are generated as config overrides of the base config (deep-copied `ProjectConfig`, validated per cell; invalid combinations, e.g. `em` with a non-closed-form family before ECM availability or `gamma` with `engine: expanded`, are skipped with a reason column).
- Per cell: fit with the cell's estimator (restarts from config), IC (`loglik, n_free, aic, bic, icl`; Part I §4.10, `n_free` from `ParamManager`), CV summary (E14-T01) if `evaluation.cv` is set, wall time, convergence, warnings (occupancy, restart dispersion).
- Execution: sequential by default; `evaluation.grid.parallel: {backend: process, workers: k}` optional via `concurrent.futures.ProcessPoolExecutor` with deterministic per-cell keys `key_for(seed, "grid", cell_index)` (results independent of scheduling). Recompilation is per structure (expected).
- Output: `grid.parquet` (one row per cell; columns = axes + metrics + `skipped_reason`), `grid.md` with best per criterion marked and a short recommendation rule documented ("lowest BIC; ties → highest CV log score").
- CLI `hsmmkit compare CONFIG --grid` (uses `evaluation.grid`); `Project.compare_grid()`.
- Gate `evaluation.grid` removed.

## Creates
- `src/hsmmkit/selection/grid.py`, CLI/`Project` wiring, `tests/integration/test_e14_t02_grid.py`
- `docs/guide/core/model_selection.md` (table schema, recipe for choosing `r`)

## Acceptance criteria
1. On the Appendix E fixture (Gaussian form), the grid over `N ∈ {2, 3, 4}` (Part I FR-M01: `N ≥ 2`) selects `N = 2` by BIC and by CV log score.
2. Table schema documented and asserted; skipped cells carry a reason.
3. Parallel and sequential execution give identical tables (excluding wall time).
4. Deterministic per seed.

## Tests
Integration on the fixture (`-m slow` for the full grid, a 2-cell smoke on PRs).

## Out of scope
Bayesian model averaging; automatic search beyond the grid.

## Connections
- Depends on: E14-T01, E13-T07
- Blocks: E17-T03
- Relates to: E7-T04
