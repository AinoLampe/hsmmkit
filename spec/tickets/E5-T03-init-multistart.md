# E5-T03 — Initialization strategies and multi-start orchestration → `FitResult`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | NFR-03, D3 | M3 | §4.8 Initialization, Multi-start; §6.3 item 6; §6.4.6; Part II §15.7 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendments 2 and 5: `estimation/map.py`, `FitResult.estimator`, clock-neutral initialization). No behavioural change for the finance profile.

## Goal
Produce valid starting points by three strategies, run `restarts` minimizations with independent keys, pick the best, and assemble a `FitResult`.

## Context (self-contained)
`estimation/init.py`, each returning a list of `θ` dicts (converted to `u` by `manager.flatten`):
- `rollvol_kmeans(data, spec, key)`: 21-step rolling realized volatility of `observations` (in clock units: `sqrt(mean(y²)/nominal_dt)` — no literal `252`), k-means with `N` clusters (NumPy implementation with the given key, 20 iterations, k-means++ seeding), sort clusters by volatility → `σ_i`; `μ_i` = cluster mean of `y_t / nominal_dt`; `λ_i` = `r_i / (nominal_dt · mean run length in steps)` (per cluster, via RLE of cluster labels); `π` = empirical transitions between cluster labels at run boundaries (row-normalized, zero diagonal, Laplace-smoothed). Explicit families: convert `λ` to their mean-sojourn parameterization via the family's `init_from_mean_sojourn`. Coefficients start at 0.
- `lattice_jitter(spec, key)`: start at config `init`/lattice values; free entries jittered by log-normal ±10 % (`exp(0.1·ε)` multiplicative for positive, additive `0.1·ε·scale` for real).
- `random(spec, key)` (scale-free, clock-neutral): with `s = sd(observations) / sqrt(nominal_dt)`: `σ ~ U[0.25, 4] · s` (sorted), `μ ~ U[−1, 1] · s`, mean sojourn `m ~ U[5, 250]` steps (log-uniform) → `λ = r_i / (m · nominal_dt)`, coefficients `N(0, 0.1)`, jump logits `N(0, 0.5)`. (For daily DAX-like data this reproduces the former annualized ranges `σ ∈ [0.04, 0.64]`, `μ ∈ [−0.16, 0.16]`.)
`estimation/map.py` (Part II §15.7: the MAP estimator is one of several later; `estimation/__init__.py` re-exports `fit`): `fit(spec, data, cfg, seed) -> FitResult` with `estimator="map"`: keys `key_for(seed, restart_idx)`; strategies from `cfg.init` cycled over `cfg.restarts` starts (first strategy gets the remainder); each restart → `minimize` (E5-T02); best finite objective wins; `restarts` DataFrame lists all; `converged` = best run converged; label handling: if `not has_identifiability_guard(spec)` → `post_sort_states` (E3-T06) + warning; `loglik` of the winner recomputed with `engine.loglik`; `diagnostics` initially `{}` (E5-T04 fills). Runs are sequential (deterministic order); `device` from cfg passed to `jax_setup`.

## Creates
- `src/regimekit/estimation/{init,map}.py` (+ `fit` re-export in `estimation/__init__.py`)
- `tests/unit/test_e5_t03_init.py`, `tests/integration/test_e5_t03_fit.py`, `tests/integration/test_e5_t03_determinism.py`

## Acceptance criteria
1. Each strategy yields `u` for which `objective.value(u)` is finite on synthetic data (GBM/NB, order 1 and 2, with covariates).
2. `rollvol_kmeans` on a 2-regime synthetic series with σ = (0.10, 0.40) recovers σ within 30 % and the correct ordering.
3. Best-of-restarts selection: a test injects one restart with a lower objective and asserts it wins; the `restarts` table has exactly `cfg.restarts` rows with the documented columns.
4. Determinism (§6.4.6): running `fit` twice with the same config/data/seed on CPU gives bitwise identical `uparams`.
5. Without ordering constraint or fixed parameters a warning is issued and states are sorted by σ.
6. `FitResult.estimator == "map"`; `from regimekit.estimation import fit` works; every init strategy produces a finite objective on the same synthetic data re-expressed under `clock: steps` (`dt = 1`, per-step parameters).

## Out of scope
Diagnostics, std errors (E5-T04); EM (E5-T06).

## Connections
- Depends on: E5-T02, E3-T06
- Blocks: E5-T04, E5-T05a, E5-T06, E7-T02b, E8-T01
- Relates to: E0-T05 (`key_for`)
