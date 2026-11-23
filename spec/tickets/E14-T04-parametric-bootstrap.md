# E14-T04 — Parametric bootstrap: standard errors and nested-model tests

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EV05 | M11 | Part II §14.12 (Parametric bootstrap); Part I §4.8 (Hessian SE), E5-T04; Hansen 1992 |

## Goal
Provide simulation-based uncertainty for parameters and for comparisons of nested models where Hessian-based and χ² approximations are unreliable.

## Context (self-contained)
`selection/bootstrap.py`:
```python
def bootstrap_se(fit: FitResult, data: PreparedDataset | SequenceBatch, R: int, seed: int, *,
                 refit_from: Literal["mle", "init"] = "mle") -> BootstrapResult
def bootstrap_lrt(fit0: FitResult, fit1: FitResult, data, R: int, seed: int) -> LRTResult
```
- `bootstrap_se`: for `r = 1..R`: simulate a dataset with the same lengths, `Δ_t` and covariate paths as `data` from `θ̂` (E12-T02 batched simulate; keys `key_for(seed, "boot", r)`), refit with the same estimator config starting from `θ̂` (`refit_from: mle`, 1 restart) or from the configured init strategies, collect `θ̂_r` (constrained). SE = sample SD per parameter element; percentile intervals (2.5/97.5); failed/non-converged refits counted and excluded (reported). Label switching handled by sorting/relabelling as in the fit (ordering constraints or E13-T06b relabelling helper).
- `bootstrap_lrt` (nested models, `fit0 ⊂ fit1`): observed `D = 2(ℓ_1 − ℓ_0)`; simulate `R` datasets from `θ̂_0`, fit both models on each, bootstrap p-value `(1 + #{D_r ≥ D}) / (R + 1)`. Docs: the χ² approximation fails when `N` differs (Hansen 1992); bootstrap is the recommended test; caveats on boundary parameters.
- Comparison with Hessian SE (E5-T04) included in `BootstrapResult.compare_hessian` when available.
- Parallelism as E14-T02 (process pool optional; deterministic keys).
- CLI `hsmmkit bootstrap CONFIG --model PATH --R 200` writes `bootstrap.parquet` and a report section.

## Creates
- `src/hsmmkit/selection/bootstrap.py`, CLI wiring, `tests/integration/test_e14_t04_bootstrap.py` (`-m slow` for the coverage check)
- `docs/guide/core/uncertainty.md`

## Acceptance criteria
1. Bootstrap SE within 20 % of the simulation-based SE (SD of estimates over 50 independent synthetic datasets) for a 2-state Gaussian/geometric model, `T = 5000`.
2. `bootstrap_lrt` of `N = 2` vs `N = 3` on 2-state data gives p > 0.05 in ≥ 4/5 seeds and p < 0.05 on 3-state data in ≥ 4/5 seeds (`R = 99`, `-m slow`).
3. Deterministic per seed; failed refits reported.

## Tests
Integration (slow), determinism.

## Out of scope
Block bootstrap of observed data (non-parametric) — backlog.

## Connections
- Depends on: E12-T02
- Blocks: E16-T05, E17-T03
- Relates to: E5-T04, E13-T06b
