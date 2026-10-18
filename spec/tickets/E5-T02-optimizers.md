# E5-T02 — Optimizers: `optax.lbfgs` with zoom line search, `adam` fallback, convergence criteria, restart record

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | D3, NFR-02 | M3 | §4.8 Optimizers, §5.4 (`FitResult.restarts`) |

## Goal
Minimize `J(u)` from one start with robust convergence handling and a per-run record.

## Context (self-contained)
```python
@dataclass(frozen=True)
class RunRecord: objective: float; loglik: float; n_iter: int; converged: bool; wall_s: float; reason: str; u: Array
def minimize(objective: Objective, u0: Array, cfg: EstimationSection, log) -> RunRecord
```
- `lbfgs`: `optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20))` driven by `optax.value_and_grad_from_state`; loop in Python with `jax.lax.while_loop`-free control (iterations are few; per-iteration jitted `update`). Convergence: `|ΔJ| ≤ tol_f · max(1, |J|)` (default 1e-9) **or** `‖∇J‖_∞ ≤ tol_g` (default 1e-6); stop at `max_iter` (default 1000) with `converged=False, reason="max_iter"`.
- `adam`: lr `1e-2` with cosine decay to 0 over `max_iter`, same convergence tests (evaluated every 10 iterations).
- Non-finite `J` or gradient at any iteration → abort *this* run with `reason="non_finite"`, `converged=False`, returning the last finite `u`. Non-finite already at `u0` → `reason="non_finite_start"`, `u=u0`. `minimize` never raises for non-finiteness; the multi-start layer (E5-T03) decides.
- Progress logging every `k` iterations (`log_every`, default 25) via structlog: `iter, J, grad_norm, elapsed`.
- Determinism: no randomness inside `minimize`.

## Creates
- `src/regimekit/estimation/optimizers.py`
- `tests/unit/test_e5_t02_optimizers.py`, `tests/benchmark/test_e5_t02_fit_bench.py`

## Acceptance criteria
1. On a 2-state GBM/geometric synthetic problem (`T=500`) from a perturbed start, `lbfgs` converges within 200 iterations to within 1e-6 of the objective found by a long `adam` run.
2. A crafted objective producing `inf` at iteration 5 aborts with `reason="non_finite"` and returns the iteration-4 `u`; no exception.
3. `max_iter=3` returns `converged=False`.
4. Benchmark: one L-BFGS fit ≤ 500 iterations at NFR-02 size completes ≤ 60 s (recorded in baseline).
5. Two runs with identical inputs give bitwise identical `u`.

## Out of scope
Initialization and multi-start (E5-T03); standard errors (E5-T04).

## Connections
- Depends on: E5-T01
- Blocks: E5-T03
- Relates to: E4-T07 (baseline file)
