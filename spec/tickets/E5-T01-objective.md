# E5-T01 — Objective: NLL + log-priors + log-det-Jacobian on `u`, jitted `value_and_grad`, non-finite guards

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M11, NFR-01, D3 | M3 | §4.8 Objective, §6.3 item 2, §6.4.3 |

## Goal
Define the single differentiable scalar the optimizer minimizes, for any family/engine combination.

## Context (self-contained)
```
J(u) = −ℓ(θ(u)) − log p(θ(u)) − log|det ∂θ/∂u|      (MAP; use_priors=True)
J(u) = −ℓ(θ(u))                                    (MLE; use_priors=False)
```
`θ = manager.unflatten(u)`; `ℓ = engine.loglik(θ, spec, space, data)`; `log p = manager.log_prior(θ)`; `log|det| = manager.log_det_jac(u)`.
```python
def make_objective(spec, space, engine, manager, data, use_priors: bool) -> Objective:
    # returns eqx.Module with .value(u), .value_and_grad(u) (jitted, spec/space/manager static), .n_free
```
Guards: `value_and_grad` returns `(J, g)`; the caller (E5-T02) checks `isfinite`; provide `check_finite(J, g, u, names) -> None | NumericalError` that names the first non-finite gradient coordinate (via `manager.names()`). Non-finite must never propagate silently.
The objective closes over `data` as a pytree argument (not static) so backtest folds reuse compilation when shapes match (E7-T02b pads `T` to buckets).

## Creates
- `src/regimekit/estimation/objective.py`
- `tests/unit/test_e5_t01_objective.py`, `tests/property/test_e5_t01_gradients.py`

## Acceptance criteria
1. Gradient check (Appendix D): `jax.grad` vs central finite differences (step 1e-6) rel ≤ 1e-5 for every combination in {GBM, Student-t} × {geometric, negbin, gamma-explicit} × {order 1, order 2 (expanded only)} × {with/without transition, duration, emission covariates} on `T=40, N=2`.
2. `use_priors=False` value equals `−engine.loglik` bitwise.
3. With priors, `J` differs from `−ℓ` by exactly `−log_prior − log_det_jac` (assert decomposition).
4. `check_finite` names the offending coordinate on a crafted `u` producing `inf`.
5. Second call with new `u`/`data` of the same shapes does not recompile.

## Out of scope
Optimization loop (E5-T02).

## Connections
- Depends on: E4-T03b, E1-T03b (also E4-T06a for the explicit combination; if not merged yet, mark that parametrization `xfail(strict=False)`)
- Blocks: E5-T02, E9-T02 (log_joint for MCMC extra)
- Relates to: E1-T03a
