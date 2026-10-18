# E6-T01 — Simulator: regime/phase paths and returns from any model given `Δ_t` and covariate paths

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-I06, NFR-03 | M3 | §5.5, §6.3 item 2 (`simulate_paths` JIT boundary), §6.4.2 |

## Goal
Sample expanded-state paths and observations from a specified model, deterministically per key, with segment bookkeeping.

## Context (self-contained)
```python
def simulate_paths(key, theta, spec, space, dt (T,), z_pi (T,K_pi)|None, z_d (T,K_d)|None, z_e (T,K_e)|None,
                   n_paths: int, init_state: Array | None) -> SimResult
# SimResult: states (n_paths, T) expanded ints, regimes (n_paths, T), returns (n_paths, T, M), log_rho used
```
- Initial expanded state: sample from `initial_log_rho` (E4-T02a) unless `init_state` given (used by forecasting: start from the filtered posterior sample).
- Per step in a `lax.scan`: `A_t = transition_fn(θ, z_pi_t, z_d_t, dt_t)`; `s_t ~ Categorical(A_t[s_{t−1}])` via `jax.random.categorical` on `log A_t` row; `y_t = emission.sample(key_t, regime_of[s_t], dt_t, θ, z_e_t)`. Keys: `key_for`-style folding per path and step (`jax.random.split` inside vmap).
- `segments_of(regimes) -> list[list[(start, end, regime)]]` (NumPy, outside jit).
- Also a convenience `simulate_dataset(...) -> PreparedDataset` (builds a synthetic `PreparedDataset` with `mask=True`, `PrepStats` zeros) for tests/recovery.
- vmap over paths; JIT boundary `simulate_paths` with static `n_paths`, `T`.

## Creates
- `src/regimekit/inference/simulate.py`
- `tests/property/test_e6_t01_simulate.py`, `tests/unit/test_e6_t01_simulate.py`

## Acceptance criteria
1. Empirical sojourn mean per regime within 3 SE of `r_i / a_i` (constant `Δ`, no covariates) over ≥ 2000 segments; minimum sojourn equals `r_i`.
2. Empirical jump frequencies at segment boundaries within 3 SE of `π`.
3. Return moments per regime within 3 SE of the GBM analytic mean/variance.
4. Same key ⇒ bitwise identical `SimResult`; different keys differ.
5. With a duration covariate `c > 0` and increasing `z_d`, sojourns shorten (monotone test on two covariate levels).

## Out of scope
Forecast aggregation (E6-T04b); CLI (E6-T05).

## Connections
- Depends on: E4-T02b, E3-T01
- Blocks: E5-T05a, E6-T04a
- Relates to: E4-T04
