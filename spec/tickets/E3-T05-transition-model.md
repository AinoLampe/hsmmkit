# E3-T05 — `TransitionModel`: jump matrix, multinomial-logit covariate link, generator mode (`expm`)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M08, FR-M09, FR-M10 (memory rows) | M1 | §4.3 Geometric/Generator, §4.4 Transitions, §4.5, Appendix E |

## Goal
Compute the per-step jump probabilities between different regimes (rows indexed by memory tuple), with covariates, and the exact CTMC alternative for HMM mode.

## Context (self-contained)
```python
class TransitionModel(Protocol):
    def param_specs(self, n_states: int, order: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]: ...
    def jump_probs(self, theta: Params, z_pi: Array | None, memory: MemoryLayout) -> Array:
        """→ (T, R, N): row r is a memory tuple (order 1: current regime i; order 2: (i, j)); zero on the current regime; rows sum to 1."""
```
`MemoryLayout` (this ticket, `model/covariates.py` or `transition.py`): `tuples (R, k−1)` admissible memory tuples (order 1: `R = N`, tuple = `(i,)`; order 2: `R = N(N−1)`, tuples `(i, j)` with `i ≠ j`), `current_of (R,)` = last element (the regime whose exit is being modelled), `target_mask (R, N)` bool = `target ≠ current`.
Multinomial logit with reference target: for row `r` with current regime `i`, targets `j ≠ i`, reference target `ref(i)` = smallest `j ≠ i`:
```
π_rj(t) = exp(a_rj + b_rjᵀ z^π_t) / Σ_{l≠i} exp(a_rl + b_rlᵀ z^π_t),   a_{r,ref} = 0, b_{r,ref} = 0
```
Specs: `transition.a (R, N−2)` real (free logits, reference dropped), `transition.b (R, N−2, K_pi)` real with `NormalPrior(0, coef_prior_sd)`; optional `DirichletPrior(alpha)` on the resulting rows (prior evaluated on `π`, with the simplex bijector's Jacobian — implement by declaring `transition.pi_rows` as domain `simplex_ref` instead of raw logits when no covariates are used; with covariates use the logit spec and the Dirichlet prior is applied to `π` at `z = 0`). Parameter counts must equal Appendix B: `N(N−2)` (order 1), `N(N−1)(N−2)` (order 2). `N = 2` → no free parameters (each row deterministic).
Generator mode (HMM only, `family=geometric`, all `r=1`, order 1): `Q_ij = λ_i π_ij`, `Q_ii = −λ_i`; `P_t = expm(Q Δ_t)` via `jax.scipy.linalg.expm`; exposed as `transition_matrix_generator(theta, lam, dt) -> (T, N, N)`; consumed by E4-T02b when `mode: generator`.

## Creates
- `src/regimekit/model/transition.py` (replaces E1-T05 stub), `src/regimekit/model/covariates.py` (`CovariateSet` helpers, `MemoryLayout`)
- `tests/unit/test_e3_t05_transition.py`, `tests/property/test_e3_t05_transition.py`
- `docs/math/transitions.md`

## Acceptance criteria
1. Rows sum to 1 ≤ 1e-12 with exactly zero on the current regime, for orders 1 and 2, `N ∈ {2,3,4}`.
2. No covariates ⇒ `jump_probs` constant over `t`.
3. `ParamManager.n_free` for the transition specs equals Appendix B counts.
4. Appendix E golden: `λ = (2, 6)`, weekly `Δ = 1/52`: `expm(QΔ)` off-diagonals `0.0357 / 0.1070` vs first-order `λΔ` = `0.0385 / 0.1154` (4 dp); per-entry difference ≤ `(λ_max Δ)²`; difference → 0 as `Δ → 0` (property over `Δ ∈ [1e-4, 1/12]`).
5. Order-2 layout excludes tuples with `i = j`; `current_of` correct.

## Out of scope
Building `A_t` over expanded states (E4-T02b).

## Connections
- Depends on: E1-T03b
- Blocks: E4-T02a, E4-T02b
- Relates to: E3-T03 (λ for generator mode)
