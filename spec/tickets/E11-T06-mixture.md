# E11-T06 — Per-state finite mixture of a base family

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EM05, FR-EM07 | M8 | Part II §14.2 (Mixture), Appendix F |

## Goal
Let every state's emission be a finite mixture of `K` components of any registered base family, with identifiability via in-state ordering and EM hooks via responsibilities.

## Context (self-contained)
```
f_i(y) = Σ_{k=1}^{K} w_ik g(y | θ_ik),   w_i ∈ simplex (bijector simplex_ref),   log f_i = logsumexp_k(log w_ik + log g(y | θ_ik))
```
- Config: `model.emission: {family: mixture, mixture: {components: K, base: <family>}, params: {...base params with leading component axis...}, base_options: {...}}`. Static `K` (same for all states in 0.2.0; per-state `K_i` via fixing surplus weights to 0 is documented, not implemented).
- `MixtureFamily(base, K)`: builds base param specs with shape `(N·K, …)` (base called with `n_states = N·K`) and `emission.weights (N, K−1)` free logits; `support`, `scale_by_dt`, `native_time` from base. `log_prob` evaluates base on `N·K` pseudo-states, reshapes to `(T, N, K)`, combines with `log w`.
- Ordering constraint within a state (identifiability): `constraint: increasing` on a chosen base parameter applies along the component axis per state (e.g. `scale` increasing over `k`), via the Part I increasing bijector applied per state row.
- `sample`: component via `Categorical(w_i)`, then base sample.
- EM hooks: `sufficient_stats(y, dt, gamma, z)` computes responsibilities `r_tik ∝ γ_ti w_ik g(y_t|θ_ik)`, returns `(Σ_t r_tik, base.sufficient_stats(y, dt, r reshaped (T, N·K), z))`; `m_step`: `w_ik ∝ Σ_t r_tik`, base `m_step` on the pooled stats; `None` if base has none. `conjugate_prior`: Dirichlet on `w` × base prior, if base has one.
- Masked rows: as base (summed per-series masks).
- `K = 1` must reduce to the base family **bitwise** (no logsumexp over a single element: short-circuit).
- Gate `model.emission.mixture` removed.

## Creates
- `src/hsmmkit/model/emission/mixture.py`, `tests/unit/test_e11_t06_mixture.py`; contract cases `mixture-gaussian-2`, `mixture-poisson-2`

## Acceptance criteria
1. `K = 1` mixture equals the base family bitwise (log_prob, sample with same key).
2. Both contract cases pass `family_contract`, including M-step consistency (check 7).
3. 2-component Gaussian mixture per state with increasing scale recovers weights within 0.05 and scales within 10 % on synthetic data (`T = 20 000`, 1 state).
4. Ordering constraint prevents label switching between components (property test: sorted after fit).
5. Gate removed.

## Tests
Unit, contract, recovery smoke (`-m slow` for the large case).

## Out of scope
Per-state variable `K_i`; Bayesian nonparametric mixtures.

## Connections
- Depends on: E11-T02
- Blocks: E11-T07
- Relates to: E11-T04 (Poisson base), E13-T02a (responsibility M-steps)
