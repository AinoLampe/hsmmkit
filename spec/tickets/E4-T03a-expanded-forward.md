# E4-T03a — `ExpandedHMMEngine`: log-space forward recursion, `loglik`, masks, initial distribution

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-I01, NFR-01, NFR-02 | M2 | §4.7.2 Forward, Appendix C, §6.3 |

## Goal
Implement the jitted forward pass over the expanded state space with `lax.scan`, returning `log_alpha (T, S)` and the log-likelihood.

## Context (self-contained)
```python
class InferenceEngine(Protocol):
    def loglik(self, params, spec, space, data: PreparedDataset) -> Array: ...
    def forward(self, params, spec, space, data) -> tuple[Array, Array]: ...   # (log_alpha (T,S), loglik)
    def forward_backward(...) -> Posterior: ...    # E4-T03b
    def viterbi(...) -> Array: ...                 # E4-T05
    def approx(self) -> ApproxInference | None: ...   # returns None (FR-X01 plug-in point)
```
Steps inside `forward` (jitted with `spec`/`space` static):
1. `log_emis = spec.emission.log_prob(y, dt, θ, z_e)` → `(T, N)`; apply mask: `jnp.where(mask.all(-1)[:, None], log_emis, 0.0)` (masked steps contribute zero density; per-series masks are summed inside the family for `M>1`).
2. `ℓ_t(s) = log_emis[t, regime_of[s]]` via gather (`log_emis @ Cᵀ` is equivalent; use `jnp.take`).
3. `log_rho = initial_log_rho(...)` (E4-T02a/b).
4. `α_1 = log_rho + ℓ_1`; scan over `t = 2..T`: `logA = log transition_fn(θ, z_pi_t, z_d_t, dt_t)`; `α_t = ℓ_t + logsumexp(α_{t−1}[:, None] + logA, axis=0)`. Carry `α_{t−1}`, stack outputs to `(T, S)`.
5. `loglik = logsumexp(α_T)`. Guard: non-finite loglik → raise `NumericalError` **outside** the jitted function (check the returned value; `jax.debug` is not used).
Covariate slices when a block is `None`: pass zeros of shape `(T, 0)`; families treat `K = 0` as no link.
`data.dt`, covariates and returns are sliced per step inside the scan via `xs=(ℓ[1:], z_pi[1:], z_d[1:], dt[1:])`.

## Creates
- `src/regimekit/engine/expanded_hmm.py` (forward + loglik), `engine/base.py` updates
- `tests/golden/test_e4_t03a_forward.py`, `tests/property/test_e4_t03a_forward.py`, `tests/benchmark/test_e4_t03a_bench.py`

## Acceptance criteria
1. Golden: `log_alpha` max-abs ≤ 1e-8 and `loglik` rel ≤ 1e-8 vs `tests/reference/ref_hmm.forward_backward` driven with E4-T02b's `A_t` (hypothesis: `N ∈ [2,4]`, `T ∈ [5,200]`, `r_i ∈ [1,3]`, order ∈ {1,2}, random covariates).
2. Masking: a fully masked step `t` leaves `α_t = logsumexp-propagated α_{t−1}` (no emission term); loglik of a dataset with a masked step equals the reference with `ℓ_t = 0`.
3. `initial="uniform"` puts zero mass on phases > 1; `initial="free"` uses `theta["initial.logits"]`.
4. `jax.grad(loglik)` w.r.t. `u` (through `ParamManager.unflatten`) matches central finite differences (step 1e-6) rel ≤ 1e-5 on a small instance (`T=30, N=2, r=2`).
5. Performance: jitted `forward` ≤ 50 ms for `T=5000, N=4, r=5, order 1, 2 covariates` on an 8-core CPU (half the 100 ms FB budget), recorded in the benchmark job.

## Out of scope
Backward pass, smoothing, `Posterior` (E4-T03b); Viterbi (E4-T05).

## Connections
- Depends on: E4-T01, E4-T02b, E3-T01
- Blocks: E4-T03b
- Relates to: E5-T01 (uses `loglik`)
