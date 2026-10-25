# ADR-0008: Approximate-inference plug-in interface reserved for factorial models

## Context

Factorial or coupled hidden Markov models (several regime chains evolving together) make exact
forward-backward intractable in general: the joint state space is the product of the chains' state spaces.
regimekit does not ship factorial inference in 0.1.x, but the engine interface must not change shape when it
is added. The E9-T02 spike prototyped the composition and measured the exact baseline.

## Decision

1. **Interface (unchanged from the original decision).** `InferenceEngine.approx()` is part of the public
   interface next to `loglik`, `forward`, `forward_backward`, `viterbi`. Exact engines return `None`. A future
   factorial engine returns an object satisfying the `ApproxInference` protocol
   (`regimekit.engine.approx`): `fit(posterior_init, theta, factorial, data) -> ApproxPosterior` with per-chain
   regime marginals `(T, N_c)` and a lower bound on `log p(y)`.
2. **Composition.** Chains compose into one `StateSpace` over the Cartesian product of their expanded states
   (`regimekit.statespace.factorial.build_factorial`, experimental). For independent chains the joint
   `A_t` is the Kronecker product of the per-chain `A_t^(c)`, built in log space so structural zeros stay
   `-inf` without NaN. The existing `ExpandedHMMEngine` runs the joint space unchanged; the spike verifies
   that it reproduces the single-chain results exactly (loglik is the sum, the joint posterior the outer
   product, both to 1e-10, `tests/spikes/test_e9_t02_factorial.py`). Coupling is a hook only: a log
   interaction factor added to `log A_t` and renormalised per row; it is neither estimated nor tested beyond
   normalisation.
3. **Algorithm choice: structured mean-field.** The approximate engine will use structured variational
   inference: one forward-backward per chain, with each chain's emission log-density replaced by its
   expectation under the other chains' current marginals, iterated to a fixed point (Ghahramani and Jordan,
   1997). Blocked Gibbs sampling is rejected as the default because it gives no deterministic objective for
   MAP/L-BFGS (the estimator of ADR-0005), is stochastic per gradient, and needs many sweeps for mixing; it may
   be added as a cross-check.
4. **Interface diff (proposed, not applied in 0.1.x).** `StateIndex.regime_of` is `(S,)`; a factorial index
   needs a per-chain regime array. The prototype keeps `StateIndex` untouched (guardrail, overview section 8)
   and carries `regime_of_chain (S, C)` in `FactorialSpace`. Adopting the design means adding
   `regime_of_chain (S, C)` and chain sizes to `StateIndex` (with `C = 1` for every existing model) and
   `collapse` per chain; parameter names get a chain prefix (`c0.emission.mu`, ...).

| | Exact (joint space) | Structured mean-field | Blocked Gibbs |
|---|---|---|---|
| Time per pass | `O(T prod_c S_c^2)` | `O(T sum_c S_c^2)` per sweep | `O(T sum_c S_c^2)` per sweep |
| Memory (states) | `prod_c S_c` per step, `O(T prod_c S_c)` stored | `O(T sum_c S_c)` | `O(T C)` samples |
| Result | exact | biased, deterministic bound | unbiased in the limit, stochastic |
| Differentiable objective | yes | yes (the bound) | no |

The exact route is a time budget: the sequential scan needs `O(T S)` memory and `O(T S^2)` time with
`S = prod_c S_c` (the optional parallel scan additionally holds `(T, S, S)` and is limited to `T S^2 <= 5e7`
by `engine/parallel.py`). Three 3-state chains give `S = 27`, `S^2 = 729`, fine for `T` in the tens of
thousands; five 4-state chains give `S = 1024`, about a million operations per step, which is out of reach
for fitting (every L-BFGS iteration needs a gradient pass).

The mean-field row assumes an additive per-chain emission (as in the spike). For a general joint emission the
expected log-emission under the other chains' marginals costs `O(T prod_c N_c)` per sweep.

## Consequences

- The `InferenceEngine` protocol is stable across 0.1.x and the eventual factorial extension; nothing changes
  for `M = 1` or single-chain users (regression-checked: `approx()` is still `None`).
- The prototype and the protocol are experimental and unwired: no config key, no CLI, no estimation.
- No batch (`B`) dimension is introduced in the core (guardrail, overview section 8).
- Open for the real implementation: estimation of coupling parameters, the mean-field bound as an objective
  for E5-T01's machinery, per-chain duration families, order > 1 chains.

## Status

accepted
