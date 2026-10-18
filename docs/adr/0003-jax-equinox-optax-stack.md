# ADR-0003: JAX + Equinox + Optax stack; no TFP dependency

## Context

regimekit needs autodiff (for MAP estimation via gradient-based optimizers), fast structured recursions
(forward-backward, Viterbi over potentially large expanded state spaces), and a way to represent model
parameters and static structure as pytrees that compose cleanly with `jax.jit`/`jax.lax.scan`. TensorFlow
Probability (TFP) offers ready-made HMM primitives but pulls in a heavy, separately-versioned dependency and
does not give the fine control needed for duration-augmented state spaces and per-step transition matrices.

## Decision

The numerical core is built directly on JAX, with Equinox (`eqx.Module`) for parameter/structure pytrees and
Optax for gradient-based optimization (L-BFGS, Adam). All probability distributions used by emission and
duration families are implemented in-house (`model/dists.py`) and validated against `scipy.stats` in tests,
rather than depending on TFP or NumPyro for the core (NumPyro remains an optional `mcmc` extra, not a core
dependency).

## Consequences

- Full control over log-space recursions, masking, and gradient safety (the "double where" pattern for NaN
  gradients) without working around a third-party HMM abstraction.
- More code to write and test for each distribution and recursion, offset by tests against `scipy` references
  (`tests/reference/`) and property tests (Appendix D tolerances).
- The `mcmc` extra (NumPyro) is available for users who want posterior sampling beyond MAP, without making it
  a hard dependency of the core engines.

## Status

accepted
