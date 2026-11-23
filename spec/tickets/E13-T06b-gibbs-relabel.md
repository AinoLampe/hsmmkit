# E13-T06b — Blocked Gibbs (FFBS + conjugate draws), post-hoc relabelling, NUTS/Gibbs agreement

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES05, NFR-17 | M10 | Part II §14.9 (b), §14.7; Stephens 2000 |

## Goal
Offer a fast sampler for fully conjugate models and make MCMC output usable when no ordering constraint fixes the labels.

## Context (self-contained)
- `estimation.mcmc.sampler: gibbs` (requires every component conjugate — reuse `ConjugateBlock` from E13-T05a; else `ConfigError` listing non-conjugate components and suggesting `nuts`).
- One sweep per sequence batch: (1) states `s ~ p(s | θ, y)` by FFBS (E13-T04, one draw per sequence); (2) sufficient statistics from the sampled path (one-hot `γ`, observed stay/advance/jump counts computed from the expanded path); (3) `θ_block ~ posterior(prior, stats)` for each block (Normal-Wishart, Gamma, Beta, Dirichlet samplers with `jax.random`). Chains are `vmap`ped; keys `key_for(seed, "gibbs", chain, sweep)`. Warmup/draws as for NUTS; output through `interop.arviz.to_inference_data` (same dims as E13-T06a, plus optional `path_samples` group thinned by `mcmc.keep_paths_every`).
- **Relabelling** (`estimation/relabel.py`, used for NUTS and Gibbs when no ordering constraint/fixed parameters identify states): pivot = MAP ordering (fit MAP first or use the chain draw with highest log-joint); for every draw choose the permutation `σ` of states minimizing `Σ_i ‖θ_{σ(i)} − θ_i^{pivot}‖²` over emission location parameters standardized by pivot scales (Hungarian matching via E14-T03 helper if present, else brute force for `N ≤ 6`); apply `σ` consistently to all state-indexed parameters (emission, duration rows, jump matrix rows and columns with topology preserved, initial distribution). Report `diagnostics["mcmc"]["relabelled_fraction"]`.
- **Agreement test**: on the Appendix E fixture (conjugate Gaussian + geometric durations with Beta priors on per-step `a`), Gibbs and NUTS posterior means agree within 2 Monte-Carlo SE.

## Creates
- `estimation/mcmc.py` (Gibbs), `estimation/relabel.py`, samplers in `model/conjugate.py`
- `tests/integration/test_e13_t06b_gibbs.py`, `tests/unit/test_e13_t06b_relabel.py` (markers `bayes`, `slow`)

## Acceptance criteria
1. Gibbs and NUTS posterior means agree within 2 MC SE on the Appendix E fixture; both within 2 posterior SD of truth, `R̂ ≤ 1.05`.
2. Relabelling: on synthetic draws with randomly permuted labels, relabelling recovers the pivot order for ≥ 99 % of draws; permuting a model's states and relabelling returns the original posterior means ≤ 1e-12.
3. Gibbs draws reproducible per seed; non-conjugate model with `sampler: gibbs` → `ConfigError` listing components.

## Tests
Integration (slow), relabel unit tests, reproducibility.

## Out of scope
Split-merge moves, HDP (E13-T09).

## Connections
- Depends on: E13-T06a, E13-T04, E13-T05a
- Blocks: E13-T07, E13-T09
- Relates to: E14-T03 (Hungarian helper)
