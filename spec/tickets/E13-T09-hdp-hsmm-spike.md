# E13-T09 — HDP-HSMM design spike (weak-limit approximation): ADR-0027 go/no-go

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | Part II §12.2 (non-goal: design spike only) | M10 | Part II §12.2, §14.9; Johnson & Willsky 2013 |

## Goal
Determine what a Bayesian nonparametric state count (HDP-HSMM, weak-limit) would require from the engine and samplers, and decide whether to pursue it after 0.5.0.

## Context (self-contained)
- Weak-limit HDP-HSMM (Johnson & Willsky 2013): truncation `L` states; global weights `β ~ Dir(γ/L, …)`; transition rows `π_i ~ Dir(α β)` with self-transitions removed (HSMM), explicit durations per state; blocked Gibbs with FFBS over segments (explicit engine) and auxiliary variables for the `β` update.
- Questions for the ADR: (1) can the existing explicit-engine FFBS (E13-T04) and conjugate blocks (E13-T05a) serve the sampler unchanged; (2) what new conjugate blocks are needed (hierarchical Dirichlet auxiliary counts); (3) memory/time at `L = 20, D_max = 200, T = 10 000`; (4) interaction with relabelling (E13-T06b) and pruning (E13-T05b) — the unused states play the role of pruned states; (5) API surface (`estimation.mcmc.sampler: hdp_gibbs`, `model.n_states` as truncation); (6) testing strategy (posterior over the number of used states on synthetic data).
- Prototype (optional, `spikes/hdp_hsmm.py`): one sampler sweep on a 3-true-state fixture with `L = 10` showing that the number of occupied states concentrates near 3.

## Creates
- `docs/adr/0027-hdp-hsmm.md` (accepted with go/no-go, interface diff and effort estimate), optional `spikes/hdp_hsmm.py`

## Acceptance criteria
1. ADR-0027 accepted with go/no-go, interface diff against the 0.3.0 APIs and an effort estimate in ticket sizes.
2. No code merged to `src/`.

## Tests
None beyond an optional spike test.

## Out of scope
Implementation.

## Connections
- Depends on: E13-T06b
- Blocks: E17-T03
- Relates to: E13-T04, E13-T05b
