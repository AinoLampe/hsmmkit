# E3-T06 — State lattice generator, ordering constraints wiring, label-order post-sorting

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-M04, FR-M05 | M1 | §4.6 (increasing bijector), §4.8 (label switching) |

## Goal
Generate fixed-parameter state lattices from parameter grids and provide the identifiability helpers (increasing constraint wiring, post-fit sorting).

## Context (self-contained)
- `lattice.build(grids: dict[str, list[float]]) -> LatticeStates`: Cartesian product of grids (e.g. `mu × sigma`) → `n_states = Π |grid|` and per-parameter `values (N,)` arrays; `to_param_cfgs()` returns `ParamCfg(mode="fixed", values=...)` per grid parameter, overriding the emission config; config validation (E1-T02a) already requires `n_states` to equal the product — this ticket adds the check function `validate_lattice(cfg)`.
- `apply_ordering(specs, name="emission.sigma")`: sets `constraint="increasing"` on the named spec (used by `GBMFamily.from_config` when `constraint: increasing`).
- `post_sort_states(theta: Params, spec: ModelSpec, by="emission.sigma") -> (theta_sorted, permutation)`: permutes all per-state parameter arrays (emission, duration, transition rows/columns incl. memory tuples) consistently; used by E5-T03 when neither an ordering constraint nor fixed parameters are configured (with a warning "states post-sorted by σ; consider `constraint: increasing`").
- `has_identifiability_guard(spec) -> bool`: true if any per-state spec has `increasing` or all emission specs are `fixed`.

## Creates
- `src/regimekit/model/lattice.py`
- `tests/unit/test_e3_t06_lattice.py`

## Acceptance criteria
1. Grids `mu: [−0.3, 0, 0.2]`, `sigma: [0.15, 0.35]` → 6 fixed states with the expected value table.
2. A GBM spec built with `constraint: increasing` yields strictly increasing `σ` for random `u` (through `ParamManager`).
3. `post_sort_states` on a 3-state spec with shuffled σ returns sorted σ and consistently permuted `mu`, `rate`, and transition rows/columns (assert `jump_probs` invariance under relabelling).

## Out of scope
Multi-start orchestration (E5-T03).

## Connections
- Depends on: E1-T03b, E3-T01
- Blocks: E5-T03
- Relates to: E3-T05 (permutation of rows), risk "label switching"
