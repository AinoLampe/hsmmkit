# E8-T01 — `ModelStore`: `<name>.rkmodel/` bundle save/load with lossless round-trip

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-P01, NFR-03, NFR-10 | M6 | §5.8 Persistence |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendment 7: manifest records `profile` and `schema_version`). No behavioural change for the finance profile.

## Goal
Persist a `FitResult` so that it can be reloaded and reproduces its log-likelihood bitwise on the same backend.

## Context (self-contained)
Bundle directory `<name>.rkmodel/`:

| File | Content |
|---|---|
| `manifest.json` | `bundle_version` (=1), regimekit version, jax version, backend/device, created_at, seed, `profile` (copied from the resolved config, `finance` in 0.1.0), `schema_version` of the embedded config |
| `config.resolved.yaml` | full configuration (E1-T02b dumper) |
| `spec.json` | `ModelSpec` structure (families by name + static hyperparameters `r`, `order`, `d_max`, `engine`, `initial`, covariate names) |
| `params.safetensors` | constrained `θ` (all arrays, key = param name) and unconstrained `u` (key `__u__`) as float64 |
| `prep_stats.json` | `PrepStats` |
| `snapshots.json` | snapshot IDs + provider metadata |
| `fit_report.json` | restarts table, convergence, diagnostics, std errors |

`safetensors` needs the `safetensors` package → **new runtime dependency**: write ADR-0011 (safetensors vs npz; chosen for zero-copy, dtype-safe, language-neutral). `ModelStore.save(fit, path)`, `load(path) -> FitResult` (rebuilds `ModelSpec` from `spec.json` via the family registries, `ParamManager` from the resolved config, verifies `manager.unflatten(u) == θ` ≤ 1e-12 and that `bundle_version ≤ CURRENT`). Loading a newer bundle → `PersistenceError` with a clear message. `load` does not require data; `loglik` is checked by the caller with a dataset.

## Creates
- `src/regimekit/persistence/store.py`, `docs/adr/0011-safetensors.md`, dependency added
- `tests/integration/test_e8_t01_bundle.py`

## Acceptance criteria
1. `load(save(fit)).params` bitwise equal; `engine.loglik` on the same data bitwise equal to `fit.loglik` (same backend).
2. All seven files present with the documented keys (including `manifest.profile` and `manifest.schema_version`); `params.safetensors` contains `__u__`; `load` rejects a manifest whose `profile` is not registered with `PersistenceError` naming it.
3. Newer `bundle_version` → `PersistenceError`; corrupted `params` (shape mismatch) → `PersistenceError` naming the parameter.
4. `decode`/`forecast` commands (E6-T02/T05) switched to this loader; their tests pass unchanged.

## Out of scope
Results persistence (already in `results.py`).

## Connections
- Depends on: E5-T03, E1-T02b
- Blocks: E8-T02, E8-T03
- Relates to: E6-T02, E6-T05
