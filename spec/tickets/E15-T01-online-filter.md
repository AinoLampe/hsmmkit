# E15-T01 — `OnlineFilter`: jitted single-step update, `OnlineState`, save/load, `spec_hash`

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ON01, FR-ON04, NFR-15, D19 | M12 | Part II §14.11, §15.2 (`OnlineState`), §15.3 (`OnlineFilter`), §16.3 item 12; Part I §4.7.2, E4-T03a, E8-T01 |

## Goal
Update filtered regime probabilities one observation at a time with exactly the arithmetic of the offline forward pass, and make the stream state persistable.

## Context (self-contained)
```python
@dataclass(frozen=True)
class OnlineState:                       # pytree
    log_alpha: Array                     # (S,) unnormalized forward log-probabilities at t
    log_norm: Array                      # () running log-likelihood ℓ_{1:t}
    t: int
    carry: Any                           # emission carry (identity for non-GARCH families)
    last_time: Any                       # last timestamp / step index (for dt under calendar clocks)
    buffer: RingBuffer | None            # fixed-lag smoothing (E15-T02), None here
    spec_hash: str                       # sha256 of spec.json + param names/shapes

class ExpandedOnlineFilter:              # online/filter.py; explicit engine → ConfigError("online filtering requires engine: expanded") in 0.4.0 (documented)
    def __init__(self, fit: FitResult, *, group: str | None = None): ...
    def start(self, initial: PreparedDataset | None = None) -> OnlineState: ...   # warm start: offline forward on `initial`, keep last α
    def update(self, state, y, dt, z) -> tuple[OnlineState, Array]: ...          # → (state', filtered regime probs (N,))
    def update_many(self, state, ys, dts, zs) -> tuple[OnlineState, Array]: ...   # lax.scan over update; (T', N)
```
- **Bitwise equality** with the offline forward pass (NFR-15) requires the same operation order: factor the per-step body of E4-T03a's scan into a function `forward_step(carry, xs)` used by **both** the offline scan and `update` (refactor in this ticket; Part I golden tests must remain within Appendix D and the offline results must not change bitwise — verify with stored outputs before/after).
- First observation: `α_1 = log ρ + ℓ_1` (same function path as offline initialization).
- Masked observation (`y` NaN) → `mask=False` → zero emission contribution, as offline.
- `dt`: caller passes it, or `update_from_time(state, time, y, z)` computes it with the configured clock from `last_time`.
- Covariates: the caller supplies `z` already lagged/standardized; helper `OnlineCovariatePreparer(prep_stats)` applies the stored transforms/standardization to raw covariate values and maintains the lag buffer (lag ≥ 1).
- Persistence: `save_state(state, path)` → `state.safetensors` + `state.json` (`t`, `last_time`, `spec_hash`, library version); `load_state(path, fit)` checks `spec_hash` equality (else `PersistenceError` naming both hashes).
- JIT: `update` jitted once per fit; no Python lists in the hot path.

## Creates
- `src/hsmmkit/online/{__init__,filter,state}.py`, refactor of `engine/expanded_hmm.py` (`forward_step`)
- `tests/golden/test_e15_t01_online_vs_offline.py`, `tests/unit/test_e15_t01_state_io.py`, `tests/benchmark/test_e15_t01_update.py`

## Acceptance criteria
1. Online filtered probabilities equal the offline filtered posteriors **bitwise** over a 5000-step replay (Gaussian, NB `r = 3`, transition and duration covariates, masked steps included), and `log_norm` equals the offline loglik bitwise.
2. Saving after step 2500, restoring, and continuing yields bitwise-identical results to an uninterrupted run; a state from a different fit is rejected via `spec_hash`.
3. Offline forward results unchanged bitwise by the `forward_step` refactor.
4. `update` per step ≤ 0.2 ms at `S = 12` on CPU (benchmark recorded).

## Tests
Golden replay, state round-trip, benchmark.

## Out of scope
Fixed-lag smoothing (E15-T02); alerts, forecasting (E15-T03); explicit-engine online filtering (backlog).

## Connections
- Depends on: E10-T08
- Blocks: E15-T02, E13-T08
- Relates to: E4-T03a, E8-T01
