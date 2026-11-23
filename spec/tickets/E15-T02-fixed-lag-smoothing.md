# E15-T02 — Fixed-lag smoothing with a ring buffer

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ON02, NFR-15 | M12 | Part II §14.11 (fixed-lag), §16.3 item 12 |

## Goal
Provide smoothed regime probabilities with a fixed delay `L` during streaming, equal to the offline smoother for all steps at least `L` before the end.

## Context (self-contained)
- `RingBuffer` (pytree): fixed-size arrays `ell (L+1, S)` (per-step emission log-terms `ℓ_u`), `inputs` (`z_pi (L+1, K_pi)`, `z_d (L+1, K_d)`, `dt (L+1,)`) — store the inputs and rebuild `log A_u` on demand (memory `O(L·(S + K))` instead of `O(L·S²)`), `log_alpha (L+1, S)`, write pointer `head`, fill count. Updated inside `update` (E15-T01) when `online.lag = L > 0`.
- `smoothed_at(state, lag=L) -> (N,)`: regime marginals at `t − L`: run the backward recursion over the buffer from `β_t = 0`: `β_u = logsumexp_{s'}[log A_{u+1}(s, s') + ℓ_{u+1}(s') + β_{u+1}(s')]` for `u = t−1 … t−L`, then `γ_{t−L} ∝ exp(α_{t−L} + β_{t−L})` collapsed with `C`. Cost `O(L S²)` per call; jitted with static `L`.
- Equality claim: `γ_{t−L}` computed this way equals the offline smoothed marginal at `t − L` **given data up to `t`** — i.e. it equals the offline smoother run on `y_{1:t}`; it equals the full-data offline smoother `γ_{t−L | T}` only approximately. NFR-15 compares against the offline smoother on the same prefix (`window(0, t+1)`) — document this precisely.
- `update(...)` returns, when `lag > 0`, `(state', filtered (N,), smoothed_lagged (N,) | NaN for t ≤ L)`.
- Batch replay helper `smoothed_series(fit, data, lag)` for evaluation (vectorized over `t`).

## Creates
- `src/hsmmkit/online/buffer.py`, edits in `online/filter.py`
- `tests/golden/test_e15_t02_fixed_lag.py`, `tests/benchmark/test_e15_t02_lag_cost.py`

## Acceptance criteria
1. For every `t ≥ L` in a 2000-step replay, `smoothed_at` equals the offline smoothed marginal at `t − L` computed on `data.window(0, t+1)` ≤ 1e-12 max-abs (Appendix G).
2. At `t = T`, `smoothed_at(lag=L)` equals the full offline smoother at `T − L` ≤ 1e-12.
3. Cost linear in `L`: per-call time at `L ∈ {10, 20, 40, 80}` fits a line with `R² ≥ 0.95` (benchmark).
4. Buffer is part of `OnlineState` save/load (round trip bitwise).

## Tests
Golden comparison against offline prefixes, benchmark, state round-trip.

## Out of scope
Alerts (E15-T03).

## Connections
- Depends on: E15-T01
- Blocks: E15-T03
- Relates to: E13-T08
