# E12-T01b — Batched engines (`vmap`) for both engines, padding semantics, per-sequence results

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-SQ01, FR-SQ03, FR-SQ04, NFR-12 | M9 | Part II §14.3, §16.3 items 9–11; Part I §4.7.2–4.7.4, E4-T03a/b, E4-T06a/b |

## Goal
Run forward-backward, log-likelihood and Viterbi over a `PaddedBatch` with one compiled function per bucket, returning per-sequence results identical to single-sequence runs.

## Context (self-contained)
`engine/batch.py`:
```python
def batched_loglik(engine, params, spec, space, pb: PaddedBatch) -> Array            # (B,) ; total = sum
def batched_forward_backward(engine, params, spec, space, pb) -> BatchedPosterior     # arrays (B, T_max, S)
def batched_viterbi(engine, params, spec, space, pb) -> Array                        # (B, T_max) expanded states (−1 on padding)
```
- Implementation: refactor each engine's jitted core into a per-sequence function taking `(params, arrays…, valid)` and `jax.vmap` it over the batch axis (`in_axes=(None, 0, …)`); `spec`/`space` static. Single-sequence public APIs call the batched core with `B = 1` (Part II §16.2: no single-sequence fast path) — Part I golden tests must stay within Appendix D.
- **Expanded engine padding**: inside the scan, `ℓ_t(s) ← where(valid_t, ℓ_t(s), 0)` and `log A_t ← where(valid_t, log A_t, log I)` (`log I` = 0 on the diagonal, `−inf` elsewhere). Then `α` is invariant on padding and `logsumexp α_{T_max} = ℓ_b`; `β` from `T_max` backwards is 0 on padding.
- **Explicit engine padding**: the right-censored final segment must end at `T_b`, not `T_max`. Implement by evaluating the censored loglik terms at index `T_b − 1` (gather) and setting emission cumulative sums constant after `T_b` (`L[t] = L[T_b]` for `t ≥ T_b`); segments that would extend into padding get `−inf`. The backward pass starts at `T_b` per sequence (masking `β` initialization by `valid`). Chunked scan (E4-T06b) applies per sequence.
- `BatchedPosterior` with `unpad()` → tuple of Part I `Posterior` objects (original lengths); `decode` on a `SequenceBatch` returns a list of `PosteriorResult` (segments per sequence, Viterbi `−1` padding stripped).
- Initial distribution per sequence: `log_rho[group_index[b]]` (groups arrive in E12-T02; here a single shared `ρ`).
- Recompilation: compile count per bucket = 1 (assert via `jax.jit(...)._cache_size()` or a compile counter wrapper).
- Benchmark (NFR-12): `B = 64, T = 1000, N = 4, S = 12, order 1` batched forward-backward ≤ 300 ms (8-core CPU), recorded in the benchmark job.

## Creates
- `src/hsmmkit/engine/batch.py`, refactors in `engine/expanded_hmm.py`, `engine/explicit_duration.py`, `engine/viterbi.py`, `inference/decode.py` (batch path)
- `tests/golden/test_e12_t01b_batched.py`, `tests/property/test_e12_t01b_padding_invariance.py`, `tests/benchmark/test_e12_t01b_bench.py`

## Acceptance criteria
1. Batched loglik equals the sum of single-sequence logliks ≤ 1e-10 relative (both engines, random `N ∈ [2,4]`, lengths 5–200, order 1 and 2, covariates).
2. Padding invariance: posteriors for `t ≤ T_b` equal the unpadded run ≤ 1e-12 max-abs, for both engines (property test over random padding amounts).
3. Compile count ≤ number of distinct buckets in a run with 3 buckets and varying `B`.
4. Part I golden/reference tests (Appendix D) still pass through the `B = 1` path.
5. NFR-12 forward-backward benchmark recorded and within budget.

## Tests
Golden, property, benchmark.

## Out of scope
Batch estimation, per-group initial distributions (E12-T02); constraints (E12-T06).

## Connections
- Depends on: E12-T01a
- Blocks: E12-T02, E12-T06, E13-T01a, E13-T01b, E13-T04
- Relates to: E4-T03a/b, E4-T06a/b, E7-T02b (padding to multiples of 256 is now subsumed by buckets)
