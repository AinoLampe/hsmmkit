# E13-T04 — Forward-filtering backward-sampling (FFBS) for both engines; `decode --samples R`

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES03 | M10 | Part II §14.7; Part I §4.7.2–4.7.3 |

## Goal
Draw exact posterior regime paths given parameters, for segment-uncertainty bands in decoding and as the state block of blocked Gibbs.

## Context (self-contained)
`engine/ffbs.py`:
```python
def ffbs_expanded(key, params, spec, space, pb: PaddedBatch, n_samples: int) -> Array    # (n_samples, B, T_max) expanded states, −1 on padding
def ffbs_explicit(key, params, spec, space, pb, n_samples) -> Array                     # (n_samples, B, T_max) regimes (segment draws expanded to steps)
```
- **Expanded**: `s_T ~ Categorical(softmax(α_T))`; for `t = T−1 … 1`: `P(s_t = s | s_{t+1}) ∝ exp(α_t(s)) A_{t+1}(s, s_{t+1})` — reverse `lax.scan` building only the column `log A_{t+1}(·, s_{t+1})` per step (gather from the per-step builder; never the full `(S, S)` per sample: use `jax.vmap` over samples of a function that builds `A_{t+1}` once per step and gathers columns — memory `O(S²)` per step shared across samples). Padding: start at `T_b` per sequence.
- **Explicit**: sample the last segment `(j, d)` from the censored terms at `T`; then repeatedly, for a segment starting at `t0`, sample the previous segment's end `t0 − 1` with `(i, d')` ∝ `exp(α_{t0−1}(i, d') + log π_ij)`; stop at `t = 0` (respect left-censoring). Implemented as a `lax.while_loop` per sample with a fixed max of `T` iterations.
- Constraints (E12-T06) are respected automatically (masked `α`, `A`).
- Keys: `key_for(seed, "ffbs", b, sample)`; deterministic.
- **Decode integration**: `decode(..., samples=R)` adds `PosteriorResult.path_samples (R, T)` and per-step segment-boundary probability `P(boundary at t)`; CLI `hsmmkit decode --samples R` writes `path_samples.parquet` (long: sample, t, regime) and boundary probabilities in `posterior.parquet`.
- Reference: enumerable 2-state/4-step example in `tests/reference/ref_ffbs.py` (all `2^4` paths, exact posterior probabilities).

## Creates
- `src/hsmmkit/engine/ffbs.py`, `tests/reference/ref_ffbs.py`, decode/CLI extensions
- `tests/golden/test_e13_t04_ffbs.py`, `tests/property/test_e13_t04_marginals.py`

## Acceptance criteria
1. Enumerable golden test: empirical path frequencies from 200 000 draws match exact path probabilities within 3 MC SE (both engines, geometric durations so the explicit engine is comparable).
2. Path-sample regime frequencies vs smoothed marginals within 3 MC SE at every `t` (random models, both engines, with labels and topology).
3. Deterministic per key; shapes and padding (`−1`) as specified.
4. `decode --samples 100` writes the documented artifacts.

## Tests
Golden (enumeration), property (marginals), CLI integration.

## Out of scope
Gibbs parameter updates (E13-T06b).

## Connections
- Depends on: E12-T01b
- Blocks: E13-T06b, E14-T05, E16-T05
- Relates to: E6-T02 (decode), E12-T06
