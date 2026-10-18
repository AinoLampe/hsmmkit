# E4-T07 — Benchmarks, GPU markers, optional `associative_scan` smoothing

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | NFR-02, NFR-06 | M2/M3 | §6.3 item 4, §6.5 benchmark job, §9 #6 |

## Goal
Record the performance guardrails as `pytest-benchmark` tests with a committed baseline, make GPU tests skip cleanly, and add the parallel-scan smoothing option behind a flag.

## Context (self-contained)
- Benchmark instance (NFR-02): `T=5000, N=4, NB r=5, order 1, 2 covariates`, GBM. Targets: jitted `forward_backward` ≤ 100 ms; explicit-duration FB with `D_max=250` ≤ 2 s; (fit ≤ 60 s benchmark added by E5-T02). Warm-up compile excluded from timing (`benchmark.pedantic` with `setup`).
- `benchmarks/baseline.json` filled from a CI run; `scripts/compare_benchmarks.py` (E0-T03) fails the nightly benchmark job on > 30 % regression.
- GPU: fixture `gpu_device` that skips when `jax.devices("gpu")` is empty; the golden tests get a `@pytest.mark.gpu` variant that runs on GPU and asserts agreement with CPU ≤ 1e-8.
- `associative_scan`: implement forward-backward smoothing via `jax.lax.associative_scan` over log-space matrix products (`(S,S)` per step, so memory `T·S²` — only enabled when `T·S² ≤ 5e7`); flag `engine.parallel_scan: bool = False` in `EstimationSection` (config) and engine kwarg; results must equal the sequential scan ≤ 1e-10. Off by default; benchmark on GPU if available.

## Creates
- `tests/benchmark/test_e4_t07_*.py`, `benchmarks/baseline.json` (populated), `tests/conftest.py` GPU fixture
- `engine/parallel.py` (`associative_scan` variant), config flag

## Acceptance criteria
1. Benchmarks run with `-m benchmark`, produce JSON, and the comparison script passes against the committed baseline.
2. GPU tests are skipped (not failed) on CPU-only runners.
3. `associative_scan` variant equals `scan` variant ≤ 1e-10 on random instances; it is not used unless the flag is set.

## Out of scope
Optimization of the engines beyond the guardrails.

## Connections
- Depends on: E4-T03b, E4-T06b
- Blocks: —
- Relates to: E0-T03, E5-T02 (fit benchmark)
