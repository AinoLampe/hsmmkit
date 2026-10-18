# E0-T03 — CI workflows (PR gates, nightly recovery + benchmark, artifacts)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | NFR-02, NFR-04 | M0 | §6.5 |

## Goal
Run the nox gates in GitHub Actions on every PR and schedule the slow suites nightly.

## Context (self-contained)
Jobs:
- `lint` (ruff, mypy), `test` (Linux, Python 3.13, CPU, `pytest -n auto -m "not slow and not live and not benchmark"` with coverage), `docs` (`mkdocs build --strict`) — on PR and `main`.
- `recovery` (nightly: `nox -s recovery_smoke`, one scenario at `T=5000`), `recovery-full` (weekly, `weekly.yml`: `nox -s recovery`, the full suite at `T=20000`, uploads `runs/recovery/`), `benchmark` (nightly, `pytest -m benchmark --benchmark-json=bench.json`, compare with `benchmarks/baseline.json`, fail if any benchmark is > 30 % slower). Benchmark failure never fails the `test` job.
- `gpu`: `workflow_dispatch` only; runs on a self-hosted runner if labelled, otherwise skipped.
Artifacts: `coverage.json`, `htmlcov/`, `bench.json`. Use `uv` caching. Structured logs are JSON when `CI=1` (E0-T05).

## Creates
- `.github/workflows/pr.yml`, `.github/workflows/nightly.yml`, `.github/workflows/weekly.yml`, `.github/workflows/gpu.yml`
- `benchmarks/baseline.json` (empty list until E4-T07 fills it), `scripts/compare_benchmarks.py`

## Acceptance criteria
1. PR pipeline completes in < 10 minutes on the skeleton.
2. Nightly and weekly workflows are scheduled (cron) and can be triggered manually.
3. `scripts/compare_benchmarks.py` exits non-zero on a synthetic 31 % regression and zero on 29 %.
4. Coverage and benchmark JSON are uploaded as artifacts.

## Tests
- `tests/unit/test_e0_t03_compare_benchmarks.py`: regression threshold logic.

## Out of scope
Release workflow (E8-T04); real benchmarks (E4-T07).

## Connections
- Depends on: E0-T02
- Blocks: — (all later tickets rely on it)
- Relates to: E4-T07, E5-T05a
