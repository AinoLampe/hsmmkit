# E5-T05b — `regimekit selftest` (fast recovery smoke test)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-C02, NFR-04 | M3 | §5.9, §6.4.4 |

## Goal
Give users a < 2-minute self-check that the installed package estimates correctly on their machine/backend.

## Context (self-contained)
- `regimekit selftest [--device] [--seed] [--quick/--full]`: runs scenario 1 of E5-T05a with `T=1500`, `restarts=2`, 1 seed (quick, default); `--full` runs all six scenarios with 1 seed. Uses the harness from `tests/recovery/harness.py` — move the harness to `src/regimekit/estimation/recovery.py` (pure functions, no pytest) and have the test suite import from there.
- Prints a table (truth / estimate / tolerance / within) and backend info (`jax.__version__`, device, x64). **Owner decision 2026-10-18:** at `T=1500` (~15 regime visits) the Appendix D recovery tolerances are not attainable, so the table is informational and the exit code is 0 only if every fitted value is finite and the determinism check passes, else 4 (`NumericalError`). Recovery tolerances are gated by the E5-T05a suite at its attainable size.
- Also logs a determinism check: fits twice and reports whether `u` is bitwise identical.

## Creates
- `src/regimekit/estimation/recovery.py` (harness moved), `selftest` command in `cli/app.py`
- `tests/integration/test_e5_t05b_selftest.py`

## Acceptance criteria
1. `regimekit selftest` completes in < 2 min on the CI runner and exits 0 (statistical rows do not gate).
2. With a monkeypatched non-finite fit (or a non-reproducible second fit) it exits 4 and prints the failing rows.
3. Output includes the backend line and the determinism result.

## Out of scope
—

## Connections
- Depends on: E5-T05a
- Blocks: E8-T02 (command parity)
- Relates to: §1.4 criterion 1
