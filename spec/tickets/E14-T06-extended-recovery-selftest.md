# E14-T06 — Extended recovery scenarios and `hsmmkit selftest`

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | NFR-04, G7–G10 | M11 | Part II §16.4 (recovery extended), Appendix G; Part I §6.4.4, E5-T05a/b |

## Goal
Extend the nightly parameter-recovery suite to the new families, structures and estimators, and make `hsmmkit selftest` a fast domain-neutral smoke test.

## Context (self-contained)
Scenarios (`tests/recovery/`, nightly `-m slow`, 5 seeds each, pass if ≥ 4/5; tolerances Appendix D + G: covariance entries ≤ 15 % rel, AR coefficients ≤ 2 SE, mean sojourn ≤ 25 %, jump probabilities ≤ 0.15 abs):
| # | Scenario | Data | Estimator |
|---|---|---|---|
| R7 | Gaussian full-cov 3-state, batch of 50 × `T = 300`, NB `r = 2` | `M = 3` | MAP and EM |
| R8 | Poisson 2-state with hazard-shaped (gamma) durations, exposure `Δ_t` irregular (calendar) | counts | GEM |
| R9 | MS-AR(2) Hamilton mean-adjusted, 2 states | `M = 1` | MAP |
| R10 | Left-to-right 4-state topology with 30 % labels, explicit per-state durations | `M = 2` | GEM + `init: labels` |
| R11 | VB on 4 configured / 3 true states | Gaussian diag | VB (pruning: exactly one state pruned) |
Part I scenarios R1–R6 keep running (re-expressed with `gaussian` in E10-T06). Synthetic generators live in `hsmmkit.presets.fixtures` (reused by presets/tutorials).
`hsmmkit selftest [--scenario R1|R7|…] [--seeds k]`: default runs a reduced R1 (Gaussian with GBM options, 1 seed, `T = 2000`) plus a reduced R7 (`B = 10`, `T = 200`) and prints a pass/fail table with the tolerance checks; total < 2 min on an 8-core CPU; exit code 0/4. `regimekit selftest` continues to run its finance scenario (unchanged in `regimekit`).

## Creates
- `tests/recovery/test_e14_t06_r7_r11.py`, generators in `src/hsmmkit/presets/fixtures.py`, `cli selftest` update
- `docs/guide/core/validation.md` (scenario table, how to read selftest output)

## Acceptance criteria
1. R7–R11 pass ≥ 4/5 seeds nightly; R1–R6 still pass.
2. `hsmmkit selftest` completes in < 2 min and exits 0 on a healthy install; a deliberately broken emission (test fixture) makes it exit 4 with the failing check named.
3. Generators are deterministic per seed and shared with presets (E16-T05).

## Tests
The recovery suite itself; selftest integration test (timed).

## Out of scope
MCMC recovery (covered by E13-T06a/b fixtures).

## Connections
- Depends on: E13-T02b, E13-T03b, E13-T05b, E12-T06, E11-T05
- Blocks: E17-T03
- Relates to: E5-T05a/b, E16-T05
