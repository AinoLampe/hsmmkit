# Contributing to regimekit

regimekit is built ticket by ticket from `spec/tickets/`, under the conventions in
`spec/tickets/00-OVERVIEW-build-order-and-conventions.md` and the harmonization decisions in
[`CLAUDE.md`](CLAUDE.md). This file summarizes the parts of that process a human contributor needs.

## Ticket template

Every ticket in `spec/tickets/` follows this shape (overview §3):

```
# <ID> — <Title>
| Size | S (≤1 d) / M (1–3 d) / L (3–5 d) |   | Requirements | FR-…, NFR-… |   | Milestone | M0…M6 |   | Concept refs | §… |
## Goal                   one sentence
## Context (self-contained)   everything needed to implement without opening the concept
## Creates                files / modules / tests / docs touched
## Acceptance criteria    testable, numbered
## Tests                  what the ticket's tests must cover (part of the ticket, never separate)
## Out of scope           explicit exclusions
## Connections            Depends on / Blocks / Relates to
```

A ticket is meant to be implementable with only its own file plus the overview open; if it isn't, that's a
spec gap worth raising.

## Definition of Done (overview §4.8)

1. Every acceptance criterion is met by a test named after the ticket (`test_<id>_...`).
2. Coverage of touched modules is not decreased; the global and per-package gates
   (`scripts/check_package_coverage.py`) pass.
3. `ruff`, `mypy --strict`, and `jaxtyped`/beartype checks (`REGIMEKIT_TYPECHECK=1`) pass.
4. Docstrings and the relevant guide/math docs are updated; an ADR is added if a decision was taken.
5. No new runtime dependency without an ADR.
6. The PR (or, for solo/scripted work, the commit body) references requirement IDs and lists deviations from
   the concept — which must be none, or an ADR plus a concept version bump.

## Branching and commit/PR naming

- Branch per ticket from `main`: `feat/<ID>-<slug>` (e.g. `feat/E4-T03a-expanded-forward`).
- Close with a squash merge onto `main`. The commit message (or PR title/description) is:

  ```
  feat(engine): E4-T03a Expanded engine: forward / loglik

  Requirements: FR-I01, NFR-01, NFR-02
  Acceptance: AC1 → test_e4_t03a_forward_matches_reference; AC2 → …; AC5 → benchmark (measured 41 ms)
  Deviations: none | ADR-00NN | CLAUDE.md §3.x
  ```
- Conventional-commit prefixes (`feat`, `fix`, `test`, `docs`, `chore`, ...); a `changelog.d/` fragment
  (`changelog.d/<ID>.<feature|bugfix|doc|removal|misc>.md`) accompanies every user-visible change.

## TDD scope

TDD is mandatory (tests written and run-to-fail before implementation) for: `model/dists`, `model/params`,
`statespace`, `engine`, `estimation/objective`, `inference/forecast` aggregation, `evaluation/metrics`, and
`data/{prepare,calendar,cache}` (overview §4.7). Other layers write tests alongside the code.

## Tolerances

Numerical tolerances (loglik/posterior agreement, gradient checks, distribution tests, parameter recovery,
simulation moments, determinism, bundle round-trips) are fixed in overview §4.7 / concept Appendix D. Never
loosen a tolerance, add retries, shrink sample sizes, or skip/xfail/delete a test to get a suite green.

## Where to look when something is ambiguous

Precedence (highest first): `CLAUDE.md` §3 harmonization decisions → overview §8 amendments → concept
(semantics: formulas, defaults, tolerances) → ticket text (signatures, file names, test design). If a ticket
and the concept genuinely disagree on semantics, the concept wins and the ticket is corrected in a separate
`docs(spec): align <ID> with concept §x.y` commit, logged in `spec/STATUS.md` → "Spec fixes".
