# E0-T04 — Docs skeleton (mkdocs) and ADR-0001…0010, 0012, 0013

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | NFR-07 | M0 | §6.6, §6.7, §6.8, §6.9 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendment 9: ADR-0012 and ADR-0013 recorded with the baseline decisions). No behavioural change for the finance profile.

## Goal
Stand up the documentation site and record the twelve baseline architecture decisions (0001–0010 from Part I, 0012–0013 from the Part II amendments) as accepted ADRs, plus CONTRIBUTING with the ticket template and Definition of Done.

## Context (self-contained)
Docs tree: `docs/guide/` (install, quickstart, config reference, data providers & ToS, forecasting, backtesting — stubs), `docs/math/` (one page per component — stubs with the heading and the governing formula), `docs/api/` (mkdocstrings), `docs/adr/`. Theme `mkdocs-material`, MathJax enabled, `mkdocstrings[python]` with google docstrings, `strict: true`.

ADR format: `docs/adr/NNNN-title.md` with sections Context / Decision / Consequences / Status ("accepted").

| ADR | Decision |
|---|---|
| 0001 | Parameters in continuous time (annualized), inference in discrete time on the grid; no intra-interval switching |
| 0002 | Library license Apache-2.0; provider data never redistributed or committed; local caches only |
| 0003 | JAX + Equinox + Optax; no TFP dependency; in-house distributions validated against scipy |
| 0004 | One generic engine over a structured state index; higher order / NB / factorial are state-space builders |
| 0005 | MAP via L-BFGS (autodiff) is the primary estimator; EM optional (v1.x) |
| 0006 | Shifted negative-binomial semantics for the phase expansion: minimum sojourn `r_i` steps |
| 0007 | Explicit-duration engine requires a regular grid (`max Δ / min Δ ≤ 1.5`, warns otherwise) |
| 0008 | Approximate-inference plug-in interface (`InferenceEngine.approx()`) reserved for factorial models (finalized in E9-T02) |
| 0009 | pydantic v2 + YAML with `extra="forbid"` for configuration (not Hydra) |
| 0010 | `typer` CLI; `pandas` in the data layer; `pyarrow` parquet |
| 0011 | *(reserved — written by E8-T01: safetensors bundle format)* |
| 0012 | Core/profile split: `regimekit` core is domain-neutral (observations, grid, covariates); everything price-, return-, provider- or strategy-specific lives in `regimekit.finance`; `config.profile` selects the profile (`finance` is the only executable profile in 0.1.0); one distribution, extras added per profile in later releases; profile import rule (overview §4.3) |
| 0013 | Three clocks: `trading`, `calendar` (Part I) and `steps` (`Δ_t ≡ 1`, parameters per step, no annualization); every family must work under every clock; durations reported in clock units (`mean_sojourn`) |

CONTRIBUTING.md contains: ticket template (overview §3), DoD (overview §4.8), branching/PR naming, TDD scope, tolerance table pointer.

## Creates
- `mkdocs.yml`, `docs/index.md`, stubs listed above, `docs/adr/0001…0010-*.md`, `docs/adr/0012-core-profile-split.md`, `docs/adr/0013-steps-clock.md`, `CONTRIBUTING.md`

## Acceptance criteria
1. `mkdocs build --strict` passes.
2. All twelve ADRs (0001–0010, 0012, 0013) exist with Status "accepted" and non-empty Consequences; 0011 is absent until E8-T01.
3. `nox -s docs` is wired to the build.

## Tests
- `tests/unit/test_e0_t04_adrs.py`: the twelve ADR files present, each with the four sections; the test is written so that E8-T01 adding `0011` does not break it (assert a set of required numbers, not a count).

## Out of scope
Real guide/math content (written per ticket; completed in E8-T04).

## Connections
- Depends on: E0-T01
- Blocks: — (E8-T04 completes)
- Relates to: every ticket that adds an ADR
