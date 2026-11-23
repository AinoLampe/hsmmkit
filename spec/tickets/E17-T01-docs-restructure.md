# E17-T01 — Documentation restructure: core and extending guides, math pages, estimator guide, deprecations, API reference

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | NFR-16, NFR-07 | M13 | Part II §16.6; Part I §6.6, E8-T04 |

## Goal
Turn the `hsmmkit` documentation into a coherent domain-neutral library manual with a clear path for extension authors.

## Context (self-contained)
Site structure (`mkdocs.yml` nav):
- **Getting started**: install (extras table, GPU notes, WSL2), quickstart (synthetic labelled batch: `presets write core/segmentation_gaussian_ltr` → fit → decode → plot, < 30 lines), concepts (regimes, segments, durations, clocks, supports, engines).
- **Core guide** (`docs/guide/core/`): data & sources, preparation & supports, clocks, families (Appendix F table generated from the registry with support / M-step / conjugate / Δ-scaling columns), durations & topology, supervision, multiple sequences, estimators (E13-T07 page), model selection & CV, uncertainty (bootstrap), diagnostics (residuals, PPC), online inference, interoperability (sklearn, pandas/xarray, hmmlearn, ArviZ), plotting, CLI reference (generated from typer), configuration reference (generated from the schema; E8-T04 script adapted).
- **Extending** (`docs/guide/extending/`): registries and entry points, writing an emission/duration family, sources, clocks, metrics, presets; running the contract suites; scaffold; publishing; stability and deprecation policy; **case study: `regimekit`** (how the finance application plugs in — clocks, transform, source, families, origin hooks, config subclassing, bundle extensions).
- **Math** (`docs/math/`): Part I pages updated for neutral notation + Part II §14.1–14.13 pages (clocks, families, batching/padding, left-censoring, topology, EM/GEM, FFBS, VB, MCMC, supervision, online, selection, segmentation metrics), each with a symbol ↔ code-name table.
- **Tutorials**: segmentation, counts (E16-T05), link to the `regimekit` finance tutorial.
- **Reference**: API (mkdocstrings for every public module), ADRs, deprecations, changelog.
Traceability check (`scripts/check_req_coverage.py`, run in the `docs` CI job): every Part II FR/NFR ID (FR-G01…NFR-18) appears on at least one page (as an HTML comment or visible "Implements" note).
Finance-specific content (GBM paragraph, providers, ToS, strategies) must not appear in `hsmmkit` docs except the case study (neutral-name scanner extended to `docs/guide/core/**`).

## Creates
- restructured `docs/`, `mkdocs.yml`, `scripts/check_req_coverage.py`, generated pages scripts
- `tests/unit/test_e17_t01_docs_links.py` (internal link check of the built site)

## Acceptance criteria
1. `mkdocs build --strict` passes; internal link check green.
2. Every Part II FR/NFR is referenced from at least one page (script in CI).
3. Generated pages (config reference, families table, CLI reference) are up to date (CI diff check).
4. The neutral-name scanner passes on `docs/guide/core/**` and `docs/math/**`.

## Tests
Docs build, link check, requirement-coverage script, staleness checks.

## Out of scope
`regimekit` documentation (E17-T05).

## Connections
- Depends on: E16-T05, E13-T07
- Blocks: E17-T04
- Relates to: E8-T04, E10-T01 (scanner)
