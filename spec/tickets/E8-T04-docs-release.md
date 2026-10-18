# E8-T04 — Documentation completion, changelog, release 0.1.0

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | NFR-07, NFR-10, §1.4 | M6 | §6.5 Release, §6.6 |

## Goal
Finish the user guide, math pages, config reference and provider ToS page; cut and publish `v0.1.0`.

## Context (self-contained)
- `docs/guide`: install (CPU default, `gpu` extra with `[tool.uv.sources]` notes, WSL2), quickstart (CLI + Python), config reference **generated from the pydantic schema** (script `scripts/gen_config_reference.py` → Markdown table per section with types/defaults; run in the `docs` nox session and checked for staleness in CI), data providers & ToS (per adapter `tos_note`), forecasting, backtesting, outputs schema, validation (recovery suite), troubleshooting (`NumericalError` cases).
- `docs/math`: one page per component (time conventions, emissions, durations, transitions, higher order, parameters, engine expanded, engine explicit, estimation, forecasting, model selection) — complete the stubs created by earlier tickets, formulas rendered with MathJax, symbol ↔ code name table on each page.
- `docs/api`: mkdocstrings nav for every public module.
- Changelog via `towncrier` fragments (`changelog.d/`), `python-semantic-release` config (conventional commits, `0.y.z`), release workflow `.github/workflows/release.yml` (on tag: `uv build`, publish artifacts to GitHub Release; PyPI publish optional/manual).
- Release checklist: §1.4 criteria (recovery suite green -- the latest weekly full run, with its date/commit and the per-category outcomes from `runs/recovery/*.json` recorded; e2e CLI flow reproducible, coverage ≥ 90 %, CI green, Appendix D checks green) recorded in `docs/releases/0.1.0.md`.

## Creates
- Docs pages above, `scripts/gen_config_reference.py`, `changelog.d/`, `towncrier.toml`, semantic-release config, release workflow, `docs/releases/0.1.0.md`, tag `v0.1.0`

## Acceptance criteria
1. `mkdocs build --strict` passes with no missing pages/links; config reference matches the schema (CI diff check).
2. `git tag v0.1.0` triggers the release workflow producing sdist + wheel artifacts.
3. `docs/releases/0.1.0.md` lists all four §1.4 success criteria with links to the CI runs proving them.

## Out of scope
Extension spikes (E9).

## Connections
- Depends on: E8-T03, E5-T05a, E4-T07
- Blocks: —
- Relates to: E0-T04, ADR on Zensical migration (future)
