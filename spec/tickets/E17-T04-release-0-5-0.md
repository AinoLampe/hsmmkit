# E17-T04 — Release `hsmmkit` 0.5.0 with changelog and roadmap page

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | NFR-10, §12.3 (0.5.0 criteria) | M13 | Part II §12.3, §17.1 |

## Goal
Publish the general-purpose release of `hsmmkit` and record what comes next.

## Context (self-contained)
- **Release checklist** `docs/releases/0.5.0.md` linking CI evidence for every §12.3 criterion of releases 0.2.0–0.5.0 (core config runs on a batch of 50 labelled sequences; family contract suite; NFR-12; EM/VB/MCMC parity table; supervised recovery; segmentation metrics vs reference; online bitwise equality; sklearn checks; plots; `regimekit` discovered via entry points; docs restructured; PyPI + conda-forge; ADRs accepted; deprecation policy).
- **Changelog** from towncrier fragments since `hsmmkit` 0.1.0, grouped by release 0.2.0–0.5.0 (intermediate tags 0.2.0/0.3.0/0.4.0 were cut at milestone ends per overview §5.9; this release summarizes).
- **Roadmap page** `docs/roadmap.md`: factorial/coupled inference (E9-T02/ADR-0008), MS-GARCH (E9-T03, in `regimekit`), online EM (E15-T05, v1.x), HDP-HSMM (ADR-0027 outcome), `calendar` clock units (Part II §19 #20), polars source plug-in, explicit-engine online filtering.
- **Publish**: tag `v0.5.0` → release workflow (E17-T02) publishes to PyPI; GitHub release with notes; conda-forge feedstock updated (version bump PR merged or auto-bot PR merged).

## Creates
- `docs/releases/0.5.0.md`, `CHANGELOG.md` update, `docs/roadmap.md`, tag `v0.5.0`

## Acceptance criteria
1. `v0.5.0` on PyPI and conda-forge; `pip install hsmmkit==0.5.0` and `conda install -c conda-forge hsmmkit=0.5.0` succeed in fresh environments (smoke-install job on the tag).
2. Release checklist links a green CI run for every criterion.
3. Roadmap page published.

## Tests
Release workflow and smoke installs.

## Out of scope
`regimekit` release (E17-T05).

## Connections
- Depends on: E17-T01, E17-T02, E17-T03
- Blocks: E17-T05
- Relates to: E8-T04
