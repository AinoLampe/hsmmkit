# E16-T05 — Presets completion and core tutorials (segmentation, counts) executed in CI

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-G06, NFR-16 | M12 | Part II §11.3 (target domains), §15.6, Appendix H; E10-T10 (preset registry) |

## Goal
Ship the remaining core presets with synthetic fixtures and two end-to-end tutorials that show `hsmmkit` without any finance concept.

## Context (self-contained)
Presets (registry kind `preset`, YAML under `src/hsmmkit/presets/core/`, fixture generators in `presets/fixtures.py`, shared with E14-T06):
| Preset | Model | Fixture |
|---|---|---|
| `core/segmentation_gaussian_ltr` (exists, E10-T10) | Appendix H: 4 states, left-to-right, explicit per-state durations, full-cov Gaussian, labels, GEM, online lag 20 | 50 labelled sequences (E10-T10 generator) |
| `core/counts_poisson_hazard` | 3 states (normal / degraded / failed-absorbing), Poisson counts with exposure under `calendar`, gamma durations, topology `matrix` with absorbing last state, MAP | 30 sequences of daily counts with irregular exposure |
| `core/msar2_hamilton` | 2-state MS-AR(2) mean-adjusted (Hamilton 1989 magnitudes), geometric durations, `steps` | single series `T = 1000` |
| `core/two_regime_gaussian` (exists) | Appendix E in Gaussian form | single series |
Each preset validates against the schema, runs under `hsmmkit validate-config`, and has a smoke test that fits it on its fixture (reduced restarts) — all gates must now be open (assert `gates.GATES` is empty for these presets).
Tutorials (`docs/tutorials/`, Markdown notebooks via `mkdocs-jupyter` or `.py` percent scripts executed by `nox -s tutorials`; executed in CI nightly and on docs PRs):
1. **Segmentation** — `presets write core/segmentation_gaussian_ltr` → fit (GEM, labels) → decode with FFBS bands → segmentation metrics (boundary F1, covering, ARI) → sequence CV → online stream on a held-out sequence with alerts → plots.
2. **Counts / reliability** — Poisson with exposure → absorbing failed state → expected remaining duration → pseudo-residuals (mid-distribution) → bootstrap SE.
Outputs are small (figures as PNG in the built docs only, not committed).

## Creates
- `src/hsmmkit/presets/core/{counts_poisson_hazard,msar2_hamilton}.yaml`, generators, `docs/tutorials/{segmentation,counts}.py`, `noxfile.py` (`tutorials`)
- `tests/integration/test_e16_t05_presets.py`

## Acceptance criteria
1. `hsmmkit presets list` shows all four core presets; each validates, has no open gate, and its smoke fit runs.
2. Both tutorials run in CI from presets in < 10 min total (CPU) and their key numbers (e.g. boundary F1 ≥ 0.8 at τ = 5 on the segmentation fixture) are asserted in the tutorial code.
3. The segmentation tutorial exercises GEM, labels, FFBS, segmentation metrics, CV, online alerts and plotting.

## Tests
Preset smoke tests; tutorial execution.

## Out of scope
Finance presets (E16-T06, `regimekit`); finance tutorial (E17-T05).

## Connections
- Depends on: E10-T10, E11-T05, E12-T05, E12-T06, E13-T03b, E13-T04, E14-T01, E14-T03, E14-T04, E14-T05, E15-T03, E16-T03
- Blocks: E17-T01, E17-T03
- Relates to: E14-T06 (generators), E15-T03, E16-T03
