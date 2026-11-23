# E17-T05 — `regimekit` on `hsmmkit` 0.5: finance documentation, finance tutorial from presets, release `regimekit` 0.5.0

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast | S | NFR-16, NFR-18, NFR-08 | M13 | Part II §12.3 (0.5.0 row), §16.6; Part I §6.6, E8-T03, E8-T04 |

## Goal
Bring the finance application up to the general `hsmmkit` release, with its own documentation and an executable tutorial.

## Context (self-contained)
- **Dependency**: `hsmmkit>=0.5,<0.6` from PyPI; remove any git/TestPyPI sources; CI matrix installs from PyPI; nightly still tests against `hsmmkit@main`.
- **Docs** (`RegimeDetectionAndForecast` site): what `regimekit` adds to `hsmmkit` (trading clock, log-returns transform, provider source + cache + ToS notes per provider (NFR-08), GBM/OU/Merton families, strategies and finance metrics, finance presets, `FinanceConfig` reference generated from the schema, migration guide from `regimekit` 0.1.x/0.2.x configs/bundles (schema 1 → 2, bundle v1 → v2), MS-GARCH spike status); links to `hsmmkit` docs for everything generic; the E8-T03 end-to-end example updated to v2 configs.
- **Tutorial** (`docs/tutorials/dax_regimes.py`, executed in CI offline): `hsmmkit presets write finance/dax_3state_gbm_nb` (with `regimekit` installed) → `regimekit fetch --offline` on shipped fixtures → fit → decode → forecast fan chart (`hsmmkit.plot`) → walk-forward backtest with `regime_long_flat` → `compare --grid` over `r` → online stream over the last 200 days with an alert on the crisis regime → two-index full-covariance variant (E11-T09).
- **Release** `v0.5.0` (PyPI publishing for `regimekit` is optional; GitHub release required), changelog since 0.4.0.

## Creates
- `regimekit` `pyproject.toml` pin, `docs/` restructure, `docs/tutorials/dax_regimes.py`, `docs/migration-0.2-to-0.5.md`, `CHANGELOG`, tag `v0.5.0`

## Acceptance criteria
1. Tutorial runs in CI offline from presets within 10 min; key outputs asserted (e.g. crisis regime occupancy within the fixture's documented range).
2. `pip install regimekit==0.5.0` (or from the GitHub release wheel) together with `hsmmkit==0.5.0` from PyPI works in a fresh environment; v0.2.0 golden values still bitwise.
3. Migration guide covers every v1 → v2 config rule (E10-T02b) and the bundle compatibility path.

## Tests
Tutorial execution; golden regression; install smoke test.

## Out of scope
New finance models.

## Connections
- Depends on: E17-T04, E16-T06, E11-T09
- Blocks: —
- Relates to: E8-T03, E10-T09
