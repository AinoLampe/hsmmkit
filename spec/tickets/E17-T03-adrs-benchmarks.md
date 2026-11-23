# E17-T03 — ADRs 0014–0021 and 0024–0027 accepted, deprecation policy in force (ADR-0025), benchmark baselines refreshed

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | NFR-14, NFR-12, NFR-02 | M13 | Part II §16.7, §16.8, §19 #19 |

## Goal
Close the architectural record for Part II, put the stability policy for plug-in authors into effect, and freeze performance baselines for the batch and EM paths.

## Context (self-contained)
- **ADR audit**: every ADR 0014–0021 and 0024–0027 exists in `docs/adr/` with status *accepted* (0014/0015 also mirrored in `RegimeDetectionAndForecast`; 0022/0023 are the existing E9 ADRs and are not touched), sections Context / Decision / Consequences / Status, and a "Implemented by" line listing tickets. Superseded/amended markers on 0012/0013 present. Test `tests/unit/test_e17_t03_adrs.py` asserts the set (extend the E0-T04 test: required numbers, not a count).
- **ADR-0025 deprecation policy** (accepted here): public API = everything exported from `hsmmkit.__all__`, `hsmmkit.testing`, the protocols (`EmissionFamily`, `DurationFamily`, `Source`, `Clock`, `Metric`, `Splitter`, `Estimator`, `OnlineFilter`, `ObservationTransform`, `OriginHook`), `PreparedDataset`, `SequenceBatch`, config schema keys, CLI commands/options, bundle format; semver from 0.3.0 (0.x minor = breaking allowed only with a two-minor deprecation cycle: deprecate in 0.y, remove in 0.(y+2)); `DeprecationWarning` naming the removal version; `docs/guide/deprecations.md` table (item, deprecated in, removal in, replacement); test helper `hsmmkit.utils.deprecation.deprecated(since, remove_in, alternative)` and a test that no deprecation past its removal version remains in the code base (compares with `__version__`). The `mcmc` extra alias is removed in 0.5.0 per its entry.
- **Benchmarks**: refresh `benchmarks/baseline.json` on the reference runner for: Part I NFR-02 set (now with `gaussian`), NFR-12 batched forward-backward and EM iteration, online `update`, fixed-lag smoothing at `L = 20`, explicit engine with duration-dependent jumps; record runner hardware in the JSON; the nightly benchmark job compares against it (> 30 % regression fails the benchmark job only).

## Creates
- ADR files/status updates, `docs/adr/0025-deprecation-policy.md`, `src/hsmmkit/utils/deprecation.py`, `docs/guide/deprecations.md` (complete)
- `benchmarks/baseline.json` (refreshed), `tests/unit/test_e17_t03_{adrs,deprecations}.py`

## Acceptance criteria
1. ADRs 0014–0021 and 0024–0027 present with status accepted; test passes.
2. Deprecation test passes; `mcmc` extra removed from `pyproject.toml` for 0.5.0 with changelog entry.
3. Baseline JSON committed with all listed benchmarks; nightly comparison green.

## Tests
ADR and deprecation tests; benchmark job.

## Out of scope
Tagging the release (E17-T04).

## Connections
- Depends on: E13-T07, E13-T09, E14-T02, E14-T04, E14-T05, E14-T06, E15-T04, E16-T01, E16-T03, E16-T04, E16-T05
- Blocks: E17-T04
- Relates to: E0-T04 (ADR test), E10-T11 (`mcmc` alias)
