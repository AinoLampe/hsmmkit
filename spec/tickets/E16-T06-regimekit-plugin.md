# E16-T06 — `regimekit` as an entry-point plug-in: registration, contract suites, `finance/*` presets

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast | S | FR-PL01, FR-PL02, FR-G06, G13 | M12 | Part II §12.3 (0.4.0 criterion), §15.6, D15 |

## Goal
Make `regimekit` the first real external plug-in of `hsmmkit`: all finance components are discovered through entry points, pass the public contract suites, and the finance presets appear in `hsmmkit presets list`.

## Context (self-contained)
- Remove in-process registration (`regimekit/_register.py`, E10-T09); declare entry points in `pyproject.toml`:
```toml
[project.entry-points."hsmmkit.clock"]       trading = "regimekit.clocks:TradingClock"
[project.entry-points."hsmmkit.transform"]   log_returns = "regimekit.transforms:LogReturns"
[project.entry-points."hsmmkit.source"]      provider = "regimekit.providers.source:ProviderSource"
[project.entry-points."hsmmkit.emission"]    gbm = "regimekit.emission.gbm:GBMFamily"
                                             ou = "regimekit.emission.ou:OUFamily"
                                             merton = "regimekit.emission.merton:MertonFamily"
[project.entry-points."hsmmkit.init"]        rollvol_kmeans = "regimekit.init:rollvol_kmeans_alias"
[project.entry-points."hsmmkit.metric"]      sharpe = "…"  max_drawdown = "…"  turnover = "…"  hit_rate = "…"
[project.entry-points."hsmmkit.preset"]      "finance/dax_3state_gbm_nb" = "regimekit.presets:DAX_3STATE"
                                             "finance/two_regime_sanity" = "regimekit.presets:TWO_REGIME"
```
  (TOML shown abridged; one key per line in the real file.) Bundle extension hooks and the strategy origin hook stay explicit Python APIs (not registry kinds) and are wired by `FinanceProject`.
- Presets: `finance/dax_3state_gbm_nb` = Appendix A (v2 form) with fixture = the offline CSV/cassette data shipped in `regimekit` (no provider data redistributed — synthetic look-alike generator if licensing forbids shipping; document per Part I NFR-08); `finance/two_regime_sanity` = Appendix E with a simulated fixture.
- Contract tests: `tests/contract/` in `RegimeDetectionAndForecast` parametrizes `family_contract` (gbm, ou, merton), `clock_contract` (trading), `source_contract` (provider via CSV adapter offline), `metric_contract` (finance metrics) — run in `RegimeDetectionAndForecast` CI and in `hsmmkit`'s `downstream` job.
- Raise the `hsmmkit` pin to the 0.4 range.

## Creates
- `pyproject.toml` of `RegimeDetectionAndForecast`: entry points, `src/regimekit/presets/`, removal of `_register.py`
- `tests/contract/test_e16_t06_contracts.py`, `tests/integration/test_e16_t06_discovery.py`

## Acceptance criteria
1. In a fresh env with `hsmmkit` and `regimekit` installed, `hsmmkit plugins list` shows every finance component with distribution `regimekit`; `hsmmkit presets list` shows `finance/*`.
2. `import regimekit` registers nothing at import time (grep + runtime check); finance CLI flows and v0.2.0 golden values still pass.
3. All finance components pass the `hsmmkit.testing` contract suites.
4. `hsmmkit validate-config` of the `finance/dax_3state_gbm_nb` preset succeeds when `regimekit` is installed and fails with the registered-names message when it is not.

## Tests
Contract suites, discovery integration, golden regression.

## Out of scope
New finance features.

## Connections
- Depends on: E16-T04, E11-T07
- Blocks: E17-T05
- Relates to: E10-T09
