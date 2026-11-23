# E10-T09 — `regimekit` on top of `hsmmkit`: delete the core copy, flatten the finance package, release `regimekit` 0.3.0

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast | M | FR-G07, FR-G08, FR-G09, NFR-18 | M7b | Part II §15.1 (regimekit tree), §15.4 (schema v2), §15.8, §19 #21 |

## Goal
Turn the `RegimeDetectionAndForecast` repository (package `regimekit`) into a pure finance application that depends on `hsmmkit`, with unchanged numbers, configs and bundles for its users.

## Context (self-contained)
1. **Dependency**: `pyproject.toml` `dependencies = ["hsmmkit>=0.1,<0.2", "pandas-datareader>=0.10", "fredapi>=0.5"]`, extra `providers = ["yfinance>=1.7", "curl_cffi>=0.15"]`; version `0.2.0`. CI installs `hsmmkit` from the pin; a nightly job installs `hsmmkit@main`.
2. **Delete** every core package now provided by `hsmmkit` (`config` core parts, `data` core parts, `model`, `statespace`, `engine`, `estimation`, `inference`, `evaluation` core, `persistence`, `utils`, `testing`, `plugins.py`, core `cli`/`project`) and their tests (they live in `hsmmkit`).
3. **Flatten** `regimekit/finance/*` → `regimekit/*` (`regimekit.providers`, `regimekit.cache`, `regimekit.clocks`, `regimekit.transforms`, `regimekit.quality`, `regimekit.emission.gbm`, `regimekit.strategies`, `regimekit.metrics`, `regimekit.config`, `regimekit.project`, `regimekit.cli`). The MS-GARCH spike (E9-T03) moves to `regimekit.emission.msgarch` (still experimental). No `regimekit.finance` compatibility shim (Part II §19 #21).
4. **Imports** rewritten to `hsmmkit.*`; errors: `ProviderError(hsmmkit.DataError)` defined in `regimekit.errors`; `register_exit_code(ProviderError, 5)`.
5. **Registration**: `regimekit/__init__.py` imports `regimekit._register` which registers (in-process, provenance `regimekit`) the `trading` clock, `log_returns` transform, `provider` source, `gbm` and `rollvol_kmeans` aliases, finance metrics, strategy hook, bundle extension hooks. (Entry points replace this in E16-T06.) `regimekit.Project` = `FinanceProject`.
6. **Configs**: `FinanceConfig` (schema v2, v1 migration) unchanged from E10-T02b; `load_config` defaults to `FinanceConfig` in the `regimekit` CLI.
7. **Bundles**: `ModelStore(suffix=".rkmodel")`; v1 bundles (0.1.x/0.2.x) load through `hsmmkit`'s v1 compatibility path plus `regimekit`'s extension hook.
8. **Tests kept here**: finance tests, provider cassettes, v0.2.0 golden values (`tests/golden/v0_2_0/`), E8-T03 end-to-end example and tutorial, plus `hsmmkit.testing` contract suites parametrized over `regimekit` components (clock `trading`, source `provider`, family `gbm`, metrics) — this is the first external use of the contract suites.
9. **Docs**: site becomes the finance guide (Part I guide content, provider ToS page, Appendix A tutorial); links to `hsmmkit` docs for core topics.

## Creates
- `RegimeDetectionAndForecast` repository restructured as above; `src/regimekit/{_register,errors}.py`
- `tests/integration/test_e10_t09_on_hsmmkit.py`, `tests/contract/test_e10_t09_contracts.py`
- `CHANGELOG` 0.2.0 ("library extracted to hsmmkit")

## Acceptance criteria
1. v0.2.0 golden values reproduced bitwise (prepared data, fit `uparams`/loglik, posteriors, forecast samples, backtest metrics and strategy frame) with `hsmmkit` 0.1.0 installed from the release artifact (`hsmmkit` 0.1.0 is the extraction release, not `regimekit` 0.1.0).
2. The v0.2.0 bundle fixture loads and reproduces its loglik bitwise; Appendix A v1 YAML runs `validate-config → fetch (offline) → fit → decode → forecast → backtest` through the `regimekit` CLI.
3. the `regimekit` package contains no module that `hsmmkit` also provides (test compares module lists); `grep -rn "from regimekit.finance" src/` is empty.
4. `trading`, `provider`, `gbm`, finance metrics pass the `hsmmkit.testing` contract suites.
5. `hsmmkit`'s `downstream` job is green against this branch; tag `v0.3.0` published (`v0.2.0` already exists).

## Tests
Golden bitwise suite, bundle compatibility, CLI flow, contract suites, module-list test.

## Out of scope
Entry-point registration (E16-T06); OU/Merton (E11-T07).

## Connections
- Depends on: E10-T08
- Blocks: E11-T07, E11-T09
- Relates to: E10-T01 (golden values), E10-T07a (bundle v2), E16-T06
