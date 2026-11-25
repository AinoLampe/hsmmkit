# ADR-0014: Two repositories: `hsmmkit` library and `regimekit` application (decouple, then extract)

## Context

Part I (0.1.0, 0.2.0) built `regimekit` as one package whose core is already domain-neutral (ADR-0012) but
still carries finance names and finance defaults (prices, log-returns, the `trading` clock, providers,
strategies). Part II (concept `extensions.md` v1.2) turns the core into a general-purpose hidden semi-Markov
library and keeps the finance model as an application of it. The finance golden values of v0.2.0 are the
safety net for that change.

## Decision

Verbatim from concept v1.2:

- **D12 Library/application split.** `hsmmkit` is domain-neutral: observations `y_t ∈ 𝒴^M`, grid `Δ_t`,
  covariates. Everything price-, return-, provider-, trading-calendar- or strategy-specific lives in the
  application package `regimekit`, which depends on `hsmmkit`. The Part I model is a `regimekit`
  configuration of `hsmmkit`.
- **D22 Two distributions, two repositories.** `hsmmkit` (extras `plot`, `sklearn`, `bayes`, `xarray`, `gpu`)
  and `regimekit` (depends on `hsmmkit`; extra `providers`). `hsmmkit` never imports or depends on
  `regimekit`.
- **D23 Decouple, then extract.** (1) Inside the `RegimeDetectionAndForecast` repository all finance-specific
  code moves behind registries and extension points until the core packages contain no finance names (guarded
  by tests: the neutral-name scanner, the finance-import rule and the `core-only` job). (2) The core is
  extracted with git history into the new `hsmmkit` repository, renamed, and `RegimeDetectionAndForecast`
  (package `regimekit`) switches to depending on it. Golden outputs of the finance model stay bitwise
  identical through both phases (`tests/golden/v0_2_0/`).

The repository `RegimeDetectionAndForecast` keeps the package, distribution and CLI `regimekit`; the new
repository is `hsmmkit`.

## Consequences

- Two release trains. `regimekit` pins `hsmmkit` to a minor range; `hsmmkit` runs a nightly `downstream` job
  against `regimekit`.
- The extraction preserves history (`git filter-repo`) and applies one rename commit.
- Until the extraction, the allow-lists of E10-T01 shrink to empty (E10-T07b) and `core-only` becomes a
  required job.
- ADR-0012 (core/profile split inside one package) is superseded when this ADR is accepted.

## Status

accepted (E10-T08: `hsmmkit` extracted with history, 0.1.0)
