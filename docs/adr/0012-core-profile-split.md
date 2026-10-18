# ADR-0012: Core/profile split

## Context

regimekit is conceived as a general-purpose regime-detection and forecasting framework (Part II), but Part I
ships only the finance profile. Building the core as if finance were the only ever consumer would bake
finance-specific concepts (prices, returns, tickers, trading calendars) into packages that should stay
domain-neutral, making a later generalization a breaking rewrite rather than an additive one.

## Decision

`regimekit` core (`config, data, model, statespace, engine, estimation, inference, evaluation, persistence`)
is domain-neutral: it speaks in terms of observations, a time grid, and covariates, never prices or returns
specifically. Everything price-, return-, provider- or trading-strategy-specific lives in
`regimekit.finance`, which sits on top of the core and may import any core package. No core package may
import `regimekit.finance`; only `cli/` and `project.py` may (the profile import rule, overview §4.3, enforced
by an AST test, E1-T01). `config.profile` selects the active profile; `finance` is the only executable profile
in 0.1.0. regimekit ships as one distribution; profile-specific extras are added in later releases as the
non-finance profiles are built out.

## Consequences

- `PreparedDataset.observations` (not `.returns`), `data/prepare.py`'s transforms, and the model layer never
  assume "observations are log-returns" — that interpretation lives entirely in `regimekit.finance`.
- Adding a second profile later (e.g. a non-financial time-series domain) is additive: a new subpackage beside
  `finance/`, no changes to `config`, `data`, `model`, `statespace`, `engine`, `estimation`, or `inference`.
- The AST-based layer/profile test (E1-T01) is load-bearing infrastructure: it must never be weakened or
  special-cased (CLAUDE.md §4 invariant 2).

## Status

accepted
