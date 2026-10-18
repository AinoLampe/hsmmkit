# ADR-0002: Apache-2.0 license; provider data never redistributed or committed

## Context

regimekit fetches market data from third-party providers (yfinance, stooq, FRED, CBOE, tiingo) whose terms of
service typically forbid redistributing raw data. The library itself, however, should be freely usable and
modifiable, including by commercial users.

## Decision

The regimekit source code is licensed under Apache-2.0. Provider data is never committed to the repository
and never redistributed by the library; it is fetched at runtime and cached only in a local, user-controlled
cache directory (`~/.cache/regimekit` by default). Test fixtures are synthetic or trimmed to a handful of rows
(NFR-08), never raw provider exports.

## Consequences

- Users are responsible for complying with each provider's own terms of service; `docs/guide/providers.md`
  documents this per provider.
- CI and tests never depend on committed provider data, keeping the repository small and license-clean.
- The cache (`data/cache.py`) is a pure local artifact store, never a distribution mechanism.

## Status

accepted
