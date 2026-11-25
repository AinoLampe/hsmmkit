# Deprecations

Deprecations are announced with a `DeprecationWarning` that names the removal version and are listed here
(policy: ADR-0025, from 0.3.0).

| Deprecated | Since | Removed in | Replacement |
|---|---|---|---|
| extra `hsmmkit[mcmc]` | 0.1.0 | 0.5.0 | `hsmmkit[bayes]` (the same packages: `numpyro`, `arviz`). Installing the alias gives no runtime signal. |
