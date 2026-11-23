# E11-T08 — Joint/product emission composition: blocks of any family, full-covariance Gaussian blocks

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-M12, FR-EM01, FR-EM07 | M8 | Part II §14.2, Appendix F (`product`); Part I §5.10, E9-T01 |

## Goal
Generalize the E9-T01 `ProductEmission` from independent per-series families to independent **blocks** of series, each block any registered family (including correlated full-covariance Gaussians and discrete families).

## Context (self-contained)
```
f_i(y_t) = Π_{b=1}^{B_e} f_i^{(b)}( y_t[cols_b] ),   blocks partition the M series
```
- Config: `model.emission: {family: product, blocks: [{family: gaussian, columns: [0, 1], covariance: full, params: …}, {family: poisson, columns: [2], params: …}]}`; `columns` are indices into `observation_columns` (or names). Every column appears exactly once (`ConfigError` otherwise). The E9-T01 form (`per_series` families) is kept as shorthand and maps to one block per series.
- Mixed supports: keep `PreparedDataset.support` (`real | count | binary | categorical`, no new literal) and add the optional field `PreparedDataset.series_support: tuple[str, ...] | None` (per-series supports; `None` = all equal to `support`). The preparer validates each block's columns against its family's support (E10-T05 rules per column). With mixed blocks, `support` holds the first non-real support and `series_support` is authoritative. Document this in `data/contracts.py`.
- `log_prob`: sum over blocks of `block.log_prob(y[:, cols_b], …)` with per-block masks (`mask[:, cols_b].all(-1)` per row for joint blocks; per-series masks inside independent blocks as in E9-T01).
- Params: block specs prefixed `emission.b{b}.<name>`; covariates per block via the block's own covariate lists.
- EM hooks: `sufficient_stats`/`m_step` = tuple of block results; `None` if any block returns `None` for `m_step` (then ECM handles that block only — expose `m_step_partial` returning per-block results with `None` holes for E13-T02b).
- `conjugate_prior`: product of block priors if all exist, else `None`.

## Creates
- `src/hsmmkit/model/emission/product.py` (generalized), `data/contracts.py` (`series_support`), preparer per-block validation
- `tests/unit/test_e11_t08_product_blocks.py`, `tests/recovery/test_e11_t08_two_series_cov.py` (`-m slow`); contract cases `product-gauss-full-block`, `product-mixed`

## Acceptance criteria
1. Two-series synthetic recovery of a 2×2 covariance per state (2 states, `T = 10 000`, 3 seeds): covariance entries within 15 % relative (Appendix G).
2. E9-T01 regression: independent per-series GBM-option Gaussian blocks reproduce the E9-T01 golden log-likelihood/posteriors bitwise.
3. A mixed block (Gaussian + Poisson) passes `family_contract`; out-of-support data in the Poisson column is rejected naming that column.
4. Masking one series of a joint full-covariance block masks the whole block for that row (documented), while independent blocks keep the other series' contribution.

## Tests
Unit, contract, recovery, E9-T01 regression.

## Out of scope
Marginalizing a partially missing joint Gaussian block (backlog); factorial regimes (E9-T02).

## Connections
- Depends on: E11-T02, E11-T04
- Blocks: E11-T09
- Relates to: E9-T01, E13-T02b
