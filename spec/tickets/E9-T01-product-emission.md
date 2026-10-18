# E9-T01 — `ProductEmission` for `M > 1` (independent per-series GBM / Student-t) and multi-primary data layer

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M12, D10 | post-0.1.0 / parallel with E8 | §5.10, §4.2 |

## Goal
Enable a small universe with independent per-series emissions sharing one regime process, without changing engine or contracts.

## Context (self-contained)
- `ProductEmission(families: tuple[EmissionFamily, ...] | EmissionFamily, n_series)`: `log_prob` returns `Σ_m log f_i^{(m)}(y_{t,m})` over series with per-series masks (`jnp.where(mask[:, m], logf_m, 0)` summed) → `(T, N)`; `sample` stacks per-series samples `(T, M)`; `param_specs` with a series axis (`(N, M)` already in E3-T01) — `ProductEmission` mainly formalizes per-series masking and optional per-series family choice; parameters may be `tied` across series via tie groups.
- Data layer: `PriceSource.symbols` with `M > 1`: master grid from the first symbol (primary), other series aligned by exact timestamp (missing → `mask=False`), `ObservationSet.prices (T0, M)` already supported; `Preparer` computes returns per column.
- Config: `model.n_series` inferred from `len(symbols)`; `emission.per_series: bool` (tie across series or not).

## Creates
- `src/regimekit/model/emission/product.py`; `Preparer`/`calendar` multi-series paths; config fields
- `tests/integration/test_e9_t01_two_series.py`, regression tests asserting `M=1` outputs unchanged

## Acceptance criteria
1. Two-series synthetic recovery (shared regimes, σ per series) meets Appendix D σ tolerance (3 seeds).
2. `M=1` path produces bitwise-identical loglik/posteriors vs. before the change (golden values stored).
3. Per-series masks: masking series 2 at step `t` leaves series 1's contribution intact (assert vs. reference).

## Out of scope
Correlated emissions, factorial regimes (E9-T02).

## Connections
- Depends on: E3-T01, E2-T06
- Blocks: E9-T04
- Relates to: E1-T01 (`M` dimension)
