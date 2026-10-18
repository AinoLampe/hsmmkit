# E3-T03 — `DurationFamily` protocol, `geometric` and `negbin` (phase representation)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M06, FR-M07, FR-M09 | M1 | §4.3, §4.4 Durations, §5.5 (`DurationFamily`), ADR-0006 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendments 2 and 8: `mean_sojourn` in clock units, `init_mean_sojourn_steps`). No behavioural change for the finance profile.

## Goal
Implement the phase-based durations used by the expanded engine: geometric (`r=1`, HMM) and shifted negative-binomial (`r_i` phases), with the log-rate covariate link and irregular `Δ_t`.

## Context (self-contained)
```python
class DurationFamily(Protocol):
    name: str; representation: Literal["phase", "explicit"]
    def param_specs(self, n_states: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]: ...
    # phase representation
    def n_phases(self, i: int) -> int: ...                   # r_i (static ints)
    def advance_prob(self, theta: Params, dt: Array, z_d: Array | None) -> Array:   # (T, N)
    # explicit representation (E3-T04); phase families raise NotImplementedError
    def log_pmf(self, theta, d_max, nominal_dt, z_entry) -> Array: ...             # (N, D_max)
    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array: ...   # (N,) in clock units (years or steps)
```
- Exit/advance rate `λ_i` per clock unit (per year under `trading`/`calendar`, per step under `steps`), parameter spec `duration.rate (N,)` positive; init from a mean sojourn `m_i` expressed in steps (`init_mean_sojourn_steps` directly; `init_mean_sojourn_days` → `days · 252 · bars_per_day / 252 = days · bars_per_day` steps on the trading clock, `days / (365.25 · nominal_dt)` steps on the calendar clock): for geometric `λ_i = 1 / (m_i · nominal_dt)`; for NB `λ_i = r_i / (m_i · nominal_dt)` (since `E[d] = r_i / a_i ≈ r_i / (λ_i Δ)`). No literal `252` outside `calendar.py`.
- Covariate link (evaluated every step): `log λ_i(t) = log λ_i + c_iᵀ z^d_t`, spec `duration.c (N, K_d)` real with default `NormalPrior(0, coef_prior_sd)`.
- Advance probability: `a_i(t) = 1 − exp(−λ_i(t) Δ_t)` (use `−expm1`).
- NB semantics (ADR-0006): regime `i` has `r_i ≥ 1` phases; chain enters phase 1, advances phase by phase with prob `a_i(t)`, exits from phase `r_i`. On a constant grid `d − r_i ~ NegBin(r_i, a_i)`, `E[d] = r_i/a_i`, `Var[d] = r_i(1−a_i)/a_i²`, minimum duration `r_i`. `r_i` are integer hyperparameters from config (`DurationCfg.r`), static.
- `mean_sojourn(theta, nominal_dt)`: report `E[d]·nominal_dt` with `E[d] = r_i / a_i(nominal_dt)` — years under `trading`/`calendar`, steps under `steps`; document that this tends to `r_i/λ_i` for small `λΔ`. (Renamed from `mean_sojourn_years`, Part II §15.7.)
- Also expose the explicit-engine equivalent for the equivalence test: `negbin_shifted_logpmf` from E1-T04 via `log_pmf` (allowed for NB even though `representation="phase"`; mark `supports_explicit=True`).

## Creates
- `src/regimekit/model/duration/{base,geometric,negbin}.py`
- `tests/unit/test_e3_t03_durations.py`, `tests/property/test_e3_t03_durations.py`
- `docs/math/durations.md`

## Acceptance criteria
1. Simulating the phase chain (simple NumPy loop in the test, constant `Δ`) for `r=3, a=0.05`: empirical mean sojourn within 3 SE of `r/a`; minimum observed `= r`.
2. `negbin` with `r=1` equals `geometric` (`advance_prob` bitwise, `log_pmf` ≤ 1e-12).
3. Irregular `Δ_t` handled: `advance_prob` shape `(T, N)` and equals `1 − exp(−λΔ_t)` per row.
4. Zero coefficients reproduce constant `λ`; positive `c` with positive `z` increases `a_i(t)` (monotonic test).
5. Appendix E: `λ=2/yr` daily → `a = 0.0079`; `λ=6/yr` → `0.0235` (4 dp).
6. Steps clock: with `dt = ones(T)` and `λ = 0.05` per step, `a = 1 − e^{−0.05}` and `mean_sojourn(theta, 1.0) == r / a` in steps; `init_mean_sojourn_steps=[20]` with `r=1` yields `λ = 0.05` (bitwise for `nominal_dt = 1`).

## Out of scope
Explicit families (E3-T04); state expansion (E4-T02a).

## Connections
- Depends on: E1-T03b, E1-T04
- Blocks: E3-T04, E4-T02a
- Relates to: E4-T06b (equivalence), E6-T01
