# E11-T07 — Finance families: Ornstein–Uhlenbeck (exact discretization) and Merton jump-diffusion

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast | M | FR-EM06, FR-EM08 | M8 (`regimekit` 0.4.0) | Part II §14.2 (OU, Merton), Appendix F; Part I §4.2 |

## Goal
Add the two planned finance emission families to `regimekit` as external `hsmmkit` families that pass the public contract suite.

## Context (self-contained)
**OU** (`regimekit.emission.ou`, registered `ou`): for an observed level `x_t` (e.g. a spread or log-VIX) with regime `i`:
```
x_t | x_{t−1}, i ~ N( m_i + (x_{t−1} − m_i) e^{−κ_i Δ_t},  σ_i² (1 − e^{−2κ_i Δ_t}) / (2κ_i) )
```
- `support="real"`, `scale_by_dt=False` (exact, nonlinear in `Δ`), `native_time="continuous"`; specs `emission.m (N, M)` real, `emission.kappa (N, M)` positive, `emission.sigma (N, M)` positive (annualized under `trading`/`calendar`, per step under `steps`).
- `x_{t−1}` is supplied as an own-lag emission covariate using the `hsmmkit` preparer lag hook (E11-T05, `requires_own_lags = 1`); the observation transform is `identity` (levels, not returns) — config example in docs; first step masked.
- Numerics: use `−expm1(−2κΔ)` for the variance and a series expansion for `κΔ < 1e-8` (limit `σ²Δ`).
**Merton** (`regimekit.emission.merton`, registered `merton`): a finite mixture over jump counts:
```
y_t | i, k ~ N( (μ_i − ½σ_i² − λ_i k̄_i) Δ_t + k m_{J,i},  σ_i² Δ_t + k s_{J,i}² ),   k ~ Poisson(λ_i Δ_t) truncated at K_max (default 5),
k̄_i = exp(m_{J,i} + ½ s_{J,i}²) − 1
```
- Implemented on `hsmmkit`'s mixture machinery (E11-T06) with weights fixed to the truncated, renormalized Poisson pmf; specs `mu, sigma, lam (positive), m_j (real), s_j (positive)` per state; `K_max` static option; the truncated tail mass `1 − Σ_{k≤K_max} Poisson(k; λΔ)` reported in fit diagnostics (warning if > 1e-6).
- `λ = 0` must reduce to `gbm` (the `k = 0` component only).
Both families: register in `regimekit._register` (entry points come with E16-T06); add contract cases to `tests/contract/` in `RegimeDetectionAndForecast` with `clocks=("steps","calendar","trading")`.

## Creates
- `src/regimekit/emission/{ou,merton}.py`, registration, `tests/unit/test_e11_t07_ou_merton.py`, contract cases
- `docs/guide/families_finance.md`

## Acceptance criteria
1. OU transition density vs the closed form (NumPy reference) ≤ 1e-12, incl. the small-`κΔ` limit.
2. Merton with `λ = 0` equals `gbm` ≤ 1e-14 (log-density); tail mass reported.
3. Both pass `hsmmkit.testing.family_contract` (all applicable checks) under all three clocks.
4. A 2-regime OU synthetic recovery (`T = 5000`, daily trading grid) recovers `κ` within 25 % and `m` within 2 SE (3 seeds, `-m slow`).

## Tests
Unit, contract, slow recovery.

## Out of scope
MS-GARCH (remains the E9-T03 spike); jump-size distributions other than Gaussian.

## Connections
- Depends on: E11-T06, E10-T09
- Blocks: E16-T06
- Relates to: E11-T05 (lag hook), E9-T03
