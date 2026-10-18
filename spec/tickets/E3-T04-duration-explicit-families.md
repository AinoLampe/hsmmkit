# E3-T04 — Explicit duration families: poisson, gamma, weibull, lognormal, nonparametric

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M06, FR-M07, FR-M09 | M3 | §4.3 (continuous-time sojourn → grid pmf, hazard form), §4.4 Durations (explicit engine) |

## Goal
Provide the `explicit` representation of `DurationFamily`: `(N, D_max)` log-pmfs on a regular grid with an entry-time covariate scale link.

## Context (self-contained)
`representation="explicit"`; `n_phases`/`advance_prob` raise `NotImplementedError`.

| Family | Parameters (annualized, per state) | Grid pmf over `d = 1..D_max` |
|---|---|---|
| `poisson` | `duration.kappa (N,)` positive, in years: `d = 1 + Poisson(κ / Δ)` | shifted Poisson truncated at `D_max`, renormalized |
| `gamma` | `shape (N,)`, `scale (N,)` positive (years) | `[F(dΔ) − F((d−1)Δ)] / F(D_max Δ)` |
| `weibull` | `shape`, `scale` (years) | same |
| `lognormal` | `mu_log (N,)` real, `sd_log (N,)` positive | same |
| `nonparametric` | `duration.logits (N, D_max)` domain `pmf` | `softmax` |

Covariate link (entry-time): `log scale_i = log scale_i^0 + c_iᵀ z^d_{t_entry}` for gamma/weibull; for lognormal on `mu_log`; for poisson on `log κ`; none for nonparametric. `log_pmf(theta, d_max, nominal_dt, z_entry)`: `z_entry` is `(K_d,)` for a single entry or `(T, K_d)` → returns `(N, D_max)` or `(T, N, D_max)` (the engine decides whether it needs the per-`t` version; without covariates always `(N, D_max)`).
Hazard helper: `hazard(log_pmf) → h_i(d) = p_i(d)/S_i(d−1)`, `S_i(d) = 1 − Σ_{u≤d} p_i(u)`, in log space with clipping.
`mean_sojourn(theta, nominal_dt)` (renamed from `mean_sojourn_years`, Part II §15.7; clock units): analytic untruncated mean (gamma `shape·scale`, weibull `scale·Γ(1+1/shape)`, lognormal `exp(μ+σ²/2)`, poisson `(1 + κ/Δ)·Δ`), plus `mean_sojourn_steps_truncated(theta, d_max, dt)` from the pmf.
`D_max` warning rule (also applied at config validation): warn if truncated mass `1 − F(D_max Δ) > 1e-6` or `d_max < 2 × mean sojourn in steps`.
Init from `init_mean_sojourn_days`: gamma `shape=2`, `scale = days/(252·2)`; weibull `shape=1.5`, `scale = mean/Γ(1+1/1.5)`; lognormal `sd_log=0.5`, `mu_log = log(mean) − sd²/2`; poisson `κ = (days−1)/252`.

## Creates
- `src/regimekit/model/duration/{poisson,gamma,weibull,lognormal,nonparametric}.py`, `duration/hazard.py`
- `tests/unit/test_e3_t04_explicit_durations.py`
- `docs/math/durations.md` (explicit section)

## Acceptance criteria
1. Every `log_pmf` row sums to 1 ≤ 1e-10 for random parameters and `D_max ∈ {50, 250}`.
2. Truncated pmf mean matches the analytic mean within the truncation error (test computes both with scipy).
3. `hazard` of a geometric pmf is constant (`= a`) ≤ 1e-10.
4. Entry-time link: zero coefficients reproduce the base pmf; `z_entry (T, K)` returns `(T, N, D_max)`.
5. `D_max` warning triggers on a fixture with `d_max = 5` and mean sojourn 20 steps.

## Out of scope
Explicit-duration engine (E4-T06a); phase families.

## Connections
- Depends on: E3-T03
- Blocks: E4-T06a
- Relates to: E1-T02a (`d_max` default and warning rule)
