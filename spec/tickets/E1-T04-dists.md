# E1-T04 — Distributions: log-pdf/pmf and samplers (`regimekit.model.dists`)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M06, FR-M07 | M0 | §4.2, §4.3, ADR-0003 |

## Goal
Provide in-house JAX distributions (no TFP) used by emission and duration families, validated against scipy.

## Context (self-contained)
All functions pure, vectorized, float64, jittable. Log-pmfs over durations return an array over `d = 1..D_max` **normalized to sum to 1 on that support** (truncation).

| Function | Definition |
|---|---|
| `normal_logpdf(y, mean, sd)` | standard |
| `student_t_logpdf(y, loc, scale, nu)` | standard, `nu > 2` |
| `poisson_shifted_logpmf(d, kappa, d_max)` | `d = 1 + Poisson(kappa)`, truncated at `D_max`, renormalized |
| `negbin_shifted_logpmf(d, r, a, d_max)` | `d − r ~ NegBin(r, a)` (failures before `r` successes with success prob `a`), support `d ≥ r`, truncated at `D_max`, renormalized; `negbin_mean(r,a)=r/a`, `negbin_var(r,a)=r(1−a)/a²` |
| `disc_gamma_logpmf(d, shape, scale, dt, d_max)` | `p(d) = [F(dΔ) − F((d−1)Δ)] / F(D_max Δ)` with `F` the Gamma cdf in years |
| `disc_weibull_logpmf(d, shape, scale, dt, d_max)` | same with Weibull cdf |
| `disc_lognormal_logpmf(d, mu_log, sd_log, dt, d_max)` | same with log-normal cdf |
| `nonparametric_logpmf(logits, d_max)` | `log softmax(logits)` over `1..D_max` |
| `geometric_logpmf(d, a, d_max)` | `(1−a)^{d−1} a`, truncated, renormalized (used by the explicit engine for `r = 1` equivalence) |
| samplers | `sample_normal`, `sample_student_t`, `sample_categorical(key, log_probs)` (Gumbel-max or `jax.random.categorical`), `sample_pmf(key, log_pmf)` returns `d` |

Cdfs from `jax.scipy.stats` / `jax.scipy.special` (`gammainc`, `erf`); Weibull cdf `1 − exp(−(x/scale)^shape)` in closed form. Use `logsumexp` for normalization; `−inf` for `d < r`.

## Creates
- `src/regimekit/model/dists.py`
- `tests/unit/test_e1_t04_dists.py` (scipy agreement), `tests/property/test_e1_t04_dists.py`

## Acceptance criteria
1. Each log-pdf/pmf agrees with the corresponding `scipy.stats` (`norm`, `t`, `poisson`, `nbinom`, `gamma`, `weibull_min`, `lognorm`, `geom`) ≤ 1e-10 on random parameters, after applying the same truncation/renormalization in the test.
2. Every pmf sums to 1 over `1..D_max` ≤ 1e-10; NB pmf is `−inf` for `d < r`.
3. Analytic means: NB `r/a` (untruncated) reproduced by the pmf mean when `D_max` truncation mass < 1e-10.
4. Sampler moments within 3 SE of analytic for `n = 20000` (normal, t, categorical, pmf).
5. All functions jittable and differentiable w.r.t. continuous parameters.

## Tests
scipy comparisons; hypothesis for normalization; sampler moment tests with fixed keys.

## Out of scope
Family classes (E3-T03/T04); emission covariate links (E3-T01).

## Connections
- Depends on: E0-T05
- Blocks: E3-T01, E3-T03, E3-T04, E4-T01
- Relates to: E1-T03a (bijectors used for parameters of these distributions)
