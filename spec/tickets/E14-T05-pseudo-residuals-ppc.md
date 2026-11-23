# E14-T05 — Pseudo-residuals and posterior predictive checks; fit-report integration

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EV06 | M11 | Part II §14.12 (Pseudo-residuals, PPC); Part I §4.10, E5-T04; Zucchini, MacDonald & Langrock 2016 |

## Goal
Give users standard model-adequacy diagnostics for every support: normal pseudo-residuals and posterior predictive checks on chosen statistics.

## Context (self-contained)
**Pseudo-residuals** (`selection/pseudo_residuals.py`): one-step predictive distribution `F_t(y) = Σ_i P(regime_t = i | y_{1:t−1}) F_i(y | Δ_t, z_t)` with predictive regime probabilities `softmax(α_{t−1}) A_t C` (E5-T04 helper).
- Continuous (`real`, `M = 1`): `u_t = F_t(y_t)`, `r_t = Φ⁻¹(u_t)`; requires family `cdf(y, dt, θ, z)` (new optional protocol method; implemented for `gaussian` (diag, per series → per-series residuals), `student_t` uv, `gbm` via alias; others → `None` and residuals skipped with a note).
- Discrete (`count`, `binary`, `categorical`): mid-distribution `u_t = ½(F_t(y_t) + F_t(y_t − 1))` with `F_t(−1) = 0`; requires `cdf` for Poisson/Bernoulli; categorical uses the ordering of category ids (documented as arbitrary → only reported, not tested).
- `M > 1` (full covariance): sequential conditional residuals (Rosenblatt transform along series order) — documented, implemented for `gaussian` full.
- Outputs: residual series, Shapiro–Wilk `W` (own implementation via the Royston approximation, scipy in tests only), Ljung–Box on `r_t` and `r_t²` (E5-T04), QQ data for plotting (E16-T03).
**PPC** (`selection/ppc.py`): simulate `R` replicates from the fitted model (E12-T02 simulate with the observed lengths, `Δ_t`, covariates; for MCMC fits draw `θ` from the posterior per replicate) and compare statistics `T(y)`: registry kind `ppc_statistic` (add to `plugins.Kind`) with defaults `acf_abs` (autocorrelation of `|y|` at lags 1–20), `sojourn_mean` (empirical mean decoded segment length per regime, decoding each replicate with the fitted `θ`), `kurtosis`; user statistics registered by name. Report two-sided tail probabilities `p = 2 min(P(T_rep ≥ T_obs), P(T_rep ≤ T_obs))`.
**Fit report**: sections "Pseudo-residuals" and "Posterior predictive checks" (enabled by `estimation.diagnostics: [pseudo_residuals, ppc]`, default `[pseudo_residuals]`).

## Creates
- `src/hsmmkit/selection/{pseudo_residuals,ppc}.py`, family `cdf` methods, report sections
- `tests/integration/test_e14_t05_residuals.py`, `tests/integration/test_e14_t05_ppc.py` (`-m slow` for seed sweeps)

## Acceptance criteria
1. Correctly specified Gaussian model: pseudo-residuals pass Shapiro–Wilk at 5 % on ≥ 90 % of 50 seeds; misspecified (Gaussian fitted to Student-t `ν = 3` data) fails on ≥ 90 %.
2. PPC flags the misspecified model (kurtosis tail probability < 0.05 in ≥ 4/5 seeds) and not the correct one (> 0.05 in ≥ 4/5).
3. Poisson mid-distribution residuals of a correct model are approximately normal (Shapiro–Wilk pass ≥ 80 % of seeds).
4. Own Shapiro–Wilk equals `scipy.stats.shapiro` W ≤ 1e-6 on random samples (dev-only).

## Tests
Integration (slow seed sweeps), unit tests for the transforms, dev-only scipy comparison.

## Out of scope
Plotting (E16-T03).

## Connections
- Depends on: E13-T04, E11-T04
- Blocks: E16-T05, E17-T03
- Relates to: E5-T04, E13-T06a (posterior draws)
