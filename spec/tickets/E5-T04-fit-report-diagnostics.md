# E5-T04 — Fit report and diagnostics: occupancy, restart dispersion, ordering check, Hessian std errors, PIT / Ljung-Box hooks

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-I03, G1 | M3 | §4.8 Standard errors, §4.10 Diagnostics, §5.4 (`FitResult.diagnostics`) |

## Goal
Attach the concept's mandatory post-fit diagnostics and optional standard errors to `FitResult`.

## Context (self-contained)
`estimation/report.py`: `diagnose(fit, spec, space, engine, data) -> dict` and `std_errors(objective, u_hat, manager) -> dict[str, Array] | None`; `map.py` (E5-T03, formerly `fit.py`) calls both when `cfg.std_errors`.
Diagnostics (each a dict entry with value + `warning: str | None`):
- `regime_occupancy (N,)`: mean smoothed regime marginal; warn if any `< 0.02` ("misspecified fixed parameters or too many states").
- `posterior_mean_duration (N,)`: mean Viterbi segment length per regime vs `mean_sojourn(theta, nominal_dt) / nominal_dt` from the family (renamed per Part II §15.7); report ratio.
- `pit_hist`: one-step-ahead PIT values `F_t(y_t)` from the filtered predictive mixture `Σ_i P(regime_t = i | y_{1:t−1}) f_i(y_t)` (predictive regime probs = `softmax(α_{t−1}) @ A_t @ C`); 10-bin histogram + KS statistic vs uniform.
- `ljung_box`: Ljung-Box Q on standardized residuals `(y_t − E[y_t|y_{1:t−1}]) / sd` at lags 10 and 20 (own implementation, chi-square p-value via `scipy` in tests only → implement p-value with `jax.scipy.special.gammainc`).
- `ordering_check`: σ sorted or not; `label_switching_risk` bool.
- `restart_dispersion`: relative gap between best two objectives; warn if `> 0.01` ("likely multimodality").
- `timings`: wall time per restart, compile time.
Standard errors: `H = jax.hessian(objective.value)(u_hat)`; `Cov_u = inv(H)`; delta method `Cov_θ = J Cov_u Jᵀ` with `J = jacfwd(manager.unflatten)`; SE per named parameter; if `H` not positive definite (min eigenvalue ≤ 0) → `std_errors=None` + diagnostic `hessian_pd=False` with caveat text.

## Creates
- `src/regimekit/estimation/report.py`
- `tests/unit/test_e5_t04_diagnostics.py`, `tests/integration/test_e5_t04_std_errors.py` (`-m slow` for the large-sample check)

## Acceptance criteria
1. Each diagnostic has a synthetic triggering case: a 4-state fit on 2-state data triggers occupancy warning; two restarts with objectives differing by 2 % trigger dispersion warning; shuffled σ triggers ordering flag; a model with wrong σ produces a non-uniform PIT (KS p < 0.01) while the true model passes (p > 0.05).
2. Std errors on a large synthetic sample (`T=20000`, 2-state GBM/geometric) within 20 % of the empirical SE over 20 simulation replications (`-m slow`).
3. Non-PD Hessian returns `None` with the caveat rather than raising.

## Out of scope
Backtest metrics (E7-T01 reuses PIT/log-score helpers → place shared helpers in `evaluation/metrics/predictive.py` stub here).

## Connections
- Depends on: E5-T03, E4-T05
- Blocks: E7-T01 (predictive helpers)
- Relates to: E4-T04
