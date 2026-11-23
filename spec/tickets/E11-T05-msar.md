# E11-T05 — Markov-switching autoregression MS-AR(p), including Hamilton's mean-adjusted form for `p ≤ 2`

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EM04, FR-M09 | M8 | Part II §14.2 (MS-AR), §19 #14; Part I §4.5 (memory builder), E4-T02a |

## Goal
Provide regime-switching autoregressions as a Gaussian family on preparer-generated own-lag covariates, plus Hamilton's mean-adjusted variant via the segment-memory builder.

## Context (self-contained)
**Regime-dependent-coefficient form** (`family: msar`, `params: {p: int ≥ 1, form: "coefficients" | "mean_adjusted" = "coefficients"}`):
```
y_t = c_i + Σ_{l=1}^{p} Φ_{i,l} y_{t−l} + ε_t,   ε_t ~ N(0, Σ_i)   (Σ_i diag or full as in gaussian)
```
- Preparer hook: when the emission declares `requires_own_lags = p`, `Preparer.transform` appends `y_{t−1..t−p}` (after the observation transform and standardization) as emission covariates named `__lag{l}_{series}`, sets `mask[t ≤ p] = False` for all series (conditioning on the first `p` observations), and records the lag columns in `cov_names["emission"]`. Lag ≥ 1 is guaranteed by construction (no look-ahead).
- Specs: `emission.intercept (N, M)`, `emission.ar (N, p, M, M)`, covariance via `gaussian` specs (`scale` or `chol`). Optional stationarity check (spectral radius of the companion matrix < 1) as a fit diagnostic (warning), not a constraint.
- `sufficient_stats`/`m_step` (weighted least squares per state): `X_t = [1, y_{t−1}ᵀ, …, y_{t−p}ᵀ]`, `B_i = (Σγ X Xᵀ)⁻¹ Σγ X yᵀ`, `Σ_i` from weighted residuals. `conjugate_prior()` → matrix-Normal-Wishart on `(B_i, Σ_i)`.
**Hamilton mean-adjusted form** (`form: mean_adjusted`, `p ≤ 2`): `y_t − μ_{s_t} = Σ_l φ_l (y_{t−l} − μ_{s_{t−l}}) + ε_t` with `φ` shared across regimes (Hamilton 1989) and `σ` per regime or shared. The density depends on `(s_t, s_{t−1}, …, s_{t−p})`, i.e. regimes at previous **time steps**. The Part I segment-memory builder excludes repeated regimes and therefore cannot express this (Part II v1.2 correction to §14.2). Build a time-step memory state space instead: builder option `memory_mode: "steps"` (`statespace/builders.py`) with tuples `(s_{t−p}, …, s_t)` allowing repeats (`N^{p+1}` states); the transition `(s_{t−p}, …, s_t) → (s_{t−p+1}, …, s_t, s_{t+1})` has probability `1 − a_{s_t}` if `s_{t+1} = s_t` and `a_{s_t} π_{s_t s_{t+1}}` otherwise (geometric durations, `r = 1`); the emission log-density is gathered by the full tuple; `collapse` maps a tuple to its last element. `p ≥ 3` → `ConfigError(path="model.emission.params.p", "mean-adjusted MS-AR supports p ≤ 2")`; `memory_mode: steps` requires geometric durations (`ConfigError` otherwise).

## Creates
- `src/hsmmkit/model/emission/msar.py`, preparer lag hook in `data/prepare.py`, `memory_mode` option in `statespace/builders.py`
- `tests/unit/test_e11_t05_msar.py`, `tests/recovery/test_e11_t05_msar2.py` (`-m slow`); contract case `msar-p2`

## Acceptance criteria
1. 1-state AR(p) log-likelihood equals `statsmodels.tsa.ar_model.AutoReg(...).fit().llf`-compatible conditional likelihood ≤ 1e-6 (dev-only comparison).
2. Recovery of a 2-state MS-AR(2) with Hamilton (1989)-like magnitudes (`T = 5000`, 5 seeds, ≥ 4 pass): AR coefficients within 2 SE, regime means within 2 SE, mean sojourn ≤ 25 % (Appendix D/G).
3. Mean-adjusted form with `p = 3` → `ConfigError`; with `p = 2` the state-space size is `N³` and matches a brute-force enumeration likelihood on `T = 8` ≤ 1e-10.
4. Lag generation never uses future values (property test with a spike injected at `t` changes nothing before `t+1`); first `p` steps masked.
5. `msar-p2` passes `family_contract` (check 7 via weighted LS).

## Tests
Unit, property (no look-ahead), brute-force mean-adjusted likelihood, slow recovery.

## Out of scope
Stationary-start likelihood (Part II §19 #14); MS-VAR with exogenous regressors beyond covariate links.

## Connections
- Depends on: E11-T02, E10-T05
- Blocks: E14-T06, E16-T05
- Relates to: E4-T02a (builder), E13-T02a (WLS M-step)
