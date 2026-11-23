# E11-T04 — Poisson, Bernoulli and Categorical emissions with covariate links, closed-form M-steps and conjugate priors

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EM03, FR-EM07, FR-G03 | M8 | Part II §14.2 (Poisson/Bernoulli/Categorical), §14.8 (conjugates), Appendix F |

## Goal
Add the three discrete-support families needed for counts, binary and categorical sequences, each with the EM and VB hooks.

## Context (self-contained)
```
poisson:      y_tm | i ~ Poisson( exp(η_im + γ_imᵀ z^e_t) · s(Δ_t) )         support count, exposure: scale_by_dt default True
bernoulli:    y_tm | i ~ Bernoulli( σ(η_im + γ_imᵀ z^e_t) )                   support binary
categorical:  y_t  | i ~ Cat( softmax([0, η_i + Γ_i z^e_t]) ),  K categories  support categorical, M = 1, first category reference
```
- Specs: `emission.eta (N, M)` real (Poisson/Bernoulli), `emission.eta (N, K−1)` (Categorical); covariate coefficients `emission.gamma (N, M, K_e)` / `(N, K−1, K_e)` with `NormalPrior(0, coef_prior_sd)`. Config `params.rate` / `params.prob` / `params.probs` with `init` in natural units converted to `η` (documented).
- `log_prob`: Poisson `y log λ − λ − lgamma(y+1)` with `λ = exp(η+γᵀz)·s(Δ)` (sum over `M`); Bernoulli `y·logit − softplus(logit)`; Categorical `log_softmax[y]`. Masked entries handled by the engine (per-series masks summed inside the family as E9-T01 does).
- `sufficient_stats(y, dt, gamma, z)` and `m_step` (no covariates): Poisson `λ_im = Σ_t γ_ti y_tm / Σ_t γ_ti s(Δ_t)`; Bernoulli `p_im = Σγ y / Σγ`; Categorical `p_ik = Σγ 1[y=k] / Σγ`; `m_step` returns `η` (log / logit / log-ratio to reference) with clipping to `[1e-12, 1−1e-12]` documented. With covariates → `None` (ECM).
- `conjugate_prior()`: Poisson → Gamma(`a0, b0`) on `λ` (per unit exposure), Bernoulli → Beta(`a0, b0`), Categorical → Dirichlet(`α0`); defaults `a0 = b0 = 1`, `α0 = 1`; used by E13-T05a/E13-T06b. Provide `posterior(stats, prior) -> prior-shaped params` and `expected_log_params(q)` (digamma forms) for VB.
- `moments` for the contract suite: Poisson mean = var = `λ`; Bernoulli `p`, `p(1−p)`; Categorical one-hot moments.
- `n_categories` from `params.probs` length or `data` (E10-T05 rule); mismatch → `ConfigError`.

## Creates
- `src/hsmmkit/model/emission/{poisson,bernoulli,categorical}.py`, conjugate helpers in `model/conjugate.py`
- `tests/unit/test_e11_t04_discrete_families.py`; contract cases `poisson`, `bernoulli`, `categorical`

## Acceptance criteria
1. All three pass `family_contract`, including check 7 (M-step closed form equals numerical optimum ≤ 1e-8).
2. Log-pmfs vs `scipy.stats.poisson/bernoulli/multinomial(1, p)` ≤ 1e-10.
3. Poisson exposure: under `calendar` with irregular `Δ_t`, MLE of `λ` equals `Σy/ΣΔ` on a 1-state fixture.
4. Conjugate posterior with zero weights returns the prior; with data matches closed-form Gamma/Beta/Dirichlet updates.
5. Data violating the support is rejected by the preparer with the E10-T05 message.

## Tests
Unit tests, contract suite, conjugate update tests.

## Out of scope
Zero-inflated / negative-binomial emissions (plug-ins); EM loop (E13-T02a).

## Connections
- Depends on: E11-T01
- Blocks: E11-T08, E13-T02a, E13-T05a, E14-T05, E16-T02
- Relates to: E10-T05 (supports), E16-T05 (counts preset)
