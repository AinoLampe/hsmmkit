# E13-T05a — `VBEstimator`: conjugate updates, structured `q(s)`, ELBO

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES04, FR-ES06 | M10 | Part II §14.8, Appendix F (conjugates); Beal 2003 |

## Goal
Provide mean-field variational Bayes over parameters with an exact structured posterior over state paths for fully conjugate models, reporting a monotone ELBO.

## Context (self-contained)
`estimation/vb.py`, estimator `vb` (expanded engine in this ticket; explicit engine with nonparametric durations allowed via `ExplicitEStats` — same pattern):
```
q(θ) Π_b q(s^{(b)}),  q(s) exact via forward-backward on expected log-parameters  log θ̃ = E_q[log θ]
```
Conjugate blocks (Appendix F), each implemented as `ConjugateBlock` with `prior`, `posterior(stats)`, `expected_log_params(q)`, `kl(q, prior)`:
| Component | `q` family | expected log-parameters |
|---|---|---|
| Gaussian (diag) | Normal-Gamma per series | `E[log N(y|μ,τ)]` via digamma/expectations |
| Gaussian (full) | Normal-Wishart | `E[log|Λ|]`, `E[(y−μ)ᵀΛ(y−μ)]` |
| Poisson rate | Gamma | `ψ(a) − log b` (exposure handled in stats) |
| Bernoulli | Beta | `ψ(a) − ψ(a+b)` |
| Categorical, jump rows (allowed targets), nonparametric pmf | Dirichlet | `ψ(α_k) − ψ(Σα)` |
| geometric exit / NB phase advance (constant `Δ`) | Beta on per-step `a_i` | `E[log a], E[log(1−a)]` |
Iteration: (1) E-step: run forward-backward with `ℓ̃_t(s) = E_q[log f]` (family `expected_log_prob`, E11-T01 protocol) and `log Ã_t` built from expected log-parameters (sub-normalized; the builder accepts log-weights directly), giving `ExpandedEStats` (E13-T01a) and `log Z̃` (the forward normalizer); (2) M-step: `q(θ_block) ← posterior(prior, stats)` per block; (3) `ELBO = log Z̃ − Σ_blocks KL(q ‖ prior)`.
- Priors: from config `prior` entries where given; otherwise weakly informative defaults per block documented in `docs/guide/estimators.md` (data-scaled for Gaussian). `use_priors: false` is invalid for VB (`ConfigError`).
- Initialization: `q` centred on an init-strategy point estimate (E5-T03) with prior pseudo-counts; restarts as MAP; best final ELBO wins.
- `FitResult(estimator="vb", params = posterior means on the constrained scale, elbo, loglik at posterior means, diagnostics["vb"] = {"elbo_trace", "q_params"})`; `uparams` from posterior means.
- Non-conjugate components → `ConfigError` in this ticket ("VB-EM hybrid available from E13-T05b").
Gate `estimation.method: vb` removed for fully conjugate models (hybrid remains gated by predicate).

## Creates
- `src/hsmmkit/estimation/vb.py`, `src/hsmmkit/model/conjugate.py` (blocks; extends E11-T04 helpers), statespace builder support for log-weights
- `tests/integration/test_e13_t05a_vb.py`, `tests/unit/test_e13_t05a_conjugate_blocks.py`

## Acceptance criteria
1. ELBO non-decreasing (violations ≤ 1e-8) on Gaussian (diag/full), Poisson and Categorical models with geometric and NB durations.
2. KL terms vs closed-form references (NumPy) ≤ 1e-10 for every block; with zero data the posterior equals the prior.
3. On `T = 20 000` synthetic data with flat-ish priors, posterior means are within 2 % of MAP estimates (Appendix G) for the Gaussian diag case.
4. ELBO ≤ log marginal likelihood estimate by importance sampling on a tiny model (`T = 10`, sanity bound).

## Tests
Unit (blocks, KL), integration (monotonicity, MAP agreement), sanity bound.

## Out of scope
Pruning, VB-EM hybrid, fit report integration (E13-T05b).

## Connections
- Depends on: E13-T01a, E11-T04
- Blocks: E13-T05b, E13-T06b
- Relates to: E11-T02 (Normal-Wishart), E13-T02a
