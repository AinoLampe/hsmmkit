# E1-T03a — Bijectors (with log-det-Jacobian) and log-priors

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M03, FR-M05, FR-M11 | M0 | §4.6, §4.8 |

## Goal
Implement the pure functions that map unconstrained `u` to constrained `θ` and back, their log-det-Jacobians, and the prior log-densities used by MAP.

## Context (self-contained)
Each bijector is an `eqx.Module` with `forward(u) -> theta`, `inverse(theta) -> u`, `log_det_jac(u) -> float` (log |det ∂θ/∂u| summed over the array), and `domain` name.

| Domain | `forward(u)` |
|---|---|
| `real` | identity |
| `positive` | `softplus(u)` (default) or `exp(u)` (`variant="exp"`) |
| `interval[lo, hi]` | `lo + (hi − lo)·sigmoid(u)` |
| `simplex_ref` (row with reference category first) | `softmax(concat([0], u))` — `u` has `n−1` entries, returns `n`; Jacobian defined on the `n−1` free coordinates |
| `nu_gt_2` | `2 + softplus(u)` |
| `pmf` (nonparametric over `D_max`) | `softmax(u)` with `u[0]` pinned to 0 for identifiability (same construction as `simplex_ref`) |
| `increasing` (ordering constraint over a vector) | `θ_0 = g(u_0)`, `θ_i = θ_{i−1} + softplus(u_i)` where `g` is the inner bijector for the base domain (`positive` for σ) |

Numerics: use `jax.nn.softplus`, `jax.nn.log_sigmoid`; inverse of softplus `u = θ + log(−expm1(−θ))` computed stably; implement `log_det_jac` analytically where it is a sum of scalar terms (identity, softplus, sigmoid, `nu_gt_2`, `increasing`); for `simplex_ref`/`pmf` compute it via `jnp.linalg.slogdet(jax.jacfwd(forward_free)(u))` on the `n−1` free coordinates (called once per objective evaluation, cheap). Every bijector is tested against the autodiff slogdet ≤ 1e-8 (Appendix D).

Priors (`params/priors.py`), evaluated on `θ`, returning summed log-density:
- `NormalPrior(mean=0, sd)`: `−0.5((θ−m)/sd)² − log(sd) − 0.5 log 2π`
- `LogNormalPrior(mean_log=log(init), sd)`: on `log θ`, includes `−log θ`
- `DirichletPrior(alpha)`: `Σ (α−1) log p` on each jump-matrix row (constant term included)
Default coefficient prior: `NormalPrior(0, coef_prior_sd=1.0)`.

## Creates
- `src/regimekit/model/params/bijectors.py`, `src/regimekit/model/params/priors.py` (make `model/params/` a package; `params.py` from the concept layout becomes `params/__init__.py` re-exporting)
- `tests/property/test_e1_t03a_bijectors.py`, `tests/unit/test_e1_t03a_priors.py`
- `docs/math/parameters.md` (bijector table)

## Acceptance criteria
1. For every bijector and random `u` (hypothesis, dims 1–6): `inverse(forward(u)) ≈ u` ≤ 1e-9 and `forward` lands in the domain (positivity, bounds, rows sum to 1, strictly increasing).
2. `log_det_jac` vs autodiff slogdet ≤ 1e-8 for all bijectors.
3. Priors match `scipy.stats.{norm,lognorm,dirichlet}.logpdf` ≤ 1e-10.
4. All functions are jittable and differentiable (`jax.grad` finite).

## Tests
Property tests (hypothesis) + scipy comparisons.

## Out of scope
`ParamSpec`/`ParamManager` (E1-T03b); which parameter uses which prior (families).

## Connections
- Depends on: E0-T05
- Blocks: E1-T03b
- Relates to: E5-T01 (objective sums `log_det_jac` and priors)
