# Estimation

MAP estimation (ADR-0005) maximizes the log-posterior in the unconstrained parameterization `u` (parameters
section):

$$
\hat{u} = \arg\max_u \; \log p(y_{1:T} \mid \theta = g(u)) + \log p_{\text{prior}}(\theta) + \log\left|\det \frac{\partial \theta}{\partial u}\right|
$$

## The objective (E5-T01)

`estimation/objective.py` builds the scalar the optimizers minimize, for any family and engine:

$$
J(u) = -\ell(\theta(u)) - \log p(\theta(u)) - \log\left|\det \frac{\partial \theta}{\partial u}\right| \quad(\text{MAP}),
\qquad J(u) = -\ell(\theta(u)) \quad(\text{MLE, } \texttt{use\_priors=False}).
$$

$\ell$ is `engine.loglik`, $\log p$ is `ParamManager.log_prior` plus the transition model's own extra term (the
Dirichlet prior on the jump matrix when covariate logits are configured), and the last term is the
change of variables from the constrained $\theta$ to the unconstrained $u$. `make_objective(spec, space, engine,
manager, data, use_priors)` returns an `Objective` with jitted `value(u)` and `value_and_grad(u)`; `spec`,
`engine` and `manager` are static, `space` and `data` are pytree arguments. Passing another dataset of the same
shapes (`value_and_grad(u, data=fold)`) reuses the compilation: the static bookkeeping in `PreparedDataset`
(`times`, `prep_stats`, `snapshot_ids`) never reaches an engine and is replaced by constants before the call.
`check_finite(J, g, u, names)` raises `NumericalError` naming the first non-finite gradient coordinate.

## Optimizers (E5-T02)

`estimation/optimizers.py`'s `minimize(objective, u0, cfg, log) -> RunRecord` drives one run from one starting
point. `cfg.optimizer` selects the algorithm:

- `lbfgs` (default): `optax.lbfgs` with a zoom line search (`optax.scale_by_zoom_linesearch`,
  `max_linesearch_steps=20`), driven by the `value_and_grad_from_state` idiom so the line search's own
  evaluations at the accepted point are reused rather than recomputed.
- `adam`: learning rate `1e-2` under `optax.cosine_decay_schedule` to `0` over `cfg.max_iter` steps. Its
  convergence tests are only evaluated every 10 iterations (its per-step objective is noisier than
  lbfgs's line-search-verified one); disabled entirely (`tol_f`/`tol_g` at or below zero) it simply runs
  the full decay schedule, which is how a "ground truth" reference run is obtained in practice — with
  early stopping enabled, a decaying-to-zero step size makes `|\Delta J| \le \texttt{tol\_f}` trivially
  true near the end of *any* fixed-length run regardless of how close `u` actually is to the optimum.

Convergence, checked once per iteration for `lbfgs` and every 10th for `adam`:

$$
|\Delta J| \le \texttt{tol\_f} \cdot \max(1, |J|) \quad\text{(`reason="tol_f"`)}, \qquad
\lVert \nabla J \rVert_\infty \le \texttt{tol\_g} \quad\text{(`reason="tol_g"`)},
$$

else the run stops at `cfg.max_iter` with `converged=False, reason="max_iter"`. A non-finite `J` or
gradient at any iteration aborts the run — never raises — returning the last known-finite `u` with
`reason="non_finite"` (or `"non_finite_start"` when `u0` itself is already non-finite). The whole
optimizer update (line search included) is fused into one `jax.jit`-ed step per iteration, since an
unfused, eagerly-dispatched line search dominates the wall clock otherwise. `minimize` has no randomness
and never touches an RNG key: identical inputs give bitwise identical output.

## Initialization and multi-start (E5-T03)

`estimation/init.py` builds a starting `theta` (constrained space) for one restart, from three
strategies (`estimation.init` config, cycled over `estimation.restarts`):

- `rolling_kmeans(data, spec, key)`: clusters a 21-step rolling realized volatility
  (`sqrt(mean(y^2) / nominal_dt)`, no literal `252`) into `n_states` regimes via 1-D k-means (k-means++
  seeding, 20 iterations; deterministic from `key` via a derived NumPy seed). Cluster centers, sorted
  ascending, become `sigma` (or `s`); the cluster mean of `y / nominal_dt` becomes `mu`/`m`; each
  cluster's mean run length (steps, via run-length encoding) becomes the duration params through
  `spec.duration.init_from_mean_sojourn`; an order-1 empirical run-boundary transition matrix
  (Laplace-smoothed) becomes `transition.pi_rows`, broadcasting the same per-current-state row to every
  memory-tuple row at `order > 1` (the label sequence alone cannot identify true higher-order
  conditionals). Everything else (coefficients, covariate-conditioned transition logits, `nu`, ...) keeps
  its config baseline.
- `lattice_jitter(spec, key)`: the config's own `init`/lattice values, with every `mode == "free"` entry
  jittered by about 10% -- multiplicative (`exp(0.1 eps)`) for `positive`, the same shifted by `2` for
  `nu_gt_2` (`2 + (value - 2) exp(0.1 eps)`, since scaling `nu` itself can jitter it below its `> 2`
  floor), additive-proportional for `real`. `constraint="increasing"` and simplex-domain specs
  (`transition.pi_rows`, nonparametric-duration `logits`) are left untouched, since jitter could break
  the ordering the `increasing` bijector's inverse needs, or land outside the simplex.
- `random(data, spec, key)`: scale-free and clock-neutral via `s = sd(observations) / sqrt(nominal_dt)`
  (the ticket text omits `data` from this strategy's signature, but the formula needs it -- resolved as a
  ticket self-inconsistency, CLAUDE.md §10). `sigma ~ U[0.25, 4] s` (sorted), `mu ~ U[-1, 1] s`, mean
  sojourn `~ LogU[5, 250]` steps via `init_from_mean_sojourn`, no-covariate jump logits `~ N(0, 0.5)`
  (`transition.pi_rows` via `softmax(concat([0], u))`, `transition.a` directly), coefficient blocks
  (`emission.gamma`/`delta`, `duration.c`, `transition.b`) `~ N(0, 0.1)`.

`estimation/map.py`'s `fit(spec, data, cfg, seed) -> FitResult` builds the engine and state space from
`spec.engine` itself, runs `cfg.restarts` sequential `minimize` calls (deterministic order, each with its
own `key_for(seed, restart_idx)`), and keeps the best finite-objective run. If
`model.lattice.has_identifiability_guard(spec)` is false (no `constraint="increasing"` and no fully fixed
emission location/scale), the winner is relabelled by ascending `sigma`/`s`
(`model.lattice.post_sort_states`) with a warning, since the label ordering is otherwise arbitrary.

## Diagnostics and standard errors (E5-T04)

`estimation/report.py`'s `diagnose(fit, spec, space, engine, data) -> dict` computes the concept's
mandatory post-fit checks, each entry `{"value": ..., "warning": str | None}`:

- `regime_occupancy`: mean smoothed regime marginal; warns below `0.02` per state.
- `posterior_mean_duration` / `posterior_mean_duration_ratio`: mean Viterbi segment length per regime
  vs. `spec.duration.mean_sojourn(theta, nominal_dt) / nominal_dt`.
- `pit_hist` / `pit_ks_statistic` / `pit_ks_pvalue`: a 10-bin histogram and Kolmogorov-Smirnov test of
  the one-step-ahead PIT values against `Uniform(0, 1)` (`evaluation/metrics/predictive.py`, shared with
  E7-T01). `pit_values` needs the emission family's `cdf` (added alongside `log_prob`/`sample` on
  `EmissionFamily`); single-series only (`series=0` by default).
- `ljung_box`: `{lag: (Q, p_value)}` at lags 10 and 20 on the Rosenblatt residuals
  `Phi^{-1}(\text{PIT}_t)` (own implementation, `jax.scipy.special.gammainc` for the chi-square p-value
  -- no `scipy` runtime dependency).
- `ordering_check` / `label_switching_risk`: whether `sigma`/`s` is ascending, and whether that matters
  (it doesn't if `has_identifiability_guard(spec)`).
- `restart_dispersion`: relative gap between the best two restart objectives; warns above `1%`.
- `timings`: total and per-restart wall time, restart count.

`std_errors(objective, u_hat, manager) -> dict[str, Array] | None`: `H = hessian(objective.value)(u_hat)`
is the observed information (`objective.value` is already the negative log-posterior/-likelihood, so no
sign flip); `Cov_u = inv(H)`; the delta method (`Cov_theta = J Cov_u J^T`, `J = jacfwd(manager.unflatten)`)
gives one SE array per `theta` key, shaped like that parameter. Returns `None` (not an exception) when
`H`'s smallest eigenvalue is non-positive or non-finite.

## EM cross-check (`estimation/em.py`, E5-T06)

For the covariate-free GBM model with geometric or negative-binomial durations, `em_fit` runs Baum-Welch as an
independent check on MAP (it ignores priors, so compare with `use_priors: false`). With smoothed weights
$w_{t,i}$ and expected transition counts $\Xi = \sum_t \xi_t$:

$$
\hat m_i = \frac{\sum_t w_{t,i} y_t}{\sum_t w_{t,i}\Delta_t}, \quad
\hat\sigma_i^2 = \frac{\sum_t w_{t,i}(y_t - \hat m_i\Delta_t)^2/\Delta_t}{\sum_t w_{t,i}}, \quad
\hat\mu_i = \hat m_i + \tfrac12\hat\sigma_i^2,
$$

$\hat a_i = \text{leave}_i / (\text{leave}_i + \text{stay}_i)$ from the expected advance-or-exit and stay counts
of the regime's phase states, $\hat\lambda_i = -\log(1-\hat a_i)/\Delta$, and each jump row is the normalised
expected exit counts. The pair $(m, \sigma)$ is a bijective reparameterization of $(\mu, \sigma)$, so the
emission update is exact in one step. The log-likelihood is non-decreasing; covariates, Student-t, the explicit
engine, order above 1 and irregular grids raise `ConfigError`.

## Batches of sequences and initial-distribution groups

For $B$ independent sequences the objective is
$J(u) = -\sum_b \ell_b(\theta(u)) - \log p(\theta) - \log|\det J|$ (`BatchedEngine.loglik` sums
`batched_loglik`, E12-T01b). The pooled data-driven inits (`rolling_kmeans`) compute the rolling scale inside each
sequence, so a window never straddles a boundary, and count sojourn runs and jump transitions per sequence.

With `initial_state: free` the initial law is a free parameter `initial.logits` over the $S_1$ first-phase states.
`initial_groups: per_group` gives it one row per group, $\rho^{(g)}$, and sequence $b$ starts from the row of its
group; `FitResult.diagnostics["initial_groups"]` lists the group names (sorted, the order of
`SequenceBatch.group_names`) and `Project.score` maps a new batch's groups to rows, with the mean of the rows as
fallback (and a warning) for a group the fit has not seen. A `B = 1` batch without groups is fitted as the single
sequence it holds, so its `uparams` equal the single-sequence fit bit for bit.

## E-step accumulators (expanded engine)

`engine.estep.expanded_estep` computes everything EM and VB need that is not a family sufficient statistic from one
forward pass and one fused backward / accumulation scan. With the pairwise posterior of a step,

$$\xi_t(s,s') = \exp\big(\log\alpha_{t-1}(s) + \log A_t(s,s') + \ell_t(s') + \log\beta_t(s') - \ell\big),$$

the `(S, S)` matrix of the current step is reduced at once with the static index arrays of the state space and added
to the carry; `(T, S, S)` is never formed (the program holds `alpha (T, S)`, the per-regime emissions `(T, N)` and
the regime marginals). Over the valid transitions of every sequence:

| Statistic | Definition |
|---|---|
| `stay[k]` | `sum xi_t(s, s)` over the states `s` of duration-rate row `k` |
| `advance[k]` | the other mass leaving `s`: the phase advance and the exits of the last phase |
| `occupancy[k]` | `sum gamma_{t-1}(s)` over the same states, so `stay + advance = occupancy` |
| `jump[r, j]` | `sum xi_t(s, entry(r, j))` over the last-phase states of jump-matrix row `r` |
| `gamma1[g]` | the first-step posterior summed over the sequences of initial group `g` |
| `gamma_regime` | regime marginals `(B, T, N)`, 0 on padding |

Tempered EM (`beta_temper = beta < 1`) runs the pass on `beta * ell`; `loglik` is always the untempered value.

## Symbols and code names

| Math | Code |
|---|---|
| `\hat u` | `FitResult.uparams` |
| objective | `estimation.objective.make_objective`, `value_and_grad` |
| fit | `hsmmkit.estimation.fit(spec, data, cfg, seed)` |
| restarts | `FitResult.restarts` (one row per start) |
| standard errors | `estimation.report.std_errors` |
| EM | `estimation.em.em_fit`, `GBMFamily.sufficient_stats` |
| E-step counts | `engine.estep.expanded_estep`, `ExpandedEStats` |
