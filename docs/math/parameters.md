# Parameters: bijectors and priors

Unconstrained parameters `u` map to constrained parameters `\theta` through a domain-specific
bijector `g`, with the objective (E5-T01) correcting for the change of variables via the
log-determinant-Jacobian:

$$
\theta = g(u), \qquad \log p(\theta) = \log p_{\text{prior}}(\theta) + \log \left| \det \frac{\partial \theta}{\partial u} \right|
$$

## Bijector table

| Domain | `forward(u)` | Notes |
|---|---|---|
| `real` | `u` | identity |
| `positive` | `softplus(u)` (default) or `exp(u)` (`variant="exp"`) | |
| `interval[lo, hi]` | `lo + (hi - lo) \cdot \text{sigmoid}(u)` | |
| `simplex_ref` | `\text{softmax}(\text{concat}([0], u))` | reference category first; `u` has `n-1` free entries, returns `n` |
| `nu_gt_2` | `2 + \text{softplus}(u)` | Student-t degrees of freedom |
| `pmf` | same construction as `simplex_ref` | nonparametric duration pmf over `1..D_max` |
| `increasing` | `\theta_0 = g(u_0)`, `\theta_i = \theta_{i-1} + \text{softplus}(u_i)` | ordering constraint; `g` is the inner bijector for the base domain (e.g. `positive` for `\sigma`) |

Log-det-Jacobians are analytic sums of scalar terms for every domain except `simplex_ref`/`pmf`,
where they are computed via `jax.jacfwd` + `jnp.linalg.slogdet` on the `n-1` free coordinates
(cheap: called once per objective evaluation). Every bijector's analytic (or autodiff-based)
`log_det_jac` is tested against a full autodiff `slogdet` reference to `\le 10^{-8}`
(`tests/property/test_e1_t03a_bijectors.py`).

## Priors

Evaluated on `\theta`, returning a summed log-density (`model/params/priors.py`):

- `NormalPrior(mean=0, sd)`: `-0.5((\theta - m)/sd)^2 - \log(sd) - 0.5 \log(2\pi)`
- `LogNormalPrior(mean_log, sd)`: the above on `\log \theta`, plus `-\log \theta` (the Jacobian of
  the log transform)
- `DirichletPrior(alpha)`: `\sum_i (\alpha - 1) \log p_i` per jump-matrix row, plus the Dirichlet
  normalizing constant `\log\Gamma(n\alpha) - n\log\Gamma(\alpha)`

Default coefficient prior: `NormalPrior(0, coef_prior_sd=1.0)`.

## Modes (`ParamManager`)

`ParamManager` maps the flat unconstrained optimizer vector `u \in \mathbb{R}^p` to the constrained
parameter pytree `\theta` (a `dict[str, Array]`), according to a per-parameter, per-state (or
uniform) estimation **mode**:

| Mode | Behaviour |
|---|---|
| `free` | Own coordinate(s) in `u`; `\theta` = `domain_bijector.forward(u_slice)`. |
| `fixed` | No coordinate in `u`; `\theta` = the spec's `value`, unchanged. |
| `tied` | One shared coordinate group per `tie_group` string; every member (within or across specs) reads the same `u` slice and is set to the identical `domain_bijector.forward(...)` — changing it changes every group member. |
| `bounded` | Like `free`, but requires `domain="interval"` and `bounds` (an `IntervalBijector(lo, hi)`). |
| `prior` | Like `free`, plus the spec's `prior` is added to `ParamManager.log_prior(theta)` for those entries. |

A spec's `mode` can be a single string (applied uniformly across its leading "state" axis) or a
tuple with one entry per state, mixing modes freely within one parameter (e.g. one state's rate
fixed while the others are free).

`constraint="increasing"` replaces the per-state independent bijector application with a single
`IncreasingBijector` spanning the whole state axis (`shape = (N,)`, one uniform mode, never
`fixed` or `tied`) — the array of `\theta` values across states is one ordering-constrained
transform of the corresponding `N` `u` coordinates, not `N` independent scalar transforms.

`n_free` (`p`) is the total number of `u` coordinates consumed: `cell_size` per `free`/`bounded`/
`prior` entry (`cell_size - 1` for `simplex_ref`/`pmf` cells, since those bijectors have one fewer
free coordinate than output dimension), `cell_size` once per distinct `tie_group`, `0` for `fixed`,
and `N` for an `increasing` spec's whole state axis.

## Identifiability helpers (`model/lattice.py`, E3-T06)

Regime labels are only identified up to permutation; three helpers guard against label switching:

- **Lattice.** `lattice.build({"mu": [...], "sigma": [...]})` takes the Cartesian product of the grids
  (first grid varies slowest): `n_states = \prod |grid|`, with `to_param_cfgs()` giving `mode: fixed` configs
  that override `model.emission.params`. `validate_lattice(cfg)` checks `model.n_states` against the product.
- **Ordering.** `apply_ordering(specs, "emission.sigma")` sets `constraint: increasing` (single series only).
- **Post-sorting.** `post_sort_states(theta, spec, by="emission.sigma")` relabels a fitted `\theta` so `by` is
  ascending and returns `(theta_sorted, perm)` with `perm[new] = old`. Emission and duration blocks are
  gathered along the state axis (pair-conditioned duration rates by the `(previous, current)` pair);
  transition blocks along memory-tuple rows and target columns. Transition logits (`a`, `b`) are defined
  relative to a reference target (the smallest `j \ne i`), which relabelling changes, so they are re-derived
  by subtracting the new reference's logit; `jump_probs` is invariant up to the permutation.
- `has_identifiability_guard(spec)` is true when some spec is `increasing`, or when every location/scale
  emission spec (`mu`, `sigma`, `m`, `s`) is `fixed`.
